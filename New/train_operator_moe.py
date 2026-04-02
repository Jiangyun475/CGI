#!/usr/bin/env python3
"""
OperatorMoE: Spectrum-Conditioned Mixture-of-Experts for CGI Prediction
========================================================================

核心思想（融合两种策略）：
  DrugOperatorNet 提出"药物是扰动算子"，产出可解释的交互谱 spectrum [B, r]。
  MoE+Target 提出用混合专家路由提升分类能力，并用路由权重条件化原子注意力。

  本模型将两者统一：
    交互谱 spectrum → MoE 路由器 → route_weights [B, K]

  即：路由信号从"两个模糊向量的拼接"升级为"有物理意义的药效团-基因交互指纹"。
  每个专家对应一类交互模式，路由具有明确的生物学语义。

完整前向流程：
  ① Gene → GeneMultiHeadReader → h_g_modes [B, r, H], h_g_global [B, H]
  ② Drug → GIN × 3 → atom_h [N, H]
           → PharmacophoreExtractor → pharma_emb [B, r, H]
  ③ PerturbationOperator:
       spectrum [B, r] = σ * (V · h_g_modes)   ← 可解释交互谱
       delta_h  [B, H] = Σ spectrum_k * U_k     ← 扰动向量
  ④ spectrum → Router → route_weights [B, K]    ← 谱驱动路由（核心融合）
  ⑤ route_weights @ expert_queries → atom query
     scatter_softmax(atom_h) → target_pool [B, H] ← 路由条件化靶向池化
  ⑥ features = [h_g_global, delta_h, target_pool]
     expert_k(features) → expert_logits [B, K]
     final = Σ route_weights_k * expert_logits_k

可解释性链条（论文 Figure）：
  原子权重 → 药效团嵌入 → 交互谱 → 专家路由 → 预测

消融控制（--ablation）：
  full         : 完整 OperatorMoE（默认）
  no_spectrum  : 路由输入改回 [V_g, V_c_perp]，即纯 MoE+Target
  no_moe       : 去掉 MoE，spectrum → delta_h → 直接 MLP 分类，即纯 DrugOperatorNet
  no_target    : 去掉靶向池化，仅用 [h_g_global, delta_h] 分类
"""

import argparse
import itertools
import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ================================================================
# 0. 工具函数
# ================================================================

_KMER_VOCAB = {}
for _i, _combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(_combo)] = _i
_KMER_VOCAB['NNNNNN'] = 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_kmer_sequence(sequence: str, k: int = 6, max_len: int = 3000) -> list:
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if any(c not in 'ACGT' for c in kmer):
            kmer = 'N' * k
        kmers.append(_KMER_VOCAB.get(kmer, 0))
    if len(kmers) > max_len:
        kmers = kmers[:max_len]
    else:
        kmers += [0] * (max_len - len(kmers))
    return kmers


def scatter_softmax(scores, batch_idx):
    """图内原子级 softmax，纯 PyTorch 实现。"""
    scores = scores.float()
    max_scores = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
    exp_scores = torch.exp(scores - max_scores[batch_idx])
    exp_sum = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)


def scatter_add(src, batch_idx, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


# ================================================================
# 1. 数据集（与 V2 一致，支持 gene_max_len 缓存）
# ================================================================

class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train', gene_max_len=3000):
        import pickle
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt')
        self.smiles_to_graph = self.data['smiles_to_graph']
        self.graph_indices = [self.data['graph_indices'][i] for i in self.indices]
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]

        suffix = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}{suffix}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 (len={gene_max_len})...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len)
                 for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
        }


def collate_fn(batch):
    all_x, all_edge_index, all_edge_attr, num_nodes_list = [], [], [], []
    offset = 0
    for item in batch:
        g = item['graph']
        n = g['x'].shape[0]
        all_x.append(g['x'])
        if g['edge_index'].shape[1] > 0:
            all_edge_index.append(g['edge_index'] + offset)
            all_edge_attr.append(g['edge_attr'])
        num_nodes_list.append(n)
        offset += n
    return {
        'x':              torch.cat(all_x, dim=0),
        'edge_index':     torch.cat(all_edge_index, dim=1) if all_edge_index
                          else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr':      torch.cat(all_edge_attr, dim=0) if all_edge_attr
                          else torch.zeros(0, 4),
        'num_nodes_list': num_nodes_list,
        'gene_ids':       torch.stack([b['gene_ids'] for b in batch]),
        'label':          torch.stack([b['label']    for b in batch]),
    }


# ================================================================
# 2. 基因编码器：GeneMultiHeadReader（来自 V2）
# ================================================================

class GeneMultiHeadReader(nn.Module):
    """
    r 个注意力头各自读取基因序列的不同方面。
    每个头对应一个药效团模式，实现模式对齐耦合。

    gene_ids [B, L]
      → embedding [B, L, H]
      → CNN×4 [B, L', H]
      → LayerNorm
      → attn_queries [r, H] → scores [B, r, L'] → softmax → attn [B, r, L']
      → weighted sum → h_g_modes [B, r, H]
      → mean → h_g_global [B, H]
    """
    def __init__(self, vocab_size=4097, hidden_dim=128, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        H = hidden_dim

        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)

        kernel_sizes = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.seq_norm = nn.LayerNorm(H)

        # 零初始化：warmup 期间避免随机 attention 破坏梯度
        self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))

        self.out_norm = nn.LayerNorm(H)
        self.dropout  = dropout

    def forward(self, gene_ids):
        """
        Returns:
          h_g_modes  [B, r, H]  — r 个模式专属基因视角
          h_g_global [B, H]     — 全局基因表示
          attn       [B, r, L'] — 注意力权重（可解释性用）
        """
        x = self.embedding(gene_ids).transpose(1, 2)             # [B, H, L]
        x = torch.cat([conv(x) for conv in self.convs], dim=1)   # [B, H, L']
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                                    # [B, L', H]
        x = self.seq_norm(x)

        scores = torch.einsum('blh,rh->brl', x, self.attn_queries) / math.sqrt(x.size(-1))
        attn   = F.softmax(scores, dim=-1)                        # [B, r, L']
        h_g_modes = torch.einsum('brl,blh->brh', attn, x)        # [B, r, H]
        h_g_modes = self.out_norm(h_g_modes)

        h_g_global = h_g_modes.mean(dim=1)                       # [B, H]
        return h_g_modes, h_g_global, attn


# ================================================================
# 3. 化学编码器：GIN × 3（来自 V2）
# ================================================================

class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge_proj = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        msg = F.relu(x[row] + self.edge_proj(edge_attr))
        return self.mlp(x + torch.zeros_like(x).index_add_(0, col, msg))


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.atom_embed = nn.Sequential(
            nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        return x  # [N_total, H]


# ================================================================
# 4. 药效团提取（来自 V2）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """r 个可学习 query × 原子交叉注意力 → r 个药效团嵌入。"""
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.queries  = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs):
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)                               # [N, H]
        V = self.val_proj(atom_h)                               # [N, H]
        scores_all = (K @ self.queries.T) / math.sqrt(d)       # [N, r]

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        # 同时保存 atom attention weights 供可解释性分析
        atom_attn = torch.zeros(num_graphs, self.num_slots, atom_h.shape[0] // num_graphs,
                                device=atom_h.device, dtype=atom_h.dtype) \
            if False else None  # 训练时不存，eval 时由外部控制

        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all  # [B, r, H], [N, r]


# ================================================================
# 5. 扰动算子（来自 V2，模式对齐耦合）
# ================================================================

class PerturbationOperator(nn.Module):
    """
    T = I + Σ_k σ_k * u_k ⊗ v_k^T
    coupling_k = v_k · h_g_modes_k  （模式对齐，V2设计）
    spectrum_k = σ_k * coupling_k
    delta_h = Σ_k spectrum_k * u_k
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_u     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.to_v     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.to_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.Tanh())

    def forward(self, pharma_emb, h_g_modes):
        """
        pharma_emb: [B, r, H]
        h_g_modes:  [B, r, H]
        Returns: delta_h [B,H], spectrum [B,r], sigma [B,r], U [B,r,H]
        """
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)  # [B, r, H]
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)  # [B, r, H]
        sigma = self.to_sigma(pharma_emb).squeeze(-1)        # [B, r]

        coupling = (V * h_g_modes).sum(-1)                   # [B, r]
        spectrum = sigma * coupling                           # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)  # [B, H]

        return delta_h, spectrum, sigma, U


# ================================================================
# 6. 完整融合模型：OperatorMoE
# ================================================================

class OperatorMoE(nn.Module):
    """
    融合架构：谱驱动路由的混合专家算子网络。

    --ablation 控制消融：
      full        : 完整 OperatorMoE（spectrum → router → MoE + target_pool）
      no_spectrum : 路由输入改为 [V_g, V_c_perp]（= 纯 MoE+Target 逻辑）
      no_moe      : 去掉 MoE，spectrum+delta_h → 直接 MLP（= 纯 DrugOperatorNet）
      no_target   : 去掉靶向池化（features 中不含 target_pool）
    """
    def __init__(self, hidden_dim=128, dropout=0.3,
                 operator_rank=8, num_experts=4, ablation='full'):
        super().__init__()
        self.ablation     = ablation
        self.operator_rank = operator_rank
        self.num_experts  = num_experts
        H = hidden_dim
        r = operator_rank
        K = num_experts

        # --- 基因编码器（V2：多头读取器）---
        self.gene_enc = GeneMultiHeadReader(
            hidden_dim=H, num_heads=r, dropout=dropout)

        # --- 化学编码器 ---
        self.atom_enc = AtomEncoder(hidden_dim=H, dropout=dropout)

        # --- 药效团提取 + 算子 ---
        self.pharma_ext = PharmacophoreExtractor(H, r)
        self.perturb_op = PerturbationOperator(H)

        # --- 路由器 ---
        if ablation == 'no_spectrum':
            # 消融：用 [V_g, V_c_perp] 路由（MoE+Target 原始方式）
            # 需要 global readout 把 sum/mean pool 压成 V_c
            self.global_readout = nn.Sequential(
                nn.Linear(H * 2, H * 2), nn.BatchNorm1d(H * 2), nn.ReLU(),
                nn.Linear(H * 2, H))
            router_in = H * 2
        else:
            # 谱驱动路由：spectrum [B, r] → [B, K]
            router_in = r

        if ablation != 'no_moe':
            self.router = nn.Sequential(
                nn.Linear(router_in, H),
                nn.LayerNorm(H),
                nn.ReLU(),
                nn.Linear(H, K),
                nn.Softmax(dim=-1))

            # 靶向池化的专家查询向量
            self.expert_queries = nn.Parameter(torch.randn(K, H) * 0.01)

        # --- 分类头 ---
        if ablation == 'no_moe':
            # 纯算子：[h_g_global, delta_h] → MLP
            self.classifier = nn.Sequential(
                nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(H, 1))
        else:
            # 特征维度：h_g_global + delta_h + target_pool（若无 target 则不含）
            feat_dim = H * 3 if ablation != 'no_target' else H * 2
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, H), nn.LayerNorm(H), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(H, 1))
                for _ in range(K)])

        self.dropout_p = dropout

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(torch.arange(B, device=device), num_nodes_t)

        # ① 基因编码
        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)
        # h_g_modes [B, r, H], h_g_global [B, H], gene_attn [B, r, L']

        # ② 化学原子编码
        atom_h = self.atom_enc(x, edge_index, edge_attr)  # [N, H]

        # ③ 药效团提取 + 算子
        pharma_emb, atom_scores = self.pharma_ext(atom_h, batch_idx, B)
        # pharma_emb [B, r, H], atom_scores [N, r]
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)
        # delta_h [B, H], spectrum [B, r]

        # ④ 路由（核心融合）
        if self.ablation == 'no_moe':
            # 直接分类，无需路由
            features = torch.cat([h_g_global, delta_h], dim=-1)
            logits   = self.classifier(features).squeeze(-1)
            return logits, spectrum, sigma, U, gene_attn, None

        if self.ablation == 'no_spectrum':
            # 消融：传统 [V_g, V_c_perp] 路由
            sum_pool  = scatter_add(atom_h, batch_idx, B)
            mean_pool = sum_pool / num_nodes_t.float().unsqueeze(1).clamp(min=1)
            h_c       = self.global_readout(torch.cat([sum_pool, mean_pool], dim=-1))
            V_g       = F.normalize(h_g_global, dim=-1)
            V_c       = F.normalize(h_c, dim=-1)
            V_c_perp  = V_c - (V_c * V_g).sum(-1, keepdim=True) * V_g
            router_in = torch.cat([V_g, V_c_perp], dim=-1)
        else:
            # 谱驱动路由（完整 OperatorMoE）
            router_in = spectrum  # [B, r]

        route_weights = self.router(router_in)  # [B, K]

        # ⑤ 路由条件化靶向池化
        if self.ablation != 'no_target':
            q           = route_weights @ self.expert_queries  # [B, H]
            scores_atom = (atom_h * q[batch_idx]).sum(-1) / math.sqrt(atom_h.size(-1))
            alpha       = scatter_softmax(scores_atom, batch_idx)
            target_pool = scatter_add(atom_h * alpha.unsqueeze(-1), batch_idx, B)
            features    = torch.cat([h_g_global, delta_h, target_pool], dim=-1)
        else:
            features    = torch.cat([h_g_global, delta_h], dim=-1)

        # ⑥ MoE 分类
        expert_logits = torch.stack(
            [exp(features).squeeze(-1) for exp in self.experts], dim=1)  # [B, K]
        logits = (route_weights * expert_logits).sum(dim=-1)             # [B]

        return logits, spectrum, sigma, U, gene_attn, route_weights


# ================================================================
# 7. 正则化损失
# ================================================================

def compute_losses(logits, labels, spectrum, sigma, U,
                   route_weights, criterion, args):
    loss_bce = criterion(logits, labels)

    # 算子正则：σ 稀疏 + U 列正交
    loss_sparse = sigma.abs().mean()
    U_n   = F.normalize(U, dim=-1)
    gram  = torch.bmm(U_n, U_n.transpose(1, 2))
    eye   = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
    loss_ortho = (gram - eye).pow(2).mean()
    loss_reg   = args.lam_sparse * loss_sparse + args.lam_ortho_modes * loss_ortho

    # MoE 负载均衡（仅 full / no_spectrum / no_target 模式）
    loss_lb = torch.tensor(0.0, device=logits.device)
    if route_weights is not None and args.ablation != 'no_moe':
        mean_route = route_weights.mean(dim=0)
        uniform    = torch.ones_like(mean_route) / args.num_experts
        loss_macro = F.mse_loss(mean_route, uniform)
        loss_entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(-1).mean()
        loss_lb    = loss_macro + 0.1 * loss_entropy

    total = loss_bce + loss_reg + args.lam_balance * loss_lb
    return total, loss_bce, loss_reg, loss_lb


# ================================================================
# 8. LR Warmup 工具
# ================================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ================================================================
# 9. 训练主循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds = OptimizedGraphDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    model = OperatorMoE(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        operator_rank=args.operator_rank,
        num_experts=args.num_experts,
        ablation=args.ablation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_operator_moe/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"{args.ablation}_r{args.operator_rank}_k{args.num_experts}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0
    base_lr = args.lr

    print(f"\n{'='*72}")
    print(f"  OperatorMoE | ablation={args.ablation}")
    print(f"  operator_rank={args.operator_rank} | num_experts={args.num_experts}")
    print(f"  gene_max_len={args.gene_max_len} | warmup={args.warmup_epochs}ep")
    print(f"  params={n_params:,} | device={args.device} | fold={args.fold}")
    print(f"{'='*72}\n")

    for epoch in range(args.epochs):
        # LR warmup（attn_queries 零初始化，需要预热）
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        total_loss = total_bce = total_reg = total_lb = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _, route_weights = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss, loss_bce, loss_reg, loss_lb = compute_losses(
                    logits, labels, spectrum, sigma, U,
                    route_weights, criterion, args)

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += loss_bce.item()
            total_reg  += loss_reg.item()
            total_lb   += loss_lb.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        all_spectra = []

        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index  = batch['edge_index'].to(device)
                edge_attr   = batch['edge_attr'].to(device)
                gene_ids    = batch['gene_ids'].to(device)

                with autocast(enabled=args.use_amp):
                    logits, spectrum, _, _, _, _ = model(
                        gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                if args.save_spectrum and spectrum is not None:
                    all_spectra.append(spectrum.cpu())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        if epoch >= args.warmup_epochs:
            scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f} LB:{total_lb/n:.4f}) | "
              f"VAL_AUC:{auroc:.4f} PRC:{auprc:.4f} F1:{f1:.4f}")

        if auroc > best_auroc:
            best_auroc   = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
            if args.save_spectrum and all_spectra:
                torch.save({
                    'spectrum': torch.cat(all_spectra, dim=0),
                    'preds':    np.array(all_preds),
                    'labels':   np.array(all_labels),
                }, save_dir / f'spectrum_{model_name}')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


# ================================================================
# 10. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OperatorMoE: 谱驱动路由混合专家算子网络')

    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--seed',          type=int, default=42)

    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int, default=10)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='LR warmup（attn_queries 零初始化需要预热）')

    parser.add_argument('--gene_max_len',  type=int, default=3000,
                        help='基因 k-mer 序列长度')
    parser.add_argument('--operator_rank', type=int, default=8,
                        help='算子秩 r = 药效团数 = 基因读取头数')
    parser.add_argument('--num_experts',   type=int, default=4,
                        help='MoE 专家数 K（full/no_spectrum/no_target 模式下有效）')

    parser.add_argument('--lam_sparse',      type=float, default=0.01,
                        help='σ 稀疏正则系数')
    parser.add_argument('--lam_ortho_modes', type=float, default=0.1,
                        help='U 列正交正则系数（建议 0.1，确保模式独立）')
    parser.add_argument('--lam_balance',     type=float, default=0.1,
                        help='MoE 负载均衡系数')

    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_spectrum', 'no_moe', 'no_target'],
                        help=(
                            'full:        完整 OperatorMoE（spectrum→router→MoE+target）\n'
                            'no_spectrum: 路由改为 [V_g,V_c_perp]（纯 MoE+Target 逻辑）\n'
                            'no_moe:      去掉 MoE，spectrum+delta_h → MLP（纯 Operator）\n'
                            'no_target:   去掉靶向池化'))

    parser.add_argument('--save_spectrum', action='store_true',
                        help='保存验证集交互谱（供 analyze_spectrum.py 分析）')
    parser.add_argument('--run_tag',       type=str, default='')

    train(parser.parse_args())
