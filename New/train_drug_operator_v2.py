#!/usr/bin/env python3
"""
DrugOperatorNet V2: 基因多头读取器 (Gene Multi-Head Reader)
============================================================

V1 → V2 的核心改动：
  [问题] GeneEncoderV1 输出单一全局向量 h_g [B, H]。
         coupling_k = v_k^T · h_g 中所有模式共享同一个粗糙表示。
         算子机制的精度受限于基因表示的信息量。

  [方案] GeneMultiHeadReader：r 个可学习查询向量对基因序列做多头注意力池化。
         每个药效团模式 k 拥有专属的基因视角 h_g_modes[:,k,:] [B, H]。
         coupling_k = v_k^T · h_g_modes[:,k,:]  ← 模式对齐耦合

  [效果]
    - 更丰富的基因-药物耦合：r 个独立读取头，各自专注于不同功能区
    - 可解释性增强：基因注意力权重 [B, r, L] 可视化哪些序列位置驱动哪种模式
    - 序列长度升级：1000 → 3000 k-mer（覆盖更多调控/功能信息）
    - 初始化稳定：attn_queries 零初始化 + 5 epoch warmup（沿用 v3 经验）

数学形式（V2）：
  h_seq = CNN(embed(gene_ids))              [B, L', H]
  score_k = h_seq @ q_k / √H              [B, L']   ← q_k ∈ R^H 可学习
  α_k = softmax(score_k)                  [B, L']
  h_g_modes_k = Σ_l α_k[l] · h_seq[l]   [B, H]
  h_g_global = mean_k(h_g_modes_k)        [B, H]

  coupling_k = v_k^T · h_g_modes_k        (比 V1 更精准的模式耦合)
  spectrum_k = σ_k · coupling_k
  Δh = Σ_k spectrum_k · u_k
  classify([h_g_global, Δh])
"""

import os
import argparse
import itertools
import random
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ================================================================
# scatter_softmax（纯 PyTorch，无需 torch_geometric）
# ================================================================

def scatter_softmax(scores, batch_idx):
    """
    图内原子级 softmax（纯 PyTorch）。
    一个 batch 内多个分子的原子被打包为 [N_total]，
    batch_idx 标记每个原子所属分子编号（如 [0,0,1,1,1,2]）。
    普通 softmax 会跨分子归一化，scatter_softmax 在每个分子内独立归一化。
    实现：减去分子内最大值（数值稳定）→ exp → 分子内求和 → 除以和。
    """
    scores = scores.float()
    max_scores = torch.zeros(batch_idx.max().item() + 1,
                             device=scores.device).index_reduce_(
                                 0, batch_idx, scores, 'amax', include_self=True)
    exp_scores = torch.exp(scores - max_scores[batch_idx])
    exp_sum = torch.zeros(batch_idx.max().item() + 1,
                          device=scores.device).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)


# ================================================================
# 0. 工具函数
# ================================================================

_KMER_VOCAB = {}
for i, combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(combo)] = i
_KMER_VOCAB['NNNNNN'] = 0

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


# ================================================================
# 1. 数据集
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

        # 缓存文件名含长度，避免与 len=1000 缓存混淆
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
            'graph': self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label': self.labels[idx],
        }

def optimized_collate_fn(batch):
    all_x, all_edge_index, all_edge_attr, num_nodes_list = [], [], [], []
    offset = 0
    for item in batch:
        graph = item['graph']
        num_nodes = graph['x'].shape[0]
        all_x.append(graph['x'])
        if graph['edge_index'].shape[1] > 0:
            all_edge_index.append(graph['edge_index'] + offset)
            all_edge_attr.append(graph['edge_attr'])
        num_nodes_list.append(num_nodes)
        offset += num_nodes
    x = torch.cat(all_x, dim=0)
    edge_index = (torch.cat(all_edge_index, dim=1)
                  if all_edge_index else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = (torch.cat(all_edge_attr, dim=0)
                 if all_edge_attr else torch.zeros(0, 4))
    return {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
        'num_nodes_list': num_nodes_list,
        'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


# ================================================================
# 2. 编码器
# ================================================================

class GeneMultiHeadReader(nn.Module):
    """
    基因多头读取器：r 个注意力头各自读取序列的不同方面。

    与 GeneEncoderV1 的区别：
      - 不做 TopK 池化（不丢弃位置信息）
      - r 个可学习查询向量 → r 个软注意力权重 → r 个专属基因视角
      - 零初始化 + warmup（沿用 train_moe_v3 稳定性经验）
      - 返回 h_g_modes [B, r, H] + h_g_global [B, H]

    前向过程：
      gene_ids [B, L]
        → embedding [B, L, H]
        → CNN×4  [B, L', H]       （保留序列维度）
        → attn_queries [r, H] × seq → scores [B, r, L']
        → softmax → attn [B, r, L']
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
        # 序列特征归一化（在位置维度上做 LayerNorm）
        self.seq_norm = nn.LayerNorm(H)

        # r 个注意力查询向量，零初始化（稳定训练）
        self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))

        # 输出归一化
        self.out_norm = nn.LayerNorm(H)
        self.dropout = dropout

    def forward(self, gene_ids):
        """
        Returns:
          h_g_modes  [B, r, H]  — r 个模式专属基因视角
          h_g_global [B, H]     — 全局基因表示（模式平均）
        """
        B = gene_ids.size(0)

        # k-mer 嵌入 → CNN 特征提取
        x = self.embedding(gene_ids).transpose(1, 2)          # [B, H, L]
        x = torch.cat([conv(x) for conv in self.convs], dim=1)  # [B, H, L']
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                                  # [B, L', H]
        x = self.seq_norm(x)                                   # 位置归一化

        # 多头注意力池化
        # scores: [B, r, L'] = x @ queries^T / √H
        scores = torch.einsum('blh,rh->brl', x, self.attn_queries) / math.sqrt(x.size(-1))
        attn = F.softmax(scores, dim=-1)                       # [B, r, L']

        # 加权聚合 → h_g_modes [B, r, H]
        h_g_modes = torch.einsum('brl,blh->brh', attn, x)
        h_g_modes = self.out_norm(h_g_modes)

        # 全局表示：各头平均
        h_g_global = h_g_modes.mean(dim=1)                    # [B, H]

        return h_g_modes, h_g_global, attn                    # attn 保留用于可解释性


class GINLayer(nn.Module):
    """
    GIN 单层：h_v ← MLP(h_v + Σ_{u∈N(v)} ReLU(h_u + W_e·e_{uv}))
      row=源节点, col=目标节点；边特征（键类型等4维）融入消息。
      BN 在 MLP 中间层，对变长图 batch 保持训练稳定。
    """
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge_proj = nn.Linear(edge_dim, dim)  # 4-dim 键特征 → hidden_dim

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index              # 源/目标原子索引
        edge_emb = self.edge_proj(edge_attr)
        msg = F.relu(x[row] + edge_emb)   # 消息：源原子嵌入 + 键嵌入
        neighbor = torch.zeros_like(x).index_add_(0, col, msg)  # 聚合到目标原子
        return self.mlp(x + neighbor)     # 更新：自身 + 邻居聚合，过 MLP


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.atom_embed = nn.Sequential(
            nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        return x  # [N_total, H]


# ================================================================
# 3. 药效团提取 + 算子（V2 改动：coupling 使用模式对齐视角）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """r 个药效团 slot，原子级交叉注意力（与 V1 相同）"""
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.queries  = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs):
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)                              # [N_total, H]
        V = self.val_proj(atom_h)                              # [N_total, H]
        scores_all = (K @ self.queries.T) / math.sqrt(d)      # [N_total, r]

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma)   # [B, r, H]


class PerturbationOperatorV2(nn.Module):
    """
    V2 改动：coupling 使用模式对齐的基因视角。

    V1: coupling_k = v_k^T · h_g          (所有模式共享单一 h_g)
    V2: coupling_k = v_k^T · h_g_modes_k  (每个模式有专属基因视角)

    这使得每个药效团模式"读取"基因序列中与自己最相关的区域。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.to_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.to_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh())

    def forward(self, pharma_emb, h_g_modes):
        """
        pharma_emb: [B, r, H]  — 药效团嵌入
        h_g_modes:  [B, r, H]  — 模式对齐的基因视角（V2 新增）

        Returns:
          delta_h:  [B, H]
          spectrum: [B, r]
          sigma:    [B, r]
          U:        [B, r, H]
        """
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)   # [B, r, H]
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)   # [B, r, H]
        sigma = self.to_sigma(pharma_emb).squeeze(-1)         # [B, r]

        # V2 核心改动：模式 k 与对应基因视角 k 做内积
        coupling = (V * h_g_modes).sum(-1)                    # [B, r]

        spectrum = sigma * coupling                            # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)   # [B, H]

        return delta_h, spectrum, sigma, U


# ================================================================
# 4. 完整模型
# ================================================================

class DrugOperatorNetV2(nn.Module):
    """
    DrugOperatorNet V2：GeneMultiHeadReader + 模式对齐耦合。

    --interaction_type 控制消融对比：
      operator     — V2 完整方法（多头基因读取 + 模式对齐算子）
      concat       — h_c ⊕ h_g_global → MLP
      ortho_concat — V_c⊥ ⊕ V_g_global → MLP
      hadamard     — h_c ⊙ h_g_global → MLP
    """
    def __init__(self, hidden_dim=128, dropout=0.3,
                 operator_rank=8, interaction_type='operator'):
        super().__init__()
        self.interaction_type = interaction_type
        r = operator_rank

        # 基因编码器：多头读取器，头数 = 算子秩
        self.gene_enc = GeneMultiHeadReader(
            hidden_dim=hidden_dim, num_heads=r, dropout=dropout)

        # 化学原子编码器
        self.atom_enc = AtomEncoder(hidden_dim=hidden_dim, dropout=dropout)

        if interaction_type == 'operator':
            self.pharma_ext = PharmacophoreExtractor(hidden_dim, r)
            self.perturb_op = PerturbationOperatorV2(hidden_dim)
            clf_in = hidden_dim * 2   # [h_g_global, Δh]
        else:
            self.drug_readout = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim))
            clf_in = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def _global_pool(self, atom_h, batch_idx, num_graphs):
        d = atom_h.shape[1]
        cnt = torch.zeros(num_graphs, device=atom_h.device).index_add_(
            0, batch_idx, torch.ones(atom_h.shape[0], device=atom_h.device))
        s = torch.zeros(num_graphs, d, device=atom_h.device,
                        dtype=atom_h.dtype).index_add_(0, batch_idx, atom_h)
        return self.drug_readout(torch.cat([s, s / cnt.unsqueeze(1).clamp(min=1)], -1))

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_tensor = torch.tensor(num_nodes_list, device=device)
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), num_nodes_tensor)

        # 基因编码：多头读取
        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)
        # h_g_modes [B, r, H], h_g_global [B, H], gene_attn [B, r, L']

        # 化学原子编码
        atom_h = self.atom_enc(x, edge_index, edge_attr)      # [N_total, H]

        delta_h = spectrum = sigma = U = None

        if self.interaction_type == 'operator':
            pharma = self.pharma_ext(atom_h, batch_idx, B)    # [B, r, H]
            delta_h, spectrum, sigma, U = self.perturb_op(pharma, h_g_modes)
            features = torch.cat([h_g_global, delta_h], dim=-1)

        elif self.interaction_type == 'concat':
            h_c = self._global_pool(atom_h, batch_idx, B)
            features = torch.cat([h_g_global, h_c], dim=-1)

        elif self.interaction_type == 'ortho_concat':
            h_c = self._global_pool(atom_h, batch_idx, B)
            V_g = F.normalize(h_g_global, dim=-1)
            V_c = F.normalize(h_c, dim=-1)
            V_c_perp = V_c - (V_c * V_g).sum(-1, keepdim=True) * V_g
            features = torch.cat([V_g, V_c_perp], dim=-1)

        elif self.interaction_type == 'hadamard':
            h_c = self._global_pool(atom_h, batch_idx, B)
            features = torch.cat([h_g_global, h_c * h_g_global], dim=-1)

        logits = self.classifier(features).squeeze(-1)

        return logits, h_g_global, delta_h, spectrum, sigma, U, gene_attn


# ================================================================
# 5. 正则化
# ================================================================

def compute_operator_regularization(sigma, U, lam_sparse, lam_ortho):
    """
    算子正则化损失（两项）：
      1. σ 稀疏：loss_sparse = mean(|σ_k|)
         让大多数模式静默，少数模式主导 → 稀疏药效团指纹，可解释性强。

      2. U 正交：loss_ortho = ||UᵀU - I||²_F
         约束 r 个输出方向两两正交（类比主成分分析的基向量正交）。
         正交 → 各模式信息不重叠 → 每个模式有独立的生物学意义。
         消融实验：lam_ortho 0.1 vs 0.0 → AUC +0.0008，且可解释性显著提升。
    """
    loss_sparse = sigma.abs().mean()
    U_n  = F.normalize(U, dim=-1)                              # 归一化到单位球
    gram = torch.bmm(U_n, U_n.transpose(1, 2))                # [B, r, r]，Gram 矩阵
    eye  = torch.eye(U.shape[1], device=U.device).unsqueeze(0)  # 目标：单位阵
    loss_ortho = (gram - eye).pow(2).mean()                    # 偏离正交的程度
    return lam_sparse * loss_sparse + lam_ortho * loss_ortho


# ================================================================
# 6. LR Warmup
# ================================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ================================================================
# 7. 训练主循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds = OptimizedGraphDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=optimized_collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=optimized_collate_fn, num_workers=4)

    model = DrugOperatorNetV2(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        operator_rank=args.operator_rank,
        interaction_type=args.interaction_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir   = Path(f'results_operator_v2/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"{args.interaction_type}_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0
    warmup_epochs = args.warmup_epochs
    base_lr       = args.lr

    print(f"\n{'='*70}")
    print(f"DrugOperatorNet V2  (GeneMultiHeadReader + 模式对齐耦合)")
    print(f"  交互类型: {args.interaction_type} | rank={args.operator_rank} | "
          f"gene_len={args.gene_max_len}")
    print(f"  Device: {args.device} | Fold: {args.fold} | warmup={warmup_epochs}ep")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        # LR warmup：前 warmup_epochs 线性从 base_lr/10 升至 base_lr
        if epoch < warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == warmup_epochs:
            set_lr(optimizer, base_lr)   # warmup 结束，交给 scheduler

        model.train()
        total_loss = total_bce = total_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            labels     = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, h_g_global, delta_h, spectrum, sigma, U, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                loss_bce = criterion(logits, labels)

                loss_reg = torch.tensor(0.0, device=device)
                if args.interaction_type == 'operator' and sigma is not None:
                    loss_reg = compute_operator_regularization(
                        sigma, U, args.lam_sparse, args.lam_ortho_modes)

                loss = loss_bce + loss_reg

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += loss_bce.item()
            total_reg  += loss_reg.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        all_spectra, all_gene_attn = [], []

        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr  = batch['edge_attr'].to(device)
                gene_ids   = batch['gene_ids'].to(device)

                with autocast(enabled=args.use_amp):
                    logits, _, _, spectrum, _, _, gene_attn = model(
                        gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                if spectrum is not None and args.save_spectrum:
                    all_spectra.append(spectrum.cpu())
                if args.save_gene_attn and gene_attn is not None:
                    # 只保存每个样本各模式的注意力熵（省内存，全权重太大）
                    entropy = -(gene_attn * (gene_attn + 1e-8).log()).sum(-1)  # [B, r]
                    all_gene_attn.append(entropy.cpu())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        # warmup 期间不更新 scheduler
        if epoch >= warmup_epochs:
            scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} (BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f}) | "
              f"VAL_AUC: {auroc:.4f} | PRC: {auprc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
            if args.save_spectrum and all_spectra:
                torch.save({
                    'spectrum':  torch.cat(all_spectra, dim=0),
                    'preds':     np.array(all_preds),
                    'labels':    np.array(all_labels),
                }, save_dir / f'spectrum_{model_name}')
            if args.save_gene_attn and all_gene_attn:
                torch.save({
                    'attn_entropy': torch.cat(all_gene_attn, dim=0),  # [N_val, r]
                    'labels':       np.array(all_labels),
                }, save_dir / f'gene_attn_{model_name}')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


# ================================================================
# 8. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--device',      type=str, default='cuda:3')
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--seed',        type=int, default=42)

    parser.add_argument('--epochs',      type=int, default=80)
    parser.add_argument('--batch_size',  type=int, default=512)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--patience',    type=int, default=10)
    parser.add_argument('--use_amp',     action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='LR warmup epoch 数（attn_queries 零初始化需要预热）')

    parser.add_argument('--gene_max_len',    type=int, default=3000,
                        help='基因 k-mer 序列长度（V2 默认 3000）')
    parser.add_argument('--interaction_type', type=str, default='operator',
                        choices=['operator', 'concat', 'ortho_concat', 'hadamard'])
    parser.add_argument('--operator_rank',   type=int, default=8,
                        help='算子秩 = 药效团数 = 基因读取头数')

    parser.add_argument('--lam_sparse',      type=float, default=0.01)
    parser.add_argument('--lam_ortho_modes', type=float, default=0.01)

    parser.add_argument('--save_spectrum',   action='store_true')
    parser.add_argument('--save_gene_attn',  action='store_true',
                        help='保存基因注意力熵（各模式聚焦程度）')
    parser.add_argument('--run_tag',         type=str, default='')

    train(parser.parse_args())
