#!/usr/bin/env python3
"""
v13: FP-Conditioned Pharmacophore (FPCond)
==========================================

核心改进：PharmacophoreExtractor 的 query 由固定可学习向量改为
Morgan 指纹（FP）条件化查询。

原来：queries [r, H] — 全局固定
v13：fp_queries = FPQueryProjector(morgan_fp) [B, r, H] — 药物子结构特异

为何 FP 作为输入（而非 Tanimoto 作为 loss）有效：
  - Tanimoto 作为 loss：强迫 σ 相似性 = 结构相似性（多跳非线性断链，失败）
  - FP 作为输入：让模型自己学习 FP片段→基因响应的映射（无先验约束）

为何 v12 gene-conditioned pharma 失败（循环对齐）：
  - pharma_k 由 h_g_modes_k 条件化 → V_k 与 h_g_modes_k 对齐
  - spectrum_k = V_k · h_g_modes_k 变成平凡高值（circular alignment）
  - REG 损失高居不下，判别力丧失

v13 为何不循环：
  - fp_queries 来自 Morgan FP（固定结构特征，非模型内部变量）
  - pharma_k 由 FP slot k 条件化
  - spectrum_k = V_k · h_g_modes_k 仍用基因模式（无循环依赖）

OOD 鲁棒性：
  - OOD novel scaffold 共享已知 fragments（FP 位有重叠）
  - 模型学习"FP位 j → 哪些原子 → 哪种基因影响"
  - 推理时对 novel drug 直接用其 FP 位中已知的 fragment 信息

可解释性（化学子结构 ↔ 基因片段）：
  - FPQueryProjector.net.weight[s*H:(s+1)*H, :] = FP位对 slot s 的贡献
  - atom_scores[i, s] = 原子 i 对 slot s 的注意力权重
  - 联合分析：哪些 FP 片段激活 slot s → 对应哪些基因 modes
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


def encode_kmer_sequence(sequence: str, k: int = 6, max_len: int = 1000,
                         stride: int = 1) -> list:
    kmers = []
    for i in range(0, len(sequence) - k + 1, stride):
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
# 1. 数据集
# ================================================================

class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train', gene_max_len=1000,
                 gene_stride=1):
        import pickle
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt',
                               weights_only=False)
        self.smiles_to_graph = self.data['smiles_to_graph']
        self.graph_indices = [self.data['graph_indices'][i] for i in self.indices]
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        if 'zscores' in self.data:
            self.zscores = torch.tensor(
                [self.data['zscores'][i] for i in self.indices], dtype=torch.float32)
        else:
            self.zscores = None
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]

        len_tag    = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        stride_tag = '' if gene_stride == 1      else f'_s{gene_stride}'
        cache_file = Path(data_dir) / \
            f'kmer_cache_fold{fold_idx}_{split}{len_tag}{stride_tag}.pt'

        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file, weights_only=True)
        else:
            bp_coverage = gene_max_len * gene_stride
            print(f"[{split.upper()}] 生成 K-mer 缓存 "
                  f"(len={gene_max_len}, stride={gene_stride}, ~{bp_coverage}bp)...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len, stride=gene_stride)
                 for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

        # ── Morgan FP 缓存（v13 新增）─────────────────────────────
        morgan_cache = Path(data_dir) / f'morgan_cache_fold{fold_idx}_{split}_r2_b2048.pt'
        if morgan_cache.exists():
            print(f"[{split.upper()}] ⚡ Morgan FP 缓存: {morgan_cache.name}")
            self.morgan_fps = torch.load(morgan_cache, weights_only=False)
        else:
            self.morgan_fps = None
            print(f"[{split.upper()}] ⚠ Morgan FP 缓存不存在，将回退到固定 queries")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
        }
        if self.zscores is not None:
            item['zscore'] = self.zscores[idx]
        if self.morgan_fps is not None:
            item['morgan_fp'] = self.morgan_fps[idx]
        return item


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
    result = {
        'x':              torch.cat(all_x, dim=0),
        'edge_index':     torch.cat(all_edge_index, dim=1) if all_edge_index
                          else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr':      torch.cat(all_edge_attr, dim=0) if all_edge_attr
                          else torch.zeros(0, 4),
        'num_nodes_list': num_nodes_list,
        'gene_ids':       torch.stack([b['gene_ids'] for b in batch]),
        'label':          torch.stack([b['label']    for b in batch]),
        'zscore':         torch.stack([b['zscore'] for b in batch])
                          if 'zscore' in batch[0] else None,
        'morgan_fp':      torch.stack([b['morgan_fp'] for b in batch])
                          if 'morgan_fp' in batch[0] else None,
    }
    return result


# ================================================================
# 2. 基因编码器
# ================================================================

class GeneMultiHeadReader(nn.Module):
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
        self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))
        self.out_norm = nn.LayerNorm(H)
        self.dropout  = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = self.seq_norm(x)

        scores = torch.einsum('blh,rh->brl', x, self.attn_queries) / math.sqrt(x.size(-1))
        attn   = F.softmax(scores, dim=-1)
        h_g_modes = torch.einsum('brl,blh->brh', attn, x)
        h_g_modes = self.out_norm(h_g_modes)

        h_g_global = h_g_modes.mean(dim=1)
        return h_g_modes, h_g_global, attn


# ================================================================
# 3. 化学编码器
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
        agg = torch.zeros_like(x).index_add_(0, col, msg)
        return self.mlp(x + agg)


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
        return x


# ================================================================
# 4. FP 查询投影器（v13 核心新增模块）
# ================================================================

class FPQueryProjector(nn.Module):
    """
    Morgan FP [B, 2048] → r 个药效团查询向量 [B, r, H]。

    瓶颈结构避免直接 2048→r*H 的大矩阵：
      Linear(2048, 64) → ReLU → Linear(64, r*H) → reshape [B, r, H]

    参数量：2048*64 + 64 + 64*(r*H) + r*H ≈ 134K（r=8, H=128）
    """
    def __init__(self, fp_dim=2048, bottleneck=64, num_slots=8, hidden_dim=128):
        super().__init__()
        self.num_slots  = num_slots
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(fp_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, num_slots * hidden_dim),
        )

    def forward(self, morgan_fp):
        # morgan_fp: [B, 2048]
        out = self.net(morgan_fp)                            # [B, r*H]
        return out.view(morgan_fp.shape[0], self.num_slots, self.hidden_dim)  # [B, r, H]


# ================================================================
# 5. 药效团提取（v13: 支持 FP 条件化查询）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """
    药效团提取器：r 个 query slot × 原子交叉注意力 → r 个药效团嵌入。

    v13 改动：forward 接受可选的 fp_queries [B, r, H]。
      - fp_queries 不为 None：用 FP 条件化查询（per-drug, per-slot）
      - fp_queries 为 None：回退到固定可学习 self.queries（原始行为）

    无循环对齐问题：
      fp_queries 来自 Morgan FP（固定结构特征），与 h_g_modes 无依赖关系，
      因此 spectrum_k = V_k · h_g_modes_k 不会变成平凡值。
    """
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots  = num_slots
        self.queries    = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)  # FP 条件化时用
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs, fp_queries=None):
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)                            # [N, H]
        V = self.val_proj(atom_h)                            # [N, H]

        if fp_queries is not None:
            # FP 条件化：每个分子、每个 slot 有独立查询向量
            Q      = self.query_proj(fp_queries)             # [B, r, H]
            Q_atom = Q[batch_idx]                            # [N, r, H]
            scores_all = (K.unsqueeze(1) * Q_atom).sum(-1) / math.sqrt(d)  # [N, r]
        else:
            # 回退：全局固定查询
            scores_all = (K @ self.queries.T) / math.sqrt(d)               # [N, r]

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all


# ================================================================
# 6. 扰动算子
# ================================================================

class PerturbationOperator(nn.Module):
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
            nn.Linear(hidden_dim // 4, 1), nn.Tanh())

    def forward(self, pharma_emb, h_g_modes):
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)
        sigma = self.to_sigma(pharma_emb).squeeze(-1)

        coupling = (V * h_g_modes).sum(-1)
        spectrum = sigma * coupling
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)

        return delta_h, spectrum, sigma, U


# ================================================================
# 7. 完整融合模型：OperatorMoE（v13: FP-Conditioned）
# ================================================================

class OperatorMoE(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3,
                 operator_rank=8, num_experts=4, ablation='full'):
        super().__init__()
        self.ablation      = ablation
        self.operator_rank = operator_rank
        self.num_experts   = num_experts
        H = hidden_dim
        r = operator_rank
        K = num_experts

        self.gene_enc   = GeneMultiHeadReader(hidden_dim=H, num_heads=r, dropout=dropout)
        self.atom_enc   = AtomEncoder(hidden_dim=H, dropout=dropout)
        self.pharma_ext = PharmacophoreExtractor(H, r)
        self.perturb_op = PerturbationOperator(H)

        # v13 核心：FP 查询投影器
        self.fp_query_proj = FPQueryProjector(fp_dim=2048, bottleneck=64,
                                              num_slots=r, hidden_dim=H)

        if ablation == 'no_spectrum':
            self.global_readout = nn.Sequential(
                nn.Linear(H * 2, H * 2), nn.BatchNorm1d(H * 2), nn.ReLU(),
                nn.Linear(H * 2, H))
            router_in = H * 2
        else:
            router_in = r

        if ablation != 'no_moe':
            self.router = nn.Sequential(
                nn.Linear(router_in, H),
                nn.LayerNorm(H),
                nn.ReLU(),
                nn.Linear(H, K),
                nn.Softmax(dim=-1))
            self.expert_queries = nn.Parameter(torch.randn(K, H) * 0.01)

        if ablation == 'no_moe':
            self.classifier = nn.Sequential(
                nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(H, 1))
        else:
            feat_dim = H * 3 if ablation != 'no_target' else H * 2
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, H), nn.LayerNorm(H), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(H, 1))
                for _ in range(K)])

        self.dropout_p = dropout

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list, morgan_fp=None):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(torch.arange(B, device=device), num_nodes_t)

        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)
        atom_h = self.atom_enc(x, edge_index, edge_attr)

        # v13: FP 条件化药效团提取
        fp_queries = None
        if morgan_fp is not None:
            fp_queries = self.fp_query_proj(morgan_fp)       # [B, r, H]

        pharma_emb, atom_scores = self.pharma_ext(atom_h, batch_idx, B, fp_queries)
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        if self.ablation == 'no_moe':
            features = torch.cat([h_g_global, delta_h], dim=-1)
            logits   = self.classifier(features).squeeze(-1)
            return logits, spectrum, sigma, U, gene_attn, None

        if self.ablation == 'no_spectrum':
            sum_pool  = scatter_add(atom_h, batch_idx, B)
            mean_pool = sum_pool / num_nodes_t.float().unsqueeze(1).clamp(min=1)
            h_c       = self.global_readout(torch.cat([sum_pool, mean_pool], dim=-1))
            V_g       = F.normalize(h_g_global, dim=-1)
            V_c       = F.normalize(h_c, dim=-1)
            V_c_perp  = V_c - (V_c * V_g).sum(-1, keepdim=True) * V_g
            router_in = torch.cat([V_g, V_c_perp], dim=-1)
        else:
            router_in = spectrum

        route_weights = self.router(router_in)

        if self.ablation != 'no_target':
            q           = route_weights @ self.expert_queries
            scores_atom = (atom_h * q[batch_idx]).sum(-1) / math.sqrt(atom_h.size(-1))
            alpha       = scatter_softmax(scores_atom, batch_idx)
            target_pool = scatter_add(atom_h * alpha.unsqueeze(-1), batch_idx, B)
            features    = torch.cat([h_g_global, delta_h, target_pool], dim=-1)
        else:
            features    = torch.cat([h_g_global, delta_h], dim=-1)

        expert_logits = torch.stack(
            [exp(features).squeeze(-1) for exp in self.experts], dim=1)
        logits = (route_weights * expert_logits).sum(dim=-1)

        return logits, spectrum, sigma, U, gene_attn, route_weights


# ================================================================
# 8. 对比损失
# ================================================================

class SpectrumDirectionCL(nn.Module):
    def __init__(self, rank: int, margin: float = 0.5):
        super().__init__()
        self.direction = nn.Parameter(torch.randn(rank))
        self.margin    = margin

    def forward(self, spectrum: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        d       = F.normalize(self.direction, dim=0)
        scores  = spectrum @ d
        targets = 2.0 * labels - 1.0
        diff    = F.relu(self.margin - scores * targets)
        weight  = (diff + 1e-4) ** 2
        return (diff * weight).mean()


# ================================================================
# 9. 损失计算
# ================================================================

def compute_losses(logits, labels, spectrum, sigma, U,
                   route_weights, criterion, args,
                   cl_module=None, soft_labels=None):
    if soft_labels is not None:
        loss_bce = F.binary_cross_entropy_with_logits(logits, soft_labels, reduction='mean')
    else:
        loss_bce = criterion(logits, labels)

    loss_sparse = sigma.abs().mean()
    U_n   = F.normalize(U, dim=-1)
    gram  = torch.bmm(U_n, U_n.transpose(1, 2))
    eye   = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
    loss_ortho = (gram - eye).pow(2).mean()
    loss_reg   = args.lam_sparse * loss_sparse + args.lam_ortho_modes * loss_ortho

    loss_lb = torch.tensor(0.0, device=logits.device)
    if route_weights is not None and args.ablation != 'no_moe':
        mean_route   = route_weights.mean(dim=0)
        uniform      = torch.ones_like(mean_route) / args.num_experts
        loss_macro   = F.mse_loss(mean_route, uniform)
        loss_entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(-1).mean()
        loss_lb      = loss_macro + 0.1 * loss_entropy

    loss_cl = torch.tensor(0.0, device=logits.device)
    if cl_module is not None and spectrum is not None:
        loss_cl = cl_module(spectrum, labels)

    total = loss_bce + loss_reg + args.lam_balance * loss_lb + args.lam_cl * loss_cl
    return total, loss_bce, loss_reg, loss_lb, loss_cl


# ================================================================
# 10. LR 工具
# ================================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ================================================================
# 11. 训练主循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds = OptimizedGraphDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len, args.gene_stride)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len, args.gene_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    has_fp = train_ds.morgan_fps is not None
    print(f"  [FP] Morgan FP 条件化: {'启用' if has_fp else '禁用（回退固定 queries）'}")

    model = OperatorMoE(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        operator_rank=args.operator_rank,
        num_experts=args.num_experts,
        ablation=args.ablation,
    ).to(device)

    cl_module = None
    if args.lam_cl > 0:
        cl_module = SpectrumDirectionCL(rank=args.operator_rank, margin=0.5).to(device)
        print(f"  [CL] SpectrumDirectionCL 已启用，lam_cl={args.lam_cl}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if cl_module is not None:
        n_params += sum(p.numel() for p in cl_module.parameters())

    all_params = list(model.parameters())
    if cl_module is not None:
        all_params += list(cl_module.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-3)
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
    print(f"  OperatorMoE v13-FPCond | ablation={args.ablation}")
    print(f"  operator_rank={args.operator_rank} | num_experts={args.num_experts}")
    print(f"  gene_max_len={args.gene_max_len} stride={args.gene_stride} | "
          f"drop_edge={args.drop_edge}")
    print(f"  params={n_params:,} | device={args.device} | fold={args.fold}")
    print(f"{'='*72}\n")

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        if cl_module is not None:
            cl_module.train()
        total_loss = total_bce = total_reg = total_lb = total_cl = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            labels     = batch['label'].to(device)
            morgan_fp  = batch['morgan_fp'].to(device) if batch['morgan_fp'] is not None else None

            # DropEdge
            if args.drop_edge > 0 and edge_index.shape[1] > 0:
                keep = torch.rand(edge_index.shape[1], device=device) > args.drop_edge
                edge_index_tr = edge_index[:, keep]
                edge_attr_tr  = edge_attr[keep]
            else:
                edge_index_tr = edge_index
                edge_attr_tr  = edge_attr

            soft_labels = None
            if args.soft_label and batch['zscore'] is not None:
                zscore = batch['zscore'].to(device)
                conf = torch.sigmoid((zscore.abs() - 2.0) / 0.5)
                soft_labels = labels * conf + (1 - labels) * (1 - conf)

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _, route_weights = model(
                    gene_ids, x, edge_index_tr, edge_attr_tr,
                    batch['num_nodes_list'], morgan_fp)
                loss, loss_bce, loss_reg, loss_lb, loss_cl = compute_losses(
                    logits, labels, spectrum, sigma, U,
                    route_weights, criterion, args,
                    cl_module=cl_module, soft_labels=soft_labels)

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += loss_bce.item()
            total_reg  += loss_reg.item()
            total_lb   += loss_lb.item()
            total_cl   += loss_cl.item()

        model.eval()
        all_preds, all_labels = [], []
        all_spectra = []

        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr  = batch['edge_attr'].to(device)
                gene_ids   = batch['gene_ids'].to(device)
                morgan_fp  = batch['morgan_fp'].to(device) if batch['morgan_fp'] is not None else None

                with autocast(enabled=args.use_amp):
                    logits, spectrum, _, _, _, _ = model(
                        gene_ids, x, edge_index, edge_attr,
                        batch['num_nodes_list'], morgan_fp)

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
        cl_str = f" CL:{total_cl/n:.4f}" if cl_module is not None else ""
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f} LB:{total_lb/n:.4f}{cl_str}) | "
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
# 12. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OperatorMoE v13: FP-Conditioned Pharmacophore')

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
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--gene_max_len',  type=int, default=1000)
    parser.add_argument('--gene_stride',   type=int, default=1)
    parser.add_argument('--operator_rank', type=int, default=8)
    parser.add_argument('--num_experts',   type=int, default=4)

    parser.add_argument('--lam_sparse',      type=float, default=0.01)
    parser.add_argument('--lam_ortho_modes', type=float, default=0.1)
    parser.add_argument('--lam_balance',     type=float, default=0.1)
    parser.add_argument('--lam_cl',          type=float, default=0.0)
    parser.add_argument('--soft_label',      action='store_true')
    parser.add_argument('--drop_edge',       type=float, default=0.0,
                        help='训练时随机丢弃边的概率（DropEdge augmentation）')

    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_spectrum', 'no_moe', 'no_target'])
    parser.add_argument('--save_spectrum', action='store_true')
    parser.add_argument('--run_tag',       type=str, default='')

    train(parser.parse_args())
