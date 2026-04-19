#!/usr/bin/env python3
"""
v12: Gene-Conditioned Pharmacophore (GCP)
=========================================

核心改进：PharmacophoreExtractor 的 query 从固定可学习向量
改为基因 mode 条件化查询（h_g_modes），使药物原子注意力由基因
片段信息驱动。

原来：固定 queries [r, H]，所有药物-基因对共用同一套探测器
v12：queries = query_proj(h_g_modes) [B, r, H]，每个基因片段
     "询问"药物分子哪些原子与自己相关。

可解释性（化学子结构 ↔ 基因片段）：
  atom_scores[N_total, r]：
    - 行 i：drug 中第 i 个原子
    - 列 s：gene mode s（对应基因 k-mer 序列的一组片段模式）
    - 值：该化学子结构对该基因片段模式的激活强度
  → 分子可视化：按 α_s,i 对原子着色 = 哪些官能团激活基因模式 s
  → 序列可视化：gene_attn[b, s, :] = 哪些 k-mer 位置属于模式 s

为何能突破性能天花板：
  OOD 药物的原子特征（原子类型/杂化/键型）在训练集中均可见；
  gene-conditioned query 聚焦化学上相关的原子，过滤 novel scaffold
  中对当前基因片段无关的噪声，降低 OOD 泛化误差。
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
    """
    stride=1: 每个位置一个 k-mer（默认，覆盖 ~max_len bp）
    stride>1: 每 stride 个位置采一个 k-mer（覆盖 ~max_len*stride bp，等效扩大感受野）
    """
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
    """
    图内原子级 softmax（纯 PyTorch，无需 torch_geometric）。

    问题背景：一个 batch 中，不同分子的原子被打包成连续的 [N_total] 张量，
    batch_idx 记录每个原子属于哪个分子（例如 [0,0,0,1,1,2,2,2,2]）。
    普通 softmax 会对整个 batch 归一化，而我们需要每个分子内部独立归一化。

    实现思路（数值稳定版 softmax）：
      1. 找到每个分子内分数的最大值（用于减去，防止 exp 溢出）
      2. 计算 exp(score - max)
      3. 在每个分子内求和
      4. 除以分子内的和

    Args:
      scores:    [N_total]，每个原子的原始得分
      batch_idx: [N_total]，每个原子对应的图编号（0-indexed）

    Returns:
      [N_total]，每个原子在其所属图内的 softmax 权重
    """
    scores = scores.float()
    # 找每个分子内的最大分数（数值稳定）
    max_scores = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
    exp_scores = torch.exp(scores - max_scores[batch_idx])  # 减去最大值后 exp
    # 在每个分子内累加 exp 值
    exp_sum = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)  # 归一化，+eps 防止除零


def scatter_add(src, batch_idx, dim_size):
    """
    将 [N_total, H] 按 batch_idx 聚合为 [B, H]（图级求和池化）。
    等价于 torch_geometric 的 scatter(src, batch_idx, dim=0, reduce='sum')。
    """
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


# ================================================================
# 1. 数据集（与 V2 一致，支持 gene_max_len 缓存）
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

        # 缓存命名：len1000_s1 是默认（向后兼容保持原名）
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
    return {
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
    """
    图同构网络（Graph Isomorphism Network）单层消息传递。

    GIN 公式：h_v^{(l+1)} = MLP( h_v^{(l)} + Σ_{u∈N(v)} msg(h_u, e_{uv}) )
      - 对每条边 (u→v)：msg = ReLU(h_u + W_edge * e_{uv})
      - 对每个节点 v：聚合所有入边消息后 + 自身，再过 MLP

    与简单 GCN 的区别：MLP（而非单线性层）确保区分不同邻域结构。
    边特征（bond type, aromaticity 等 4-dim）通过 edge_proj 融入消息。
    """
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        # 节点更新 MLP：BN 确保训练稳定
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        # 边特征投影：将 4-dim 键特征映射到 hidden_dim
        self.edge_proj = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index  # row=源原子, col=目标原子（u→v 中 u=row, v=col）
        # 消息：源原子嵌入 + 边嵌入，非线性激活
        msg = F.relu(x[row] + self.edge_proj(edge_attr))  # [E, H]
        # 聚合：将消息累加到目标原子（scatter_add，按 col 聚合）
        agg = torch.zeros_like(x).index_add_(0, col, msg)  # [N, H]
        # 更新：节点自身 + 聚合消息，再过 MLP（残差形式）
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
        return x  # [N_total, H]


# ================================================================
# 4. 药效团提取（来自 V2）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """
    v12 Gene-Conditioned Pharmacophore Extractor

    v12改进：query 由固定可学习向量改为基因 mode 条件化查询。

    原来 (baseline)：
      query_s = self.queries[s]          [H]  — 全局固定，所有对共享
    v12：
      query_{b,s} = query_proj(h_g_modes[b, s])  [H]  — 基因特异

    Slot s 的计算（v12）：
      Q_{b,s}  = query_proj(h_g_modes[b, s])   [H]
      scores_s  = K_b · Q_{b,s} / √d            [n_atoms_b]
      α_s       = scatter_softmax(scores_s)      [N_total]
      pharma_s  = Σ_i α_{s,i} * V_i             [B, H]

    可解释性（化学子结构 ↔ 基因片段）：
      atom_scores[i, s] = 原子 i 在基因模式 s 驱动下的注意力强度
      → 化学子结构（原子局部环境）↔ 基因片段（gene mode s 对应的 k-mer 区域）
    """
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)  # 基因mode → query空间
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)  # 原子 → Key
        self.val_proj   = nn.Linear(hidden_dim, hidden_dim)  # 原子 → Value
        self.norm       = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs, gene_queries):
        """
        Args:
          atom_h:       [N_total, H]  所有分子原子嵌入（打包）
          batch_idx:    [N_total]     每个原子所属分子编号
          num_graphs:   int           batch 中分子数 B
          gene_queries: [B, r, H]    基因 mode 嵌入（h_g_modes）

        Returns:
          pharma:     [B, r, H]    r 个基因条件化药效团嵌入
          scores_all: [N_total, r] 原子对每个基因模式的注意力分（可解释性）
        """
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)                         # [N_total, H]
        V = self.val_proj(atom_h)                         # [N_total, H]
        Q = self.query_proj(gene_queries)                 # [B, r, H]

        # 将基因 query 广播到原子级别：Q_atom[i, s] = Q[batch_idx[i], s, :]
        Q_atom = Q[batch_idx]                             # [N_total, r, H]

        # 向量化计算所有原子-slot的分数：[N_total, r]
        scores_all = (K.unsqueeze(1) * Q_atom).sum(-1) / math.sqrt(d)

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)

        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)  # [N_total]
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all  # [B, r, H], [N_total, r]


# ================================================================
# 5. 扰动算子（来自 V2，模式对齐耦合）
# ================================================================

class PerturbationOperator(nn.Module):
    """
    扰动算子：将药物编码为基因表达空间上的低秩扰动。

    数学形式（秩-r 算子）：
      T = I + Σ_{k=1}^{r} σ_k · u_k ⊗ v_k^T

    各符号含义：
      u_k ∈ R^H  : 第 k 个药效团的"输出方向"（扰动后的方向）
      v_k ∈ R^H  : 第 k 个药效团的"输入方向"（对哪类基因特征敏感）
      σ_k ∈ [-1,1]: 第 k 个模式的强度（Tanh 保证有界，正=激活，负=抑制）

    模式对齐耦合（V2 设计，区别于 V1）：
      coupling_k = v_k · h_g_modes_k   ← 药物方向 v_k 与基因专属视角 k 的内积
                                          衡量"药物此模式与基因此方向的对齐程度"
      spectrum_k = σ_k × coupling_k    ← 强度 × 对齐 = 有效交互谱

    实际扰动：
      Δh = Σ_k spectrum_k · u_k        ← 加权叠加各模式的输出方向
      （对应 T 作用到 h_g 后的增量部分）

    可解释性：
      spectrum [B, r] 是该（药物, 基因）对的"机理指纹"，
      可直接分析正/负样本区别、药物聚类等（见 visualize_spectrum.py）。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 从药效团嵌入学习输出方向 u（两层 MLP + 归一化）
        self.to_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        # 从药效团嵌入学习输入方向 v（同上）
        self.to_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        # 从药效团嵌入学习模式强度 σ（瓶颈 MLP + Tanh → 有界 [-1, 1]）
        self.to_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), nn.Tanh())

    def forward(self, pharma_emb, h_g_modes):
        """
        Args:
          pharma_emb: [B, r, H]  药效团嵌入（来自 PharmacophoreExtractor）
          h_g_modes:  [B, r, H]  基因多视角编码（来自 GeneMultiHeadReader）
                                  第 k 个视角对应第 k 个药效团模式

        Returns:
          delta_h:  [B, H]    净扰动向量（算子增量部分）
          spectrum: [B, r]    交互谱（可解释性核心）
          sigma:    [B, r]    各模式强度
          U:        [B, r, H] 各模式输出方向（归一化）
        """
        # 学习输出/输入方向并归一化（单位向量，使得 coupling 仅反映方向相似度）
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)  # [B, r, H]，输出方向
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)  # [B, r, H]，输入方向
        sigma = self.to_sigma(pharma_emb).squeeze(-1)        # [B, r]，模式强度

        # V2 核心：模式对齐耦合（逐元素乘 + 在 H 维求和 = 内积）
        coupling = (V * h_g_modes).sum(-1)                   # [B, r]，方向对齐度
        spectrum = sigma * coupling                           # [B, r]，有效交互谱

        # 求和得净扰动：Δh = Σ_k spectrum_k * u_k
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

        # ③ 基因条件化药效团提取 + 算子
        # v12: h_g_modes 作为 query，每个基因片段模式"询问"药物哪些原子相关
        pharma_emb, atom_scores = self.pharma_ext(atom_h, batch_idx, B, h_g_modes)
        # pharma_emb [B, r, H], atom_scores [N, r] ← 化学子结构×基因片段注意力
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
# 7. 对比损失模块（谱方向感知 CL）
# ================================================================

class SpectrumDirectionCL(nn.Module):
    """
    方向感知对比损失，作用于交互谱 spectrum [B, r]。

    直觉：正样本（CGI=1）的交互谱应该在谱空间中指向一个一致的"有效方向"，
    负样本则应偏离该方向。这与 FocalDirectionAwareCL（旧框架 train_summean.py）
    的设计思路一致，但直接作用于 DrugOperatorNet 输出的谱表示，而非
    旧框架的 V_g / V_c_perp 中间变量。

    与正交正则的区别：
    - 正交正则（lam_ortho）约束 U 列向量互相正交，是结构约束（各模式独立）
    - CL（lam_cl）约束正负样本在谱空间的方向分离，是语义约束（正负可区分）
    两者互补，不重叠。

    Loss = E[ focal_weight * relu(margin - score * target) ]
    其中 score = spectrum @ d（谱在可学习方向上的投影）
         target = +1（正样本），-1（负样本）
         focal_weight = (diff + eps)^2（困难样本加权，类似 Focal Loss）
    """
    def __init__(self, rank: int, margin: float = 0.5):
        super().__init__()
        # 可学习的"有效方向"向量，维度与谱维度 r 一致
        self.direction = nn.Parameter(torch.randn(rank))
        self.margin    = margin

    def forward(self, spectrum: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum: [B, r]  交互谱
            labels:   [B]     二值标签（0/1）
        Returns:
            scalar loss
        """
        d       = F.normalize(self.direction, dim=0)           # [r] 单位方向向量
        scores  = spectrum @ d                                  # [B] 每个样本在方向上的投影
        targets = 2.0 * labels - 1.0                           # +1（正）/ -1（负）
        diff    = F.relu(self.margin - scores * targets)        # 违反 margin 的程度
        weight  = (diff + 1e-4) ** 2                           # 困难样本聚焦（类 Focal）
        return (diff * weight).mean()


# ================================================================
# 8. 正则化损失
# ================================================================

def compute_losses(logits, labels, spectrum, sigma, U,
                   route_weights, criterion, args,
                   cl_module=None, soft_labels=None):
    """
    计算总损失 = BCE + 算子正则 + MoE 负载均衡。

    各损失项说明：
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. BCE（分类主损失）                                          │
    │    标准二元交叉熵，优化预测概率与真实标签的对齐。             │
    │                                                              │
    │ 2. 算子正则（仅 operator 模式）                              │
    │    a) σ 稀疏（lam_sparse）：                                 │
    │       loss_sparse = E[|σ_k|]                                 │
    │       让大多数模式保持"静默"，少数主导模式激活，提升可解释性。│
    │    b) U 正交（lam_ortho_modes）：                            │
    │       loss_ortho = ||UᵀU - I||²_F                           │
    │       约束 r 个输出方向两两正交，确保各模式独立互不冗余。    │
    │       消融实验验证：lam_ortho=0.1 vs 0.0，+0.0008 AUC。     │
    │                                                              │
    │ 4. 方向感知 CL（lam_cl > 0 时启用）                         │
    │       loss_cl = SpectrumDirectionCL(spectrum, labels)        │
    │       约束正负样本在谱空间方向分离，与正交正则互补。         │
    │       注：CL 是语义约束，正交是结构约束，两者不重叠。        │
    │                                                              │
    │ 5. Soft Label BCE（--soft_label 时启用）                     │
    │       soft = sigmoid((|zscore| - 2) / 0.5) × 方向           │
    │       用 z-score 置信度替换硬标签，降低边界区噪声影响。      │
    │                                                              │
    │ 3. MoE 负载均衡（仅 full/no_spectrum/no_target 模式）        │
    │    a) 宏观均衡（MSE）：路由均值接近均匀分布，防止专家坍缩。  │
    │    b) 负熵（0.1 权重）：最大化路由熵，鼓励路由多样性。       │
    │    注：no_moe 模式无路由，此项为 0。                         │
    └──────────────────────────────────────────────────────────────┘
    """
    # ── 1. 主分类损失 ────────────────────────────────────────────
    if soft_labels is not None:
        # Soft label BCE：用 z-score 置信度替换硬标签（0/1 → 连续值）
        # soft_labels 已经是 [0,1] 之间的置信度，直接用 BCEWithLogitsLoss 的 per-sample 版本
        loss_bce = F.binary_cross_entropy_with_logits(logits, soft_labels, reduction='mean')
    else:
        loss_bce = criterion(logits, labels)

    # ── 2. 算子正则 ──────────────────────────────────────────────
    # σ 稀疏：鼓励少数模式主导，其余接近 0
    loss_sparse = sigma.abs().mean()
    # U 正交：Gram 矩阵 UᵀU 应接近单位阵（各列两两正交且单位长度）
    U_n   = F.normalize(U, dim=-1)                             # [B, r, H]，归一化
    gram  = torch.bmm(U_n, U_n.transpose(1, 2))               # [B, r, r]，Gram 矩阵
    eye   = torch.eye(U.shape[1], device=U.device).unsqueeze(0)  # [1, r, r]
    loss_ortho = (gram - eye).pow(2).mean()                    # F-范数²
    loss_reg   = args.lam_sparse * loss_sparse + args.lam_ortho_modes * loss_ortho

    # ── 3. MoE 负载均衡（no_moe 时跳过）────────────────────────
    loss_lb = torch.tensor(0.0, device=logits.device)
    if route_weights is not None and args.ablation != 'no_moe':
        mean_route = route_weights.mean(dim=0)                  # [K]，batch 内路由均值
        uniform    = torch.ones_like(mean_route) / args.num_experts  # [K]，均匀分布
        # 宏观均衡：batch 级路由接近均匀分布
        loss_macro   = F.mse_loss(mean_route, uniform)
        # 样本级路由熵（负号→最大化熵→多样性路由）
        loss_entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(-1).mean()
        loss_lb      = loss_macro + 0.1 * loss_entropy

    # ── 4. 方向感知 CL（可选）────────────────────────────────────
    loss_cl = torch.tensor(0.0, device=logits.device)
    if cl_module is not None and spectrum is not None:
        loss_cl = cl_module(spectrum, labels)

    total = loss_bce + loss_reg + args.lam_balance * loss_lb + args.lam_cl * loss_cl
    return total, loss_bce, loss_reg, loss_lb, loss_cl


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
        args.data_dir, args.fold, 'train', args.gene_max_len, args.gene_stride)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len, args.gene_stride)
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

    # CL 模块（可选，lam_cl > 0 时启用）
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
    print(f"  [v12 GCP] Gene-Conditioned Pharmacophore | ablation={args.ablation}")
    print(f"  operator_rank={args.operator_rank} | num_experts={args.num_experts}")
    print(f"  gene_max_len={args.gene_max_len} stride={args.gene_stride} "
        f"(~{args.gene_max_len*args.gene_stride}bp) | warmup={args.warmup_epochs}ep")
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
        if cl_module is not None:
            cl_module.train()
        total_loss = total_bce = total_reg = total_lb = total_cl = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)

            # Soft label：用 z-score 置信度替换硬标签（--soft_label 启用时）
            soft_labels = None
            if args.soft_label and batch['zscore'] is not None:
                zscore = batch['zscore'].to(device)
                # 置信度 = sigmoid((|z| - 2) / 0.5)：边界区 (|z|≈2) → 0.5，强信号 (|z|≥4) → 1.0
                conf = torch.sigmoid((zscore.abs() - 2.0) / 0.5)
                # 保持方向：正样本 soft=conf，负样本 soft=1-conf
                soft_labels = labels * conf + (1 - labels) * (1 - conf)

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _, route_weights = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
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

    parser.add_argument('--gene_max_len',  type=int, default=1000,
                        help='k-mer token 数量（序列覆盖 = max_len × stride bp）')
    parser.add_argument('--gene_stride',   type=int, default=1,
                        help='k-mer 采样步长（1=逐位，2=每2bp采一次，以此类推）')
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
    parser.add_argument('--lam_cl',          type=float, default=0.0,
                        help='方向感知 CL 系数（0=禁用，建议先试 0.1）')
    parser.add_argument('--soft_label',      action='store_true',
                        help='用 z-score 置信度替换硬标签，降低边界区噪声影响')

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
