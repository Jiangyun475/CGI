#!/usr/bin/env python3
"""
v14: TF-Mediated Bilinear + Bayesian Probe Adaptation
======================================================

第一性原理：药物→基因的因果链

    drug  →  [靶蛋白结合]  →  [信号级联]  →  TF 活性改变
                                                    ↓
    gene  ←  [启动子结合]  ←  TF

因果链的两端才是我们的输入：drug SMILES 和 gene DNA。
中间的 TF 环节是**隐变量**，也是真正解释交互方向的机制。

v14 的核心改变：把这个隐变量显式建模为共享潜在空间。

    drug_tf  [B, r] ∈ [-1, 1]^r  — "drug 如何改变各 TF 通路的活性"
                drug_tf[k] > 0 : 激活 TF 概念 k
                drug_tf[k] < 0 : 抑制 TF 概念 k

    gene_tf  [B, r] ∈ [-1, 1]^r  — "各 TF 通路如何调控该基因"
                gene_tf[k] > 0 : TF 概念 k 上调该基因
                gene_tf[k] < 0 : TF 概念 k 下调该基因

    logit = Σ_k W_k · drug_tf_k · gene_tf_k
            物理含义：drug 激活 TF_k (drug_tf_k>0) × TF_k 上调 gene (gene_tf_k>0)
                      → 预测上调（正 logit）

与 v1-v13 的根本区别：
    之前：drug 是"算子"，gene 是被作用向量，两者不在同一空间，
          需要 PerturbationOperator 做 V·h_g 的事后对齐
    v14：drug 和 gene 都投影到同一个"TF 活性空间"，
          交互是内积，无需对齐模块

泛化保证：
    drug_tf = GIN(SMILES) → MLP → Tanh : novel scaffold 通过局部结构插值
    gene_tf = CNN(DNA)    → MLP → Tanh : 任意基因序列可用（非 ID lookup）

Bayesian Probe Adaptation（Few-Shot 整合，--eval_fewshot 开启）：
    给定 K 个 probe 对 {(gene_j, label_j)}:
        G  = {gene_tf_j * W_scale}  ∈ R^{K×r}
        d  = {2*label_j - 1}         ∈ {-1,+1}^K
    闭合形式后验（脊回归）：
        drug_tf_adapted = drug_tf_prior
                        + G^T (G G^T + λI)^{-1} (d - G · drug_tf_prior)
    物理含义：probe 告诉模型"drug 确实上调 gene_j"，而 gene_j 的 TF 结合特征
    已知，因此可以反推 drug 在对应 TF 通路上的活性，无需梯度下降。
"""

import argparse
import itertools
import math
import random
from collections import defaultdict
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


def encode_kmer_sequence(sequence, k=6, max_len=1000, stride=1):
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
    max_s = torch.zeros(batch_idx.max().item() + 1, device=scores.device
                        ).index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
    exp_s = torch.exp(scores - max_s[batch_idx])
    exp_sum = torch.zeros(batch_idx.max().item() + 1, device=scores.device
                          ).index_add_(0, batch_idx, exp_s)
    return exp_s / (exp_sum[batch_idx] + 1e-8)


def scatter_add(src, batch_idx, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


# ================================================================
# 1. 数据集（新增 drug_id 字段，支持 few-shot 分组）
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
        self.graph_indices   = [self.data['graph_indices'][i] for i in self.indices]
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        self.zscores = (torch.tensor(
            [self.data['zscores'][i] for i in self.indices], dtype=torch.float32)
            if 'zscores' in self.data else None)
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]

        len_tag    = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        stride_tag = '' if gene_stride == 1      else f'_s{gene_stride}'
        cache_file = Path(data_dir) / \
            f'kmer_cache_fold{fold_idx}_{split}{len_tag}{stride_tag}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file, weights_only=True)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存...")
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
            'drug_id':  self.graph_indices[idx],   # SMILES key → drug identity
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
        'drug_id':        [b['drug_id'] for b in batch],
    }


# ================================================================
# 2. Drug Encoder：GIN × 3 → pool → MLP → Tanh → drug_tf [B, r]
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


class DrugTFEncoder(nn.Module):
    """
    药物 → TF 激活向量。

    GIN 提取原子级特征，全局 sum+mean 池化得到分子表示，
    然后通过两层 MLP + Tanh 映射到 r 维 TF 活性空间。

    drug_tf[k] ∈ [-1, 1]：
      +1 = 该药物强烈激活 TF 概念 k
      -1 = 该药物强烈抑制 TF 概念 k
       0 = 该药物对 TF 概念 k 无显著影响
    """
    def __init__(self, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        H = hidden_dim
        self.atom_embed = nn.Sequential(
            nn.Linear(31, H), nn.BatchNorm1d(H), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(H) for _ in range(3)])
        self.norms      = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(), nn.Dropout(dropout))
        self.tf_proj = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, r_tf), nn.Tanh())
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, num_nodes_list, batch_idx):
        h = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            h = F.dropout(F.relu(norm(h + gin(h, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        # atom_h saved for interpretability (atom → TF contribution)
        atom_h = h  # [N, H]

        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=x.device)
        sum_pool  = scatter_add(atom_h, batch_idx, B)
        mean_pool = sum_pool / num_nodes_t.float().unsqueeze(1).clamp(min=1)
        mol_h = self.readout(torch.cat([sum_pool, mean_pool], dim=-1))  # [B, H]
        drug_tf = self.tf_proj(mol_h)                                   # [B, r]
        return drug_tf, atom_h


# ================================================================
# 3. Gene Encoder：k-mer CNN → attn pool → MLP → Tanh → gene_tf [B, r]
# ================================================================

class GeneTFEncoder(nn.Module):
    """
    基因 → TF 结合向量。

    多尺度 k-mer CNN 捕捉序列上不同长度的模式（类 TF 结合基序），
    注意力池化聚焦在最相关的序列区段，
    MLP + Tanh 映射到与药物共享的 TF 活性空间。

    gene_tf[k] ∈ [-1, 1]：
      +1 = 该基因被 TF 概念 k 强烈上调
      -1 = 该基因被 TF 概念 k 强烈下调
       0 = 该基因不由 TF 概念 k 调控

    泛化保证：这是一个 DNA 序列到 TF binding 的函数映射，
    对任何未见过的基因序列都可以计算，不依赖基因 ID lookup。
    """
    def __init__(self, vocab_size=4097, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        H = hidden_dim
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes   = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes])
        self.seq_norm = nn.LayerNorm(H)
        # 单一注意力查询：聚焦"TF-binding relevant"序列区段
        self.attn_query = nn.Parameter(torch.zeros(H))
        self.out_norm   = nn.LayerNorm(H)
        self.tf_proj = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, r_tf), nn.Tanh())
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)             # [B, H, L]
        x = torch.cat([conv(x) for conv in self.convs], dim=1)   # [B, H, L']
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                                    # [B, L', H]
        x = self.seq_norm(x)

        # 注意力池化：attn_query 学会聚焦最相关的序列区段
        scores = (x @ self.attn_query) / math.sqrt(x.size(-1))  # [B, L']
        attn   = F.softmax(scores, dim=-1)                       # [B, L']
        h = (attn.unsqueeze(-1) * x).sum(1)                     # [B, H]
        h = self.out_norm(h)

        gene_tf = self.tf_proj(h)                                # [B, r]
        return gene_tf, attn                                     # attn for interpretability


# ================================================================
# 4. 主模型：TFBilinearNet
# ================================================================

class TFBilinearNet(nn.Module):
    """
    TF 介导双线性交互网络。

    核心公式：
      logit = Σ_k W_k · drug_tf_k · gene_tf_k

    其中 W_k 是第 k 个 TF 概念的可学习重要性权重。

    Bayesian Probe Adaptation（推理时）：
      给定 K 个 probe 对 {(gene_j, label_j)}，用脊回归闭合形式
      更新 drug_tf，无需梯度下降，O(K³) 复杂度。
    """
    def __init__(self, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        self.r_tf     = r_tf
        self.drug_enc = DrugTFEncoder(hidden_dim, r_tf, dropout)
        self.gene_enc = GeneTFEncoder(4097, hidden_dim, r_tf, dropout)
        # TF 重要性权重：初始化为 1/√r，使 logit 的初始范围 ≈ [-1, 1]
        self.W_scale  = nn.Parameter(torch.ones(r_tf) / math.sqrt(r_tf))

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(torch.arange(B, device=device), num_nodes_t)

        drug_tf, atom_h  = self.drug_enc(x, edge_index, edge_attr, num_nodes_list, batch_idx)
        gene_tf, gene_attn = self.gene_enc(gene_ids)

        # 核心：TF 介导双线性交互
        logit = (drug_tf * gene_tf * self.W_scale).sum(-1)   # [B]
        return logit, drug_tf, gene_tf, atom_h, gene_attn

    def adapt_drug_tf(self, drug_tf_prior, probe_gene_tfs, probe_directions, lam=0.1):
        """
        Bayesian 后验更新：用 K 个 probe pair 精化 drug_tf。

        原理：probe (gene_j, direction_j) 告诉我们
            Σ_k W_k · drug_tf_k · gene_tf_j_k ≈ direction_j  （即 G·x ≈ d）
        这是关于 drug_tf 的线性约束。K 个 probe → K 个约束。

        闭合形式脊回归：
            x* = x_prior + G^T (G G^T + λI)^{-1} (d - G x_prior)
        其中 G = probe_gene_tfs × W_scale  ∈ R^{K×r}

        λ 平衡"相信 probe"（λ→0）与"保持结构先验"（λ→∞）。

        Args:
            drug_tf_prior:  [B, r] 来自 GIN 结构的先验
            probe_gene_tfs: [B, K, r] K 个 probe 基因的 TF 向量
            probe_directions: [B, K] ∈ {-1, +1}（2*label-1）
            lam: 正则化强度（控制先验 vs probe 的权重）

        Returns:
            drug_tf_adapted: [B, r] 后验估计
        """
        B, K, r = probe_gene_tfs.shape

        # 缩放 gene_tf 向量（与前向传播的计算方式一致）
        W = self.W_scale.unsqueeze(0).unsqueeze(0)      # [1, 1, r]
        G = probe_gene_tfs * W                          # [B, K, r]
        d = probe_directions.float().unsqueeze(-1)      # [B, K, 1]

        # 先验预测残差
        prior_pred = torch.bmm(G, drug_tf_prior.unsqueeze(-1))   # [B, K, 1]
        residual   = d - prior_pred                               # [B, K, 1]

        # G G^T + λI  [B, K, K]
        GGT = torch.bmm(G, G.transpose(1, 2)) + \
              lam * torch.eye(K, device=G.device).unsqueeze(0)

        # Solve (G G^T + λI) α = residual
        alpha = torch.linalg.solve(GGT, residual)       # [B, K, 1]

        # drug_tf_adapted = prior + G^T α
        update = torch.bmm(G.transpose(1, 2), alpha).squeeze(-1)  # [B, r]
        return drug_tf_prior + update                              # [B, r]


# ================================================================
# 5. 损失函数
# ================================================================

def compute_losses(logit, labels, drug_tf, gene_tf, args, criterion):
    """
    total = BCE
          + lam_sparse * (drug_tf L1 + gene_tf L1)   ← 稀疏：每次只激活少数 TF 通路
          + lam_ortho  * batch TF 去相关损失           ← 多样性：不同 TF 概念独立
    """
    loss_bce    = criterion(logit, labels)

    # L1 稀疏：每个药物/基因只激活少数 TF 通路
    loss_sparse = drug_tf.abs().mean() + gene_tf.abs().mean()

    # Batch TF 去相关：不同 TF 维度在 batch 内应不相关
    # 防止所有 TF 概念退化为同一个维度
    drug_centered  = drug_tf - drug_tf.mean(0, keepdim=True)   # [B, r]
    corr_drug = (drug_centered.T @ drug_centered) / (drug_tf.shape[0] - 1)  # [r, r]
    eye   = torch.eye(args.r_tf, device=drug_tf.device)
    loss_ortho = (corr_drug - eye).pow(2).mean()

    total = loss_bce + args.lam_sparse * loss_sparse + args.lam_ortho * loss_ortho
    return total, loss_bce, loss_sparse, loss_ortho


# ================================================================
# 6. Few-Shot Probe Evaluation
# ================================================================

def collect_val_representations(model, val_loader, device, args):
    """
    预计算验证集中所有 pair 的 (drug_tf, gene_tf, label, drug_id)。
    drug_tf 基于纯结构（GIN），不使用任何 probe 信息。
    """
    model.eval()
    drug_tfs, gene_tfs, labels_all, drug_ids = [], [], [], []

    with torch.no_grad(), autocast(enabled=args.use_amp):
        for batch in val_loader:
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            B = len(batch['num_nodes_list'])
            num_nodes_t = torch.tensor(batch['num_nodes_list'], device=device)
            batch_idx   = torch.repeat_interleave(
                torch.arange(B, device=device), num_nodes_t)

            drug_tf, _ = model.drug_enc(x, edge_index, edge_attr,
                                        batch['num_nodes_list'], batch_idx)
            gene_tf, _ = model.gene_enc(gene_ids)

            drug_tfs.append(drug_tf.cpu())
            gene_tfs.append(gene_tf.cpu())
            labels_all.append(batch['label'])
            drug_ids.extend(batch['drug_id'])

    return {
        'drug_tfs': torch.cat(drug_tfs),   # [N_val, r]
        'gene_tfs': torch.cat(gene_tfs),   # [N_val, r]
        'labels':   torch.cat(labels_all), # [N_val]
        'drug_ids': drug_ids,              # list of N_val SMILES strings
    }


def evaluate_fewshot(model, val_repr, device, args,
                     K_list=(0, 1, 5, 10, 20), n_trials=10, seed=0):
    """
    K-shot 评估：对每个测试药物，随机选 K 个 pair 作 probe，
    用 Bayesian adapter 更新 drug_tf，在剩余 pairs 上计算 AUC。

    K=0: 纯结构预测（standard chemical cold split）
    K>0: K 个 probe 信息提升预测

    n_trials: 每个 K 对每个药物重复 probe 选择次数（取平均）
    """
    model.eval()
    rng = np.random.RandomState(seed)

    # 按药物分组
    drug_to_idx = defaultdict(list)
    for i, did in enumerate(val_repr['drug_ids']):
        drug_to_idx[did].append(i)

    drug_tfs = val_repr['drug_tfs']
    gene_tfs = val_repr['gene_tfs']
    labels   = val_repr['labels']

    auc_results = {}
    model_W = model.W_scale.cpu()

    for K in K_list:
        all_preds, all_lbls = [], []

        for drug_id, idxs in drug_to_idx.items():
            n = len(idxs)
            if K == 0:
                # 纯结构预测
                dtf = drug_tfs[idxs]          # [n, r]
                gtf = gene_tfs[idxs]          # [n, r]
                logits = (dtf * gtf * model_W).sum(-1)  # [n]
                preds  = torch.sigmoid(logits).tolist()
                lbls   = labels[idxs].tolist()
                all_preds.extend(preds)
                all_lbls.extend(lbls)
                continue

            if n <= K:
                continue  # 需要至少 K+1 个 pair

            # 多次随机 probe 选择，结果取平均
            pair_pred_sums  = [0.0] * n
            pair_pred_cnt   = [0]   * n
            trial_count = min(n_trials, max(1, n // (K + 1)))

            for _ in range(trial_count):
                shuffled = rng.permutation(n)
                probe_local  = shuffled[:K]
                target_local = shuffled[K:]

                dtf_prior = drug_tfs[idxs[0]].unsqueeze(0).to(device)  # [1, r]
                probe_gtf = gene_tfs[[idxs[j] for j in probe_local]
                                     ].unsqueeze(0).to(device)          # [1, K, r]
                probe_dir = (2 * labels[[idxs[j] for j in probe_local]
                                        ].float() - 1
                             ).unsqueeze(0).to(device)                  # [1, K]

                with torch.no_grad():
                    dtf_adapted = model.adapt_drug_tf(
                        dtf_prior, probe_gtf, probe_dir, lam=args.adapt_lam)  # [1, r]

                dtf_adapted_cpu = dtf_adapted.cpu().squeeze(0)         # [r]
                for j in target_local:
                    gtf_j = gene_tfs[idxs[j]]
                    logit = (dtf_adapted_cpu * gtf_j * model_W).sum().item()
                    pred  = 1.0 / (1.0 + math.exp(-logit))
                    pair_pred_sums[j] += pred
                    pair_pred_cnt[j]  += 1

            for j in range(n):
                if pair_pred_cnt[j] > 0:
                    all_preds.append(pair_pred_sums[j] / pair_pred_cnt[j])
                    all_lbls.append(labels[idxs[j]].item())

        if len(set(all_lbls)) == 2:
            auc_results[K] = roc_auc_score(all_lbls, all_preds)
        else:
            auc_results[K] = float('nan')

    return auc_results


# ================================================================
# 7. LR 工具
# ================================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ================================================================
# 8. 训练主循环
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

    model = TFBilinearNet(
        hidden_dim=args.hidden_dim,
        r_tf=args.r_tf,
        dropout=args.dropout,
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
    model_name = f"tfbilinear_r{args.r_tf}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0
    base_lr = args.lr

    print(f"\n{'='*72}")
    print(f"  TFBilinearNet | r_tf={args.r_tf} | hidden_dim={args.hidden_dim}")
    print(f"  lam_sparse={args.lam_sparse} | lam_ortho={args.lam_ortho}")
    print(f"  drop_edge={args.drop_edge} | params={n_params:,}")
    print(f"  device={args.device} | fold={args.fold}")
    print(f"{'='*72}\n")

    for epoch in range(args.epochs):
        # LR warmup
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        total_loss = total_bce = total_sparse = total_ortho = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            labels     = batch['label'].to(device)

            # DropEdge
            if args.drop_edge > 0 and edge_index.shape[1] > 0:
                keep = torch.rand(edge_index.shape[1], device=device) > args.drop_edge
                edge_index = edge_index[:, keep]
                edge_attr  = edge_attr[keep]

            with autocast(enabled=args.use_amp):
                logit, drug_tf, gene_tf, _, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss, loss_bce, loss_sparse, loss_ortho = compute_losses(
                    logit, labels, drug_tf, gene_tf, args, criterion)

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

            total_loss   += loss.item()
            total_bce    += loss_bce.item()
            total_sparse += loss_sparse.item()
            total_ortho  += loss_ortho.item()

        # 标准验证（K=0）
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad(), autocast(enabled=args.use_amp):
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr  = batch['edge_attr'].to(device)
                gene_ids   = batch['gene_ids'].to(device)
                logit, _, _, _, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                all_preds.extend(torch.sigmoid(logit).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        if epoch >= args.warmup_epochs:
            scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} "
              f"SP:{total_sparse/n:.4f} "
              f"OR:{total_ortho/n:.4f}) | "
              f"VAL_AUC:{auroc:.4f} PRC:{auprc:.4f} F1:{f1:.4f}")

        if auroc > best_auroc:
            best_auroc   = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC (K=0): {best_auroc:.4f}")

    # ── Few-Shot 评估 ─────────────────────────────────────────────
    if args.eval_fewshot:
        print("\n[Few-Shot] 加载最优模型，运行 K-shot 评估...")
        model.load_state_dict(torch.load(save_dir / model_name, map_location=device))
        val_repr = collect_val_representations(model, val_loader, device, args)
        K_list = [0, 1, 2, 5, 10, 20]
        auc_by_K = evaluate_fewshot(model, val_repr, device, args,
                                    K_list=K_list, n_trials=20, seed=42)
        print("\n[Few-Shot 结果] AUC vs K probes：")
        print(f"{'K':>6}  {'AUC':>8}")
        for K in K_list:
            auc = auc_by_K.get(K, float('nan'))
            print(f"{K:>6}  {auc:>8.4f}")
        print(f"\n  K=0  → K=20 提升：{auc_by_K.get(20,0) - auc_by_K.get(0,0):+.4f}")

    print(f"\n模型保存于: {save_dir / model_name}")


# ================================================================
# 9. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFBilinearNet v14')

    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--seed',          type=int, default=42)

    parser.add_argument('--epochs',        type=int, default=100)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int, default=15)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--gene_max_len',  type=int, default=1000)
    parser.add_argument('--r_tf',         type=int, default=32,
                        help='TF 概念数（潜在 TF 通路数），建议 16-64')

    parser.add_argument('--lam_sparse',    type=float, default=0.01,
                        help='drug_tf/gene_tf L1 稀疏正则（鼓励少数 TF 通路激活）')
    parser.add_argument('--lam_ortho',     type=float, default=0.05,
                        help='batch TF 去相关损失（保证 TF 概念互相独立）')
    parser.add_argument('--drop_edge',     type=float, default=0.0,
                        help='训练时 DropEdge 概率')

    parser.add_argument('--eval_fewshot',  action='store_true',
                        help='训练后运行 K-shot probe evaluation')
    parser.add_argument('--adapt_lam',     type=float, default=0.1,
                        help='Bayesian adapter 的脊正则系数（控制先验 vs probe 权重）')

    parser.add_argument('--run_tag',       type=str, default='')

    train(parser.parse_args())
