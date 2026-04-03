#!/usr/bin/env python3
"""
extract_representations.py
===========================
从已训练的 DrugOperatorNet (no_moe) 模型中提取：
  - spectrum [N_val, r]：交互谱（可解释性核心）
  - atom_attn [N_val, r, max_atoms]：原子注意力权重（药效团热图）
  - gene_attn [N_val, r, L']：基因注意力权重（调控位点）
  - smiles [N_val]：对应 SMILES 字符串
  - labels [N_val]：真实标签
  - preds [N_val]：模型预测概率

用法:
  python analyze/extract_representations.py \
    --data_dir /path/to/MCF7 \
    --model_path results_operator_moe/MCF7/no_moe_r8_k4_Fold0_v1.pt \
    --output_dir analyze/cache/MCF7 \
    --fold 0 --device cuda:0

输出 (output_dir/):
  representations.npz：所有中间表示
  summary.json：基本统计信息
"""

import sys
import argparse
import json
import math
import itertools
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────
# 复制模型定义（独立于训练脚本，避免 import 依赖）
# ────────────────────────────────────────────────────────────────────

_KMER_VOCAB = {}
for _i, _combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(_combo)] = _i
_KMER_VOCAB['NNNNNN'] = 0


def encode_kmer_sequence(sequence, k=6, max_len=1000):
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
    scores = scores.float()
    max_s = torch.zeros(batch_idx.max().item() + 1, device=scores.device)
    max_s.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
    exp_s = torch.exp(scores - max_s[batch_idx])
    exp_sum = torch.zeros(batch_idx.max().item() + 1, device=scores.device)
    exp_sum.index_add_(0, batch_idx, exp_s)
    return exp_s / (exp_sum[batch_idx] + 1e-8)


def scatter_add(src, batch_idx, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


class GeneMultiHeadReader(nn.Module):
    def __init__(self, vocab_size=4097, hidden_dim=128, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        H = hidden_dim
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes])
        self.seq_norm = nn.LayerNorm(H)
        self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))
        self.out_norm = nn.LayerNorm(H)
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = self.seq_norm(x)
        scores = torch.einsum('blh,rh->brl', x, self.attn_queries) / math.sqrt(x.size(-1))
        attn = F.softmax(scores, dim=-1)
        h_g_modes = torch.einsum('brl,blh->brh', attn, x)
        h_g_modes = self.out_norm(h_g_modes)
        h_g_global = h_g_modes.mean(dim=1)
        return h_g_modes, h_g_global, attn  # attn: [B, r, L']


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
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        return x


class PharmacophoreExtractor(nn.Module):
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs, return_attn=False):
        """
        Args:
          atom_h:      [N_total, H]  所有分子原子嵌入
          batch_idx:   [N_total]     每个原子所属分子编号
          num_graphs:  int           batch 分子数 B
          return_attn: bool          是否返回原子注意力权重（推理/可视化时用）

        Returns:
          pharma:      [B, r, H]    r 个药效团嵌入（归一化后）
          scores_all:  [N_total, r] 原子对每个 slot 的原始得分
          atom_attn:   [N_total, r] 图内归一化的注意力权重（仅 return_attn=True 时）
        """
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)                              # [N_total, H]
        V = self.val_proj(atom_h)                              # [N_total, H]
        # 所有原子 × 所有 slot 的相似度（一次矩阵乘，效率高）
        scores_all = (K @ self.queries.T) / math.sqrt(d)      # [N_total, r]

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)

        # 逐 slot 做图内 softmax + 加权聚合
        # 为保证每个分子内独立归一化，必须用 scatter_softmax（不能用普通 softmax）
        attn_weights = []
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)  # [N_total]，图内归一化
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))
            if return_attn:
                attn_weights.append(alpha)  # 保存每个 slot 的原子权重

        if return_attn:
            # [N_total, r]：每个原子在每个药效团 slot 上的注意力权重
            # → 用于 visualize_pharmacophore.py 的分子热图
            return self.norm(pharma), scores_all, torch.stack(attn_weights, dim=1)
        return self.norm(pharma), scores_all


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
        U = F.normalize(self.to_u(pharma_emb), dim=-1)
        V = F.normalize(self.to_v(pharma_emb), dim=-1)
        sigma = self.to_sigma(pharma_emb).squeeze(-1)
        coupling = (V * h_g_modes).sum(-1)
        spectrum = sigma * coupling
        delta_h = (spectrum.unsqueeze(-1) * U).sum(dim=1)
        return delta_h, spectrum, sigma, U


class DrugOperatorNet(nn.Module):
    """no_moe 配置的主模型（用于推理）"""
    def __init__(self, hidden_dim=128, dropout=0.3, operator_rank=8):
        super().__init__()
        self.operator_rank = operator_rank
        H, r = hidden_dim, operator_rank
        self.gene_enc = GeneMultiHeadReader(hidden_dim=H, num_heads=r, dropout=dropout)
        self.atom_enc = AtomEncoder(hidden_dim=H, dropout=dropout)
        self.pharma_ext = PharmacophoreExtractor(H, r)
        self.perturb_op = PerturbationOperator(H)
        self.classifier = nn.Sequential(
            nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(H, 1))

    def forward_with_attn(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        """
        完整前向传播，返回所有中间表示（仅推理时调用，训练不用）。

        与训练时 forward 的区别：
          1. 调用 PharmacophoreExtractor(return_attn=True) 获取原子注意力权重
          2. 返回 dict 包含所有可解释性所需的中间量

        返回 dict 各字段：
          preds:          [B]          预测概率
          spectrum:       [B, r]       交互谱（药效团-基因耦合强度）
          sigma:          [B, r]       各模式幅度（Tanh 输出，有界）
          gene_attn:      [B, r, L']   基因序列注意力权重（L'=卷积后序列长度）
          atom_attn:      [N_total, r] 原子对每个药效团 slot 的注意力（图内归一化）
          batch_idx:      [N_total]    每个原子所属分子编号（拆分用）
          num_nodes_list: [B]          每个分子的原子数（atom_attn 拆分用）
        """
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        # batch_idx[i] = 第 i 个原子属于第几个分子（0-indexed）
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), num_nodes_t)

        # ① 基因编码：r 个多头视角
        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)
        # gene_attn: [B, r, L'] → 基因序列上哪些位置对哪个模式最重要

        # ② 原子编码：3 层 GIN
        atom_h = self.atom_enc(x, edge_index, edge_attr)  # [N_total, H]

        # ③ 药效团提取（带注意力权重输出）
        pharma_emb, _, atom_attn = self.pharma_ext(
            atom_h, batch_idx, B, return_attn=True)
        # atom_attn: [N_total, r] → 原子 i 对药效团 slot s 的贡献权重

        # ④ 扰动算子：计算交互谱和净扰动
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        # ⑤ 分类
        features = torch.cat([h_g_global, delta_h], dim=-1)  # [B, 2H]
        logits = self.classifier(features).squeeze(-1)        # [B]
        preds = torch.sigmoid(logits)                         # [B]，概率值

        return {
            'preds': preds,                     # [B]
            'spectrum': spectrum,               # [B, r]
            'sigma': sigma,                     # [B, r]
            'gene_attn': gene_attn,             # [B, r, L']
            'atom_attn': atom_attn,             # [N_total, r]
            'batch_idx': batch_idx,             # [N_total]
            'num_nodes_list': num_nodes_list,   # 每个分子的原子数
        }


# ────────────────────────────────────────────────────────────────────
# 数据集（轻量版，只加载验证集，并保留 SMILES）
# ────────────────────────────────────────────────────────────────────

class ValDatasetWithSMILES(Dataset):
    def __init__(self, data_dir, fold_idx=0, gene_max_len=1000):
        data_dir = Path(data_dir)
        with open(data_dir / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        val_indices = splits[fold_idx][1]

        cell_line = data_dir.name
        data = torch.load(data_dir / f'preprocessed_graphs_{cell_line}.pt',
                          map_location='cpu')

        self.smiles_to_graph = data['smiles_to_graph']
        self.smiles_list = [data['graph_indices'][i] for i in val_indices]
        self.labels = torch.tensor(
            [data['labels'][i] for i in val_indices], dtype=torch.float32)

        suffix = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        cache_file = data_dir / f'kmer_cache_fold{fold_idx}_val{suffix}.pt'
        if cache_file.exists():
            self.gene_ids = torch.load(cache_file)
        else:
            gene_seqs = [data['gene_sequences'][i] for i in val_indices]
            print("生成 k-mer 缓存...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len)
                 for seq in tqdm(gene_seqs)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles_list[idx],
            'graph': self.smiles_to_graph[self.smiles_list[idx]],
            'gene_ids': self.gene_ids[idx],
            'label': self.labels[idx],
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
        'smiles': [b['smiles'] for b in batch],
        'x': torch.cat(all_x, dim=0),
        'edge_index': torch.cat(all_edge_index, dim=1) if all_edge_index
                      else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr': torch.cat(all_edge_attr, dim=0) if all_edge_attr
                     else torch.zeros(0, 4),
        'num_nodes_list': num_nodes_list,
        'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


# ────────────────────────────────────────────────────────────────────
# 主提取函数
# ────────────────────────────────────────────────────────────────────

def extract(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print(f"\n[1/4] 加载验证集: {args.data_dir} fold={args.fold}")
    dataset = ValDatasetWithSMILES(args.data_dir, args.fold, args.gene_max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    print(f"  验证集大小: {len(dataset)}")

    # 加载模型
    print(f"\n[2/4] 加载模型: {args.model_path}")
    model = DrugOperatorNet(
        hidden_dim=args.hidden_dim,
        dropout=0.0,  # 推理时关闭 dropout
        operator_rank=args.operator_rank,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    # 兼容 OperatorMoE 的权重（no_moe 配置）
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 推理
    print(f"\n[3/4] 提取表示...")
    all_preds, all_labels, all_smiles = [], [], []
    all_spectrum, all_sigma = [], []
    all_gene_attn = []
    # atom_attn 因分子大小不同，按分子存储
    all_atom_attn_list = []  # list of [num_atoms, r]
    all_num_nodes = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting'):
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            gene_ids = batch['gene_ids'].to(device)

            out = model.forward_with_attn(
                gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

            B = len(batch['num_nodes_list'])
            preds = out['preds'].cpu().numpy()
            labels = batch['label'].numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_smiles.extend(batch['smiles'])
            all_spectrum.append(out['spectrum'].cpu().numpy())
            all_sigma.append(out['sigma'].cpu().numpy())
            # gene_attn: [B, r, L'] → 只存均值（节省内存）
            # 完整的 L' 维度太大，按需存
            all_gene_attn.append(out['gene_attn'].mean(dim=-1).cpu().numpy())  # [B, r]

            # atom_attn: 按分子分割
            atom_attn_np = out['atom_attn'].cpu().numpy()  # [N_total, r]
            batch_idx_np = out['batch_idx'].cpu().numpy()
            for b in range(B):
                mask = batch_idx_np == b
                all_atom_attn_list.append(atom_attn_np[mask])  # [n_atoms_b, r]
                all_num_nodes.append(batch['num_nodes_list'][b])

    # 合并
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_spectrum = np.concatenate(all_spectrum, axis=0)   # [N, r]
    all_sigma = np.concatenate(all_sigma, axis=0)         # [N, r]
    all_gene_attn_mean = np.concatenate(all_gene_attn, axis=0)  # [N, r]

    auroc = roc_auc_score(all_labels, all_preds)
    print(f"\n  AUC: {auroc:.4f} (验证集确认一致性)")

    # 保存
    print(f"\n[4/4] 保存到: {output_dir}")
    np.savez_compressed(
        output_dir / 'representations.npz',
        preds=all_preds,
        labels=all_labels,
        spectrum=all_spectrum,
        sigma=all_sigma,
        gene_attn_mean=all_gene_attn_mean,
    )

    # SMILES 和原子注意力单独存（变长）
    np.save(output_dir / 'smiles.npy', np.array(all_smiles, dtype=object))
    np.save(output_dir / 'num_nodes.npy', np.array(all_num_nodes))

    # 原子注意力：每个分子保存为 object array（各分子原子数不同）
    atom_attn_arr = np.empty(len(all_atom_attn_list), dtype=object)
    for i, a in enumerate(all_atom_attn_list):
        atom_attn_arr[i] = a
    np.save(output_dir / 'atom_attn.npy', atom_attn_arr, allow_pickle=True)

    summary = {
        'cell_line': Path(args.data_dir).name,
        'fold': args.fold,
        'model_path': str(args.model_path),
        'n_val': int(len(all_labels)),
        'n_pos': int(all_labels.sum()),
        'n_neg': int((1 - all_labels).sum()),
        'auc': float(auroc),
        'operator_rank': args.operator_rank,
        'spectrum_shape': list(all_spectrum.shape),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 完成")
    print(f"  n_val={len(all_labels)}, pos={all_labels.sum():.0f}, "
          f"neg={(1-all_labels).sum():.0f}")
    print(f"  AUC={auroc:.4f}")
    print(f"  spectrum.shape={all_spectrum.shape}")
    print(f"  输出目录: {output_dir}")


# ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--operator_rank', type=int, default=8)
    parser.add_argument('--gene_max_len', type=int, default=1000)
    args = parser.parse_args()
    extract(args)
