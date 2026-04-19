#!/usr/bin/env python3
"""
v18: TFBilinear + TanimotoCL + Promoter DNA Gene Encoder
=========================================================

v16 基础 (L2 norm × sqrt(r)) + 两项关键改进：

1. TanimotoCL（Tanimoto 软标签对比学习）
   - 对批内药物对，计算 Morgan FP Tanimoto 相似度 T_ij
   - 目标：cos(drug_tf_i, drug_tf_j) ≈ 2·T_ij - 1  (仅对 T_ij >= threshold 的对)
   - 物理意义：结构相似的药物应激活相似的 TF 组合
   - 设计原则：三种"同基因同效应"情况：
       (1) 同骨架/同靶点 → 高 Tanimoto → 有效 CL，强制相似 drug_tf
       (2) 不同骨架汇聚通路 → 低 Tanimoto → 不强制，避免噪声
       (3) 非特异效应 → 低 Tanimoto → 不强制
     threshold=0.3 正确过滤 (2)(3)，只约束 (1)

2. 启动子 DNA 基因编码器（如果 promoter_sequences.json 存在）
   - 用 TSS 上游 2kb + 下游 200bp 的启动子 DNA 替换 mRNA 序列
   - 物理意义：TF 结合位点在启动子区（TATA box, BRE 等），不在 mRNA
   - 截断策略：取最后 max_len 个 k-mer（保留 TSS 近端核心启动子区）
   - 若启动子数据不可用，自动回退到 mRNA 序列
"""

import argparse
import json
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

# RDKit for Morgan fingerprints
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs as _DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress RDKit deprecation warnings

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


def encode_kmer_sequence(sequence, k=6, max_len=1000, stride=1, from_end=False):
    """
    编码 DNA/RNA 序列为 k-mer ID 序列。
    from_end=True: 从序列末端截取 max_len 个 k-mer（用于启动子序列，保留 TSS 近端）
    """
    kmers = []
    for i in range(0, len(sequence) - k + 1, stride):
        kmer = sequence[i:i+k].upper()
        if any(c not in 'ACGT' for c in kmer):
            kmer = 'N' * k
        kmers.append(_KMER_VOCAB.get(kmer, 0))

    if len(kmers) > max_len:
        if from_end:
            kmers = kmers[-max_len:]   # 保留 TSS 近端（启动子末端）
        else:
            kmers = kmers[:max_len]    # 保留序列前端（mRNA 5' 端）
    else:
        kmers += [0] * (max_len - len(kmers))
    return kmers


def smiles_to_morgan_fp(smiles: str, radius=2, n_bits=2048) -> np.ndarray:
    """计算 Morgan 指纹（ECFP4），返回 bool numpy 数组。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=bool)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=bool)
    _DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


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
# 1. 数据集（新增 Morgan FP、启动子序列支持）
# ================================================================

PROMOTER_JSON = Path('/home/data/jiangyun/cgi_data_pipeline/outputs/gene_data/promoter_sequences.json')
MRNA_JSON     = Path('/home/data/jiangyun/cgi_data_pipeline/outputs/gene_data/gene_sequences.json')


def _load_promoter_map():
    """
    构建 mRNA序列 → 启动子序列 的映射。
    通过 gene_sequences.json (gene_id→mRNA) 和 promoter_sequences.json (gene_id→promoter) 关联。
    """
    if not PROMOTER_JSON.exists() or not MRNA_JSON.exists():
        return {}
    with open(MRNA_JSON) as f:
        mrna_map = json.load(f)   # {gene_id: mrna_seq}
    with open(PROMOTER_JSON) as f:
        prom_map = json.load(f)   # {gene_id: promoter_seq}
    # 反向：mRNA前100bp → promoter_seq（用前100bp作唯一键）
    result = {}
    for gid, mseq in mrna_map.items():
        if gid in prom_map and prom_map[gid]:
            result[mseq[:100]] = prom_map[gid]
    return result


class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train', gene_max_len=1000,
                 gene_stride=1, use_promoter=True, morgan_radius=2, morgan_bits=2048):
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

        # ── 启动子 / mRNA 序列选择 ──────────────────────────────
        prom_map = _load_promoter_map() if use_promoter else {}
        n_prom = 0
        final_seqs = []
        for seq in gene_sequences:
            key = seq[:100]
            if key in prom_map:
                final_seqs.append(prom_map[key])
                n_prom += 1
            else:
                final_seqs.append(seq)

        from_end = use_promoter and len(prom_map) > 0
        seq_type = 'prom' if from_end else 'mrna'
        if split == 'train':
            pct = n_prom / len(final_seqs) * 100 if final_seqs else 0
            print(f"[{split.upper()}] 启动子序列覆盖率: {n_prom}/{len(final_seqs)} ({pct:.1f}%)")

        # ── K-mer 缓存 ─────────────────────────────────────────
        len_tag    = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        stride_tag = '' if gene_stride == 1      else f'_s{gene_stride}'
        cache_file = Path(data_dir) / \
            f'kmer_cache_{seq_type}_fold{fold_idx}_{split}{len_tag}{stride_tag}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file, weights_only=True)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 ({seq_type})...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len, stride=gene_stride,
                                      from_end=from_end)
                 for seq in tqdm(final_seqs)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

        # ── Morgan 指纹预计算（唯一 SMILES）────────────────────
        fp_cache = Path(data_dir) / f'morgan_fp_r{morgan_radius}_b{morgan_bits}.pt'
        if fp_cache.exists():
            if split == 'train':
                print(f"[{split.upper()}] ⚡ Morgan FP 缓存: {fp_cache.name}")
            self.fp_dict = torch.load(fp_cache, weights_only=True)
        else:
            if split == 'train':
                print(f"[{split.upper()}] 计算 Morgan FP ({len(self.smiles_to_graph)} SMILES)...")
            fp_dict = {}
            for smi in tqdm(self.smiles_to_graph.keys(), disable=(split != 'train')):
                fp_dict[smi] = torch.tensor(
                    smiles_to_morgan_fp(smi, morgan_radius, morgan_bits), dtype=torch.bool)
            torch.save(fp_dict, fp_cache)
            self.fp_dict = fp_dict

        self.morgan_bits = morgan_bits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smiles = self.graph_indices[idx]
        fp = self.fp_dict.get(smiles,
                              torch.zeros(self.morgan_bits, dtype=torch.bool))
        item = {
            'graph':    self.smiles_to_graph[smiles],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
            'drug_id':  smiles,
            'fp':       fp,
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
        'fp':             torch.stack([b['fp'] for b in batch]),  # [B, 2048] bool
    }


# ================================================================
# 2. Drug Encoder
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
    def __init__(self, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        H = hidden_dim
        self.r_tf = r_tf
        self.atom_embed = nn.Sequential(
            nn.Linear(31, H), nn.BatchNorm1d(H), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(H) for _ in range(3)])
        self.norms      = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(), nn.Dropout(dropout))
        self.tf_proj = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, r_tf))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, num_nodes_list, batch_idx):
        h = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            h = F.dropout(F.relu(norm(h + gin(h, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        atom_h = h

        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=x.device)
        sum_pool  = scatter_add(atom_h, batch_idx, B)
        mean_pool = sum_pool / num_nodes_t.float().unsqueeze(1).clamp(min=1)
        mol_h = self.readout(torch.cat([sum_pool, mean_pool], dim=-1))
        drug_tf_raw = self.tf_proj(mol_h)
        drug_tf = F.normalize(drug_tf_raw, dim=-1) * math.sqrt(self.r_tf)
        return drug_tf, atom_h


# ================================================================
# 3. Gene Encoder
# ================================================================

class GeneTFEncoder(nn.Module):
    def __init__(self, vocab_size=4097, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        H = hidden_dim
        self.r_tf = r_tf
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes   = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes])
        self.seq_norm = nn.LayerNorm(H)
        self.attn_query = nn.Parameter(torch.zeros(H))
        self.out_norm   = nn.LayerNorm(H)
        self.tf_proj = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, r_tf))
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = self.seq_norm(x)

        scores = (x @ self.attn_query) / math.sqrt(x.size(-1))
        attn   = F.softmax(scores, dim=-1)
        h = (attn.unsqueeze(-1) * x).sum(1)
        h = self.out_norm(h)

        gene_tf_raw = self.tf_proj(h)
        gene_tf = F.normalize(gene_tf_raw, dim=-1) * math.sqrt(self.r_tf)
        return gene_tf, attn


# ================================================================
# 4. TanimotoCL
# ================================================================

class TanimotoCL(nn.Module):
    """
    Tanimoto 软标签对比学习。

    对批内所有药物对，若 T(i,j) >= threshold，则
    强制 cos(drug_tf_i, drug_tf_j) ≈ 2·T(i,j) - 1

    将 Morgan FP Tanimoto 相似度映射到 [-1, 1]：
      T=1.0 → target= 1.0 (完全相同骨架)
      T=0.3 → target=-0.4 (中等相似)
      T=0.0 → target=-1.0 (完全不同)

    仅约束高相似对（T >= threshold），低相似对不强制解相关。
    """
    def __init__(self, threshold=0.3):
        super().__init__()
        self.threshold = threshold

    def forward(self, drug_tf, fps, drug_ids):
        """
        drug_tf: [B, r]
        fps:     [B, 2048] bool tensor
        drug_ids: list of B SMILES strings
        """
        # ── 唯一药物聚合 ─────────────────────────────────────────
        uid_to_first = {}   # smiles → first batch index
        uid_list     = []
        for i, did in enumerate(drug_ids):
            if did not in uid_to_first:
                uid_to_first[did] = i
                uid_list.append(did)
        uid_to_idx = {did: j for j, did in enumerate(uid_list)}
        sample_uid = [uid_to_idx[did] for did in drug_ids]

        D = len(uid_list)
        if D < 2:
            return torch.tensor(0.0, device=drug_tf.device)

        # 每个唯一药物取均值 drug_tf
        uid_t = torch.tensor(sample_uid, device=drug_tf.device)
        dtf_uniq = torch.zeros(D, drug_tf.size(1),
                               device=drug_tf.device, dtype=drug_tf.dtype)
        dtf_uniq.index_add_(0, uid_t, drug_tf.detach() if False else drug_tf)
        counts = torch.bincount(uid_t, minlength=D).float().unsqueeze(1)
        dtf_uniq = dtf_uniq / counts.clamp(min=1)

        # 取每个唯一药物的 FP（首次出现）
        first_idx = [uid_to_first[uid] for uid in uid_list]
        fp_uniq = fps[first_idx].float()  # [D, 2048]

        # ── Tanimoto 计算 ─────────────────────────────────────────
        dot   = fp_uniq @ fp_uniq.T                                  # [D, D]
        norms = fp_uniq.sum(1)                                        # [D]
        denom = norms.unsqueeze(1) + norms.unsqueeze(0) - dot        # [D, D]
        tanimoto = dot / (denom + 1e-8)                              # [D, D]

        # ── 余弦相似度 ────────────────────────────────────────────
        dtf_norm = F.normalize(dtf_uniq.float(), dim=1)              # [D, r]
        cosine   = dtf_norm @ dtf_norm.T                             # [D, D]

        # ── 损失：仅对高相似对 ────────────────────────────────────
        triu_mask = torch.triu(torch.ones(D, D, dtype=torch.bool,
                               device=drug_tf.device), diagonal=1)
        pos_mask = (tanimoto >= self.threshold) & triu_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=drug_tf.device)

        target = 2.0 * tanimoto - 1.0          # maps [0,1] → [-1,1]
        loss = F.mse_loss(cosine[pos_mask], target[pos_mask])
        return loss


# ================================================================
# 5. 主模型
# ================================================================

class TFBilinearNet(nn.Module):
    def __init__(self, hidden_dim=128, r_tf=32, dropout=0.3):
        super().__init__()
        self.r_tf     = r_tf
        self.drug_enc = DrugTFEncoder(hidden_dim, r_tf, dropout)
        self.gene_enc = GeneTFEncoder(4097, hidden_dim, r_tf, dropout)
        self.W_scale  = nn.Parameter(torch.ones(r_tf) / math.sqrt(r_tf))

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(torch.arange(B, device=device), num_nodes_t)

        drug_tf, atom_h    = self.drug_enc(x, edge_index, edge_attr, num_nodes_list, batch_idx)
        gene_tf, gene_attn = self.gene_enc(gene_ids)

        logit = (drug_tf * gene_tf * self.W_scale).sum(-1)
        return logit, drug_tf, gene_tf, atom_h, gene_attn

    def adapt_drug_tf(self, drug_tf_prior, probe_gene_tfs, probe_directions, lam=0.1):
        B, K, r = probe_gene_tfs.shape
        drug_tf_prior    = drug_tf_prior.float()
        probe_gene_tfs   = probe_gene_tfs.float()
        probe_directions = probe_directions.float()

        W = self.W_scale.float().unsqueeze(0).unsqueeze(0)
        G = probe_gene_tfs * W
        d = probe_directions.unsqueeze(-1)

        prior_pred = torch.bmm(G, drug_tf_prior.unsqueeze(-1))
        residual   = d - prior_pred
        GGT = torch.bmm(G, G.transpose(1, 2)) + \
              lam * torch.eye(K, device=G.device).unsqueeze(0)
        alpha  = torch.linalg.solve(GGT, residual)
        update = torch.bmm(G.transpose(1, 2), alpha).squeeze(-1)
        return drug_tf_prior + update


# ================================================================
# 6. 损失函数
# ================================================================

def compute_losses(logit, labels, drug_tf, gene_tf, fps, drug_ids,
                   args, criterion, tan_module=None):
    loss_bce = criterion(logit, labels)

    drug_centered = drug_tf - drug_tf.mean(0, keepdim=True)
    corr_drug = (drug_centered.T @ drug_centered) / (drug_tf.shape[0] - 1)
    eye = torch.eye(args.r_tf, device=drug_tf.device)
    loss_ortho = (corr_drug - eye).pow(2).mean()

    if tan_module is not None and args.lam_tan > 0:
        loss_tan = tan_module(drug_tf, fps, drug_ids)
    else:
        loss_tan = torch.zeros(1, device=logit.device)[0]

    total = loss_bce + args.lam_ortho * loss_ortho + args.lam_tan * loss_tan
    return total, loss_bce, loss_ortho, loss_tan


# ================================================================
# 7. 验证集收集 + Few-Shot 评估
# ================================================================

def collect_val_representations(model, val_loader, device, args):
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
        'drug_tfs': torch.cat(drug_tfs),
        'gene_tfs': torch.cat(gene_tfs),
        'labels':   torch.cat(labels_all),
        'drug_ids': drug_ids,
    }


def evaluate_fewshot(model, val_repr, device, args,
                     K_list=(0, 1, 5, 10, 20), n_trials=10, seed=0):
    model.eval()
    rng = np.random.RandomState(seed)
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
                dtf = drug_tfs[idxs]
                gtf = gene_tfs[idxs]
                logits = (dtf * gtf * model_W).sum(-1)
                all_preds.extend(torch.sigmoid(logits).tolist())
                all_lbls.extend(labels[idxs].tolist())
                continue
            if n <= K:
                continue
            pair_pred_sums = [0.0] * n
            pair_pred_cnt  = [0]   * n
            trial_count = min(n_trials, max(1, n // (K + 1)))
            for _ in range(trial_count):
                shuffled     = rng.permutation(n)
                probe_local  = shuffled[:K]
                target_local = shuffled[K:]
                dtf_prior = drug_tfs[idxs[0]].unsqueeze(0).to(device)
                probe_gtf = gene_tfs[[idxs[j] for j in probe_local]].unsqueeze(0).to(device)
                probe_dir = (2 * labels[[idxs[j] for j in probe_local]].float() - 1
                             ).unsqueeze(0).to(device)
                with torch.no_grad():
                    dtf_adapted = model.adapt_drug_tf(
                        dtf_prior, probe_gtf, probe_dir, lam=args.adapt_lam)
                dtf_cpu = dtf_adapted.cpu().squeeze(0)
                for j in target_local:
                    gtf_j = gene_tfs[idxs[j]]
                    logit = (dtf_cpu * gtf_j * model_W).sum().item()
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
# 8. LR 工具
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
        args.data_dir, args.fold, 'train', args.gene_max_len,
        use_promoter=args.use_promoter)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val', args.gene_max_len,
        use_promoter=args.use_promoter)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    model = TFBilinearNet(
        hidden_dim=args.hidden_dim,
        r_tf=args.r_tf,
        dropout=args.dropout,
    ).to(device)

    tan_module = TanimotoCL(threshold=args.tan_threshold).to(device) \
                 if args.lam_tan > 0 else None

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

    seq_label = 'prom' if args.use_promoter and PROMOTER_JSON.exists() else 'mrna'
    print(f"\n{'='*72}")
    print(f"  TFBilinearNet v18 (TanimotoCL + {seq_label}) | r_tf={args.r_tf}")
    print(f"  lam_ortho={args.lam_ortho} | lam_tan={args.lam_tan} | tan_thr={args.tan_threshold}")
    print(f"  drop_edge={args.drop_edge} | params={n_params:,}")
    print(f"  device={args.device} | fold={args.fold}")
    print(f"{'='*72}\n")

    best_auroc, patience_cnt = 0.0, 0
    base_lr = args.lr

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        if tan_module is not None:
            tan_module.train()
        total_loss = total_bce = total_ortho = total_tan = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            labels     = batch['label'].to(device)
            fps        = batch['fp'].to(device)
            drug_ids   = batch['drug_id']

            if args.drop_edge > 0 and edge_index.shape[1] > 0:
                keep = torch.rand(edge_index.shape[1], device=device) > args.drop_edge
                edge_index = edge_index[:, keep]
                edge_attr  = edge_attr[keep]

            with autocast(enabled=args.use_amp):
                logit, drug_tf, gene_tf, _, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss, loss_bce, loss_ortho, loss_tan = compute_losses(
                    logit, labels, drug_tf, gene_tf, fps, drug_ids,
                    args, criterion, tan_module)

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

            total_loss  += loss.item()
            total_bce   += loss_bce.item()
            total_ortho += loss_ortho.item()
            total_tan   += loss_tan.item() if torch.is_tensor(loss_tan) else loss_tan

        # 验证
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
              f"OR:{total_ortho/n:.4f} "
              f"TAN:{total_tan/n:.4f}) | "
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

    if args.eval_fewshot:
        print("\n[Few-Shot] 加载最优模型，运行 K-shot 评估...")
        model.load_state_dict(torch.load(save_dir / model_name, map_location=device))
        val_repr = collect_val_representations(model, val_loader, device, args)
        K_list = [0, 1, 2, 5, 10, 20]
        auc_by_K = evaluate_fewshot(model, val_repr, device, args,
                                    K_list=K_list, n_trials=20, seed=42)
        print("\n[Few-Shot 结果] AUC vs K probes：")
        print(f"{'K':>6}  {'AUC':>8}  {'delta':>8}")
        auc0 = auc_by_K.get(0, float('nan'))
        for K in K_list:
            auc = auc_by_K.get(K, float('nan'))
            print(f"{K:>6}  {auc:>8.4f}  {auc-auc0:>+8.4f}")

    print(f"\n模型保存于: {save_dir / model_name}")


# ================================================================
# 10. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFBilinearNet v18 (TanimotoCL + Promoter)')

    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--seed',          type=int, default=42)

    parser.add_argument('--epochs',        type=int, default=120)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int, default=15)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--gene_max_len',  type=int, default=1000)
    parser.add_argument('--r_tf',          type=int, default=32)

    parser.add_argument('--lam_ortho',     type=float, default=0.05)
    parser.add_argument('--lam_tan',       type=float, default=0.1,
                        help='Tanimoto CL 损失权重')
    parser.add_argument('--tan_threshold', type=float, default=0.3,
                        help='最低 Tanimoto 阈值（高于此才约束 drug_tf 相似）')
    parser.add_argument('--drop_edge',     type=float, default=0.0)

    parser.add_argument('--use_promoter',  action='store_true', default=True,
                        help='使用启动子序列替代 mRNA（如果可用）')
    parser.add_argument('--no_promoter',   dest='use_promoter', action='store_false')

    parser.add_argument('--eval_fewshot',  action='store_true')
    parser.add_argument('--adapt_lam',     type=float, default=0.1)
    parser.add_argument('--run_tag',       type=str, default='')

    train(parser.parse_args())
