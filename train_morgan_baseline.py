#!/usr/bin/env python3
"""
Morgan Fingerprint Baseline
============================
用途：验证我们 GIN 化学编码器的有效性。

设计原则（单变量对照）：
  - 化学编码：GIN×3 → 替换为 Morgan 指纹（ECFP4, radius=2, 2048-bit）→ Linear(2048, H)
  - 基因编码：GeneEncoderV1（与所有方法完全一致）
  - 下游架构：Ortho 剥离 + MLP 分类头（与 train_baseline_bce.py 完全一致）
  - 训练设置：AdamW + ReduceLROnPlateau + early stop + AMP

Morgan 指纹是药物 ML 的传统金标准（ECFP4），广泛用于各类 QSAR/ADMET 任务。
若我们的 GIN 在 chemical cold split 下优于 Morgan FP，说明端到端图学习
能捕捉到固定指纹遗漏的结构-活性关系。

依赖：rdkit（生成指纹），其余与项目一致。
"""

import argparse
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')   # 屏蔽 RDKit DEPRECATION WARNING
RDLogger.DisableLog('rdApp.warning')

# ================================================================
# 0. K-mer 工具
# ================================================================

_KMER_VOCAB = {}
for i, combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(combo)] = i
_KMER_VOCAB['NNNNNN'] = 0

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_kmer_sequence(sequence: str, k: int = 6, max_len: int = 1000) -> list:
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

def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * n_bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)

# ================================================================
# 1. 数据集（预计算 Morgan 指纹）
# ================================================================

class MorganGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train',
                 morgan_radius=2, morgan_bits=2048):
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

        # --- 基因 K-mer 缓存 ---
        cache_gene = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}.pt'
        if cache_gene.exists():
            print(f"[{split.upper()}] ⚡ 基因缓存: {cache_gene.name}")
            self.gene_ids = torch.load(cache_gene)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)],
                dtype=torch.long)
            torch.save(self.gene_ids, cache_gene)

        # --- Morgan 指纹缓存 ---
        cache_morgan = Path(data_dir) / \
            f'morgan_cache_fold{fold_idx}_{split}_r{morgan_radius}_b{morgan_bits}.pt'
        if cache_morgan.exists():
            print(f"[{split.upper()}] ⚡ Morgan 缓存: {cache_morgan.name}")
            self.morgan_fps = torch.load(cache_morgan)
        else:
            print(f"[{split.upper()}] 生成 Morgan 指纹（r={morgan_radius}, bits={morgan_bits}）...")
            # graph_indices 存的就是 SMILES 字符串，直接用
            fps = []
            for gi in tqdm(self.graph_indices):
                fps.append(smiles_to_morgan(gi, morgan_radius, morgan_bits))
            self.morgan_fps = torch.tensor(fps, dtype=torch.float32)
            torch.save(self.morgan_fps, cache_morgan)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'morgan_fp': self.morgan_fps[idx],
            'gene_ids':  self.gene_ids[idx],
            'label':     self.labels[idx],
        }

def collate_fn(batch):
    return {
        'morgan_fp': torch.stack([b['morgan_fp'] for b in batch]),
        'gene_ids':  torch.stack([b['gene_ids']  for b in batch]),
        'label':     torch.stack([b['label']      for b in batch]),
    }

# ================================================================
# 2. 模型：Morgan FP 化学编码器 + GeneEncoderV1 + Ortho + MLP
# ================================================================

class GeneEncoderV1(nn.Module):
    """与所有方法完全一致的基因编码器（k-mer CNN + TopK）"""
    def __init__(self, vocab_size=4097, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        kernel_sizes = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // len(kernel_sizes), k, padding=k // 2)
            for k in kernel_sizes])
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.k = 10
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        features = torch.cat([conv(x) for conv in self.convs], dim=1)
        features = F.dropout(F.relu(features), p=self.dropout, training=self.training)
        values, _ = torch.topk(features, k=self.k, dim=2)
        return self.aggregation(values.mean(dim=2))


class MorganChemEncoder(nn.Module):
    """Morgan 指纹 → 可训练投影 → 化学表示（替代 GIN）"""
    def __init__(self, fp_dim=2048, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, morgan_fp):
        return self.proj(morgan_fp)


class MorganModel(nn.Module):
    """
    Morgan FP 对照模型。
    单变量：仅化学编码器不同（Morgan FP 替代 GIN），其余与 train_baseline_bce.py 完全一致。
    """
    def __init__(self, hidden_dim=128, dropout=0.3, fp_dim=2048, use_ortho=True):
        super().__init__()
        self.use_ortho = use_ortho
        self.gene_enc  = GeneEncoderV1(hidden_dim=hidden_dim, dropout=dropout)
        self.chem_enc  = MorganChemEncoder(fp_dim=fp_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, gene_ids, morgan_fp):
        h_g = self.gene_enc(gene_ids)
        h_c = self.chem_enc(morgan_fp)

        V_g = F.normalize(h_g, dim=-1)
        V_c = F.normalize(h_c, dim=-1)

        if self.use_ortho:
            V_c_out = V_c - (V_c * V_g).sum(dim=-1, keepdim=True) * V_g
        else:
            V_c_out = V_c

        logits = self.classifier(torch.cat([V_g, V_c_out], dim=-1)).squeeze(1)
        return logits, V_g

# ================================================================
# 3. 训练
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader = DataLoader(
        MorganGraphDataset(args.data_dir, args.fold, 'train',
                           args.morgan_radius, args.morgan_bits),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(
        MorganGraphDataset(args.data_dir, args.fold, 'val',
                           args.morgan_radius, args.morgan_bits),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4)

    model = MorganModel(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        fp_dim=args.morgan_bits, use_ortho=not args.no_ortho).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    best_auroc, patience_cnt = 0.0, 0
    save_dir = Path(f'results_morgan/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    ortho_str = "" if not args.no_ortho else "_noortho"
    model_name = f"morgan_r{args.morgan_radius}_Fold{args.fold}{ortho_str}{tag}.pt"

    print(f"\nMorgan Baseline | r={args.morgan_radius} bits={args.morgan_bits} "
          f"ortho={not args.no_ortho} | Device: {args.device} | Fold: {args.fold}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_spread = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            morgan_fp = batch['morgan_fp'].to(device)
            gene_ids  = batch['gene_ids'].to(device)
            labels    = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, V_g = model(gene_ids, morgan_fp)
                loss_bce = criterion(logits, labels)

                # loss_spread（与 baseline 一致，防 V_g 坍缩）
                B = labels.size(0)
                mask = 1.0 - torch.eye(B, device=device)
                loss_spread = (torch.matmul(V_g, V_g.T) * mask).mean()

                loss = loss_bce + args.lam_spread * loss_spread

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss   += loss.item()
            total_bce    += loss_bce.item()
            total_spread += loss_spread.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                morgan_fp = batch['morgan_fp'].to(device)
                gene_ids  = batch['gene_ids'].to(device)
                with autocast(enabled=args.use_amp):
                    logits, _ = model(gene_ids, morgan_fp)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} | L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} Spr:{total_spread/n:.3f}) | "
              f"VAL_AUC: {auroc:.4f} | PRC: {auprc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--lam_spread',    type=float, default=0.1)
    parser.add_argument('--morgan_radius', type=int, default=2,
                        help='Morgan 指纹半径 (ECFP4=radius2, ECFP6=radius3)')
    parser.add_argument('--morgan_bits',   type=int, default=2048,
                        help='Morgan 指纹位数')
    parser.add_argument('--no_ortho',      action='store_true')
    parser.add_argument('--patience',      type=int, default=10)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--run_tag',       type=str, default='')
    train(parser.parse_args())
