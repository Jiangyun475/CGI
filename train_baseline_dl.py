#!/usr/bin/env python3
"""
深度学习基准 —— 晚期融合模型 (GraphDTA 变体)
=============================================
对比目的：
    证明"靶向池化 + 正交剥离 + 对比学习"相对于传统晚期融合的增益。

架构设计：
    化学侧: GIN (3层) + Sum/Mean 全局池化       ← 无靶向注意力，无 h_g 输入
    基因侧: 1D-CNN (多尺度) + TopK 池化         ← 与主模型 GeneEncoderV1 完全相同
    融合:   Concat(h_g, h_c) → MLP             ← 无正交剥离，无对比学习
    损失:   纯 BCEWithLogitsLoss               ← 无 lam_var，无 lam_cl

公平性保证：
    - 相同的 DataLoader / collate_fn（含 edge_attr）
    - 相同的 chemical_cold_splits.pkl
    - 相同的 hidden_dim=128, dropout=0.3, lr=2e-4
    - 相同的 early stopping patience=10
    - 直接复用已有的 kmer_cache 文件

用法：
    python train_baseline_dl.py --data_dir /path/to/MCF7 --fold 0 --device cuda:0
"""

import os
import argparse
import itertools
import random
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
# 0. 随机种子
# ================================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ================================================================
# 1. 数据集（与主模型完全相同，含 edge_attr）
# ================================================================

_BASES      = ['A', 'C', 'G', 'T']
_KMER_VOCAB = {''.join(combo): i+1
               for i, combo in enumerate(itertools.product(_BASES, repeat=6))}
_KMER_VOCAB['N' * 6] = 0


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


class CGIDataset(Dataset):
    """直接复用主模型的 kmer_cache，无需重新计算。"""

    def __init__(self, data_dir, fold_idx=0, split='train'):
        import pickle
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)

        self.indices          = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line             = Path(data_dir).name
        preprocessed_file     = Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt'
        self.data             = torch.load(preprocessed_file)
        self.smiles_to_graph  = self.data['smiles_to_graph']
        self.graph_indices    = [self.data['graph_indices'][i] for i in self.indices]
        self.labels           = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        gene_sequences        = [self.data['gene_sequences'][i] for i in self.indices]

        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ 复用 K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存...")
            ids = [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)]
            self.gene_ids = torch.tensor(ids, dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx].item(),
        }


def collate_fn(batch):
    """与主模型完全相同，保留 edge_attr。"""
    graphs   = [item['graph'] for item in batch]
    labels   = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    all_x, all_ei, all_ea, num_nodes_list = [], [], [], []
    offset = 0
    for g in graphs:
        n = g['x'].shape[0]
        e = g['edge_index'].shape[1]
        num_nodes_list.append(n)
        all_x.append(g['x'])
        if e > 0:
            all_ei.append(g['edge_index'] + offset)
            all_ea.append(g['edge_attr'])
        offset += n

    x          = torch.cat(all_x, dim=0) if all_x else torch.zeros(len(batch), 31)
    edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.zeros(2, 0, dtype=torch.long)
    edge_attr  = torch.cat(all_ea, dim=0) if all_ea else torch.zeros(0, 4)
    gene_ids   = torch.stack([item['gene_ids'] for item in batch], dim=0)
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'num_nodes_list': num_nodes_list, 'gene_ids': gene_ids, 'label': labels}


# ================================================================
# 2. 模型
# ================================================================

class GeneEncoderV1(nn.Module):
    """
    与主模型完全相同。
    vocab_size = 4097（4^6=4096 个 6-mer + padding_idx=0）
    输出: [B, hidden_dim]
    """
    def __init__(self, vocab_size=4097, embed_dim=128, hidden_dim=128, k=10, dropout=0.3):
        super().__init__()
        self.k         = k
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs     = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim // 4, ks, padding=ks // 2)
            for ks in [6, 8, 10, 12]
        ])
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, gene_ids):
        x        = self.embedding(gene_ids).transpose(1, 2)            # [B, E, 1000]
        features = torch.cat([conv(x) for conv in self.convs], dim=1)  # [B, H, L]
        values, _ = torch.topk(features, k=self.k, dim=2)              # [B, H, k]
        return self.aggregation(values.mean(dim=2))                     # [B, H]



class GINLayer(nn.Module):
    # 增加 edge_dim 参数，默认对应你 preprocess_graphs.py 里的 4 维边特征
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        # 新增：边特征投影层，将低维边特征映射到高维隐藏层空间
        self.edge_proj = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # 新增：计算边嵌入
        edge_emb = self.edge_proj(edge_attr)
        # 融合机制：将源节点特征与对应的边特征相加，通过 ReLU 激活后作为传递的消息
        msg = F.relu(x[row] + edge_emb)
        # 聚合：将所有流入目标节点 col 的消息相加
        neighbor_feat = torch.zeros_like(x).index_add_(0, col, msg)
        
        return self.mlp(x + neighbor_feat)

class ChemEncoderBaseline(nn.Module):
    """
    化学图编码器：GIN × 3 + Sum/Mean 全局池化。
    无靶向注意力，不接收 h_g，对应主模型 pool_type='sum_mean' 的消融。
    输出: [B, hidden_dim]
    """
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.atom_embed = nn.Sequential(
            nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout    = dropout
        self.readout    = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        x      = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)

        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(
            torch.arange(len(num_nodes_list), device=device), num_nodes_t)

        sum_pool  = torch.zeros(len(num_nodes_list), x.size(1), device=device
                                ).index_add_(0, batch_idx, x)
        mean_pool = sum_pool / num_nodes_t.unsqueeze(1).clamp(min=1.0)
        return self.readout(torch.cat([sum_pool, mean_pool], dim=1))


class LateFusionBaseline(nn.Module):
    """
    晚期融合基线。
    与主模型（PaperModel）相比，消融了：
        - 靶向注意力池化（ChemEncoder 不看 h_g）
        - 正交剥离（直接 Concat h_g 和 h_c，而非 h_g 和 V_c_perp）
        - 对比学习损失
    """
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.gene_enc   = GeneEncoderV1(hidden_dim=hidden_dim, dropout=dropout)
        self.chem_enc   = ChemEncoderBaseline(hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        h_g    = self.gene_enc(gene_ids)
        h_c    = self.chem_enc(x, edge_index, edge_attr, num_nodes_list)
        logits = self.classifier(torch.cat([h_g, h_c], dim=-1)).squeeze(1)
        return logits


# ================================================================
# 3. 训练循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_ds     = CGIDataset(args.data_dir, args.fold, 'train')
    val_ds       = CGIDataset(args.data_dir, args.fold, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model     = LateFusionBaseline(hidden_dim=args.hidden_dim,
                                   dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_baseline_dl/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_auroc, patience_cnt = 0.0, 0

    print(f"\n{'='*60}")
    print(f"🚀 深度学习基线（晚期融合）")
    print(f"   细胞系: {Path(args.data_dir).name} | Fold: {args.fold} | Device: {args.device}")
    print(f"   Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    print(f"   消融对比: 无靶向池化 | 无正交剥离 | 无对比学习")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            edge_attr = batch['edge_attr'].to(device) # 新增
            gene_ids   = batch['gene_ids'].to(device, non_blocking=True)
            labels     = batch['label'].to(device, non_blocking=True)

            with autocast(enabled=args.use_amp):
                logits = model(gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss   = criterion(logits, labels)

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device) # 新增
                gene_ids   = batch['gene_ids'].to(device)
                with autocast(enabled=args.use_amp):
                    logits = model(gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        scheduler.step(auroc)

        print(f"Ep {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"VAL_AUC: {auroc:.4f} | PRC: {auprc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc, patience_cnt = auroc, 0
            torch.save(model.state_dict(), save_dir / f'best_fold{args.fold}.pt')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"🛑 早停 (patience={args.patience}) | Best AUC: {best_auroc:.4f}")
                break

    print(f"\n✅ Fold {args.fold} 完成 | Best VAL_AUC: {best_auroc:.4f}")


# ================================================================
# 4. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   required=True)
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--fold',       type=int,   default=0)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int,   default=128)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--use_amp',    action='store_true')
    parser.add_argument('--seed',       type=int,   default=42)
    train(parser.parse_args())
