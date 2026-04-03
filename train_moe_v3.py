#!/usr/bin/env python3
"""
MoE V3：修复 GeneEncoderV2 训练不稳定
========================================
相比 V2 的两处改动：

1. attn_query 零初始化
   原因：randn 初始化的 query 在 len=3000~8000 的序列上，早期会强烈聚焦到
   随机位置，导致 V_g 质量极不稳定，AUC 周期性跳水。
   修复：nn.init.zeros_(self.attn_query)，初始所有位置得分相等，
   注意力从近似均匀分布出发，随训练逐渐学习有意义的聚焦。

2. 线性 Warmup LR Scheduler
   原因：注意力参数从均匀分布出发，初期梯度很小，用全 lr 会在注意力
   刚开始分化时产生大幅震荡。
   修复：前 warmup_epochs 个 epoch 线性从 lr/10 升到 lr，
   之后交给 ReduceLROnPlateau 正常管理。
"""

import argparse
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ================================================================
# 0. 工具函数 & 数据集
# ================================================================

_KMER_VOCAB = {}
for i, combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(combo)] = i
_KMER_VOCAB['NNNNNN'] = 0

GENE_MAX_LEN  = 2000
GENE_INNER_DIM = 128
GENE_STRIDES   = [3, 2, 2]   # 3000→1000→500→250，第一层 stride=3 对齐长度

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_kmer_sequence(sequence: str, k: int = 6, max_len: int = GENE_MAX_LEN) -> list:
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if any(c not in 'ACGT' for c in kmer): kmer = 'N' * k
        kmers.append(_KMER_VOCAB.get(kmer, 0))
    if len(kmers) > max_len: kmers = kmers[:max_len]
    else: kmers += [0] * (max_len - len(kmers))
    return kmers

class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train'):
        import pickle
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt')
        self.smiles_to_graph = self.data['smiles_to_graph']
        self.graph_indices = [self.data['graph_indices'][i] for i in self.indices]
        self.labels = torch.tensor([self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}_len{GENE_MAX_LEN}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ 加载 K-mer 缓存 ({GENE_MAX_LEN}): {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 (len={GENE_MAX_LEN})...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'graph': self.smiles_to_graph[self.graph_indices[idx]],
                'gene_ids': self.gene_ids[idx],
                'label': self.labels[idx]}

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
    x          = torch.cat(all_x, dim=0)
    edge_index = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros(2, 0, dtype=torch.long)
    edge_attr  = torch.cat(all_edge_attr, dim=0)  if all_edge_attr  else torch.zeros(0, 4)
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'num_nodes_list': num_nodes_list,
            'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
            'label':    torch.stack([b['label']    for b in batch])}

# ================================================================
# 1. 模型定义
# ================================================================

class LayerNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GeneEncoderV2(nn.Module):
    """
    层次化 CNN + 注意力池化。
    [修复] attn_query 零初始化，避免早期随机聚焦导致 V_g 不稳定。
    """
    def __init__(self, vocab_size=4097, out_dim=128,
                 inner_dim=GENE_INNER_DIM, strides=GENE_STRIDES, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.strides = strides

        self.embedding = nn.Embedding(vocab_size, inner_dim, padding_idx=0)

        self.hier_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(inner_dim, inner_dim, kernel_size=7, stride=s, padding=3),
                LayerNorm1d(inner_dim),
                nn.ReLU()
            ) for s in strides
        ])

        # [修复 1] 零初始化：初始所有位置得分相等，注意力从均匀分布出发
        self.attn_query = nn.Parameter(torch.zeros(inner_dim))

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, gene_ids):
        pad_mask = (gene_ids != 0)
        for s in self.strides:
            pad_mask = pad_mask[:, ::s]

        x = self.embedding(gene_ids)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)

        for conv in self.hier_convs:
            x = conv(x)

        x = x.transpose(1, 2)   # [B, L', inner_dim]

        query  = F.normalize(self.attn_query, dim=0)
        scores = (x * query).sum(dim=-1) / (x.size(-1) ** 0.5)
        scores = scores.masked_fill(~pad_mask, float('-inf'))

        all_pad = (~pad_mask).all(dim=1, keepdim=True)
        scores  = scores.masked_fill(all_pad, 0.0)

        attn_weights = torch.softmax(scores, dim=-1)
        V_g_raw = (attn_weights.unsqueeze(-1) * x).sum(dim=1)

        return self.proj(V_g_raw)


class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge_proj = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        row, col  = edge_index
        edge_emb  = self.edge_proj(edge_attr)
        msg       = F.relu(x[row] + edge_emb)
        neighbor  = torch.zeros_like(x).index_add_(0, col, msg)
        return self.mlp(x + neighbor)


class ChemEncoder(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.atom_embed = nn.Sequential(
            nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        num_nodes_tensor = torch.tensor(num_nodes_list, device=device)
        batch_idx = torch.repeat_interleave(
            torch.arange(len(num_nodes_list), device=device), num_nodes_tensor)
        sum_pool  = torch.zeros(len(num_nodes_list), x.size(1), device=device).index_add_(0, batch_idx, x)
        mean_pool = sum_pool / num_nodes_tensor.float().unsqueeze(1).clamp(min=1)
        return self.readout(torch.cat([sum_pool, mean_pool], dim=1))


class MoEPaperModel(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.gene_enc = GeneEncoderV2(out_dim=hidden_dim,
                                      inner_dim=GENE_INNER_DIM,
                                      strides=GENE_STRIDES,
                                      dropout=dropout)
        self.chem_enc = ChemEncoder(hidden_dim=hidden_dim, dropout=dropout)

        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_experts)
        ])

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        h_g = self.gene_enc(gene_ids)
        h_c = self.chem_enc(x, edge_index, edge_attr, num_nodes_list)

        V_g = F.normalize(h_g, dim=-1)
        V_c = F.normalize(h_c, dim=-1)
        V_c_perp = V_c - (V_c * V_g).sum(dim=-1, keepdim=True) * V_g

        feat = torch.cat([V_g, V_c_perp], dim=-1)

        route_weights = self.router(feat)
        expert_logits = torch.stack(
            [expert(feat).squeeze(-1) for expert in self.experts], dim=1)
        final_logits  = (route_weights * expert_logits).sum(dim=-1)

        return final_logits, V_g, route_weights

# ================================================================
# 2. 训练
# ================================================================

def get_lr(epoch, warmup_epochs, base_lr):
    """线性 warmup：前 warmup_epochs 个 epoch 从 base_lr/10 升到 base_lr"""
    if epoch < warmup_epochs:
        return base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    return base_lr


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader = DataLoader(
        OptimizedGraphDataset(args.data_dir, args.fold, 'train'),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=optimized_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        OptimizedGraphDataset(args.data_dir, args.fold, 'val'),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=optimized_collate_fn, num_workers=4, pin_memory=True)

    model = MoEPaperModel(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        num_experts=args.num_experts).to(device)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # ReduceLROnPlateau 负责 warmup 结束后的衰减
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    best_auroc, patience_cnt = 0.0, 0
    save_dir = Path(f'results_moe_v3/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"moe_v3_k{args.num_experts}_Fold{args.fold}{tag}.pt"

    print(f"\n🚀 MoE V3 | experts={args.num_experts} | gene_len={GENE_MAX_LEN} | "
          f"warmup={args.warmup_epochs}ep | Device: {args.device} | Fold: {args.fold}")

    for epoch in range(args.epochs):

        # [修复 2] 线性 warmup：手动设置 lr
        if epoch < args.warmup_epochs:
            cur_lr = get_lr(epoch, args.warmup_epochs, args.lr)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        total_loss = total_bce = total_lb = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, V_g, route_weights = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                loss_bce = criterion_bce(logits, labels)

                mean_route = route_weights.mean(dim=0)
                uniform    = torch.ones_like(mean_route) / args.num_experts
                loss_lb    = F.mse_loss(mean_route, uniform)

                loss = loss_bce + args.lam_balance * loss_lb

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
            total_lb   += loss_lb.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index  = batch['edge_index'].to(device)
                edge_attr   = batch['edge_attr'].to(device)
                gene_ids    = batch['gene_ids'].to(device)
                with autocast(enabled=args.use_amp):
                    logits, _, _ = model(
                        gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        # warmup 结束后才让 plateau scheduler 工作
        if epoch >= args.warmup_epochs:
            plateau_sched.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} | lr:{cur_lr:.2e} | L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} LB:{total_lb/n:.4f}) | "
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

    print(f"\n✅ 最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='线性 warmup 的 epoch 数，期间 lr 从 lr/10 升到 lr')
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--num_experts',   type=int, default=4)
    parser.add_argument('--lam_balance',   type=float, default=0.01)
    parser.add_argument('--patience',      type=int, default=10)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--run_tag',       type=str, default='')
    train(parser.parse_args())
