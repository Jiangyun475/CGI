#!/usr/bin/env python3
"""
MoE Classifier: BCE + Load Balancing
=====================================
架构：GINLayer × 3 → sum/mean pool → Ortho → MoE 分类头
路由器直连 BCE 梯度，每个专家独立预测，加权融合输出。
无外挂 CL 模块。
"""

import argparse
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
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ 加载 K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存...")
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

class GeneEncoderV1(nn.Module):
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
    """
    机制感知多分支分类器。
    路由器和专家集群均直连 BCE 梯度，路由器有明确监督信号。
    """
    def __init__(self, hidden_dim=128, dropout=0.3, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.gene_enc = GeneEncoderV1(hidden_dim=hidden_dim, dropout=dropout)
        self.chem_enc = ChemEncoder(hidden_dim=hidden_dim, dropout=dropout)

        # 路由器：输入 [V_g, V_c_perp]，输出 K 个专家的软权重
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # 专家集群：每个专家是独立的分类器
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

        # 正交剥离
        V_c_perp = V_c - (V_c * V_g).sum(dim=-1, keepdim=True) * V_g

        feat = torch.cat([V_g, V_c_perp], dim=-1)  # [B, 2H]

        # 路由权重 [B, K]
        route_weights = self.router(feat)

        # 每个专家独立预测 → [B, K]
        expert_logits = torch.stack(
            [expert(feat).squeeze(-1) for expert in self.experts], dim=1
        )

        # 加权融合 → [B]
        final_logits = (route_weights * expert_logits).sum(dim=-1)

        return final_logits, V_g, route_weights


# ================================================================
# 2. 训练
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader = DataLoader(
        OptimizedGraphDataset(args.data_dir, args.fold, 'train'),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=optimized_collate_fn, num_workers=4)
    val_loader = DataLoader(
        OptimizedGraphDataset(args.data_dir, args.fold, 'val'),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=optimized_collate_fn, num_workers=4)

    model     = MoEPaperModel(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        num_experts=args.num_experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    best_auroc, patience_cnt = 0.0, 0
    save_dir = Path(f'results_moe/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"moe_k{args.num_experts}_Fold{args.fold}{tag}.pt"

    print(f"\n🚀 MoE Classifier | experts={args.num_experts} | "
          f"Device: {args.device} | Fold: {args.fold} | AMP: {args.use_amp}")

    for epoch in range(args.epochs):
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

                # Load Balancing Loss：
                # 宏观均衡：batch 级别各专家分配接近均匀
                mean_route  = route_weights.mean(dim=0)
                uniform     = torch.ones_like(mean_route) / args.num_experts
                loss_macro  = F.mse_loss(mean_route, uniform)
                # 微观尖锐：单样本路由权重越稀疏越好（熵越小越好）
                loss_entropy = -(route_weights * torch.log(route_weights + 1e-8)).sum(dim=-1).mean()
                loss_lb      = loss_macro + 0.1 * loss_entropy

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
        scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} | L:{total_loss/n:.3f} "
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
    parser.add_argument('--data_dir',     type=str, required=True)
    parser.add_argument('--device',       type=str, default='cuda:0')
    parser.add_argument('--fold',         type=int, default=0)
    parser.add_argument('--epochs',       type=int, default=80)
    parser.add_argument('--batch_size',   type=int, default=512)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--hidden_dim',   type=int, default=128)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--num_experts',  type=int, default=4)
    parser.add_argument('--lam_balance',  type=float, default=0.1,
                        help='Load Balancing Loss 权重，防止专家坍缩')
    parser.add_argument('--patience',     type=int, default=10)
    parser.add_argument('--use_amp',      action='store_true')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--run_tag',      type=str, default='')
    train(parser.parse_args())
