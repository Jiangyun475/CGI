#!/usr/bin/env python3
"""
MoE Classifier V2：GeneEncoderV2（层次化 CNN + 注意力池化）
=============================================================
与 train_moe.py 的唯一区别：GeneEncoderV1 → GeneEncoderV2

GeneEncoderV2 改动：
  - max_len: 1000 → 8000（覆盖 ~98% 的基因序列）
  - 内部维度: 128 → 256（更强的表达能力）
  - 池化: Top-K → 注意力池化 + padding mask（消除填充位置的干扰）
  - 新增 3 层 stride=2 层次化 CNN（感受野指数增大，显存友好）
  - 输出仍为 128 维（最终投影回 128，兼容 MoEPaperModel）

k-mer 缓存文件名：kmer_cache_fold{N}_{split}_len8000.pt
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

GENE_MAX_LEN = 8000        # 覆盖约 98% 的基因序列
GENE_INNER_DIM = 128       # CNN 内部维度，平衡速度与表达能力
GENE_STRIDES   = [4, 2, 2] # 8000→2000→1000→500，第一层激进压缩

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
        # 独立的缓存文件，不覆盖旧的 len1000 缓存
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}_len8000.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ 加载 K-mer 缓存 (len8000): {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 (len8000)，首次运行需要几分钟...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'graph': self.smiles_to_graph[self.graph_indices[idx]],
                'gene_ids': self.gene_ids[idx],
                'label': self.labels[idx],
                'sample_idx': idx}   # 全局索引，用于 gene_cache 查表

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
            'gene_ids':    torch.stack([b['gene_ids']    for b in batch]),
            'label':       torch.stack([b['label']       for b in batch]),
            'sample_idx':  torch.tensor([b['sample_idx'] for b in batch], dtype=torch.long)}

# ================================================================
# 1. 模型定义
# ================================================================

class LayerNorm1d(nn.Module):
    """
    专为 CNN 1D [B, C, L] 设计的 LayerNorm 包装器。
    在 C 维度上独立归一化，每个 token 各算各的，
    padding 位置不会污染真实 token 的统计量。
    """
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, L] → [B, L, C]
        x = self.norm(x)       # 在 C 维度归一化，每个 token 独立
        x = x.transpose(1, 2)  # [B, L, C] → [B, C, L]
        return x


class GeneEncoderV2(nn.Module):
    """
    层次化 CNN + 注意力池化基因编码器。

    数据流：
      [B, 8000] k-mer ids
        → Embedding [B, 256, 8000]
        → stride=2 Conv × 3  →  [B, 256, 1000]   感受野指数增大
        → 注意力池化（padding mask 屏蔽填充位置）
        → Linear 256→128  →  V_g [B, 128]

    padding mask：
      gene_ids == 0 的位置是填充，注意力分数设为 -inf，
      softmax 后权重精确为 0，不污染聚合结果。
    """
    def __init__(self, vocab_size=4097, out_dim=128,
                 inner_dim=GENE_INNER_DIM, strides=GENE_STRIDES, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.strides = strides

        # 词嵌入，padding_idx=0 保证填充位置梯度不更新
        self.embedding = nn.Embedding(vocab_size, inner_dim, padding_idx=0)

        # 层次化降采样：stride=[4,2,2]，长度 8000→2000→1000→500
        # kernel=7, padding=3 在任意 stride 下均保持输出长度 = ceil(L/stride)
        # LayerNorm1d 在 C 维度独立归一化，padding 位置不污染真实 token
        self.hier_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(inner_dim, inner_dim, kernel_size=7, stride=s, padding=3),
                LayerNorm1d(inner_dim),
                nn.ReLU()
            ) for s in strides
        ])

        # 注意力池化：可学习 query 向量
        self.attn_query = nn.Parameter(torch.randn(inner_dim))

        # 最终投影：256 → 128，与化学编码器维度对齐
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, gene_ids):
        # ── padding mask ──────────────────────────────────────────
        # True = 有效位置，False = 填充位置
        # 每经过一次 stride=2，mask 长度减半（取步长=2 的子集）
        pad_mask = (gene_ids != 0)               # [B, 8000]
        for s in self.strides:
            pad_mask = pad_mask[:, ::s]          # 8000→2000→1000→500

        # ── 特征提取 ──────────────────────────────────────────────
        x = self.embedding(gene_ids)        # [B, 8000, 256]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)              # [B, 256, 8000]

        for conv in self.hier_convs:
            x = conv(x)                    # 逐步：→[B,256,4000]→[B,256,2000]→[B,256,1000]

        # x: [B, 256, 1000]，转为 [B, 1000, 256] 方便注意力计算
        x = x.transpose(1, 2)             # [B, 1000, 256]

        # ── 注意力池化 ────────────────────────────────────────────
        # query: [256] → 与每个位置做内积 → 分数 [B, 1000]
        query  = F.normalize(self.attn_query, dim=0)
        scores = (x * query).sum(dim=-1) / (x.size(-1) ** 0.5)  # [B, 1000]

        # 填充位置分数设为 -inf，softmax 后权重精确为 0
        scores = scores.masked_fill(~pad_mask, float('-inf'))

        # 防止极端情况：某序列全部被截断为填充（理论上不会发生）
        # 用 clamp 保护，避免全 -inf 导致 NaN
        all_pad = (~pad_mask).all(dim=1, keepdim=True)  # [B, 1]
        scores  = scores.masked_fill(all_pad, 0.0)

        attn_weights = torch.softmax(scores, dim=-1)    # [B, 1000]

        # 加权求和
        V_g_raw = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # [B, 256]

        return self.proj(V_g_raw)   # [B, 128]


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
        self.num_experts   = num_experts
        self._gene_cache   = None
        self._cache_built  = False
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

    def build_gene_cache(self, all_gene_ids, device, batch_size=256):
        """
        预计算数据集中所有唯一基因的 V_g，存入 self._gene_cache。
        每个 epoch 开始前调用一次，用 no_grad 提速。
        gene_enc 的梯度通过 optimizer.step() 在下一次 build_gene_cache 时生效。
        """
        # 找出唯一基因序列及其在原始索引中的位置
        unique_ids, inverse_idx = torch.unique(all_gene_ids, dim=0, return_inverse=True)
        n = unique_ids.size(0)

        v_list = []
        self.gene_enc.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = unique_ids[start:start+batch_size].to(device)
                with autocast(enabled=True):
                    v = F.normalize(self.gene_enc(chunk), dim=-1)
                v_list.append(v.cpu())
        self.gene_enc.train()

        # unique_vg[i] = 第 i 个唯一基因的 V_g
        unique_vg = torch.cat(v_list, dim=0)          # [N_unique, H]
        # 还原成与原始数据集等长的查找表
        self._gene_cache  = unique_vg[inverse_idx]    # [N_total, H]  存在 CPU
        self._cache_built = True

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list,
                sample_indices=None):
        """
        sample_indices: 当前 batch 在数据集中的全局索引，用于查 gene_cache。
                        若为 None（推理阶段），则直接调用 gene_enc。
        """
        if self._cache_built and sample_indices is not None:
            # 直接从缓存取 V_g，无需过 gene_enc
            V_g = self._gene_cache[sample_indices].to(x.device)
        else:
            V_g = F.normalize(self.gene_enc(gene_ids), dim=-1)

        h_c = self.chem_enc(x, edge_index, edge_attr, num_nodes_list)
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

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_dataset = OptimizedGraphDataset(args.data_dir, args.fold, 'train')
    train_loader  = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=optimized_collate_fn, num_workers=4, pin_memory=True,
        sampler=None)
    val_loader = DataLoader(
        OptimizedGraphDataset(args.data_dir, args.fold, 'val'),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=optimized_collate_fn, num_workers=4, pin_memory=True)

    # 预先把整个训练集的 gene_ids 收集成一个大 tensor，供 build_gene_cache 使用
    print("收集训练集全部基因序列...", flush=True)
    all_train_gene_ids = train_dataset.gene_ids   # [N_train, 8000]，已在 Dataset 里缓存

    model     = MoEPaperModel(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        num_experts=args.num_experts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    best_auroc, patience_cnt = 0.0, 0
    save_dir = Path(f'results_moe_v2/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"moe_v2_k{args.num_experts}_Fold{args.fold}{tag}.pt"

    print(f"\n🚀 MoE V2 (GeneEncoderV2) | experts={args.num_experts} | "
          f"gene_len=8000 | Device: {args.device} | Fold: {args.fold}")

    for epoch in range(args.epochs):
        # ── 每个 epoch 开始前预计算全部唯一基因的 V_g ──────────────
        print(f"[Ep {epoch+1}] 预计算基因缓存...", end=' ', flush=True)
        model.build_gene_cache(all_train_gene_ids, device, batch_size=512)
        print("完成", flush=True)

        model.train()
        total_loss = total_bce = total_lb = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x           = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)
            sample_idx  = batch['sample_idx']   # 留在 CPU，用于查表

            with autocast(enabled=args.use_amp):
                logits, V_g, route_weights = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'],
                    sample_indices=sample_idx)

                loss_bce = criterion_bce(logits, labels)

                # 宏观均衡
                mean_route   = route_weights.mean(dim=0)
                uniform      = torch.ones_like(mean_route) / args.num_experts
                loss_macro   = F.mse_loss(mean_route, uniform)
                # 微观尖锐：最小化单样本路由熵
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
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--epochs',      type=int, default=80)
    parser.add_argument('--batch_size',  type=int, default=256,
                        help='V2 显存占用更大，默认从 512 降到 256')
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--lam_balance', type=float, default=0.1)
    parser.add_argument('--patience',    type=int, default=10)
    parser.add_argument('--use_amp',     action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--run_tag',     type=str, default='')
    train(parser.parse_args())
