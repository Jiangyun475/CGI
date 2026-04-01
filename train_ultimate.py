#!/usr/bin/env python3
"""
Ultimate CGI Model - Paper Edition (带消融实验与可解释性接口)
=====================================================
论文核心贡献点与消融开关：
1. 混合靶向池化 (Hybrid Pooling) -> --pool_type [hybrid | sum_mean | target]
2. 正交剥离机制 (Orthogonal Stripping) -> --disable_ortho
3. 焦点方向对比学习 (Focal Direction CL) -> --disable_cl
"""

import os
import argparse
import itertools
import random
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.utils import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    from entmax import entmax15
    HAS_ENTMAX = True
except ImportError:
    HAS_ENTMAX = False
    print("⚠️ 提示: 未安装 entmax，将回退到普通 Softmax。请运行 pip install entmax")

# ================================================================
# 0. 全局随机种子
# ================================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================================================
# 1. 数据预处理与 DataLoader (极速缓存版)
# ================================================================
_BASES = ['A', 'C', 'G', 'T']
_KMER_VOCAB = {''.join(combo): i+1 for i, combo in enumerate(itertools.product(_BASES, repeat=6))}
_KMER_VOCAB['N' * 6] = 0

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
        preprocessed_file = Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt'
        
        self.data = torch.load(preprocessed_file)
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
            gene_ids_list = [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)]
            self.gene_ids = torch.tensor(gene_ids_list, dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'graph': self.smiles_to_graph[self.graph_indices[idx]],
                'gene_ids': self.gene_ids[idx], 'label': self.labels[idx].item()}

def optimized_collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    all_x, all_edge_index, all_edge_attr, num_nodes_list = [], [], [], []
    offset = 0
    for graph in graphs:
        num_nodes = graph['x'].shape[0]
        num_edges = graph['edge_index'].shape[1]
        num_nodes_list.append(num_nodes)
        all_x.append(graph['x'])
        if num_edges > 0:
            all_edge_index.append(graph['edge_index'] + offset)
            all_edge_attr.append(graph['edge_attr'])
        offset += num_nodes

    x = torch.cat(all_x, dim=0) if all_x else torch.zeros(len(batch), 31)
    edge_index = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.cat(all_edge_attr, dim=0) if all_edge_attr else torch.zeros(0, 4)
    gene_ids = torch.stack([item['gene_ids'] for item in batch], dim=0)
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'num_nodes_list': num_nodes_list, 'gene_ids': gene_ids, 'label': labels}

# ================================================================
# 2. 模型架构 (内置消融实验开关与 Attention 提取)
# ================================================================
class GeneEncoderV1(nn.Module):
    def __init__(self, vocab_size=4097, embed_dim=128, hidden_dim=128, k=10, dropout=0.3):
        super().__init__()
        self.k = k
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, hidden_dim // 4, ks, padding=ks//2) for ks in [6, 8, 10, 12]])
        self.aggregation = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        features = torch.cat([conv(x) for conv in self.convs], dim=1)
        values, _ = torch.topk(features, k=self.k, dim=2)
        return self.aggregation(values.mean(dim=2))

# class GINLayer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
#     def forward(self, x, edge_index):
#         row, col = edge_index
#         neighbor_feat = torch.zeros_like(x).index_add_(0, col, x[row])
#         return self.mlp(x + neighbor_feat)
class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge_proj = nn.Linear(edge_dim, dim) # 边特征投影
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # 将边特征融入邻居节点特征
        edge_emb = self.edge_proj(edge_attr)
        msg = F.relu(x[row] + edge_emb) 
        neighbor_feat = torch.zeros_like(x).index_add_(0, col, msg)
        return self.mlp(x + neighbor_feat)


class ChemEncoder_Ablation(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3, pool_type='hybrid'):
        super().__init__()
        self.pool_type = pool_type
        self.atom_embed = nn.Sequential(nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout
        self.attn_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        # 根据消融策略动态改变输出层维度
        out_dim_multiplier = 3 if pool_type == 'hybrid' else (2 if pool_type == 'sum_mean' else 1)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * out_dim_multiplier, hidden_dim * 2), 
            nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), 
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr,num_nodes_list, h_g):
        device = x.device
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))), p=self.dropout, training=self.training)
            
        num_nodes_tensor = torch.tensor(num_nodes_list, device=device)
        batch_idx = torch.repeat_interleave(torch.arange(len(num_nodes_list), device=device), num_nodes_tensor)
        
        # 1. 全局特征
        sum_pool = torch.zeros(len(num_nodes_list), x.size(1), device=device).index_add_(0, batch_idx, x)
        mean_pool = sum_pool / num_nodes_tensor.unsqueeze(1).clamp(min=1.0)
        
        # 2. 靶向注意力 (获取原子级权重 alpha)
        # q = self.attn_proj(h_g) 
        # scores = (x * q[batch_idx]).sum(dim=-1) / math.sqrt(x.size(-1)) 
        # alpha = softmax(scores, batch_idx) # 🔥 返回给外部画热力图用！
        
        # 修改为 (加入 tau = 0.1):
        tau = 0.1  # 你可以后续测试 0.05, 0.1, 0.5
        q = self.attn_proj(h_g) 
        scores = (x * q[batch_idx]).sum(dim=-1) / math.sqrt(x.size(-1)) 
        # 核心修改：缩放 scores
        alpha = softmax(scores / tau, batch_idx)




        x_weighted = x * alpha.unsqueeze(-1) 
        target_pool = torch.zeros(len(num_nodes_list), x.size(1), device=device).index_add_(0, batch_idx, x_weighted)
        
        # 消融选择
        if self.pool_type == 'hybrid':
            h_c = torch.cat([sum_pool, mean_pool, target_pool], dim=1)
        elif self.pool_type == 'sum_mean':
            h_c = torch.cat([sum_pool, mean_pool], dim=1)
        elif self.pool_type == 'target':
            h_c = target_pool

        return self.readout(h_c), alpha

class PaperModel(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3, pool_type='hybrid', use_ortho=True):
        super().__init__()
        self.use_ortho = use_ortho
        self.gene_enc = GeneEncoderV1(hidden_dim=hidden_dim, dropout=dropout)
        self.chem_enc = ChemEncoder_Ablation(hidden_dim=hidden_dim, dropout=dropout, pool_type=pool_type)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, gene_ids, x, edge_index,edge_attr, num_nodes_list):
        h_g = self.gene_enc(gene_ids)
        h_c, alpha = self.chem_enc(x, edge_index, edge_attr, num_nodes_list, h_g)
        
        V_g = F.normalize(h_g, dim=-1)
        V_c = F.normalize(h_c, dim=-1)
        
        if self.use_ortho:
            dot = (V_c * V_g).sum(dim=-1, keepdim=True)
            V_c_perp = V_c - (dot * V_g)
        else:
            V_c_perp = V_c # 消融：关闭正交剥离，直接使用 V_c
            
        logits = self.classifier(torch.cat([V_g, V_c_perp], dim=-1)).squeeze(1)
        # 返回 alpha 用于画图
        return logits, V_g, V_c, V_c_perp, alpha 

# ================================================================
# 3. 损失函数 (Focal CL)
# ================================================================
class FocalDirectionAwareCL(nn.Module):
    def __init__(self, dim=128, margin=0.5):
        super().__init__()
        self.margin = margin
        self.direction = nn.Parameter(torch.randn(dim))
        self.gene_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.chem_perp_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, V_g, V_c_perp, labels):
        proj_g = F.normalize(self.gene_proj(V_g), dim=1)
        proj_c = F.normalize(self.chem_perp_proj(V_c_perp), dim=1)
        direction = F.normalize(self.direction, dim=0)
        
        h_fused = (proj_g + proj_c) / 2.0
        dir_scores = torch.matmul(h_fused, direction)
        
        dir_targets = 2 * labels - 1 
        diff = F.relu(self.margin - dir_scores * dir_targets)
        hard_weight = (diff + 1e-4) ** 2.0  # Focal 机制挖掘困难样本
        return (diff * hard_weight).mean()

# ================================================================
# 4. 训练主循环
# ================================================================
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    
    train_loader = DataLoader(OptimizedGraphDataset(args.data_dir, args.fold, 'train'),
                              batch_size=args.batch_size, shuffle=True, collate_fn=optimized_collate_fn, num_workers=4)
    val_loader = DataLoader(OptimizedGraphDataset(args.data_dir, args.fold, 'val'),
                            batch_size=args.batch_size, shuffle=False, collate_fn=optimized_collate_fn, num_workers=4)

    model = PaperModel(hidden_dim=args.hidden_dim, dropout=args.dropout, pool_type=args.pool_type, use_ortho=not args.disable_ortho).to(device)
    cl_module = FocalDirectionAwareCL(dim=args.hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(cl_module.parameters()), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    best_auroc, patience_cnt = 0.0, 0
    save_dir = Path(f'results_paper/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建保存的模型名（携带消融参数 + fold + 可选标签）
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"model_ortho{not args.disable_ortho}_cl{not args.disable_cl}_{args.pool_type}_Fold{args.fold}{tag}.pt"

    print(f"\n🚀 论文训练版本启动 | Device: {args.device} | AMP: {args.use_amp}")
    print(f"📊 [消融配置] 混合池化: {args.pool_type} | 正交: {not args.disable_ortho} | 对比学习: {not args.disable_cl}")

    for epoch in range(args.epochs):
        model.train()
        cl_module.train()
        total_loss, total_bce, total_var, total_cl = 0, 0, 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            x, edge_index = batch['x'].to(device), batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device) # 新增
            gene_ids, labels = batch['gene_ids'].to(device), batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, V_g, V_c, V_c_perp, _ = model(gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                
                loss_bce = criterion_bce(logits, labels)
                
                # 条件方差惩罚 (只有正交且为正样本时才受罚)
                loss_var = torch.tensor(0.0, device=device)
                if not args.disable_ortho:
                    pos_mask = (labels == 1.0).float()
                    norms = V_c_perp.norm(dim=1)
                    loss_var = (pos_mask * torch.relu(1.0 - norms)).sum() / (pos_mask.sum() + 1e-8)
                
                # 对比学习
                loss_cl = torch.tensor(0.0, device=device)
                if not args.disable_cl:
                    loss_cl = cl_module(V_g, V_c_perp, labels)
                
                loss = loss_bce + (args.lam_var * loss_var) + (args.lam_cl * loss_cl)

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_var += loss_var.item()
            total_cl += loss_cl.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index = batch['x'].to(device), batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device)
                gene_ids, labels = batch['gene_ids'].to(device), batch['label'].numpy()
                with autocast(enabled=args.use_amp):
                    logits, _, _, _, _ = model(gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels)
                
        auroc, auprc = roc_auc_score(all_labels, all_preds), average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        scheduler.step(auroc)
        
        print(f"Ep {epoch+1:02d} | L: {total_loss/len(train_loader):.3f} (BCE:{total_bce/len(train_loader):.3f} Var:{total_var/len(train_loader):.3f} CL:{total_cl/len(train_loader):.3f}) | "
              f"VAL_AUC: {auroc:.4f} | PRC: {auprc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lam_var', type=float, default=0.2)
    parser.add_argument('--lam_cl', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    # 🌟 论文专属消融开关
    parser.add_argument('--pool_type', type=str, default='hybrid', choices=['hybrid', 'sum_mean', 'target'], help='池化策略')
    parser.add_argument('--disable_ortho', action='store_true', help='关闭正交剥离')
    parser.add_argument('--disable_cl', action='store_true', help='关闭对比学习')
    parser.add_argument('--run_tag', type=str, default='', help='自定义标签附加到模型名，如 tau01、exp1')
    
    train(parser.parse_args())