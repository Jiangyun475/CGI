#!/usr/bin/env python3
"""
多细胞系联合训练（Multi-task DrugOperatorNet）
================================================

设计逻辑：
  - 药物编码器（GIN + PharmacophoreExtractor）：4个细胞系共享
    同一药物在不同细胞系中学到的化学特征应该共享，强迫GIN学通用化学规律
  - 基因编码器（GeneMultiHeadReader）：4个细胞系共享
    基因序列是固定的，跨细胞系共享即可
  - 扰动算子（PerturbationOperator）：每个细胞系独立
    不同细胞系（MCF7/A375/A549/VCAP）的转录响应空间不同
    用细胞系特异的 U 矩阵生成头实现

期望收益：
  - 训练数据 ×4（从单细胞系 ~160K 到 ~640K 样本）
  - Drug GIN 见到4倍的药物-基因对，化学泛化能力大幅提升
  - Chemical Cold Split 下同一未见药物在4个细胞系都是测试，但训练时另外3个细胞系提供跨细胞泛化信号

训练策略：
  - 每个 batch 从所有细胞系随机采样（按比例）
  - 各细胞系独立计算 loss，共同反传
  - 总loss = Σ_c (BCE_c + reg_c) / n_cells

用法：
  python New/train_multitask.py \
    --cells MCF7 A375 A549 VCAP \
    --device cuda:0 --fold 0 \
    --epochs 80 --batch_size 512 --lr 2e-4 \
    --lam_ortho_modes 0.1 --lam_sparse 0.01 \
    --patience 10 --seed 42 --use_amp
"""

import os, sys, pickle, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 复用 train_operator_moe 里的基础组件
sys.path.insert(0, str(Path(__file__).parent))


# ── 基础工具（从 train_operator_moe 复制，避免循环导入）────────────

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def encode_kmer_sequence(sequence, k=6, max_len=1000):
    base2idx = {'A':0,'T':1,'G':2,'C':3,'U':1,'a':0,'t':1,'g':2,'c':3,'u':1}
    kmer_ids = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        idx, valid = 0, True
        for ch in kmer:
            if ch not in base2idx: valid = False; break
            idx = idx * 4 + base2idx[ch]
        kmer_ids.append(idx if valid else 0)
    if len(kmer_ids) > max_len: kmer_ids = kmer_ids[:max_len]
    else: kmer_ids += [0] * (max_len - len(kmer_ids))
    return kmer_ids

def scatter_softmax(scores, batch_idx):
    scores = scores.float()
    max_s = torch.zeros(batch_idx.max()+1, device=scores.device).index_reduce_(
        0, batch_idx, scores, 'amax', include_self=True)
    exp_s = torch.exp(scores - max_s[batch_idx])
    exp_sum = torch.zeros(batch_idx.max()+1, device=scores.device).index_add_(
        0, batch_idx, exp_s)
    return exp_s / (exp_sum[batch_idx] + 1e-8)

def scatter_add(src, batch_idx, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


# ================================================================
# 1. 数据集（带细胞系标签）
# ================================================================

class CellDataset(Dataset):
    def __init__(self, data_dir, fold_idx, split, gene_max_len, cell_id):
        with open(Path(data_dir)/'chemical_cold_splits.pkl','rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split=='train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir)/f'preprocessed_graphs_{cell_line}.pt')
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        self.cell_id = cell_id  # 整数，标识细胞系

        gene_seqs = [self.data['gene_sequences'][i] for i in self.indices]
        suffix = '' if gene_max_len==1000 else f'_len{gene_max_len}'
        cache = Path(data_dir)/f'kmer_cache_fold{fold_idx}_{split}{suffix}.pt'
        if cache.exists():
            self.gene_ids = torch.load(cache)
        else:
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(s, max_len=gene_max_len) for s in tqdm(gene_seqs)],
                dtype=torch.long)
            torch.save(self.gene_ids, cache)

        self.smiles_to_graph = self.data['smiles_to_graph']
        self.graph_indices   = [self.data['graph_indices'][i] for i in self.indices]

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
            'cell_id':  self.cell_id,
        }


def multitask_collate(batch):
    all_x, all_ei, all_ea, nl = [], [], [], []
    offset = 0
    for item in batch:
        g = item['graph']
        n = g['x'].shape[0]
        all_x.append(g['x'])
        if g['edge_index'].shape[1] > 0:
            all_ei.append(g['edge_index'] + offset)
            all_ea.append(g['edge_attr'])
        nl.append(n); offset += n
    return {
        'x':          torch.cat(all_x),
        'edge_index': torch.cat(all_ei, dim=1) if all_ei else torch.zeros(2,0,dtype=torch.long),
        'edge_attr':  torch.cat(all_ea) if all_ea else torch.zeros(0,4),
        'num_nodes_list': nl,
        'gene_ids':   torch.stack([b['gene_ids'] for b in batch]),
        'label':      torch.stack([b['label']    for b in batch]),
        'cell_id':    torch.tensor([b['cell_id'] for b in batch], dtype=torch.long),
    }


# ================================================================
# 2. 模型：共享编码器 + 细胞系特异算子头
# ================================================================

class GINLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_proj = nn.Linear(4, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2), nn.BatchNorm1d(dim*2), nn.ReLU(),
            nn.Linear(dim*2, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, num_nodes):
        src, dst = edge_index
        e = F.relu(self.edge_proj(edge_attr))
        agg = torch.zeros_like(x).index_add_(0, dst, F.relu(x[src] + e))
        return self.mlp((1 + self.eps) * x + agg)


class MultiTaskDrugOperator(nn.Module):
    """
    共享 GIN 药物编码器 + 共享 GeneMultiHeadReader
    + 每个细胞系独立的 PerturbationOperator 头
    """
    def __init__(self, n_cells, hidden_dim=128, operator_rank=8,
                 dropout=0.3, gene_vocab=4096, gene_max_len=1000):
        super().__init__()
        self.r = operator_rank
        self.H = hidden_dim
        self.n_cells = n_cells

        # ── 共享药物编码器（GIN × 3）────────────────────────────
        self.atom_emb = nn.Embedding(120, hidden_dim)
        self.gin1 = GINLayer(hidden_dim)
        self.gin2 = GINLayer(hidden_dim)
        self.gin3 = GINLayer(hidden_dim)
        self.drug_norm = nn.LayerNorm(hidden_dim)

        # ── 共享药效团提取器（→ U矩阵材料）──────────────────────
        # 为每个细胞系独立生成 U，但共享底层 atom cross-attention
        self.pharma_q = nn.Parameter(torch.zeros(operator_rank, hidden_dim))
        nn.init.trunc_normal_(self.pharma_q, std=0.02)
        self.pharma_proj = nn.Linear(hidden_dim, hidden_dim)

        # 细胞系特异 sigma head（U 是共享的，sigma 让每个细胞系权重不同）
        self.sigma_heads = nn.ModuleList([
            nn.Linear(hidden_dim, operator_rank) for _ in range(n_cells)])

        # ── 共享基因编码器（GeneMultiHeadReader）────────────────
        self.gene_emb = nn.Embedding(gene_vocab+1, hidden_dim, padding_idx=0)
        self.gene_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, k, padding=k//2)
            for k in [6, 8, 10, 12]])
        self.gene_q = nn.Parameter(torch.zeros(operator_rank, hidden_dim))
        nn.init.trunc_normal_(self.gene_q, std=0.02)
        self.gene_attn_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── 细胞系特异分类头 ──────────────────────────────────
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for _ in range(n_cells)])

        self.dropout = nn.Dropout(dropout)

    def encode_drug(self, x, edge_index, edge_attr, num_nodes_list):
        """共享药物编码，返回 h_drug [B,H] 和原子特征 [N_total,H]"""
        h = F.relu(self.atom_emb(x.squeeze(-1).long()))
        h = self.gin1(h, edge_index, edge_attr, sum(num_nodes_list))
        h = self.gin2(h, edge_index, edge_attr, sum(num_nodes_list))
        h = self.gin3(h, edge_index, edge_attr, sum(num_nodes_list))
        # 图级别池化
        B = len(num_nodes_list)
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=h.device),
            torch.tensor(num_nodes_list, device=h.device))
        h_pool = scatter_add(h, batch_idx, B)           # [B, H]
        h_drug = self.drug_norm(h_pool)
        return h_drug, h, batch_idx

    def encode_gene(self, gene_ids):
        """共享基因编码，返回 h_gene [B,H], V_modes [B,r,H]"""
        B = gene_ids.shape[0]
        x = self.gene_emb(gene_ids).permute(0,2,1)      # [B,H,L]
        feats = [F.relu(conv(x)) for conv in self.gene_convs]
        gf = torch.stack(feats).mean(0).permute(0,2,1)  # [B,L,H]
        h_gene = gf.mean(1)                              # [B,H]
        Q = self.gene_q.unsqueeze(0).expand(B,-1,-1)    # [B,r,H]
        K = self.gene_attn_proj(gf)                      # [B,L,H]
        attn = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / (self.H**0.5), -1)
        V_modes = torch.bmm(attn, gf)                   # [B,r,H]
        return h_gene, V_modes

    def forward_cell(self, h_drug, h_atoms, batch_idx, h_gene, V_modes, cell_idx):
        """单细胞系前向：使用 cell_idx 的独立算子头"""
        B, r, H = h_drug.shape[0], self.r, self.H

        # 药效团提取（U矩阵，共享）
        Q = self.pharma_q.unsqueeze(0).expand(B,-1,-1)   # [B,r,H]
        K = self.pharma_proj(h_atoms)                     # [N_total,H]
        # 按分子计算 cross-attention
        U_list = []
        for i in range(B):
            mask = batch_idx == i
            k_i = K[mask]                                 # [n_i, H]
            q_i = Q[i]                                    # [r, H]
            a_i = torch.softmax(q_i @ k_i.T / (H**0.5), -1)  # [r, n_i]
            U_list.append(a_i @ k_i)                     # [r, H]
        U = torch.stack(U_list)                           # [B, r, H]

        # 细胞系特异 sigma
        sigma = torch.sigmoid(self.sigma_heads[cell_idx](h_drug))  # [B, r]

        # 交互谱
        coupling = (U * V_modes).sum(-1)                 # [B, r]
        spectrum = sigma * coupling                       # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(1)   # [B, H]

        # 分类
        feat   = self.dropout(torch.cat([h_gene, delta_h], -1))
        logits = self.classifiers[cell_idx](feat).squeeze(-1)

        return logits, spectrum, sigma, U

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list, cell_ids):
        """
        统一前向：处理混合 batch（多细胞系样本混在一起）
        cell_ids: [B] 每个样本的细胞系编号
        返回: dict {cell_id: (logits, spectrum, sigma, U)}
        """
        h_drug, h_atoms, batch_idx = self.encode_drug(
            x, edge_index, edge_attr, num_nodes_list)
        h_gene, V_modes = self.encode_gene(gene_ids)

        results = {}
        for cid in cell_ids.unique():
            mask = cell_ids == cid
            if mask.sum() == 0: continue
            # 提取该细胞系的样本
            bi_mask = torch.zeros(batch_idx.max()+1, dtype=torch.bool, device=cell_ids.device)
            bi_mask[mask] = True
            atom_mask = bi_mask[batch_idx]

            # 重映射 batch_idx 到子集
            idx_map = torch.full((mask.max()+1,), -1, device=cell_ids.device)
            idx_map[torch.where(mask)[0]] = torch.arange(mask.sum(), device=cell_ids.device)
            sub_batch_idx = idx_map[batch_idx[atom_mask]]

            logits, spectrum, sigma, U = self.forward_cell(
                h_drug[mask], h_atoms[atom_mask], sub_batch_idx,
                h_gene[mask], V_modes[mask], cid.item())
            results[cid.item()] = (logits, spectrum, sigma, U)

        return results


# ================================================================
# 3. 训练
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    DATA_ROOT = Path(args.data_root)
    cell_names = args.cells                          # e.g. ['MCF7','A375','A549','VCAP']
    n_cells = len(cell_names)

    # 数据集
    train_sets, val_sets = [], []
    for cid, cell in enumerate(cell_names):
        train_sets.append(CellDataset(DATA_ROOT/cell, args.fold, 'train', args.gene_max_len, cid))
        val_sets.append(  CellDataset(DATA_ROOT/cell, args.fold, 'val',   args.gene_max_len, cid))

    train_ds = ConcatDataset(train_sets)
    # 验证集保持各细胞系独立，按细胞系分开评估
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=multitask_collate, num_workers=4)
    val_loaders  = [DataLoader(vs, batch_size=args.batch_size, shuffle=False,
                               collate_fn=multitask_collate, num_workers=2)
                    for vs in val_sets]

    model = MultiTaskDrugOperator(
        n_cells=n_cells, hidden_dim=args.hidden_dim,
        operator_rank=args.operator_rank, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  多细胞联合训练 | cells={cell_names}")
    print(f"  total_train={len(train_ds):,} | params={n_params:,}")
    print(f"  Fold: {args.fold} | Device: {args.device}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # 调度器基于所有细胞系的平均 AUC
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_multitask')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"multitask_{'_'.join(cell_names)}_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    best_mean_auc, patience_cnt = 0.0, 0

    for epoch in range(args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            lr = args.lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs-1, 1))
            for pg in optimizer.param_groups: pg['lr'] = lr
        elif epoch == args.warmup_epochs:
            for pg in optimizer.param_groups: pg['lr'] = args.lr

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x        = batch['x'].to(device)
            ei       = batch['edge_index'].to(device)
            ea       = batch['edge_attr'].to(device)
            gene_ids = batch['gene_ids'].to(device)
            labels   = batch['label'].to(device)
            cell_ids = batch['cell_id'].to(device)

            with autocast(enabled=args.use_amp):
                results = model(gene_ids, x, ei, ea, batch['num_nodes_list'], cell_ids)
                loss = torch.tensor(0.0, device=device)
                for cid, (logits, spectrum, sigma, U) in results.items():
                    mask = cell_ids == cid
                    # BCE
                    loss_bce = criterion(logits, labels[mask])
                    # 正交正则
                    U_n = F.normalize(U, dim=-1)
                    gram = torch.bmm(U_n, U_n.transpose(1,2))
                    eye  = torch.eye(args.operator_rank, device=device).unsqueeze(0)
                    loss_ortho = (gram - eye).pow(2).mean()
                    loss_sparse = sigma.abs().mean()
                    loss += (loss_bce +
                             args.lam_ortho_modes * loss_ortho +
                             args.lam_sparse * loss_sparse)
                loss = loss / n_cells

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

        # 验证：各细胞系独立评估
        model.eval()
        aucs = []
        with torch.no_grad():
            for cid, vl in enumerate(val_loaders):
                preds, labs = [], []
                for batch in vl:
                    x  = batch['x'].to(device)
                    ei = batch['edge_index'].to(device)
                    ea = batch['edge_attr'].to(device)
                    gids = batch['gene_ids'].to(device)
                    cids = batch['cell_id'].to(device)
                    with autocast(enabled=args.use_amp):
                        res = model(gids, x, ei, ea, batch['num_nodes_list'], cids)
                    if cid in res:
                        logits, _, _, _ = res[cid]
                        preds.extend(torch.sigmoid(logits).cpu().numpy())
                        labs.extend(batch['label'].numpy())
                if preds:
                    aucs.append(roc_auc_score(labs, preds))

        mean_auc = np.mean(aucs)
        scheduler.step(mean_auc)
        n = len(train_loader)
        auc_str = ' '.join([f'{cell_names[i]}:{aucs[i]:.4f}' for i in range(len(aucs))])
        print(f"Ep {epoch+1:02d} | L:{total_loss/n:.3f} | {auc_str} | mean:{mean_auc:.4f}")

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优平均 AUC: {best_mean_auc:.4f}")
    return best_mean_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str,
        default='/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended')
    parser.add_argument('--cells',      nargs='+', default=['MCF7','A375','A549','VCAP'])
    parser.add_argument('--device',     type=str,  default='cuda:0')
    parser.add_argument('--fold',       type=int,  default=0)
    parser.add_argument('--seed',       type=int,  default=42)
    parser.add_argument('--epochs',     type=int,  default=80)
    parser.add_argument('--batch_size', type=int,  default=512)
    parser.add_argument('--lr',         type=float,default=2e-4)
    parser.add_argument('--hidden_dim', type=int,  default=128)
    parser.add_argument('--dropout',    type=float,default=0.3)
    parser.add_argument('--patience',   type=int,  default=10)
    parser.add_argument('--use_amp',    action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--gene_max_len', type=int,  default=1000)
    parser.add_argument('--operator_rank',type=int,  default=8)
    parser.add_argument('--lam_sparse',   type=float,default=0.01)
    parser.add_argument('--lam_ortho_modes',type=float,default=0.1)
    parser.add_argument('--run_tag',    type=str,  default='')
    train(parser.parse_args())
