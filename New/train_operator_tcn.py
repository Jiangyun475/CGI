#!/usr/bin/env python3
"""
OperatorNet-TCN: 空洞 TCN 基因编码器 + 算子网络（无 MoE）
============================================================

与 train_operator_moe.py --ablation no_moe 的唯一区别：
  基因编码器从 GeneMultiHeadReader（标准多尺度 CNN）
  换成 GeneTCNReader（TCN 风格空洞卷积 + 多槽注意力池化）

TCN vs 标准 CNN 的核心差异：
  标准 CNN：4 个不同 kernel 的并行卷积，感受野最大 ~17 k-mer
  TCN：     5 层串行空洞卷积 + 残差，感受野 63 k-mer（≈68 bp）

感受野计算（kernel=3）：
  Layer 1: dilation=1,  RF=3
  Layer 2: dilation=2,  RF=7
  Layer 3: dilation=4,  RF=15
  Layer 4: dilation=8,  RF=31
  Layer 5: dilation=16, RF=63  ← 覆盖完整启动子核心区

为什么串行堆叠优于并行分支：
  并行：4 支各自独立，concat 后是"4 种浅层视角的拼接"，没有层次积累
  串行：每层在上一层的基础上扩大感受野，高层特征融合了低层局部+全局信息

化学侧与分类侧与 no_moe 完全一致（单变量对比）：
  PharmacophoreExtractor + PerturbationOperator → spectrum [B,r] → MLP

训练命令：
  python New/train_operator_tcn.py \\
    --data_dir .../MCF7 --device cuda:0 \\
    --operator_rank 8 --epochs 80 --batch_size 512 --lr 2e-4 \\
    --lam_sparse 0.01 --lam_ortho_modes 0.1 \\
    --warmup_epochs 5 --patience 10 --seed 42 \\
    --use_amp --save_spectrum --run_tag v1
"""

import argparse
import itertools
import math
import random
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


def scatter_softmax(scores, batch_idx):
    scores = scores.float()
    max_scores = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
    exp_scores = torch.exp(scores - max_scores[batch_idx])
    exp_sum = torch.zeros(
        batch_idx.max().item() + 1, device=scores.device
    ).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)


def scatter_add(src, batch_idx, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, batch_idx, src)


# ================================================================
# 1. 数据集
# ================================================================

class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train', gene_max_len=1000):
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

        suffix = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}{suffix}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 (len={gene_max_len})...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len)
                 for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
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
        'x':              torch.cat(all_x, dim=0),
        'edge_index':     torch.cat(all_edge_index, dim=1) if all_edge_index
                          else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr':      torch.cat(all_edge_attr, dim=0) if all_edge_attr
                          else torch.zeros(0, 4),
        'num_nodes_list': num_nodes_list,
        'gene_ids':       torch.stack([b['gene_ids'] for b in batch]),
        'label':          torch.stack([b['label']    for b in batch]),
    }


# ================================================================
# 2. 基因编码器：GeneTCNReader
#    TCN 风格串行空洞卷积 + 多槽注意力池化
# ================================================================

class DilatedResBlock(nn.Module):
    """
    单个空洞残差块（TCN 基本单元）。

    结构：
      x → Conv1d(dilation=d) → LayerNorm → GELU → Dropout → + x（残差）

    关键细节：
      padding = dilation * (kernel-1) // 2  确保输出长度 = 输入长度（same padding）
      LayerNorm 在序列维度，与 BatchNorm 不同，对长序列更稳定
    """
    def __init__(self, hidden_dim: int, kernel: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel,
            padding=padding, dilation=dilation)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, H]"""
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, H]
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + x  # 残差连接保留低层局部特征


class GeneTCNReader(nn.Module):
    """
    TCN 基因编码器：串行空洞卷积 + 多槽注意力池化。

    感受野（kernel=3）：
      dilations=[1,2,4,8,16] → RF = 1+2*(1+2+4+8+16) = 63 k-mer ≈ 68 bp
      dilations=[1,2,4,8]    → RF = 1+2*(1+2+4+8)    = 31 k-mer ≈ 36 bp

    前向流程：
      gene_ids [B, L]
        → embedding [B, L, H]
        → input_proj [B, L, H]       （从 vocab_dim 投影到 hidden_dim）
        → TCN blocks × N [B, L, H]   （串行空洞，感受野递增）
        → seq_norm [B, L, H]
        → attn_queries [r, H] → scores [B, r, L] → softmax → attn [B, r, L]
        → weighted sum → h_g_modes [B, r, H]
        → mean → h_g_global [B, H]
    """
    def __init__(self, vocab_size: int = 4097, hidden_dim: int = 128,
                 num_heads: int = 8, dropout: float = 0.3,
                 dilations: list = None, kernel: int = 3):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.num_heads = num_heads
        H = hidden_dim

        # k-mer 嵌入
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)

        # 串行 TCN 块（每块感受野翻倍）
        self.tcn_blocks = nn.ModuleList([
            DilatedResBlock(H, kernel=kernel, dilation=d, dropout=dropout * 0.5)
            for d in dilations
        ])

        self.seq_norm = nn.LayerNorm(H)

        # 多槽注意力查询（零初始化，warmup 后才有效）
        self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))

        self.out_norm = nn.LayerNorm(H)
        self.dropout  = dropout

        # 记录感受野信息（用于打印）
        rf = 1 + sum((kernel - 1) * d for d in dilations)
        self.receptive_field = rf
        self.dilations = dilations

    def forward(self, gene_ids: torch.Tensor):
        """
        Returns:
          h_g_modes  [B, r, H]  — r 个模式专属基因视角
          h_g_global [B, H]     — 全局基因表示
          attn       [B, r, L]  — 注意力权重（可解释性）
        """
        # 嵌入
        x = self.embedding(gene_ids)                       # [B, L, H]
        x = F.dropout(x, p=self.dropout * 0.3, training=self.training)

        # 串行 TCN 块
        for block in self.tcn_blocks:
            x = block(x)                                   # [B, L, H]

        x = self.seq_norm(x)

        # 多槽注意力池化
        scores    = torch.einsum('blh,rh->brl', x, self.attn_queries) \
                    / math.sqrt(x.size(-1))                # [B, r, L]
        attn      = F.softmax(scores, dim=-1)              # [B, r, L]
        h_g_modes = torch.einsum('brl,blh->brh', attn, x) # [B, r, H]
        h_g_modes = self.out_norm(h_g_modes)

        h_g_global = h_g_modes.mean(dim=1)                # [B, H]

        return h_g_modes, h_g_global, attn


# ================================================================
# 3. 化学编码器（与 no_moe 完全一致）
# ================================================================

class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim))
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
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=self.dropout, training=self.training)
        return x


# ================================================================
# 4. 药效团提取 + 扰动算子（与 no_moe 完全一致）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.queries  = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs):
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)
        V = self.val_proj(atom_h)
        scores_all = (K @ self.queries.T) / math.sqrt(d)  # [N, r]

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all  # [B, r, H], [N, r]


class PerturbationOperator(nn.Module):
    """
    模式对齐耦合（V2 设计）：
      coupling_k = v_k · h_g_modes_k  （每模式专属基因视角）
      spectrum_k = σ_k * coupling_k
      delta_h    = Σ_k spectrum_k * u_k
    """
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
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)  # [B, r, H]
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)  # [B, r, H]
        sigma = self.to_sigma(pharma_emb).squeeze(-1)        # [B, r]
        coupling = (V * h_g_modes).sum(-1)                   # [B, r]
        spectrum = sigma * coupling                           # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)  # [B, H]
        return delta_h, spectrum, sigma, U


# ================================================================
# 5. 完整模型：OperatorNetTCN
# ================================================================

class OperatorNetTCN(nn.Module):
    """
    DrugOperatorNet + TCN 基因编码器。

    相较于 train_operator_moe.py --ablation no_moe，
    唯一变化是将 GeneMultiHeadReader（标准多尺度CNN）
    替换为 GeneTCNReader（串行空洞TCN）。
    其余组件完全相同，保证单变量对比有效性。
    """
    def __init__(self, hidden_dim=128, dropout=0.3,
                 operator_rank=8, dilations=None, kernel=3):
        super().__init__()
        r = operator_rank

        self.gene_enc  = GeneTCNReader(
            hidden_dim=hidden_dim, num_heads=r, dropout=dropout,
            dilations=dilations, kernel=kernel)
        self.atom_enc  = AtomEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.pharma_ext = PharmacophoreExtractor(hidden_dim, r)
        self.perturb_op = PerturbationOperator(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

        self.dropout_p = dropout

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=device)
        batch_idx   = torch.repeat_interleave(
            torch.arange(B, device=device), num_nodes_t)

        # 基因编码（TCN）
        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)

        # 化学原子编码
        atom_h = self.atom_enc(x, edge_index, edge_attr)

        # 药效团提取 + 算子
        pharma_emb, _ = self.pharma_ext(atom_h, batch_idx, B)
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        # 分类
        features = torch.cat([h_g_global, delta_h], dim=-1)
        logits   = self.classifier(features).squeeze(-1)

        return logits, spectrum, sigma, U, gene_attn


# ================================================================
# 6. 正则化
# ================================================================

def compute_reg_loss(sigma, U, lam_sparse, lam_ortho):
    loss_sparse = sigma.abs().mean()
    U_n  = F.normalize(U, dim=-1)
    gram = torch.bmm(U_n, U_n.transpose(1, 2))
    eye  = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
    loss_ortho = (gram - eye).pow(2).mean()
    return lam_sparse * loss_sparse + lam_ortho * loss_ortho


# ================================================================
# 7. LR Warmup
# ================================================================

def get_lr(opt):
    return opt.param_groups[0]['lr']

def set_lr(opt, lr):
    for pg in opt.param_groups:
        pg['lr'] = lr


# ================================================================
# 8. 训练
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    dilations = list(map(int, args.dilations.split(',')))

    train_ds = OptimizedGraphDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = OperatorNetTCN(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        operator_rank=args.operator_rank,
        dilations=dilations, kernel=args.kernel,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rf       = model.gene_enc.receptive_field

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_operator_tcn/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    dil_str    = args.dilations.replace(',', '-')
    model_name = f"tcn_d{dil_str}_k{args.kernel}_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0
    base_lr = args.lr

    print(f"\n{'='*72}")
    print(f"  OperatorNet-TCN")
    print(f"  基因编码器: TCN | kernel={args.kernel} | dilations={dilations}")
    print(f"  感受野: {rf} k-mer ≈ {rf} bp（含重叠）")
    print(f"  operator_rank={args.operator_rank} | gene_max_len={args.gene_max_len}")
    print(f"  params={n_params:,} | warmup={args.warmup_epochs}ep | device={args.device}")
    print(f"{'='*72}\n")

    for epoch in range(args.epochs):
        # warmup
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        total_loss = total_bce = total_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss_bce = criterion(logits, labels)
                loss_reg = compute_reg_loss(sigma, U, args.lam_sparse, args.lam_ortho_modes)
                loss     = loss_bce + loss_reg

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

            total_loss += loss.item()
            total_bce  += loss_bce.item()
            total_reg  += loss_reg.item()

        # 验证
        model.eval()
        all_preds, all_labels, all_spectra = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x         = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr  = batch['edge_attr'].to(device)
                gene_ids   = batch['gene_ids'].to(device)

                with autocast(enabled=args.use_amp):
                    logits, spectrum, _, _, _ = model(
                        gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                if args.save_spectrum and spectrum is not None:
                    all_spectra.append(spectrum.cpu())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        if epoch >= args.warmup_epochs:
            scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} (BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f}) | "
              f"VAL_AUC:{auroc:.4f} PRC:{auprc:.4f} F1:{f1:.4f}")

        if auroc > best_auroc:
            best_auroc   = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
            if args.save_spectrum and all_spectra:
                torch.save({
                    'spectrum': torch.cat(all_spectra, dim=0),
                    'preds':    np.array(all_preds),
                    'labels':   np.array(all_labels),
                }, save_dir / f'spectrum_{model_name}')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


# ================================================================
# 9. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OperatorNet-TCN: 串行空洞TCN基因编码器')

    parser.add_argument('--data_dir',     type=str, required=True)
    parser.add_argument('--device',       type=str, default='cuda:0')
    parser.add_argument('--fold',         type=int, default=0)
    parser.add_argument('--seed',         type=int, default=42)

    parser.add_argument('--epochs',       type=int, default=80)
    parser.add_argument('--batch_size',   type=int, default=512)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--hidden_dim',   type=int, default=128)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--patience',     type=int, default=10)
    parser.add_argument('--use_amp',      action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--gene_max_len', type=int, default=1000)
    parser.add_argument('--operator_rank', type=int, default=8,
                        help='算子秩 r = 药效团数 = 基因读取头数')

    # TCN 专属参数
    parser.add_argument('--dilations', type=str, default='1,2,4,8,16',
                        help='空洞率列表，逗号分隔。'
                             '默认 1,2,4,8,16 → RF=63 k-mer；'
                             '1,2,4,8 → RF=31；1,2,4 → RF=15')
    parser.add_argument('--kernel',   type=int, default=3,
                        help='TCN 卷积核大小（建议3或5，奇数保证same padding）')

    parser.add_argument('--lam_sparse',      type=float, default=0.01)
    parser.add_argument('--lam_ortho_modes', type=float, default=0.1)

    parser.add_argument('--save_spectrum', action='store_true')
    parser.add_argument('--run_tag',       type=str, default='')

    train(parser.parse_args())
