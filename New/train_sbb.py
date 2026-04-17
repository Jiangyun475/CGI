#!/usr/bin/env python3
"""
SBB + TanimotoConsistency: 共享生物学基 & 结构-谱一致性
=========================================================

在 train_operator_moe.py (no_moe 基准) 基础上新增两个创新组件：

1. SharedBiologicalBasis (SBB, --use_sbb)
   ─────────────────────────────────────
   原设计：GeneMultiHeadReader 的 r 个 attn_queries 与
           PharmacophoreExtractor 的 r 个 slot queries 完全独立初始化，
           在算子层才隐式对齐。算子需同时承担"如何扰动"和"拉齐两空间"两个任务。

   SBB 设计：共用一套可学习基 B ∈ R^{r×H}，两个编码器都用 B[j] 作为第 j 个查询。
             药物和基因天然在同一生物学坐标系下被描述，算子只需学习强度 σ，
             无需再做空间对齐。

   数学上：原本 drug_slot_j 和 gene_query_j 随机游走后被 V_j·h_g_modes_j 拉齐；
           SBB 把这个对齐从"事后"提前到"事前"，并让 r 个维度共享语义。

2. TanimotoConsistencyLoss (--lam_tanimoto)
   ─────────────────────────────────────────
   原 SpectrumDirectionCL 失败原因：BCE 已完全覆盖了"方向分离"信号，CL 是冗余监督。

   Tanimoto 设计：结构相似的药物应激活相似的生物学程序（相似 σ 指纹）。
                  这个信号 BCE 根本看不到——BCE 只能看到 (drug, gene) 对的标签，
                  看不到 drug-drug 之间的结构关系。

   实现：用 RDKit 计算 Morgan FP（2048-bit），无任何预训练，纯算法变换。
         batch 内计算 Tanimoto_ij vs cosine(σ_i, σ_j)，MSE 对齐两者。

   loss_tanimoto = MSE( cosine(σ_i, σ_j), Tanimoto(drug_i, drug_j) )  ∀ i≠j in batch

消融实验设计（对比基准：train_operator_moe.py，no_moe 配置）：
  base  : --ablation no_moe                   （原论文基准）
  +SBB  : --ablation no_moe --use_sbb          （SBB 单独贡献）
  +Tan  : --ablation no_moe --lam_tanimoto 0.1 （Tanimoto 单独贡献）
  +Both : --ablation no_moe --use_sbb --lam_tanimoto 0.1
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
# 0. 工具函数（与 train_operator_moe.py 完全一致）
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
# 新增：Morgan 指纹计算（无预训练，纯 RDKit 算法）
# ================================================================

def compute_morgan_fps(smiles_list, radius=2, n_bits=2048):
    """从 SMILES 列表计算 Morgan 指纹，返回 [N, n_bits] float32 tensor。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
    return torch.tensor(np.array(fps), dtype=torch.float32)


# ================================================================
# 1. 数据集（增加 Morgan FP 字段）
# ================================================================

class OptimizedGraphDataset(Dataset):
    def __init__(self, data_dir, fold_idx=0, split='train', gene_max_len=1000,
                 load_morgan=True):
        import pickle
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt',
                               weights_only=False)
        self.smiles_to_graph = self.data['smiles_to_graph']
        self.graph_indices = [self.data['graph_indices'][i] for i in self.indices]
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        if 'zscores' in self.data:
            self.zscores = torch.tensor(
                [self.data['zscores'][i] for i in self.indices], dtype=torch.float32)
        else:
            self.zscores = None

        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]
        suffix = '' if gene_max_len == 1000 else f'_len{gene_max_len}'
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}{suffix}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file, weights_only=True)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存 (len={gene_max_len})...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len)
                 for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

        # Morgan FP（新增）：缓存到磁盘避免重复计算
        self.morgan_fps = None
        if load_morgan:
            morgan_cache = Path(data_dir) / f'morgan_fps_fold{fold_idx}_{split}.pt'
            if morgan_cache.exists():
                print(f"[{split.upper()}] ⚡ Morgan FP 缓存: {morgan_cache.name}")
                self.morgan_fps = torch.load(morgan_cache, weights_only=True)
            else:
                print(f"[{split.upper()}] 计算 Morgan FP...")
                self.morgan_fps = compute_morgan_fps(self.graph_indices)
                torch.save(self.morgan_fps, morgan_cache)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'graph':    self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label':    self.labels[idx],
        }
        if self.zscores is not None:
            item['zscore'] = self.zscores[idx]
        if self.morgan_fps is not None:
            item['morgan_fp'] = self.morgan_fps[idx]
        return item


def collate_fn(batch):
    all_x, all_edge_index, all_edge_attr, num_nodes_list = [], [], [], []
    offset = 0
    for item in batch:
        g = item['graph']
        all_x.append(g['x'])
        num_nodes = g['x'].shape[0]
        num_nodes_list.append(num_nodes)
        if g['edge_index'].shape[1] > 0:
            all_edge_index.append(g['edge_index'] + offset)
            all_edge_attr.append(g['edge_attr'])
        offset += num_nodes

    result = {
        'x':            torch.cat(all_x, dim=0),
        'edge_index':   torch.cat(all_edge_index, dim=1) if all_edge_index
                        else torch.zeros(2, 0, dtype=torch.long),
        'edge_attr':    torch.cat(all_edge_attr, dim=0) if all_edge_attr
                        else torch.zeros(0, 4),
        'gene_ids':     torch.stack([item['gene_ids'] for item in batch]),
        'label':        torch.stack([item['label'] for item in batch]),
        'num_nodes_list': num_nodes_list,
        'zscore':       torch.stack([item['zscore'] for item in batch])
                        if 'zscore' in batch[0] else None,
        'morgan_fp':    torch.stack([item['morgan_fp'] for item in batch])
                        if 'morgan_fp' in batch[0] else None,
    }
    return result


# ================================================================
# 2. 基因编码器（支持 SharedBiologicalBasis）
# ================================================================

class GeneMultiHeadReader(nn.Module):
    """
    r 个注意力头各自读取基因序列。
    shared_basis: 若提供，用共享基 B[j] 替代独立的 attn_queries[j]。
    """
    def __init__(self, vocab_size=4097, hidden_dim=128, num_heads=8, dropout=0.3,
                 shared_basis=None):
        super().__init__()
        self.num_heads = num_heads
        H = hidden_dim
        self.shared_basis = shared_basis  # SharedBiologicalBasis 或 None

        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.seq_norm = nn.LayerNorm(H)

        # 独立 attn_queries 仅在 shared_basis=None 时使用
        if shared_basis is None:
            self.attn_queries = nn.Parameter(torch.zeros(num_heads, H))
        self.out_norm = nn.LayerNorm(H)
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x = self.seq_norm(x)

        queries = self.shared_basis.basis if self.shared_basis is not None \
                  else self.attn_queries                                       # [r, H]
        scores = torch.einsum('blh,rh->brl', x, queries) / math.sqrt(x.size(-1))
        attn   = F.softmax(scores, dim=-1)
        h_g_modes = torch.einsum('brl,blh->brh', attn, x)
        h_g_modes = self.out_norm(h_g_modes)
        h_g_global = h_g_modes.mean(dim=1)
        return h_g_modes, h_g_global, attn


# ================================================================
# 3. 化学编码器（GIN 与原版相同）
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
# 4. 药效团提取（支持 SharedBiologicalBasis）
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """
    r 个 slot query × 原子注意力 → r 个药效团嵌入。
    shared_basis: 若提供，用共享基 B[j] 替代独立的 queries[j]。
    """
    def __init__(self, hidden_dim, num_slots, shared_basis=None):
        super().__init__()
        self.num_slots = num_slots
        self.shared_basis = shared_basis

        if shared_basis is None:
            self.queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs):
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)
        V = self.val_proj(atom_h)

        queries = self.shared_basis.basis if self.shared_basis is not None \
                  else self.queries                                            # [r, H]
        scores_all = (K @ queries.T) / math.sqrt(d)

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all


# ================================================================
# 新增：SharedBiologicalBasis
# ================================================================

class SharedBiologicalBasis(nn.Module):
    """
    r 个生物学程序的共享基，同时用于基因编码器和药物药效团提取器。

    零初始化（与 GeneMultiHeadReader 的 attn_queries 保持一致），
    warmup 期间梯度缓慢建立，避免早期随机注意力破坏训练稳定性。
    """
    def __init__(self, rank, hidden_dim):
        super().__init__()
        self.basis = nn.Parameter(torch.zeros(rank, hidden_dim))


# ================================================================
# 5. 扰动算子（与原版相同）
# ================================================================

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
        U     = F.normalize(self.to_u(pharma_emb), dim=-1)
        V     = F.normalize(self.to_v(pharma_emb), dim=-1)
        sigma = self.to_sigma(pharma_emb).squeeze(-1)
        coupling = (V * h_g_modes).sum(-1)
        spectrum = sigma * coupling
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)
        return delta_h, spectrum, sigma, U


# ================================================================
# 6. 完整模型（no_moe 为基准，新增 use_sbb 开关）
# ================================================================

class OperatorNet(nn.Module):
    """
    DrugOperatorNet（no_moe 消融版）+ SBB 选项。
    no_moe 已是论文基准最优配置，在此基础上引入 SBB。
    """
    def __init__(self, hidden_dim=128, dropout=0.3, operator_rank=8, use_sbb=False):
        super().__init__()
        self.operator_rank = operator_rank
        H = hidden_dim
        r = operator_rank

        # 共享生物学基（可选）
        self.sbb = SharedBiologicalBasis(r, H) if use_sbb else None

        self.gene_enc  = GeneMultiHeadReader(
            hidden_dim=H, num_heads=r, dropout=dropout, shared_basis=self.sbb)
        self.atom_enc  = AtomEncoder(hidden_dim=H, dropout=dropout)
        self.pharma_ext = PharmacophoreExtractor(H, r, shared_basis=self.sbb)
        self.perturb_op = PerturbationOperator(H)

        self.classifier = nn.Sequential(
            nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(H, 1))
        self.dropout_p = dropout

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        B = len(num_nodes_list)
        num_nodes_t = torch.tensor(num_nodes_list, device=x.device)
        batch_idx   = torch.repeat_interleave(
            torch.arange(B, device=x.device), num_nodes_t)

        h_g_modes, h_g_global, gene_attn = self.gene_enc(gene_ids)
        atom_h = self.atom_enc(x, edge_index, edge_attr)
        pharma_emb, atom_scores = self.pharma_ext(atom_h, batch_idx, B)
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        features = torch.cat([h_g_global, delta_h], dim=-1)
        logits   = self.classifier(features).squeeze(-1)
        return logits, spectrum, sigma, U, gene_attn


# ================================================================
# 新增：Tanimoto 谱一致性损失
# ================================================================

class TanimotoConsistencyLoss(nn.Module):
    """
    结构相似的药物应激活相似的生物学程序（相似 σ 指纹）。

    给定 batch 中的 Morgan FP [B, 2048] 和 sigma [B, r]：
      T_ij = Tanimoto(mfp_i, mfp_j)        ← 药物结构相似性（无预训练，纯算法）
      C_ij = cosine(σ_i, σ_j)              ← 谱空间相似性
      loss = MSE(C_ij, T_ij)  over i≠j

    为什么这能工作而 SpectrumDirectionCL 不能：
      - CL 监督的是 (drug, gene) 对的方向，BCE 已完全覆盖。
      - Tanimoto 监督的是 drug-drug 之间的关系，BCE 看不到这个信号。
      - 结构相似的药物在化学冷分割下可能出现在不同 fold，
        但它们的谱指纹应当一致，这个约束直接提升冷分割泛化能力。
    """
    def __init__(self):
        super().__init__()

    def forward(self, sigma: torch.Tensor, morgan_fp: torch.Tensor) -> torch.Tensor:
        """
        sigma:     [B, r]    模式强度指纹
        morgan_fp: [B, 2048] Morgan 指纹（binary float）
        """
        B = sigma.size(0)
        if B < 2:
            return sigma.sum() * 0.0

        # Tanimoto 相似度矩阵 [B, B]（binary 向量的 Tanimoto = Jaccard）
        dot  = morgan_fp @ morgan_fp.T                          # [B, B] 交集大小
        sums = morgan_fp.sum(dim=1)                             # [B]    每个药物的 bit 数
        union = sums.unsqueeze(1) + sums.unsqueeze(0) - dot     # [B, B] 并集大小
        tanimoto = dot / (union + 1e-8)                         # [B, B]

        # σ 的余弦相似度矩阵 [B, B]
        sigma_n = F.normalize(sigma, dim=-1)                    # [B, r]
        cosine  = sigma_n @ sigma_n.T                           # [B, B]

        # 只取上三角（排除对角线和重复）
        mask = torch.triu(torch.ones(B, B, device=sigma.device, dtype=torch.bool), diagonal=1)
        loss = F.mse_loss(cosine[mask], tanimoto[mask])
        return loss


# ================================================================
# 7. 损失计算
# ================================================================

def compute_losses(logits, labels, spectrum, sigma, U,
                   criterion, args, tanimoto_loss_fn=None, morgan_fp=None):
    loss_bce = criterion(logits, labels)

    loss_sparse = sigma.abs().mean()
    U_n   = F.normalize(U, dim=-1)
    gram  = torch.bmm(U_n, U_n.transpose(1, 2))
    eye   = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
    loss_ortho = (gram - eye).pow(2).mean()
    loss_reg   = args.lam_sparse * loss_sparse + args.lam_ortho_modes * loss_ortho

    loss_tan = torch.tensor(0.0, device=logits.device)
    if tanimoto_loss_fn is not None and morgan_fp is not None and args.lam_tanimoto > 0:
        loss_tan = tanimoto_loss_fn(sigma, morgan_fp)

    total = loss_bce + loss_reg + args.lam_tanimoto * loss_tan
    return total, loss_bce, loss_reg, loss_tan


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

    load_morgan = args.lam_tanimoto > 0
    train_ds = OptimizedGraphDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len, load_morgan=load_morgan)
    val_ds   = OptimizedGraphDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len, load_morgan=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    model = OperatorNet(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        operator_rank=args.operator_rank,
        use_sbb=args.use_sbb,
    ).to(device)

    tanimoto_loss_fn = TanimotoConsistencyLoss().to(device) if args.lam_tanimoto > 0 else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_sbb/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    sbb_str    = "_sbb" if args.use_sbb else ""
    tan_str    = f"_tan{args.lam_tanimoto}" if args.lam_tanimoto > 0 else ""
    model_name = f"sbb_r{args.operator_rank}_Fold{args.fold}{sbb_str}{tan_str}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0
    base_lr = args.lr

    print(f"\n{'='*72}")
    print(f"  OperatorNet-SBB | use_sbb={args.use_sbb} | lam_tanimoto={args.lam_tanimoto}")
    print(f"  operator_rank={args.operator_rank} | gene_max_len={args.gene_max_len}")
    print(f"  params={n_params:,} | device={args.device} | fold={args.fold}")
    print(f"{'='*72}\n")

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            cur_lr = base_lr / 10 * (1 + 9 * epoch / max(args.warmup_epochs - 1, 1))
            set_lr(optimizer, cur_lr)
        elif epoch == args.warmup_epochs:
            set_lr(optimizer, base_lr)

        model.train()
        total_loss = total_bce = total_reg = total_tan = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)
            morgan_fp   = batch['morgan_fp'].to(device) \
                          if batch['morgan_fp'] is not None else None

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _ = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])
                loss, loss_bce, loss_reg, loss_tan = compute_losses(
                    logits, labels, spectrum, sigma, U,
                    criterion, args,
                    tanimoto_loss_fn=tanimoto_loss_fn, morgan_fp=morgan_fp)

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
            total_tan  += loss_tan.item()

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
                    logits, _, _, _, _ = model(
                        gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        if epoch >= args.warmup_epochs:
            scheduler.step(auroc)

        n = len(train_loader)
        tan_str_log = f" Tan:{total_tan/n:.4f}" if tanimoto_loss_fn else ""
        print(f"Ep {epoch+1:02d} [lr={get_lr(optimizer):.2e}] | "
              f"L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f}{tan_str_log}) | "
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

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


# ================================================================
# 10. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OperatorNet + SBB + Tanimoto')

    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--device',        type=str, default='cuda:0')
    parser.add_argument('--fold',          type=int, default=0)
    parser.add_argument('--seed',          type=int, default=42)

    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int, default=10)
    parser.add_argument('--use_amp',       action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--gene_max_len',  type=int, default=1000)
    parser.add_argument('--operator_rank', type=int, default=8)

    parser.add_argument('--lam_sparse',      type=float, default=0.01)
    parser.add_argument('--lam_ortho_modes', type=float, default=0.1)

    # 新增：SBB 开关
    parser.add_argument('--use_sbb',  action='store_true',
                        help='启用共享生物学基（GeneReader 和 PharmacophoreExtractor 共用 B∈R^{r×H}）')

    # 新增：Tanimoto 一致性损失
    parser.add_argument('--lam_tanimoto', type=float, default=0.0,
                        help='Tanimoto 谱一致性损失系数（0=禁用，建议先试 0.1）')

    parser.add_argument('--run_tag',  type=str, default='')

    train(parser.parse_args())
