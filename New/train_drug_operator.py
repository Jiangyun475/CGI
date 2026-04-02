#!/usr/bin/env python3
"""
DrugOperatorNet: Drug-as-Perturbation-Operator Network for CGI Prediction
=========================================================================

核心科学前提 (Core Scientific Premise):
  药物分子不是一个静态特征向量——它是一个作用于基因表达状态空间的
  **扰动算子 (Perturbation Operator)**。

  传统方法: encode(drug) → h_c,  encode(gene) → h_g,  classify(h_c ⊕ h_g)
  本方法:   encode(drug) → T_drug (算子),  encode(gene) → h_g (状态),
            Δh = T_drug(h_g) - h_g  (扰动向量),  classify(h_g, Δh)

数学形式 (Mathematical Formulation):
  药物算子采用恒等 + 低秩扰动的形式:
    T_drug = I + U · diag(σ) · V^T

  其中 U, V ∈ R^{d×r} 和 σ ∈ R^r 由药物分子的**药效团级注意力
  (Pharmacophore-level Cross-Attention)** 生成:
    1. GIN → 原子嵌入 H_atoms
    2. r 个可学习的药效团查询向量 Q 对原子做交叉注意力 → r 个药效团嵌入 P
    3. 每个药效团 P_k → (u_k, v_k, σ_k): 扰动方向、耦合方向、扰动幅度

  扰动计算:
    coupling_k = v_k^T · h_g              (基因对第 k 模式的耦合强度)
    spectrum_k = σ_k · coupling_k          (第 k 模式的激活强度)
    Δh = Σ_k spectrum_k · u_k             (总扰动向量)

  "交互谱 (Interaction Spectrum)" S = [s_1, ..., s_r] 是药物-基因互作
  的 r 维可解释指纹:
    - 每个维度对应一种独立的"作用模式 (Mode of Action)"
    - 药效团注意力权重指示哪些原子/亚结构驱动哪种模式
    - σ_k 的大小反映该模式的内在强度

消融对比模式 (Ablation Baselines via --interaction_type):
  operator     : [提出方法] 药物作为低秩扰动算子
  concat       : [基线] 拼接 h_c ⊕ h_g → MLP
  ortho_concat : [基线] 正交剥离 V_c⊥ ⊕ V_g → MLP (现有 PaperModel)
  hadamard     : [基线] 元素积 h_c ⊙ h_g + h_g → MLP

附加控制:
  --operator_rank R      扰动模式数 (默认 8)
  --lam_sparse           σ 的 L1 稀疏正则 (鼓励少数模式主导)
  --lam_ortho_modes      U 列向量正交正则 (鼓励独立模式)
  --save_spectrum         保存交互谱用于下游可解释性分析
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
def scatter_softmax(scores, batch_idx):
    """图内原子级 softmax，纯 PyTorch 实现（无需 torch_geometric）。"""
    max_scores = torch.zeros(batch_idx.max().item() + 1,
                             device=scores.device).index_reduce_(
                                 0, batch_idx, scores, 'amax', include_self=True)
    exp_scores = torch.exp(scores - max_scores[batch_idx])
    exp_sum = torch.zeros(batch_idx.max().item() + 1,
                          device=scores.device).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ================================================================
# 0. 工具函数 & 随机种子
# ================================================================

_KMER_VOCAB = {}
for i, combo in enumerate(itertools.product('ACGT', repeat=6), 1):
    _KMER_VOCAB[''.join(combo)] = i
_KMER_VOCAB['NNNNNN'] = 0

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# ================================================================
# 1. 数据集 (与现有代码完全兼容)
# ================================================================

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
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]

        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}.pt'
        if cache_file.exists():
            print(f"[{split.upper()}] ⚡ 加载 K-mer 缓存: {cache_file.name}")
            self.gene_ids = torch.load(cache_file)
        else:
            print(f"[{split.upper()}] 生成 K-mer 缓存...")
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq) for seq in tqdm(gene_sequences)],
                dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'graph': self.smiles_to_graph[self.graph_indices[idx]],
            'gene_ids': self.gene_ids[idx],
            'label': self.labels[idx],
        }

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
    x = torch.cat(all_x, dim=0)
    edge_index = (torch.cat(all_edge_index, dim=1)
                  if all_edge_index else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = (torch.cat(all_edge_attr, dim=0)
                 if all_edge_attr else torch.zeros(0, 4))
    return {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
        'num_nodes_list': num_nodes_list,
        'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

# ================================================================
# 2. 共享编码器 (与现有模型一致，确保公平对比)
# ================================================================

class GeneEncoderV1(nn.Module):
    """多尺度 CNN + TopK 池化的基因序列编码器"""
    def __init__(self, vocab_size=4097, hidden_dim=128, k=10, dropout=0.3):
        super().__init__()
        self.k = k
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        kernel_sizes = [6, 8, 10, 12]
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(self, gene_ids):
        x = self.embedding(gene_ids).transpose(1, 2)       # [B, d, L]
        features = torch.cat([conv(x) for conv in self.convs], dim=1)  # [B, d, L']
        features = F.dropout(F.relu(features), p=self.dropout, training=self.training)
        values, _ = torch.topk(features, k=self.k, dim=2)  # [B, d, k]
        return self.aggregation(values.mean(dim=2))          # [B, d]


class GINLayer(nn.Module):
    """带边特征的 GIN 消息传递层"""
    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge_proj = nn.Linear(edge_dim, dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_emb = self.edge_proj(edge_attr)
        msg = F.relu(x[row] + edge_emb)
        neighbor = torch.zeros_like(x).index_add_(0, col, msg)
        return self.mlp(x + neighbor)


class AtomEncoder(nn.Module):
    """GIN ×3 原子级编码器，返回原子嵌入 (不做全局池化)"""
    def __init__(self, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.atom_embed = nn.Sequential(
            nn.Linear(31, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.gin_layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        for gin, norm in zip(self.gin_layers, self.norms):
            x = F.dropout(
                F.relu(norm(x + gin(x, edge_index, edge_attr))),
                p=self.dropout, training=self.training)
        return x  # [N_total_atoms, d]

# ================================================================
# 3. 核心创新模块: 药效团算子 (Pharmacophore Perturbation Operator)
# ================================================================

class PharmacophoreExtractor(nn.Module):
    """
    从原子嵌入中提取 r 个药效团级表示。
    使用可学习的查询向量对原子做交叉注意力 (Cross-Attention)，
    类似 Slot Attention，但每个 slot 对应一种"作用模式"。

    输出:
      pharma_emb: [B, r, d]  — r 个药效团嵌入
      attn_map:   [B, r, N_max] — 每个药效团对每个原子的注意力权重 (可解释性)
    """
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h, batch_idx, num_graphs):
        """
        atom_h:    [N_total, d]
        batch_idx: [N_total] — 每个原子属于哪个图
        num_graphs: int — batch 中图的数量
        """
        d = atom_h.shape[1]
        K = self.key_proj(atom_h)    # [N_total, d]，统一计算，不在循环内重复
        V = self.val_proj(atom_h)    # [N_total, d]

        # 所有 slot 的 scores 一次矩阵乘得到：[N_total, r]
        scores_all = (K @ self.queries.T) / math.sqrt(d)

        pharma = torch.zeros(num_graphs, self.num_slots, d,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)    # [N_total]
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma)   # [B, r, d]


class PerturbationOperator(nn.Module):
    """
    从药效团嵌入生成低秩扰动算子，并作用于基因状态向量。

    数学形式:
      T = I + Σ_k σ_k · u_k ⊗ v_k^T    (低秩扰动)
      Δh = T(h_g) - h_g = Σ_k σ_k · (v_k^T h_g) · u_k

    返回:
      delta_h:  [B, d]  — 总扰动向量
      spectrum: [B, r]  — 交互谱 (每个模式的激活强度)
      sigma:    [B, r]  — 各模式内在幅度
      U:        [B, r, d] — 扰动方向矩阵 (用于正交正则化)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))   # 扰动方向
        self.to_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))   # 耦合方向
        self.to_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh())                            # 幅度，限制在 [-1, 1]

    def forward(self, pharma_emb, h_g):
        """
        pharma_emb: [B, r, d]
        h_g:        [B, d]
        """
        U = F.normalize(self.to_u(pharma_emb), dim=-1)    # [B, r, d]
        V = F.normalize(self.to_v(pharma_emb), dim=-1)    # [B, r, d]
        sigma = self.to_sigma(pharma_emb).squeeze(-1)      # [B, r]

        # 耦合: 基因状态在每个模式的耦合方向上的投影
        coupling = (V * h_g.unsqueeze(1)).sum(-1)          # [B, r]

        # 交互谱: 幅度 × 耦合
        spectrum = sigma * coupling                         # [B, r]

        # 总扰动: 按谱加权叠加各模式的扰动方向
        delta_h = (spectrum.unsqueeze(-1) * U).sum(dim=1)  # [B, d]

        return delta_h, spectrum, sigma, U

# ================================================================
# 4. 完整模型 (含消融基线)
# ================================================================

class DrugOperatorNet(nn.Module):
    """
    统一架构: 通过 --interaction_type 切换不同的交互建模方式，
    共享完全相同的编码器骨架，确保消融对比公平。

    Modes:
      'operator'     — 药物算子 (本文提出)
      'concat'       — h_c ⊕ h_g → MLP
      'ortho_concat' — V_c⊥ ⊕ V_g → MLP (现有方法)
      'hadamard'     — h_c ⊙ h_g + h_g → MLP
    """
    def __init__(self, hidden_dim=128, dropout=0.3,
                 operator_rank=8, interaction_type='operator'):
        super().__init__()
        self.interaction_type = interaction_type
        self.hidden_dim = hidden_dim

        # ---- 共享编码器 ----
        self.gene_enc = GeneEncoderV1(hidden_dim=hidden_dim, dropout=dropout)
        self.atom_enc = AtomEncoder(hidden_dim=hidden_dim, dropout=dropout)

        # ---- 交互模块 (按类型构建) ----
        if interaction_type == 'operator':
            self.pharma_ext = PharmacophoreExtractor(hidden_dim, operator_rank)
            self.perturb_op = PerturbationOperator(hidden_dim)
            clf_in = hidden_dim * 2   # [h_g, Δh]
        else:
            # concat / ortho_concat / hadamard 都需要全局池化
            self.drug_readout = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim))
            clf_in = hidden_dim * 2   # [h_g, h_c / V_c⊥ / h_c⊙h_g]

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _global_pool(self, atom_h, batch_idx, num_graphs):
        """Sum + Mean 全局池化 → 单一药物向量"""
        d = atom_h.shape[1]
        num_nodes_t = torch.zeros(num_graphs, device=atom_h.device)
        num_nodes_t.index_add_(0, batch_idx,
                               torch.ones(atom_h.shape[0], device=atom_h.device))
        sum_pool = torch.zeros(num_graphs, d, device=atom_h.device,
                               dtype=atom_h.dtype).index_add_(0, batch_idx, atom_h)
        mean_pool = sum_pool / num_nodes_t.unsqueeze(1).clamp(min=1)
        return self.drug_readout(torch.cat([sum_pool, mean_pool], dim=-1))  # [B, d]

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        device = x.device
        B = len(num_nodes_list)
        num_nodes_tensor = torch.tensor(num_nodes_list, device=device)
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), num_nodes_tensor)

        # ==== 编码 ====
        h_g = self.gene_enc(gene_ids)                           # [B, d]
        atom_h = self.atom_enc(x, edge_index, edge_attr)        # [N_total, d]

        # ==== 交互 ====
        # 初始化可选返回值
        delta_h = spectrum = sigma = U = None

        if self.interaction_type == 'operator':
            pharma = self.pharma_ext(atom_h, batch_idx, B)      # [B, r, d]
            delta_h, spectrum, sigma, U = self.perturb_op(pharma, h_g)
            features = torch.cat([h_g, delta_h], dim=-1)        # [B, 2d]

        elif self.interaction_type == 'concat':
            h_c = self._global_pool(atom_h, batch_idx, B)
            features = torch.cat([h_g, h_c], dim=-1)

        elif self.interaction_type == 'ortho_concat':
            h_c = self._global_pool(atom_h, batch_idx, B)
            V_g = F.normalize(h_g, dim=-1)
            V_c = F.normalize(h_c, dim=-1)
            V_c_perp = V_c - (V_c * V_g).sum(-1, keepdim=True) * V_g
            features = torch.cat([V_g, V_c_perp], dim=-1)

        elif self.interaction_type == 'hadamard':
            h_c = self._global_pool(atom_h, batch_idx, B)
            h_inter = h_c * h_g   # 元素积
            features = torch.cat([h_g, h_inter], dim=-1)

        logits = self.classifier(features).squeeze(-1)          # [B]

        return logits, h_g, delta_h, spectrum, sigma, U

# ================================================================
# 5. 正则化损失
# ================================================================

def compute_operator_regularization(sigma, U, lam_sparse, lam_ortho):
    """
    sigma: [B, r]  — 各模式幅度
    U:     [B, r, d] — 扰动方向矩阵

    L_sparse: L1 on |σ|，鼓励只有少数模式主导
    L_ortho:  ||U^T U - I||_F，鼓励各模式方向正交 (独立)
    """
    loss_sparse = sigma.abs().mean()

    # Gram 矩阵: [B, r, r]
    U_normed = F.normalize(U, dim=-1)
    gram = torch.bmm(U_normed, U_normed.transpose(1, 2))
    eye = torch.eye(U.shape[1], device=U.device).unsqueeze(0)
    loss_ortho = (gram - eye).pow(2).mean()

    return lam_sparse * loss_sparse + lam_ortho * loss_ortho

# ================================================================
# 6. 训练主循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- 数据 ----
    train_ds = OptimizedGraphDataset(args.data_dir, args.fold, 'train')
    val_ds   = OptimizedGraphDataset(args.data_dir, args.fold, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=optimized_collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=optimized_collate_fn, num_workers=4)

    # ---- 模型 ----
    model = DrugOperatorNet(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        operator_rank=args.operator_rank,
        interaction_type=args.interaction_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 模型参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if args.use_amp else None

    # ---- 保存路径 ----
    save_dir = Path(f'results_operator/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"{args.interaction_type}_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0

    print(f"\n{'='*70}")
    print(f"🚀 DrugOperatorNet 训练")
    print(f"   交互类型: {args.interaction_type} | 算子秩: {args.operator_rank}")
    print(f"   设备: {args.device} | AMP: {args.use_amp} | Fold: {args.fold}")
    print(f"   稀疏正则: {args.lam_sparse} | 正交正则: {args.lam_ortho_modes}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            x          = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr  = batch['edge_attr'].to(device)
            gene_ids   = batch['gene_ids'].to(device)
            labels     = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, h_g, delta_h, spectrum, sigma, U = model(
                    gene_ids, x, edge_index, edge_attr, batch['num_nodes_list'])

                loss_bce = criterion(logits, labels)

                # 算子正则化 (仅 operator 模式)
                loss_reg = torch.tensor(0.0, device=device)
                if args.interaction_type == 'operator' and sigma is not None:
                    loss_reg = compute_operator_regularization(
                        sigma, U, args.lam_sparse, args.lam_ortho_modes)

                loss = loss_bce + loss_reg

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

        # ---- 验证 ----
        model.eval()
        all_preds, all_labels = [], []
        all_spectra = []   # 用于可解释性保存

        with torch.no_grad():
            for batch in val_loader:
                x          = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr  = batch['edge_attr'].to(device)
                gene_ids   = batch['gene_ids'].to(device)

                with autocast(enabled=args.use_amp):
                    logits, _, _, spectrum, _, _ = model(
                        gene_ids, x, edge_index, edge_attr,
                        batch['num_nodes_list'])

                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                if spectrum is not None and args.save_spectrum:
                    all_spectra.append(spectrum.cpu())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        preds_bin = (np.array(all_preds) > 0.5).astype(int)
        f1 = f1_score(all_labels, preds_bin)
        scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} | L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f}) | "
              f"VAL_AUC: {auroc:.4f} | PRC: {auprc:.4f} | F1: {f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
            # 保存交互谱
            if args.save_spectrum and all_spectra:
                spectra_tensor = torch.cat(all_spectra, dim=0)
                torch.save({
                    'spectrum': spectra_tensor,
                    'preds': np.array(all_preds),
                    'labels': np.array(all_labels),
                }, save_dir / f'spectrum_{model_name}')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"⏹ Early stop at epoch {epoch+1}")
                break

    print(f"\n✅ 最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")


# ================================================================
# 7. 入口
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DrugOperatorNet: Drug-as-Perturbation-Operator for CGI')

    # 数据与设备
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--fold',        type=int, default=0)
    parser.add_argument('--seed',        type=int, default=42)

    # 训练超参
    parser.add_argument('--epochs',      type=int, default=80)
    parser.add_argument('--batch_size',  type=int, default=512)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--patience',    type=int, default=10)
    parser.add_argument('--use_amp',     action='store_true')

    # 🔬 核心消融参数
    parser.add_argument('--interaction_type', type=str, default='operator',
                        choices=['operator', 'concat', 'ortho_concat', 'hadamard'],
                        help='交互建模方式')
    parser.add_argument('--operator_rank', type=int, default=8,
                        help='扰动算子的秩 (药效团/作用模式数)')

    # 算子正则化
    parser.add_argument('--lam_sparse',      type=float, default=0.01,
                        help='σ 的 L1 稀疏正则权重')
    parser.add_argument('--lam_ortho_modes', type=float, default=0.01,
                        help='U 列正交正则权重')

    # 可解释性输出
    parser.add_argument('--save_spectrum', action='store_true',
                        help='保存最优 epoch 的交互谱用于下游分析')
    parser.add_argument('--run_tag', type=str, default='')

    train(parser.parse_args())
