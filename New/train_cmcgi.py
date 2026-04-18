#!/usr/bin/env python3
"""
CMCGI: Cross-Modal Conditional Chemical-Gene Interaction
=========================================================

与 train_cross_modal.py 的核心区别（三处修复 + 两处创新）：

【修复1】单次 CNN 前向（原版双次计算 gene CNN，浪费 2×）
  原版: gene_reader(gene_ids, None) → gene_rough → gene_reader(gene_ids, drug_rough)
  新版: get_seq_features(gene_ids) → seq_feat [B, L', H]（只算一次）
        → gene_rough = seq_feat.mean(1)
        → attn_pool(seq_feat, drug_query)（复用 seq_feat）

【修复2】base_queries 改为 kaiming 初始化（原版零初始化）
  零初始化 + drug_cond_bias 在 warmup 前几步也为零
  → 注意力全均匀 → 梯度极小 → query 长时间不更新
  新版: kaiming_normal_(base_queries, nonlinearity='relu')
        即使 drug_cond_bias 还未收敛，base_queries 本身已提供方向信号

【修复3】可学习温度参数 log_tau（原版固定 1/sqrt(H)）
  用 log_tau（per head）控制注意力锐度：
    attn = softmax(scores / exp(log_tau))
  初始化为 log(sqrt(H))，训练中模型可通过减小 tau 锐化注意力
  这是让 gene_attn 从"弥散"变"尖锐"的关键机制

【创新1】pharma_emb slot 正交正则（原版只对 U 向量）
  原版的 loss_ortho 作用在 U [B, r, H]（输出方向），
  不约束 pharma slot 本身的多样性 → sigma2/sigma3 仍可高度相关
  新版在 pharma_emb 的 slot 均值上加 Gram 矩阵正则：
    S = pharma_emb.mean(0) [r, H]
    ortho_loss += (S_n @ S_n.T - I).pow(2).mean()
  这直接约束每个药效团探测器学到不同的化学特征

【创新2】Top-k sparse attention（可选）
  gene_attn 默认 softmax（软，所有位置有权重）
  开启 --gene_topk K 后，每个 (batch, head) 只保留 topK 个位置：
    scores 其余位置填 -inf → softmax 后其余为 0
  生成稀疏注意力热图，方便可解释性分析

完整前向流程：
  ① atom_h  = GIN(drug)                               [N, H]
     seq_feat = Embed + CNN(gene) (single pass)        [B, L', H]
     drug_rough = mean_pool(atom_h)                    [B, H]
     gene_rough = seq_feat.mean(1)                     [B, H]

  ② pharma_emb = SlotAttn(atom_h,  query=base+f(gene_rough))  [B, r, H]
     h_g_modes  = SeqAttn (seq_feat, query=base+f(drug_rough)) [B, r, H]

  ③ pharma', h_g' = BiCrossAttn(pharma_emb, h_g_modes)

  ④ delta_h, spectrum, sigma = BilateralOp(pharma', h_g')

  ⑤ logit = MLP(cat[h_g'.mean(1), delta_h])

Loss = BCE + λ_ortho*(U-ortho + slot-ortho) + λ_sparse*|sigma| + λ_cl*CL
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_operator_moe import (
    set_seed, OptimizedGraphDataset, collate_fn,
    AtomEncoder, SpectrumDirectionCL,
    scatter_softmax, scatter_add,
)


# ================================================================
# 1. 基因编码器（单次 CNN + 药物条件化 attention pooling）
# ================================================================

class EfficientGeneEncoder(nn.Module):
    """
    两阶段设计（解决 train_cross_modal.py 的双次 CNN 浪费）：
      Stage 1: get_seq_features(gene_ids) → seq_feat [B, L', H]
               纯序列特征提取，只算一次
      Stage 2: pool(seq_feat, drug_query) → h_g_modes [B, r, H], gene_attn [B, r, L']
               使用药物条件化的 query 做 attention pooling

    核心改动（vs GeneMultiHeadReader）：
      - base_queries: kaiming_normal 初始化（非零）
      - query += MLP(drug_rough)：药物条件化偏置
      - log_tau [r]: 可学习每头温度，控制注意力锐度
      - 可选 top-k sparse attention
    """
    def __init__(self, vocab_size=4097, hidden_dim=128, num_heads=8,
                 dropout=0.3, gene_topk=0):
        super().__init__()
        H, r = hidden_dim, num_heads
        self.num_heads = r
        self.gene_topk = gene_topk

        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes   = [6, 8, 10, 12]
        self.convs     = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.seq_norm = nn.LayerNorm(H)

        # base_queries: kaiming 初始化，为每个头提供不同的初始方向
        self.base_queries = nn.Parameter(torch.empty(r, H))
        nn.init.kaiming_normal_(self.base_queries, nonlinearity='relu')

        # 药物条件化投影：drug_rough [B, H] → r 个偏置向量 [B, r, H]
        self.drug_cond = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, r * H),
        )

        # 可学习温度（log scale，初始化为 log(sqrt(H))）
        # tau = exp(log_tau)；tau 越小，注意力越尖锐
        self.log_tau = nn.Parameter(torch.full((r,), math.log(math.sqrt(H))))

        self.out_norm = nn.LayerNorm(H)
        self.dropout  = dropout

    def get_seq_features(self, gene_ids):
        """Stage 1：纯 CNN 特征提取，不做 attention pooling。"""
        x = self.embedding(gene_ids).transpose(1, 2)            # [B, H, L]
        x = torch.cat([conv(x) for conv in self.convs], dim=1)  # [B, H, L']
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                                   # [B, L', H]
        return self.seq_norm(x)                                  # [B, L', H]

    def pool(self, seq_feat, drug_rough=None):
        """
        Stage 2：用药物条件化的 query 对序列特征做 attention pooling。

        Args:
          seq_feat:   [B, L', H]  来自 get_seq_features()
          drug_rough: [B, H]      药物粗粒度摘要（None 时为无条件版本）

        Returns:
          h_g_modes: [B, r, H]
          gene_attn: [B, r, L']
        """
        B = seq_feat.size(0)

        # 构造 per-sample query
        queries = self.base_queries.unsqueeze(0).expand(B, -1, -1)  # [B, r, H]
        if drug_rough is not None:
            bias    = self.drug_cond(drug_rough).view(B, self.num_heads, -1)
            queries = queries + bias                                  # [B, r, H]

        # Attention scores，per-head 可学习温度缩放
        tau = self.log_tau.exp().clamp(min=0.1)                      # [r] 防止过小
        scores = torch.einsum('blh,brh->brl', seq_feat, queries)     # [B, r, L']
        scores = scores / tau.view(1, -1, 1)

        # 可选 top-k sparse attention
        if self.gene_topk > 0:
            k = min(self.gene_topk, scores.size(-1))
            topk_vals, topk_idx = scores.topk(k, dim=-1)
            sparse = torch.full_like(scores, float('-inf'))
            sparse.scatter_(-1, topk_idx, topk_vals)
            gene_attn = F.softmax(sparse, dim=-1)
        else:
            gene_attn = F.softmax(scores, dim=-1)                    # [B, r, L']

        h_g_modes = torch.einsum('brl,blh->brh', gene_attn, seq_feat)  # [B, r, H]
        return self.out_norm(h_g_modes), gene_attn


# ================================================================
# 2. 基因条件化药效团提取器
# ================================================================

class GeneCondPharmExt(nn.Module):
    """
    药效团 slot attention，query 被基因粗粒度摘要条件化。

    核心：base_queries[s] + gene_cond_proj(gene_rough)[s]
    → 不同基因对同一药物提取不同药效团组合
    → atom_alpha[原子] 对不同基因是不同的

    与原版 PharmacophoreExtractor 区别：
      - query 从固定参数变为 gene-conditioned
      - 同样暴露 atom_scores [N, r] 用于可解释性
    """
    def __init__(self, hidden_dim: int, num_slots: int):
        super().__init__()
        H, r = hidden_dim, num_slots
        self.num_slots = r
        self.base_queries = nn.Parameter(torch.randn(r, H) * 0.02)
        self.key_proj     = nn.Linear(H, H)
        self.val_proj     = nn.Linear(H, H)
        self.norm         = nn.LayerNorm(H)
        self.gene_cond    = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, r * H),
        )

    def forward(self, atom_h, batch_idx, num_graphs, gene_rough=None):
        B, H = num_graphs, atom_h.size(1)
        K = self.key_proj(atom_h)    # [N, H]
        V = self.val_proj(atom_h)    # [N, H]

        queries = self.base_queries.unsqueeze(0).expand(B, -1, -1)  # [B, r, H]
        if gene_rough is not None:
            bias    = self.gene_cond(gene_rough).view(B, self.num_slots, H)
            queries = queries + bias

        # per-atom query（每个原子获取其所在分子的 query）
        q_atom     = queries[batch_idx]                                   # [N, r, H]
        scores_all = (K.unsqueeze(1) * q_atom).sum(-1) / math.sqrt(H)   # [N, r]

        pharma = torch.zeros(B, self.num_slots, H,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all  # [B, r, H], [N, r]


# ================================================================
# 3. 双向 Cross-Attention 精炼
# ================================================================

class BidirectionalCrossAttn(nn.Module):
    """
    pharma_emb [B, r, H] 和 h_g_modes [B, r, H] 互相 attend（r×r attention）。

    这是计算量最小的精炼：r=8，attention 矩阵只有 8×8。
    但语义最强：药效团 token 可以直接"看到"哪些基因模式与自己相关，反之亦然。
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        H = hidden_dim
        self.d2g = nn.MultiheadAttention(H, num_heads, dropout=0.1, batch_first=True)
        self.g2d = nn.MultiheadAttention(H, num_heads, dropout=0.1, batch_first=True)
        self.norm_d  = nn.LayerNorm(H)
        self.norm_g  = nn.LayerNorm(H)
        self.ffn_d   = nn.Sequential(nn.Linear(H, H * 2), nn.GELU(), nn.Linear(H * 2, H))
        self.ffn_g   = nn.Sequential(nn.Linear(H, H * 2), nn.GELU(), nn.Linear(H * 2, H))
        self.norm_d2 = nn.LayerNorm(H)
        self.norm_g2 = nn.LayerNorm(H)

    def forward(self, pharma, h_g):
        p_a, _ = self.d2g(pharma, h_g, h_g)
        p_out  = self.norm_d2(self.norm_d(pharma + p_a) +
                              self.ffn_d(self.norm_d(pharma + p_a)))

        g_a, _ = self.g2d(h_g, pharma, pharma)
        g_out  = self.norm_g2(self.norm_g(h_g + g_a) +
                              self.ffn_g(self.norm_g(h_g + g_a)))

        return p_out, g_out


# ================================================================
# 4. 双侧扰动算子
# ================================================================

class BilateralSigmaOp(nn.Module):
    """
    sigma 由 pharma 和 h_g 双侧点积计算（非单侧 tanh(MLP(pharma))）：
      sigma_j = tanh( q_j(pharma_j) · k_j(h_g_j) / sqrt(d) )
    这使 sigma 真正反映该（药物, 基因）对在模式 j 上的互相激活强度。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        H, d = hidden_dim, hidden_dim // 4
        self.to_u    = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.to_v    = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.sigma_q = nn.Linear(H, d)
        self.sigma_k = nn.Linear(H, d)

    def forward(self, pharma, h_g):
        U = F.normalize(self.to_u(pharma), dim=-1)        # [B, r, H]
        V = F.normalize(self.to_v(pharma), dim=-1)        # [B, r, H]
        q = self.sigma_q(pharma)                           # [B, r, d]
        k = self.sigma_k(h_g)                             # [B, r, d]
        sigma    = torch.tanh((q * k).sum(-1) / math.sqrt(q.size(-1)))  # [B, r]
        coupling = (V * h_g).sum(-1)                       # [B, r]
        spectrum = sigma * coupling                        # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(1)    # [B, H]
        return delta_h, spectrum, sigma, U


# ================================================================
# 5. 完整 CMCGI 模型
# ================================================================

class CMCGI(nn.Module):
    """
    Cross-Modal Conditional CGI 模型。
    前向流程见文件顶部 docstring。
    """
    def __init__(self, hidden_dim=128, operator_rank=8, dropout=0.3, gene_topk=0):
        super().__init__()
        H, r = hidden_dim, operator_rank

        self.atom_enc    = AtomEncoder(H, dropout)
        self.gene_enc    = EfficientGeneEncoder(4097, H, r, dropout, gene_topk)
        self.pharma_ext  = GeneCondPharmExt(H, r)
        self.cross_attn  = BidirectionalCrossAttn(H, num_heads=4)
        self.perturb_op  = BilateralSigmaOp(H)
        self.classifier  = nn.Sequential(
            nn.Linear(H * 2, H), nn.LayerNorm(H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(H, 1),
        )

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        B = gene_ids.size(0)
        device = x.device
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device),
            torch.tensor(num_nodes_list, device=device)
        )

        # ① 粗粒度摘要（各自独立提取，计算量最小）
        atom_h   = self.atom_enc(x, edge_index, edge_attr)          # [N, H]
        seq_feat = self.gene_enc.get_seq_features(gene_ids)         # [B, L', H]（单次）

        cnt        = torch.tensor(num_nodes_list, device=device, dtype=torch.float32)
        drug_rough = scatter_add(atom_h, batch_idx, B) / cnt.unsqueeze(1)  # [B, H]
        gene_rough = seq_feat.mean(1)                                       # [B, H]

        # ② 条件化 attention（双向互相感知）
        pharma_emb, atom_scores = self.pharma_ext(atom_h, batch_idx, B, gene_rough)
        h_g_modes,  gene_attn  = self.gene_enc.pool(seq_feat, drug_rough)

        # ③ Cross-attention 精炼（r×r，计算量极小）
        pharma_emb, h_g_modes = self.cross_attn(pharma_emb, h_g_modes)

        # ④ 双侧算子
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        # ⑤ 分类
        h_g_global = h_g_modes.mean(1)                              # [B, H]
        logits     = self.classifier(torch.cat([h_g_global, delta_h], dim=-1))

        return logits.squeeze(-1), spectrum, sigma, U, gene_attn, atom_scores


# ================================================================
# 正交正则辅助函数
# ================================================================

def ortho_loss_fn(matrix, device):
    """
    Gram 矩阵正则：鼓励 matrix [r, H] 的行向量两两正交。
    loss = mean((S_n @ S_n.T - I)^2)
    """
    r = matrix.size(0)
    S = F.normalize(matrix, dim=-1)
    G = S @ S.T
    I = torch.eye(r, device=device)
    return (G - I).pow(2).mean()


# ================================================================
# 训练主函数
# ================================================================

def train(args):
    set_seed(args.seed)
    device   = torch.device(args.device)
    data_dir = Path(args.data_dir)
    save_dir = Path('results_new_models') / data_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    tag        = f"_{args.run_tag}" if args.run_tag else ""
    model_name = (f"cmcgi_r{args.operator_rank}_k{args.gene_topk}"
                  f"_Fold{args.fold}{tag}.pt")

    train_ds = OptimizedGraphDataset(data_dir, args.fold, 'train', args.gene_max_len)
    val_ds   = OptimizedGraphDataset(data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = CMCGI(
        hidden_dim    = args.hidden_dim,
        operator_rank = args.operator_rank,
        dropout       = args.dropout,
        gene_topk     = args.gene_topk,
    ).to(device)

    cl_module = None
    if args.lam_cl > 0:
        cl_module = SpectrumDirectionCL(rank=args.operator_rank, margin=0.5).to(device)

    params = list(model.parameters())
    if cl_module:
        params += list(cl_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CMCGI] 参数量: {n_params:,}  rank={args.operator_rank}"
          f"  topk={args.gene_topk}  fold={args.fold}")

    best_auc, patience_cnt = 0.0, 0
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step  = 0
    base_lr      = args.lr

    for epoch in range(1, args.epochs + 1):
        model.train()
        if cl_module: cl_module.train()
        ep_loss = ep_bce = ep_ortho = 0.0

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                lr = base_lr * global_step / warmup_steps
                for pg in optimizer.param_groups: pg['lr'] = lr

            gene_ids = batch['gene_ids'].to(device)
            labels   = batch['label'].to(device)
            x  = batch['x'].to(device)
            ei = batch['edge_index'].to(device)
            ea = batch['edge_attr'].to(device)
            nnl = batch['num_nodes_list']

            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, gene_attn, atom_scores = \
                    model(gene_ids, x, ei, ea, nnl)

                loss_bce = criterion(logits, labels)

                # 正交正则 ①：U 向量（输出方向）按 batch 平均
                U_mean  = U.mean(0)                     # [r, H]
                lo_U    = ortho_loss_fn(U_mean, device)

                # 正交正则 ②：pharma slot（药效团探测器 base_queries）
                lo_slot = ortho_loss_fn(model.pharma_ext.base_queries, device)

                # 正交正则 ③：gene base_queries（基因阅读头）
                lo_gene = ortho_loss_fn(model.gene_enc.base_queries, device)

                loss_ortho = lo_U + lo_slot + lo_gene

                # sigma 稀疏正则（鼓励模式激活具有选择性）
                loss_sparse = sigma.abs().mean()

                # 对比损失（可选）
                loss_cl = (cl_module(spectrum, labels)
                           if cl_module else torch.tensor(0., device=device))

                loss = (loss_bce
                        + args.lam_ortho * loss_ortho
                        + args.lam_sparse * loss_sparse
                        + args.lam_cl * loss_cl)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            ep_loss  += loss.item()
            ep_bce   += loss_bce.item()
            ep_ortho += loss_ortho.item()

        # ── 验证 ───────────────────────────────────────────────────
        model.eval()
        if cl_module: cl_module.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                gene_ids = batch['gene_ids'].to(device)
                x  = batch['x'].to(device)
                ei = batch['edge_index'].to(device)
                ea = batch['edge_attr'].to(device)
                nnl = batch['num_nodes_list']
                with autocast(enabled=args.use_amp):
                    logits, _, _, _, _, _ = model(gene_ids, x, ei, ea, nnl)
                all_p.append(torch.sigmoid(logits).cpu())
                all_l.append(batch['label'])

        probs   = torch.cat(all_p).numpy()
        lbls    = torch.cat(all_l).numpy()
        val_auc = roc_auc_score(lbls, probs)
        val_prc = average_precision_score(lbls, probs)
        val_f1  = f1_score(lbls, probs > 0.5)

        scheduler.step(val_auc)
        lr_cur = optimizer.param_groups[0]['lr']
        n = len(train_loader)
        print(f"Ep {epoch:3d} [lr={lr_cur:.2e}]"
              f" | loss={ep_loss/n:.4f} bce={ep_bce/n:.4f} ortho={ep_ortho/n:.4f}"
              f" | AUC={val_auc:.4f} PRC={val_prc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save({'model': model.state_dict(), 'args': vars(args)},
                       save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop @ ep{epoch}，最优 AUC={best_auc:.4f}")
                break

    print(f"最优 AUC={best_auc:.4f}  →  {save_dir / model_name}")
    return best_auc


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CMCGI: Cross-Modal Conditional CGI')
    p.add_argument('--data_dir',      required=True,
                   help='数据目录，如 /path/to/datasets/MCF7')
    p.add_argument('--device',        default='cuda:0')
    p.add_argument('--fold',          type=int,   default=0)
    p.add_argument('--epochs',        type=int,   default=80)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--hidden_dim',    type=int,   default=128)
    p.add_argument('--dropout',       type=float, default=0.3)
    p.add_argument('--operator_rank', type=int,   default=8)
    p.add_argument('--gene_max_len',  type=int,   default=1000)
    p.add_argument('--gene_topk',     type=int,   default=0,
                   help='gene attention top-k 稀疏化，0=关闭（用全 softmax）')
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--lam_ortho',     type=float, default=0.1,
                   help='正交正则系数（U + slot + gene_query）')
    p.add_argument('--lam_sparse',    type=float, default=0.01,
                   help='sigma 稀疏正则')
    p.add_argument('--lam_cl',        type=float, default=0.0,
                   help='谱方向对比损失（0=关闭）')
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--use_amp',       action='store_true')
    p.add_argument('--run_tag',       default='')
    args = p.parse_args()
    train(args)
