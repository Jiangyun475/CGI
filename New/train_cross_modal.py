#!/usr/bin/env python3
"""
Cross-Modal Conditional Interaction（跨模态互相条件化）
=======================================================

【核心问题】
  当前 DrugOperatorNet 的两个编码器完全独立：
    PharmacophoreExtractor 使用固定 r 个 query slot，
    不管面对哪个基因，提取的 pharma_emb 完全相同。
    GeneMultiHeadReader 使用固定 r 个 attn_query，
    不管面对哪个药物，提取的 h_g_modes 完全相同。
  两者唯一交汇：coupling = (V * h_g_modes).sum(-1) ← 一次末尾点积。
  这是完全的晚期融合（Late Fusion）。

【生物学动机：诱导契合（Induced Fit）】
  药物-靶标互作不是"药物盖印章"——而是双向适应过程：
  1. 药物的哪个官能团"重要"，取决于目标基因的调控区域特征
  2. 基因序列的哪段位置"重要"，取决于药物的化学机制
  这在结构生物学中叫"诱导契合"：结合时配体和受体的构象都会调整。

【设计：四步互相条件化】

  Step 1 | 粗粒度摘要（初步信息交换前的概览）
    drug_rough = scatter_mean(atom_h, batch_idx)    [B, H]
    gene_rough = CNN(gene_ids).mean(dim=1)           [B, H]

  Step 2 | 条件化 Query（让每个编码器"知道对方是谁"）
    pharma_queries[b,s] = base_drug_queries[s] + MLP_d(gene_rough[b])  ← 药效团探测器被基因定制
    gene_queries[b,s]   = base_gene_queries[s]  + MLP_g(drug_rough[b]) ← 基因阅读头被药物定制

  Step 3 | 条件化 Slot Attention（用条件化的 query 重新提取）
    pharma_emb = CondSlotAttn(atom_h, pharma_queries)  [B, r, H]  药物表示已包含基因上下文
    h_g_modes  = CondSeqAttn(seq_feat, gene_queries)   [B, r, H]  基因表示已包含药物上下文

  Step 4 | 双向 Cross-Attention 精炼（让两个已知对方的表示再深度交互）
    pharma' = pharma_emb + CrossAttn(Q=pharma_emb, KV=h_g_modes)  药效团进一步看基因
    h_g'    = h_g_modes  + CrossAttn(Q=h_g_modes,  KV=pharma_emb) 基因进一步看药效团

  + 双侧 Sigma（因为输入已经互相感知，自然结合双侧 sigma）
    sigma = Tanh(dot(q(pharma'), k(h_g')) / √d)    完全双侧激活强度

【为什么对 Chemical Cold Split 有帮助】
  cold split 下新药物的 pharma_emb 在分布外（未见过的分子）。
  如果 pharma_emb 被基因条件化，新药物的表示被"锚定"到基因上下文：
  即使药物是全新的，基因（训练见过）充当约束，
  迫使药物表示落在生物学合理的方向上，减少 OOD 效应。

【参数增量估算（相对主模型 918K）】
  Step 2 条件化 MLP：2 × (H→H) = 2 × 128² ≈ 33K
  Step 4 Cross-Attention（4 heads，H=128）：2 × 4×(3×H²/4) ≈ 49K
  双侧 sigma：2 × (H→H/4) ≈ 8K
  总计约 +90K（+9.8%），在可接受范围内。
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
    GINLayer, AtomEncoder, SpectrumDirectionCL,
)
from torch_geometric.utils import scatter_softmax


# ================================================================
# 1. 条件化基因阅读器（gene query 被药物条件化）
# ================================================================

class CondGeneReader(nn.Module):
    """
    GeneMultiHeadReader 的条件化版本。

    改动：r 个 attn_query 不再固定，而是在全局固定 query 上叠加"药物条件偏置"：
      query_s[b] = base_query_s + drug_cond_proj(drug_rough[b])_s

    这样，"读取基因的哪些位置" 会随药物上下文动态调整。

    同时暴露 seq_feat（CNN 特征，attention 前），供 CrossModalRefinement 使用。
    """
    def __init__(self, vocab_size=4097, hidden_dim=128, num_heads=8, dropout=0.3):
        super().__init__()
        H, r = hidden_dim, num_heads
        self.num_heads = r
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=0)
        kernel_sizes   = [6, 8, 10, 12]
        self.convs     = nn.ModuleList([
            nn.Conv1d(H, H // len(kernel_sizes), ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.seq_norm = nn.LayerNorm(H)

        # 基础 query（全局共享，零初始化如原版）
        self.base_queries = nn.Parameter(torch.zeros(r, H))
        # ── NEW: 药物条件化投影 ──────────────────────────────────
        # drug_rough [B, H] → r 个偏置向量 [B, r, H]
        self.drug_cond = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, r * H),
        )
        self.out_norm = nn.LayerNorm(H)
        self.dropout  = dropout

    def forward(self, gene_ids, drug_rough=None):
        """
        Args:
          gene_ids:   [B, L]
          drug_rough: [B, H]  药物粗粒度摘要，None 时退化为原版（固定 query）

        Returns:
          h_g_modes: [B, r, H]
          h_g_global:[B, H]
          seq_feat:  [B, L', H]  暴露给 CrossModalRefinement 使用
          attn:      [B, r, L']
        """
        B = gene_ids.size(0)
        x = self.embedding(gene_ids).transpose(1, 2)           # [B, H, L]
        x = torch.cat([conv(x) for conv in self.convs], dim=1) # [B, H, L']
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                                  # [B, L', H]
        seq_feat = self.seq_norm(x)                             # 暴露此处

        # 构造 query：固定基础 + 药物条件偏置
        queries = self.base_queries.unsqueeze(0).expand(B, -1, -1)  # [B, r, H]
        if drug_rough is not None:
            cond_bias = self.drug_cond(drug_rough).view(B, self.num_heads, -1)  # [B, r, H]
            queries   = queries + cond_bias

        # Attention（per-sample queries → einsum 方式）
        scores = torch.einsum('blh,brh->brl', seq_feat, queries) / math.sqrt(seq_feat.size(-1))
        attn   = F.softmax(scores, dim=-1)                      # [B, r, L']
        h_g_modes = torch.einsum('brl,blh->brh', attn, seq_feat)  # [B, r, H]
        h_g_modes = self.out_norm(h_g_modes)
        h_g_global = h_g_modes.mean(1)                          # [B, H]
        return h_g_modes, h_g_global, seq_feat, attn


# ================================================================
# 2. 条件化药效团提取器（drug query 被基因条件化）
# ================================================================

class CondPharmacophoreExtractor(nn.Module):
    """
    PharmacophoreExtractor 的条件化版本。

    改动：r 个 slot query 不再固定，而是：
      query_s[b] = base_query_s + gene_cond_proj(gene_rough[b])_s

    "探测哪类药效团"会随基因上下文动态调整。
    """
    def __init__(self, hidden_dim: int, num_slots: int):
        super().__init__()
        H, r = hidden_dim, num_slots
        self.num_slots = r
        self.base_queries = nn.Parameter(torch.randn(r, H) * 0.02)
        self.key_proj     = nn.Linear(H, H)
        self.val_proj     = nn.Linear(H, H)
        self.norm         = nn.LayerNorm(H)
        # ── NEW: 基因条件化投影 ──────────────────────────────────
        self.gene_cond = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, r * H),
        )

    def forward(self, atom_h, batch_idx, num_graphs, gene_rough=None):
        """
        Args:
          atom_h:     [N_total, H]
          batch_idx:  [N_total]
          num_graphs: int
          gene_rough: [B, H]  基因粗粒度摘要，None 时退化为原版
        """
        B, H = num_graphs, atom_h.size(1)
        K  = self.key_proj(atom_h)   # [N, H]
        V  = self.val_proj(atom_h)   # [N, H]

        # 构造 per-sample query：[B, r, H]
        queries = self.base_queries.unsqueeze(0).expand(B, -1, -1)
        if gene_rough is not None:
            cond_bias = self.gene_cond(gene_rough).view(B, self.num_slots, H)
            queries   = queries + cond_bias

        # per-atom query（batch_idx 索引展开）：[N, r, H]
        q_per_atom = queries[batch_idx]                           # [N, r, H]
        scores_all = (K.unsqueeze(1) * q_per_atom).sum(-1) / math.sqrt(H)  # [N, r]

        pharma = torch.zeros(B, self.num_slots, H,
                             device=atom_h.device, dtype=atom_h.dtype)
        for s in range(self.num_slots):
            alpha = scatter_softmax(scores_all[:, s], batch_idx)
            pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))

        return self.norm(pharma), scores_all   # [B, r, H], [N, r]


# ================================================================
# 3. 双向 Cross-Attention 精炼
# ================================================================

class CrossModalRefinement(nn.Module):
    """
    双向跨模态注意力精炼。

    基本原理：
      pharma_emb 和 h_g_modes 经过条件化提取后已经互相感知（粗粒度）。
      此模块让两者"再深度交互一轮"：
        pharma' = pharma + CrossAttn(Q=pharma, KV=h_g)   药效团 token 看基因 token
        h_g'    = h_g    + CrossAttn(Q=h_g,    KV=pharma) 基因 token 看药效团 token

      残差连接保证即使 attention 权重为零，信息不丢失。
      LayerNorm 保证训练稳定性。

    注：这是 r×r 的注意力（r=8），计算量极小。
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        H = hidden_dim
        self.drug2gene = nn.MultiheadAttention(H, num_heads, dropout=0.1, batch_first=True)
        self.gene2drug = nn.MultiheadAttention(H, num_heads, dropout=0.1, batch_first=True)
        self.norm_d    = nn.LayerNorm(H)
        self.norm_g    = nn.LayerNorm(H)
        # FFN（Transformer-style，增强非线性）
        self.ffn_d = nn.Sequential(nn.Linear(H, H * 2), nn.ReLU(), nn.Linear(H * 2, H))
        self.ffn_g = nn.Sequential(nn.Linear(H, H * 2), nn.ReLU(), nn.Linear(H * 2, H))
        self.norm_d2 = nn.LayerNorm(H)
        self.norm_g2 = nn.LayerNorm(H)

    def forward(self, pharma_emb, h_g_modes):
        """
        Args:
          pharma_emb: [B, r, H]
          h_g_modes:  [B, r, H]

        Returns:
          pharma_out: [B, r, H]  基因感知的药效团表示
          gene_out:   [B, r, H]  药物感知的基因表示
        """
        # 药效团 → 看基因（residual）
        p_attn, _ = self.drug2gene(pharma_emb, h_g_modes, h_g_modes)
        p_out     = self.norm_d(pharma_emb + p_attn)
        p_out     = self.norm_d2(p_out + self.ffn_d(p_out))

        # 基因 → 看药效团（residual）
        g_attn, _ = self.gene2drug(h_g_modes, pharma_emb, pharma_emb)
        g_out     = self.norm_g(h_g_modes + g_attn)
        g_out     = self.norm_g2(g_out + self.ffn_g(g_out))

        return p_out, g_out


# ================================================================
# 4. 双侧扰动算子（pharma 和 h_g 均已跨模态感知后使用）
# ================================================================

class BilateralPerturbationOperator(nn.Module):
    """
    在 CrossModalRefinement 后使用的算子。

    输入（pharma_emb', h_g_modes'）已经互相感知，因此：
    - U = to_u(pharma') : 输出方向，已含基因上下文
    - V = to_v(pharma') : 输入方向，已含基因上下文
    - coupling = (V * h_g')       : h_g' 已含药物上下文，天然双侧
    - sigma = dot(q(pharma'), k(h_g')) / √d : 显式双侧强度

    所有量自然成为双侧属性，无需额外改动 U/V 的来源。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        d = H // 4
        self.to_u   = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.to_v   = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.sigma_q = nn.Sequential(nn.Linear(H, d), nn.ReLU())
        self.sigma_k = nn.Sequential(nn.Linear(H, d), nn.ReLU())

    def forward(self, pharma_emb, h_g_modes):
        U = F.normalize(self.to_u(pharma_emb), dim=-1)    # [B, r, H]
        V = F.normalize(self.to_v(pharma_emb), dim=-1)    # [B, r, H]
        q = self.sigma_q(pharma_emb)                       # [B, r, d]
        k = self.sigma_k(h_g_modes)                        # [B, r, d]
        sigma    = torch.tanh((q * k).sum(-1) / math.sqrt(q.size(-1)))  # [B, r]
        coupling = (V * h_g_modes).sum(-1)                 # [B, r]
        spectrum = sigma * coupling                        # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(1)    # [B, H]
        return delta_h, spectrum, sigma, U


# ================================================================
# 5. 完整跨模态模型
# ================================================================

class CrossModalOperatorNet(nn.Module):
    """
    完整跨模态条件化算子网络。

    前向流程（4步互相条件化）：

    ① 粗粒度摘要（一次 CNN + GIN，不做 attention pooling）
       atom_h     = GIN(drug)               [N, H]
       seq_feat   = CNN(gene_ids)           [B, L', H]  before pooling
       drug_rough = mean(atom_h per mol)    [B, H]
       gene_rough = seq_feat.mean(1)        [B, H]

    ② 条件化 Slot Attention
       pharma_emb = CondPharmExt(atom_h, gene_rough)   [B, r, H]  药效团按基因调整
       h_g_modes  = CondGeneReader(seq_feat, drug_rough)[B, r, H]  基因按药物调整

    ③ Cross-Attention 精炼（双向 r×r）
       pharma', h_g' = CrossModalRefinement(pharma_emb, h_g_modes)

    ④ 双侧算子（U, V, sigma 全部在互相感知后的表示上计算）
       delta_h, spectrum = BilateralOp(pharma', h_g')

    ⑤ 分类
       h_g_global = h_g'.mean(1)
       logit = MLP(cat([h_g_global, delta_h]))
    """
    def __init__(self, hidden_dim=128, operator_rank=8, dropout=0.3):
        super().__init__()
        H, r = hidden_dim, operator_rank

        self.atom_encoder = AtomEncoder(H, dropout)
        self.pharma_ext   = CondPharmacophoreExtractor(H, r)
        self.gene_reader  = CondGeneReader(4097, H, r, dropout)
        self.cross_refine = CrossModalRefinement(H, num_heads=4)
        self.perturb_op   = BilateralPerturbationOperator(H)
        self.classifier   = nn.Sequential(
            nn.Linear(H * 2, H), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        B = gene_ids.size(0)
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=x.device),
            torch.tensor(num_nodes_list, device=x.device)
        )

        # ① 粗粒度摘要
        atom_h     = self.atom_encoder(x, edge_index, edge_attr)      # [N, H]
        drug_rough = torch.zeros(B, atom_h.size(1), device=x.device)
        drug_rough.index_add_(0, batch_idx, atom_h)
        counts     = torch.tensor(num_nodes_list, device=x.device, dtype=torch.float32)
        drug_rough = drug_rough / counts.unsqueeze(1)                  # [B, H] 均值

        # gene CNN (rough) — gene_reader 内部完成 CNN，暴露 seq_feat
        # 先做一次 gene_reader 获取 seq_feat（无条件，用于 gene_rough）
        _, _, seq_feat, _ = self.gene_reader(gene_ids, drug_rough=None)
        gene_rough = seq_feat.mean(1)                                  # [B, H]

        # ② 条件化 Slot Attention
        pharma_emb, _ = self.pharma_ext(atom_h, batch_idx, B, gene_rough)  # [B, r, H]
        h_g_modes, h_g_global_init, _, gene_attn = self.gene_reader(gene_ids, drug_rough)  # [B, r, H]

        # ③ Cross-Attention 精炼
        pharma_emb, h_g_modes = self.cross_refine(pharma_emb, h_g_modes)

        # ④ 双侧算子
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        # ⑤ 分类（使用精炼后的基因全局表示）
        h_g_global = h_g_modes.mean(1)                                # [B, H]
        logits     = self.classifier(torch.cat([h_g_global, delta_h], dim=-1))

        return logits.squeeze(-1), spectrum, sigma, U, gene_attn, None


# ================================================================
# 训练循环（与 train_bilateral_sigma.py 结构一致）
# ================================================================

def train(args):
    set_seed(args.seed)
    device   = torch.device(args.device)
    data_dir = Path(args.data_dir)
    save_dir = Path('results_new_models') / data_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    tag       = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"cross_modal_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    train_ds     = OptimizedGraphDataset(data_dir, args.fold, 'train', args.gene_max_len)
    val_ds       = OptimizedGraphDataset(data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = CrossModalOperatorNet(
        hidden_dim=args.hidden_dim,
        operator_rank=args.operator_rank,
        dropout=args.dropout,
    ).to(device)

    cl_module = None
    if args.lam_cl > 0:
        cl_module = SpectrumDirectionCL(rank=args.operator_rank, margin=0.5).to(device)

    params = list(model.parameters())
    if cl_module: params += list(cl_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [CrossModalOperatorNet] 参数量: {n_params:,}")

    best_auc, patience_cnt, base_lr = 0.0, 0, args.lr
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if cl_module: cl_module.train()
        ep_loss = 0.0

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                lr = base_lr * global_step / warmup_steps
                for pg in optimizer.param_groups: pg['lr'] = lr

            gene_ids = batch['gene_ids'].to(device)
            labels   = batch['labels'].to(device)
            x  = batch['x'].to(device)
            ei = batch['edge_index'].to(device)
            ea = batch['edge_attr'].to(device)
            nnl = batch['num_nodes_list']

            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _, _ = model(gene_ids, x, ei, ea, nnl)
                loss_bce   = criterion(logits, labels)
                loss_sp    = sigma.abs().mean()
                U_n        = F.normalize(U, dim=-1)
                gram       = torch.bmm(U_n, U_n.transpose(1, 2))
                eye        = torch.eye(U.size(1), device=device).unsqueeze(0)
                loss_ortho = (gram - eye).pow(2).mean()
                loss_cl    = cl_module(spectrum, labels) if cl_module else torch.tensor(0., device=device)
                loss = (loss_bce
                        + args.lam_sparse * loss_sp
                        + args.lam_ortho_modes * loss_ortho
                        + args.lam_cl * loss_cl)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            ep_loss += loss.item()

        # ── 验证 ─────────────────────────────────────────────────
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
                all_l.append(batch['labels'])

        probs   = torch.cat(all_p).numpy()
        lbls    = torch.cat(all_l).numpy()
        val_auc = roc_auc_score(lbls, probs)
        val_prc = average_precision_score(lbls, probs)
        val_f1  = f1_score(lbls, probs > 0.5)

        scheduler.step(val_auc)
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:3d} [lr={lr_cur:.2e}] | L:{ep_loss/len(train_loader):.3f} | "
              f"VAL_AUC:{val_auc:.4f} PRC:{val_prc:.4f} F1:{val_f1:.4f}")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save({'model': model.state_dict(), 'args': args},
                       save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop @ ep{epoch}，最优 AUC: {best_auc:.4f}")
                break

    print(f"最优 AUC: {best_auc:.4f}  模型: {save_dir / model_name}")
    return best_auc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CrossModal Conditional Operator Net')
    p.add_argument('--data_dir',        required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--fold',            type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=80)
    p.add_argument('--batch_size',      type=int,   default=512)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--hidden_dim',      type=int,   default=128)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--operator_rank',   type=int,   default=8)
    p.add_argument('--gene_max_len',    type=int,   default=1000)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--lam_sparse',      type=float, default=0.01)
    p.add_argument('--lam_ortho_modes', type=float, default=0.1)
    p.add_argument('--lam_cl',          type=float, default=0.0)
    p.add_argument('--patience',        type=int,   default=10)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--use_amp',         action='store_true')
    p.add_argument('--run_tag',         default='cross_modal')
    args = p.parse_args()
    train(args)
