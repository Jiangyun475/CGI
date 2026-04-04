#!/usr/bin/env python3
"""
双侧条件化 Sigma（Bilateral Conditioned Sigma）
================================================

【设计起点与原理】
  现有 PerturbationOperator 中三个关键量全部由药物单侧决定：
    U     = to_u(pharma_emb)         # 扰动输出方向，纯药物
    V     = to_v(pharma_emb)         # 扰动输入方向，纯药物
    sigma = to_sigma(pharma_emb)     # 模式激活强度，纯药物
  基因仅在最后 coupling = (V * h_g_modes).sum(-1) 被动参与一次点积。

  sigma 代表"第 k 个互作模式有多强"。
  生物学现实：这个强度不只取决于药物有没有这个药效团，
  还取决于基因对这个模式的"受体敏感性"——即双方的匹配程度。
  类比：酶-底物反应速率 = f(酶活性, 底物浓度) × 亲和力，是双侧属性。

  本文件的改动（单变量对照实验）：
    Old: sigma_k = Tanh(MLP(pharma_emb_k))             ← 药物单侧
    New: sigma_k = Tanh(dot(q(pharma_emb_k), k(h_g_modes_k)) / √d) ← 双侧内积

  这是最小的结构修改，保持 U、V、coupling 不变，
  只让 sigma 变成"药效团 slot k" × "基因模式 k" 的联合属性。

【与 train_cross_modal.py 的关系】
  本文件是"仅 sigma 双侧化"的最小实验。
  train_cross_modal.py 是更完整的方案：连 U、V、以及提取 query 都双侧化。
  对比两者的结果，可以量化"哪一层的双侧化贡献最大"。

【预期效果】
  基于双侧内积在 protein-ligand affinity 预测领域的一致有效性，
  预计 MCF7 Fold0 AUC +0.002~+0.008。
"""

import argparse
import itertools
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.utils import scatter

# ── 从主文件导入所有共享基础组件 ──────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_operator_moe import (
    set_seed, _KMER_VOCAB, encode_kmer_sequence,
    OptimizedGraphDataset, collate_fn,
    GINLayer, AtomEncoder, GeneMultiHeadReader, PharmacophoreExtractor,
    SpectrumDirectionCL,
)
from torch_geometric.utils import scatter_softmax


# ================================================================
# NEW: 双侧条件化 PerturbationOperator
# ================================================================

class BilateralPerturbationOperator(nn.Module):
    """
    双侧条件化扰动算子。

    与基础版 PerturbationOperator 的唯一区别：
      sigma 由"药物 query"和"基因 key"的双侧内积决定，而非纯药物 MLP。

    数学形式：
      q_k = Linear_q(pharma_emb_k)           [B, d]  药物侧 query
      k_k = Linear_k(h_g_modes_k)            [B, d]  基因侧 key
      sigma_k = Tanh( dot(q_k, k_k) / √d )   [B]    双侧激活强度

    其余（U、V、coupling）保持不变：
      U = to_u(pharma_emb)                   输出方向（仍药物单侧）
      V = to_v(pharma_emb)                   输入方向（仍药物单侧）
      coupling_k = (V_k · h_g_modes_k)       对齐度（点积）
      spectrum_k = sigma_k × coupling_k      有效交互谱
      Δh = Σ_k spectrum_k · u_k              净扰动向量

    局限：U 和 V 仍药物单侧。如需完全双侧，使用 train_cross_modal.py。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        H = hidden_dim
        d = H // 4  # 投影维度（比 H 小，降低参数量）

        self.to_u = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.to_v = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))

        # ── NEW: 双侧 sigma ──────────────────────────────────────
        # 药物侧：pharma_emb_k → query 向量
        self.sigma_q = nn.Sequential(nn.Linear(H, d), nn.ReLU())
        # 基因侧：h_g_modes_k → key 向量
        self.sigma_k = nn.Sequential(nn.Linear(H, d), nn.ReLU())
        # 注：d 维内积比 H 维内积更稳定，ReLU 保证非负基础再 Tanh 归一到 [-1,1]

    def forward(self, pharma_emb, h_g_modes):
        """
        Args:
          pharma_emb: [B, r, H]  药效团嵌入
          h_g_modes:  [B, r, H]  基因多视角编码

        Returns:
          delta_h:  [B, H]    净扰动向量
          spectrum: [B, r]    交互谱
          sigma:    [B, r]    模式强度（现在是双侧）
          U:        [B, r, H] 输出方向
        """
        U = F.normalize(self.to_u(pharma_emb), dim=-1)   # [B, r, H]
        V = F.normalize(self.to_v(pharma_emb), dim=-1)   # [B, r, H]

        # ── NEW: 双侧 sigma 计算 ─────────────────────────────────
        q = self.sigma_q(pharma_emb)    # [B, r, d]  药物侧 query
        k = self.sigma_k(h_g_modes)     # [B, r, d]  基因侧 key
        # 逐 slot 内积 / √d → Tanh 有界 [-1, 1]
        sigma = torch.tanh(
            (q * k).sum(dim=-1) / math.sqrt(q.size(-1))   # [B, r]
        )

        # coupling 保持不变（药物 V 与基因 h_g 的对齐度）
        coupling = (V * h_g_modes).sum(dim=-1)            # [B, r]
        spectrum = sigma * coupling                        # [B, r]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)  # [B, H]

        return delta_h, spectrum, sigma, U


# ================================================================
# NEW: 主模型（替换 perturb_op 为双侧版本）
# ================================================================

class BilateralOperatorNet(nn.Module):
    """
    DrugOperatorNet（no_moe 配置）+ 双侧 sigma。

    架构完全与 train_operator_moe.py --ablation no_moe 一致，
    唯一区别：PerturbationOperator → BilateralPerturbationOperator。
    """
    def __init__(self, hidden_dim=128, operator_rank=8, dropout=0.3):
        super().__init__()
        H, r = hidden_dim, operator_rank

        self.atom_encoder  = AtomEncoder(H, dropout)
        self.pharma_ext    = PharmacophoreExtractor(H, r)
        self.gene_reader   = GeneMultiHeadReader(4097, H, r, dropout)

        # ── 核心改动：使用双侧 sigma ─────────────────────────────
        self.perturb_op    = BilateralPerturbationOperator(H)

        self.classifier = nn.Sequential(
            nn.Linear(H * 2, H), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
        )
        self.hidden_dim = H

    def forward(self, gene_ids, x, edge_index, edge_attr, num_nodes_list):
        B = gene_ids.size(0)
        # batch_idx：每个原子属于哪个分子
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=x.device),
            torch.tensor(num_nodes_list, device=x.device)
        )

        # Drug
        atom_h = self.atom_encoder(x, edge_index, edge_attr)
        pharma_emb, _ = self.pharma_ext(atom_h, batch_idx, B)

        # Gene
        h_g_modes, h_g_global, gene_attn = self.gene_reader(gene_ids)

        # Bilateral Operator
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)

        # 分类
        logits = self.classifier(torch.cat([h_g_global, delta_h], dim=-1))
        return logits.squeeze(-1), spectrum, sigma, U, gene_attn, None


# ================================================================
# 训练循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device    = torch.device(args.device)
    data_dir  = Path(args.data_dir)
    save_dir  = Path('results_new_models') / data_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    tag       = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"bilateral_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    train_ds = OptimizedGraphDataset(data_dir, args.fold, 'train', args.gene_max_len)
    val_ds   = OptimizedGraphDataset(data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = BilateralOperatorNet(
        hidden_dim=args.hidden_dim,
        operator_rank=args.operator_rank,
        dropout=args.dropout,
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
    print(f"  [BilateralOperatorNet] 参数量: {n_params:,}")

    best_auc, patience_cnt, base_lr = 0.0, 0, args.lr
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if cl_module:
            cl_module.train()
        ep_loss = 0.0

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                lr = base_lr * global_step / warmup_steps
                for pg in optimizer.param_groups: pg['lr'] = lr

            gene_ids = batch['gene_ids'].to(device)
            labels   = batch['labels'].to(device)
            x        = batch['x'].to(device)
            ei       = batch['edge_index'].to(device)
            ea       = batch['edge_attr'].to(device)
            nnl      = batch['num_nodes_list']

            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U, _, _ = model(gene_ids, x, ei, ea, nnl)

                loss_bce  = criterion(logits, labels)
                loss_sp   = sigma.abs().mean()
                U_n       = F.normalize(U, dim=-1)
                gram      = torch.bmm(U_n, U_n.transpose(1, 2))
                eye       = torch.eye(U.size(1), device=device).unsqueeze(0)
                loss_ortho = (gram - eye).pow(2).mean()
                loss_cl   = cl_module(spectrum, labels) if cl_module else torch.tensor(0., device=device)

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
        probs = torch.cat(all_p).numpy()
        lbls  = torch.cat(all_l).numpy()
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


# ================================================================
# 入口
# ================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='BilateralSigma Operator Net')
    p.add_argument('--data_dir',        required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--fold',            type=int, default=0)
    p.add_argument('--epochs',          type=int, default=80)
    p.add_argument('--batch_size',      type=int, default=512)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--hidden_dim',      type=int, default=128)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--operator_rank',   type=int, default=8)
    p.add_argument('--gene_max_len',    type=int, default=1000)
    p.add_argument('--warmup_epochs',   type=int, default=5)
    p.add_argument('--lam_sparse',      type=float, default=0.01)
    p.add_argument('--lam_ortho_modes', type=float, default=0.1)
    p.add_argument('--lam_cl',          type=float, default=0.0)
    p.add_argument('--patience',        type=int, default=10)
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--use_amp',         action='store_true')
    p.add_argument('--run_tag',         default='bilateral')
    args = p.parse_args()
    train(args)
