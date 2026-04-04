#!/usr/bin/env python3
"""
多尺度谱分解（Multi-Scale Spectrum Decomposition）
==================================================

【设计起点】
  当前模型中 r=8 个谱模式是"平等"的：
    - 无层次结构，所有模式用相同方式参与竞争
    - 每个 slot 独立激活，彼此没有依赖关系
    - 正则项（正交+稀疏）只在事后约束，不在结构上引导层次

  但药物-基因互作天然存在层次：
    粗粒度（是否有互作？激活还是抑制？）→ 细粒度（具体哪条通路？哪个结合位点？）

【核心思想：层次化谱】

  将 r=8 个模式分为两级：
    粗粒度 r_c=2：捕捉全局药物-基因相容性（激活 vs 抑制）
    细粒度 r_f=6：捕捉具体机理模式，其激活强度由粗粒度谱条件化

  条件化机制（Coarse-to-Fine Gate）：
    spectrum_c ∈ R^{r_c}
    gate_f = σ(W_gate · spectrum_c)  ∈ R^{r_f}   ← 粗粒度谱决定细粒度激活"权限"
    sigma_f_final = sigma_f_base × gate_f          ← 细粒度强度被粗粒度调制

  直觉：
    如果粗粒度谱说"这个药物根本不影响这个基因"（spectrum_c ≈ 0），
    那么所有细粒度模式的激活门（gate_f ≈ 0.5）也应该被压制。
    如果粗粒度谱指示"激活模式很强"，细粒度的激活通路被开放。

【结构层级】

  粗粒度（r_c=2 slots）：
    PharmacophoreExtractor_coarse → pharma_emb_c [B, r_c, H]
    GeneMultiHeadReader_coarse    → h_g_modes_c  [B, r_c, H]
    PerturbationOperator_coarse   → spectrum_c   [B, r_c]，delta_h_c

  细粒度（r_f=6 slots，sigma 被 spectrum_c 调制）：
    PharmacophoreExtractor_fine   → pharma_emb_f [B, r_f, H]
    GeneMultiHeadReader_fine      → h_g_modes_f  [B, r_f, H]
    FineOperator（条件化）        → spectrum_f   [B, r_f]，delta_h_f

  全谱 = cat([spectrum_c, spectrum_f]) [B, r]
  全扰动 = delta_h_c + delta_h_f        [B, H]

【为什么可能有效】
  1. 粗粒度模式接受更强监督（BCE 梯度 + 正交约束），更快收敛到全局相容性
  2. 细粒度模式在粗粒度确认"有互作"后再发挥，避免浪费容量在"无互作"样本上
  3. 对 Chemical Cold Split：新药物的粗粒度激活更容易泛化（只需判断"有无"），
     细粒度在粗粒度约束下搜索空间更小，OOD 效应更小

【超参】
  --r_coarse 2  粗粒度模式数
  --r_fine   6  细粒度模式数
  合计 r=8（与基础模型一致，参数量可比）

  参数估算：
    多一套 PharmExt+GeneReader（粗粒度）：+约 60K
    gate 层：Linear(r_c, r_f) = 12 参数（极小）
    总计 +约 60K（+6.5%）
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
    AtomEncoder, PharmacophoreExtractor, GeneMultiHeadReader,
    PerturbationOperator, SpectrumDirectionCL,
)
from torch_geometric.utils import scatter_softmax


# ================================================================
# NEW: 条件化细粒度算子（sigma 被粗粒度谱调制）
# ================================================================

class FinePerturbationOperator(nn.Module):
    """
    细粒度扰动算子。

    与基础 PerturbationOperator 的区别：
      sigma 额外乘以一个"粗粒度门"（coarse gate），
      该门由粗粒度交互谱 spectrum_c 决定。

    数学形式：
      sigma_base_k = Tanh(MLP(pharma_emb_k))           药物单侧基础强度
      gate_k       = Sigmoid(W_gate[k,:] · spectrum_c) 粗粒度条件门
      sigma_k      = sigma_base_k × gate_k             调制后的强度

    这样：如果 spectrum_c ≈ 0（粗粒度认为无互作），
    gate ≈ Sigmoid(0) = 0.5 仍然开放，所以是软门而非硬截断。
    如果想要更强的条件化，可以将 gate_k 改为 ReLU 或将 spectrum_c 通过 LayerNorm。
    """
    def __init__(self, hidden_dim: int, r_fine: int, r_coarse: int):
        super().__init__()
        H = hidden_dim
        self.to_u     = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.to_v     = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.to_sigma = nn.Sequential(
            nn.Linear(H, H // 4), nn.ReLU(),
            nn.Linear(H // 4, 1), nn.Tanh(),
        )
        # ── NEW: 粗粒度条件化门 ──────────────────────────────────
        # spectrum_c [B, r_c] → gate [B, r_f]
        # 用 LayerNorm 稳定 spectrum_c 的尺度（它经过了 Tanh，范围 [-1,1]）
        self.coarse_norm = nn.LayerNorm(r_coarse)
        self.gate_proj   = nn.Linear(r_coarse, r_fine)  # r_c → r_f，参数极少

    def forward(self, pharma_emb, h_g_modes, spectrum_c):
        """
        Args:
          pharma_emb:  [B, r_f, H]
          h_g_modes:   [B, r_f, H]
          spectrum_c:  [B, r_c]   粗粒度交互谱（来自粗粒度算子）

        Returns:
          delta_h:  [B, H]
          spectrum: [B, r_f]
          sigma:    [B, r_f]  调制后的细粒度强度
          U:        [B, r_f, H]
        """
        U         = F.normalize(self.to_u(pharma_emb), dim=-1)   # [B, r_f, H]
        V         = F.normalize(self.to_v(pharma_emb), dim=-1)   # [B, r_f, H]
        sigma_base = self.to_sigma(pharma_emb).squeeze(-1)        # [B, r_f]

        # 粗粒度条件门（软门）
        gate     = torch.sigmoid(self.gate_proj(self.coarse_norm(spectrum_c)))  # [B, r_f]
        sigma    = sigma_base * gate                               # [B, r_f]  调制

        coupling = (V * h_g_modes).sum(dim=-1)                    # [B, r_f]
        spectrum = sigma * coupling                               # [B, r_f]
        delta_h  = (spectrum.unsqueeze(-1) * U).sum(dim=1)       # [B, H]
        return delta_h, spectrum, sigma, U


# ================================================================
# NEW: 多尺度谱模型
# ================================================================

class MultiScaleOperatorNet(nn.Module):
    """
    层次化多尺度谱分解模型。

    前向流程：

    ① 粗粒度通路（r_c=2 个模式）
       pharma_c = PharmExt_c(atom_h)         [B, r_c, H]
       h_g_c    = GeneReader_c(gene_ids)     [B, r_c, H]
       spectrum_c, delta_h_c = Op_c(pharma_c, h_g_c)

    ② 细粒度通路（r_f=6 个模式，被 spectrum_c 条件化）
       pharma_f = PharmExt_f(atom_h)         [B, r_f, H]
       h_g_f    = GeneReader_f(gene_ids)     [B, r_f, H]
       gate_f   = σ(W · spectrum_c)          [B, r_f]
       spectrum_f, delta_h_f = FinOp(pharma_f, h_g_f, spectrum_c)

    ③ 合并
       spectrum = cat([spectrum_c, spectrum_f]) [B, r]
       delta_h  = delta_h_c + delta_h_f        [B, H]

    ④ 分类
       logit = MLP(cat([h_g_global_f, delta_h]))
       （用细粒度基因视角作为全局表示，包含更丰富信息）
    """
    def __init__(self, hidden_dim=128, r_coarse=2, r_fine=6, dropout=0.3):
        super().__init__()
        H = hidden_dim
        self.r_c = r_coarse
        self.r_f = r_fine

        self.atom_encoder = AtomEncoder(H, dropout)

        # 粗粒度提取器（r_c 个 slot/head）
        self.pharma_c = PharmacophoreExtractor(H, r_coarse)
        self.gene_c   = GeneMultiHeadReader(4097, H, r_coarse, dropout)

        # 细粒度提取器（r_f 个 slot/head）
        self.pharma_f = PharmacophoreExtractor(H, r_fine)
        self.gene_f   = GeneMultiHeadReader(4097, H, r_fine, dropout)

        # 算子
        self.op_coarse = PerturbationOperator(H)
        self.op_fine   = FinePerturbationOperator(H, r_fine, r_coarse)

        self.classifier = nn.Sequential(
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

        atom_h = self.atom_encoder(x, edge_index, edge_attr)   # [N, H]

        # ① 粗粒度通路
        pharma_c, _       = self.pharma_c(atom_h, batch_idx, B)    # [B, r_c, H]
        h_g_c, _, attn_c  = self.gene_c(gene_ids)                  # [B, r_c, H]
        delta_h_c, spectrum_c, sigma_c, U_c = self.op_coarse(pharma_c, h_g_c)

        # ② 细粒度通路（sigma 被 spectrum_c 条件化）
        pharma_f, _       = self.pharma_f(atom_h, batch_idx, B)    # [B, r_f, H]
        h_g_f, h_g_global, attn_f = self.gene_f(gene_ids)          # [B, r_f, H]
        delta_h_f, spectrum_f, sigma_f, U_f = self.op_fine(pharma_f, h_g_f, spectrum_c)

        # ③ 合并谱和扰动
        spectrum = torch.cat([spectrum_c, spectrum_f], dim=1)  # [B, r]
        sigma    = torch.cat([sigma_c,    sigma_f],    dim=1)  # [B, r]
        U        = torch.cat([U_c,        U_f],        dim=1)  # [B, r, H]
        delta_h  = delta_h_c + delta_h_f                       # [B, H]

        # ④ 分类
        logits = self.classifier(torch.cat([h_g_global, delta_h], dim=-1))
        return logits.squeeze(-1), spectrum, sigma, U, attn_f, None


# ================================================================
# 训练循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device   = torch.device(args.device)
    data_dir = Path(args.data_dir)
    save_dir = Path('results_new_models') / data_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    tag       = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"multiscale_c{args.r_coarse}f{args.r_fine}_Fold{args.fold}{tag}.pt"

    train_ds     = OptimizedGraphDataset(data_dir, args.fold, 'train', args.gene_max_len)
    val_ds       = OptimizedGraphDataset(data_dir, args.fold, 'val',   args.gene_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = MultiScaleOperatorNet(
        hidden_dim=args.hidden_dim,
        r_coarse=args.r_coarse,
        r_fine=args.r_fine,
        dropout=args.dropout,
    ).to(device)

    cl_module = None
    if args.lam_cl > 0:
        cl_module = SpectrumDirectionCL(
            rank=args.r_coarse + args.r_fine, margin=0.5).to(device)

    params = list(model.parameters())
    if cl_module: params += list(cl_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [MultiScaleOperatorNet] r_coarse={args.r_coarse}, r_fine={args.r_fine}, "
          f"参数量: {n_params:,}")

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
    p = argparse.ArgumentParser(description='Multi-Scale Spectrum Operator Net')
    p.add_argument('--data_dir',        required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--fold',            type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=80)
    p.add_argument('--batch_size',      type=int,   default=512)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--hidden_dim',      type=int,   default=128)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--r_coarse',        type=int,   default=2)
    p.add_argument('--r_fine',          type=int,   default=6)
    p.add_argument('--gene_max_len',    type=int,   default=1000)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--lam_sparse',      type=float, default=0.01)
    p.add_argument('--lam_ortho_modes', type=float, default=0.1)
    p.add_argument('--lam_cl',          type=float, default=0.0)
    p.add_argument('--patience',        type=int,   default=10)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--use_amp',         action='store_true')
    p.add_argument('--run_tag',         default='multiscale')
    args = p.parse_args()
    train(args)
