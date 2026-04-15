#!/usr/bin/env python3
"""
interp/extract.py
=================
从论文模型 (no_moe, cl01) 提取可解释性所需的所有中间表示。

输出 results/{cell}/representations.npz 包含：
  spectrum    [N, r]       交互谱：模式 j 在该 pair 上的激活强度
  sigma       [N, r]       药物模式幅度（药物内在属性，与基因无关）
  U           [N, r, H]    药物生成的 r 个模式方向向量
  h_g_modes   [N, r, H]    基因在每个模式子空间的表示
  preds       [N]          模型预测概率
  labels      [N]          真实标签
  gene_ids    [N]          Entrez Gene ID（从 dataset.csv 读取）
  smiles      [N]          药物 SMILES（从 dataset.csv 读取）
  compound_ids [N]         BRD compound ID

用法:
  python interp/extract.py \
    --cell MCF7 \
    --model_path results_operator_moe/MCF7/no_moe_r8_k4_Fold0_cl01.pt \
    --fold 0 \
    --device cuda:3
"""

import sys, os, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── 从训练脚本直接导入模型和数据集 ────────────────────────────────
from New.train_operator_moe import (
    OperatorMoE, OptimizedGraphDataset, collate_fn
)

DATA_ROOT = "/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"


def main(args):
    device = torch.device(args.device)
    out_dir = os.path.join("interp/results", args.cell)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 加载数据集 ─────────────────────────────────────────────
    data_dir = os.path.join(DATA_ROOT, args.cell)
    df = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

    with open(os.path.join(data_dir, "chemical_cold_splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    train_idx, val_idx = splits[args.fold]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    val_ds = OptimizedGraphDataset(
        data_dir=data_dir,
        split="val",
        fold_idx=args.fold,
        gene_max_len=1000,
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    # SMILES per val sample（直接从 dataset.graph_indices 取）
    smiles_list = val_ds.graph_indices  # list[str], len=N

    # ── 2. 加载模型 ───────────────────────────────────────────────
    model = OperatorMoE(
        hidden_dim=128,
        dropout=0.3,
        operator_rank=args.rank,
        ablation="no_moe",
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    print(f"Val set: {len(val_ds)} samples")

    # ── 3. 推理，收集中间表示 ─────────────────────────────────────
    all_spectrum, all_sigma, all_U, all_hg = [], [], [], []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting"):
            x          = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr  = batch["edge_attr"].to(device)
            gene_ids   = batch["gene_ids"].to(device)
            labels     = batch["label"]

            # 获取 h_g_modes（基因模式表示）
            h_g_modes, h_g_global, gene_attn = model.gene_enc(gene_ids)
            # h_g_modes: [B, r, H]

            # 获取药物表示
            B = len(batch["num_nodes_list"])
            num_nodes_t = torch.tensor(batch["num_nodes_list"], device=device)
            batch_idx   = torch.repeat_interleave(
                torch.arange(B, device=device), num_nodes_t)
            atom_h      = model.atom_enc(x, edge_index, edge_attr)
            pharma_emb, _ = model.pharma_ext(atom_h, batch_idx, B)
            # pharma_emb: [B, r, H]

            # 算子分解
            delta_h, spectrum, sigma, U = model.perturb_op(pharma_emb, h_g_modes)
            # spectrum [B,r], sigma [B,r], U [B,r,H]

            # 分类
            features = torch.cat([h_g_global, delta_h], dim=-1)
            logits   = model.classifier(features).squeeze(-1)
            preds    = torch.sigmoid(logits)

            all_spectrum.append(spectrum.cpu().float().numpy())
            all_sigma.append(sigma.cpu().float().numpy())
            all_U.append(U.cpu().float().numpy())
            all_hg.append(h_g_modes.cpu().float().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    spectrum  = np.concatenate(all_spectrum)   # [N, r]
    sigma     = np.concatenate(all_sigma)      # [N, r]
    U         = np.concatenate(all_U)          # [N, r, H]
    h_g_modes = np.concatenate(all_hg)         # [N, r, H]
    preds     = np.concatenate(all_preds)      # [N]
    labels    = np.concatenate(all_labels)     # [N]

    auc = roc_auc_score(labels, preds)
    print(f"Val AUC: {auc:.4f}")

    # ── 4. 附加 gene_id / smiles / compound_id ────────────────────
    gene_ids_arr = val_df["gene_id"].values.astype(np.int64)
    compound_ids = val_df["compound_id"].values
    smiles_arr   = np.array(smiles_list)  # 已从 dataset.graph_indices 取得

    # ── 5. 保存 ───────────────────────────────────────────────────
    save_path = os.path.join(out_dir, "representations.npz")
    np.savez_compressed(
        save_path,
        spectrum     = spectrum,      # [N, r]
        sigma        = sigma,         # [N, r]
        U            = U,             # [N, r, H]
        h_g_modes    = h_g_modes,     # [N, r, H]
        preds        = preds,         # [N]
        labels       = labels,        # [N]
        gene_ids     = gene_ids_arr,  # [N]
        smiles       = smiles_arr,    # [N]
        compound_ids = compound_ids,  # [N]
    )
    print(f"Saved to {save_path}")
    print(f"  spectrum:  {spectrum.shape}")
    print(f"  U:         {U.shape}")
    print(f"  h_g_modes: {h_g_modes.shape}")
    print(f"  gene_ids unique: {len(np.unique(gene_ids_arr))}")
    print(f"  compounds unique: {len(np.unique(compound_ids))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell",       default="MCF7")
    parser.add_argument("--model_path", default="results_operator_moe/MCF7/no_moe_r8_k4_Fold0_cl01.pt")
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--rank",       type=int, default=8)
    parser.add_argument("--device",     default="cuda:3")
    main(parser.parse_args())
