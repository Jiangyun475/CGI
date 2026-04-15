#!/usr/bin/env python3
"""
interp/drug_cluster.py
======================
药物谱模式聚类分析。

核心思路：
  每个药物的"模式指纹" = 其 sigma 向量（r=8 维）的均值（跨所有 val 对）
  sigma 是药物内在属性，表示该药物在每个谱模式上的激活幅度。

分析内容：
  1. UMAP 降维可视化（药物 sigma 空间）
  2. 已知机制药物标注（手工标注 + 自动颜色）
  3. Mode 激活 bar chart（各模式强度分布）
  4. 每个模式 top 药物的化学结构（可选，需 RDKit）

输出：
  results/{cell}/drug_cluster/
    drug_sigma_umap.png
    mode_activation_profile.png
    drug_sigma.csv           # 药物级别数据

用法:
  python interp/drug_cluster.py --cell MCF7
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap not available, skipping UMAP plot")

# 已知机制的标志性药物（BRD compound ID → (name, mechanism_class)）
# 从 LINCS/L1000 数据库已知的化合物映射
KNOWN_DRUGS = {
    # 示例：用 compound_id 来标注
    # 实际运行后从 drug_sigma.csv 里找有代表性的化合物来标注
}

# 机制类别颜色方案
MOA_COLORS = {
    "kinase inhibitor":      "#E74C3C",
    "HDAC inhibitor":        "#3498DB",
    "proteasome inhibitor":  "#2ECC71",
    "DNA damage":            "#9B59B6",
    "hormone":               "#F39C12",
    "other":                 "#95A5A6",
}


def main(args):
    data_path = os.path.join("interp/results", args.cell, "representations.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} 不存在，请先运行 extract.py")
        sys.exit(1)

    out_dir = os.path.join("interp/results", args.cell, "drug_cluster")
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    sigma        = d["sigma"]         # [N, r]
    labels       = d["labels"]        # [N]
    compound_ids = d["compound_ids"]  # [N]
    smiles       = d["smiles"]        # [N]

    r = sigma.shape[1]

    # ── 1. 每个药物聚合 sigma（跨 gene 均值）────────────────────
    unique_cids = np.unique(compound_ids)
    drug_sigma = []
    drug_smiles_list = []
    drug_pos_rate = []  # 该药物的正样本比例（粗略 MOA 指标）

    for cid in unique_cids:
        mask = compound_ids == cid
        drug_sigma.append(sigma[mask].mean(axis=0))
        # SMILES（取第一个）
        smi_arr = smiles[mask]
        drug_smiles_list.append(smi_arr[0] if len(smi_arr) > 0 else "")
        # 正样本比例
        drug_pos_rate.append(labels[mask].mean())

    drug_sigma    = np.array(drug_sigma)    # [D, r]
    drug_pos_rate = np.array(drug_pos_rate) # [D]
    D = len(unique_cids)
    print(f"Unique drugs: {D}, operator_rank: {r}")

    # ── 2. 保存药物级别数据 ───────────────────────────────────────
    drug_df = pd.DataFrame(drug_sigma, columns=[f"sigma_mode{j}" for j in range(r)])
    drug_df["compound_id"] = unique_cids
    drug_df["smiles"]      = drug_smiles_list
    drug_df["pos_rate"]    = drug_pos_rate
    drug_df.to_csv(os.path.join(out_dir, "drug_sigma.csv"), index=False)
    print(f"Drug sigma saved to {out_dir}/drug_sigma.csv")

    # ── 3. UMAP 可视化 ────────────────────────────────────────────
    if HAS_UMAP and D >= 20:
        print("Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(30, D-1), min_dist=0.1)
        emb = reducer.fit_transform(drug_sigma)  # [D, 2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 左图：按正样本比例着色（高 pos_rate = 该药效强）
        sc = axes[0].scatter(emb[:, 0], emb[:, 1],
                             c=drug_pos_rate, cmap="RdYlBu_r",
                             s=15, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=axes[0], label="Positive rate (fraction of significant gene responses)")
        axes[0].set_title(f"Drug sigma UMAP — {args.cell} (colored by activity)", fontsize=13)
        axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # 右图：按主导模式着色（argmax of sigma per drug）
        dominant_mode = np.argmax(drug_sigma, axis=1)  # [D]
        colors_mode = cm.tab10(dominant_mode / r)
        axes[1].scatter(emb[:, 0], emb[:, 1], c=colors_mode,
                        s=15, alpha=0.7, linewidths=0)
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cm.tab10(j/r), markersize=8, label=f"Mode {j}")
            for j in range(r)
        ]
        axes[1].legend(handles=legend_elements, loc="best", fontsize=8)
        axes[1].set_title(f"Drug sigma UMAP — {args.cell} (colored by dominant mode)", fontsize=13)
        axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "drug_sigma_umap.png"), dpi=150)
        plt.close()
        print(f"UMAP saved to {out_dir}/drug_sigma_umap.png")

        # 保存 embedding 供后续标注
        np.save(os.path.join(out_dir, "umap_embedding.npy"), emb)

    # ── 4. 模式激活分布图 ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for j in range(r):
        ax = axes[j]
        vals = drug_sigma[:, j]
        ax.hist(vals, bins=50, color=cm.tab10(j/r), alpha=0.8, edgecolor="none")
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={vals.mean():.3f}")
        ax.set_title(f"Mode {j}  (std={vals.std():.3f})", fontsize=11)
        ax.set_xlabel("sigma amplitude")
        ax.set_ylabel("# drugs")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle(f"Drug Mode Activation Distribution — {args.cell}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mode_activation_distribution.png"), dpi=150)
    plt.close()
    print(f"Mode distribution saved to {out_dir}/mode_activation_distribution.png")

    # ── 5. 每个模式 top-20 活跃药物（高 sigma_j 的药物）────────
    print("\nTop-10 drugs per mode (highest sigma):")
    top_drug_lines = []
    for j in range(r):
        top_idx = np.argsort(drug_sigma[:, j])[-10:][::-1]
        top_cids = unique_cids[top_idx]
        top_vals = drug_sigma[top_idx, j]
        line = f"Mode {j}: " + ", ".join(f"{c}({v:.3f})" for c, v in zip(top_cids, top_vals))
        print(f"  {line}")
        top_drug_lines.append(line + "\n")

    with open(os.path.join(out_dir, "top_drugs_per_mode.txt"), "w") as f:
        f.writelines(top_drug_lines)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="MCF7")
    main(parser.parse_args())
