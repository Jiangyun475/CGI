#!/usr/bin/env python3
"""
interp/case_study.py
====================
Case study 可视化：选取代表性药物，展示其谱模式指纹。

选取逻辑：每个显著模式选 1 个已知机制的代表性药物：
  Mode 2 (Metabolic)     → trimidox (Ribonucleotide reductase inhibitor)
  Mode 3 (Golgi/nuclear) → calcipotriol (Vitamin D receptor agonist)
  Mode 4 (Mito apoptosis)→ fenretinide (Apoptosis stimulant)
  Mode 5 (NF-kB/MAPK)   → trametinib (MEK inhibitor)

图表：
  A. 多药物 sigma 指纹对比（grouped bar）
  B. 归一化热图（各药物在各模式的 z-score）

用法:
  python interp/case_study.py --cell MCF7
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Mode 标签（来自 GO 富集结果）
MODE_LABELS = [
    "Mode 0\nKinase\nSignaling",
    "Mode 1\nCell\nCycle",
    "Mode 2\nMetabolic\nProcess",
    "Mode 3\nGolgi /\nNuclear",
    "Mode 4\nMito\nApoptosis",
    "Mode 5\nNF-κB /\nMAPK",
    "Mode 6\nRTK\n(weak)",
    "Mode 7\nApoptosis\nRegulation",
]

MODE_LABELS_SHORT = [
    "M0\nKinase",
    "M1\nCellCycle",
    "M2\nMetabolic",
    "M3\nGolgi",
    "M4\nMitoApop",
    "M5\nNF-κB",
    "M6\nRTK",
    "M7\nApoptosis",
]

# 验证药物：每个模式一个代表
CASE_STUDY_DRUGS = {
    # Mode 2: metabolic
    "trimidox\n(RNR inhibitor)":           "BRD-K11801786",
    "tiaprofenic acid\n(COX inhibitor)":   "BRD-A72988804",
    # Mode 3: vesicular/nuclear receptor
    "calcipotriol\n(VitD receptor)":       "BRD-K96390176",
    "clobetasol\n(Glucocorticoid)":        "BRD-A26095496",
    # Mode 4: mitochondrial apoptosis
    "fenretinide\n(Apoptosis stimulant)":  "BRD-K89085489",
    # Mode 5: NF-kB / MAPK signaling
    "trametinib\n(MEK inhibitor)":         "BRD-K12343256",
    "atorvastatin\n(HMGCR inhibitor)":     "BRD-U88459701",
}


def main(args):
    data_path = os.path.join("interp/results", args.cell, "drug_cluster/drug_sigma.csv")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run drug_cluster.py first.")
        sys.exit(1)

    out_dir = os.path.join("interp/results", args.cell, "case_study")
    os.makedirs(out_dir, exist_ok=True)

    drug_df = pd.read_csv(data_path).set_index("compound_id")
    sigma_cols = [f"sigma_mode{j}" for j in range(8)]

    # 全局 mean/std 用于 z-score
    all_sigma = drug_df[sigma_cols].values
    global_mean = all_sigma.mean(axis=0)
    global_std  = all_sigma.std(axis=0) + 1e-8

    # 收集 case study 药物数据
    drug_names, sigma_matrix = [], []
    found_drugs = {}
    for name, brd in CASE_STUDY_DRUGS.items():
        if brd in drug_df.index:
            sigma_matrix.append(drug_df.loc[brd, sigma_cols].values.astype(float))
            drug_names.append(name)
            found_drugs[name] = brd
        else:
            print(f"  WARNING: {brd} ({name.split(chr(10))[0]}) not in val set")

    if not drug_names:
        print("No case study drugs found in validation set.")
        sys.exit(1)

    sigma_matrix = np.array(sigma_matrix)   # [D, 8]
    z_matrix = (sigma_matrix - global_mean) / global_std  # z-score

    n_drugs = len(drug_names)
    r = 8

    # ── Figure A: Grouped bar chart (z-score normalised) ─────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x     = np.arange(r)
    width = 0.8 / n_drugs
    colors = cm.Set2(np.linspace(0, 1, n_drugs))

    for i, (name, z_row) in enumerate(zip(drug_names, z_matrix)):
        offset = (i - n_drugs/2 + 0.5) * width
        ax.bar(x + offset, z_row, width, label=name,
               color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS_SHORT, fontsize=9)
    ax.set_ylabel("Z-score vs all drugs", fontsize=11)
    ax.set_title(f"Drug Spectral Mode Fingerprints (Z-score) — {args.cell}",
                 fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(2, color="gray", linewidth=0.7, alpha=0.4, linestyle="--")
    ax.axhline(-2, color="gray", linewidth=0.7, alpha=0.4, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "case_study_sigma.png"), dpi=150)
    plt.close()

    # ── Figure B: Heatmap of z-scores ────────────────────────────
    # Determine symmetric colormap limits based on actual data
    z_abs_max = min(6.0, max(3.0, np.ceil(np.abs(z_matrix).max())))
    fig, ax = plt.subplots(figsize=(12, max(4, n_drugs * 0.7 + 1)))
    im = ax.imshow(z_matrix, cmap="RdBu_r", vmin=-z_abs_max, vmax=z_abs_max, aspect="auto")
    plt.colorbar(im, ax=ax, label="Z-score vs all drugs")

    ax.set_xticks(range(r))
    ax.set_xticklabels(MODE_LABELS_SHORT, fontsize=9)
    ax.set_yticks(range(n_drugs))
    ax.set_yticklabels(drug_names, fontsize=9)
    ax.set_title(f"Drug Mode Z-score — {args.cell}", fontsize=12)

    # 标注数值
    for i in range(n_drugs):
        for j in range(r):
            val = z_matrix[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 1.5 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "case_study_zscore_heatmap.png"), dpi=150)
    plt.close()

    # ── 文字摘要 ─────────────────────────────────────────────────
    lines = [f"# Case Study: Drug Spectral Mode Fingerprints\n\n"]
    lines.append("| Drug | BRD ID | Dominant Mode | Max Sigma | Z-score |\n")
    lines.append("|------|--------|--------------|-----------|--------|\n")
    for i, name in enumerate(drug_names):
        brd      = found_drugs[name]
        dom_j    = int(np.argmax(z_matrix[i]))   # dominant by z-score
        max_z    = z_matrix[i, dom_j]
        max_sig  = sigma_matrix[i, dom_j]
        name_1l  = name.replace("\n", " ")
        lines.append(f"| {name_1l} | {brd} | Mode {dom_j} ({MODE_LABELS[dom_j].split(chr(10))[1]}) | {max_sig:.3f} | {max_z:+.2f} |\n")
    lines.append("\n## Mode Labels (from GO enrichment)\n")
    for j, lbl in enumerate(MODE_LABELS):
        lines.append(f"- {lbl.replace(chr(10), ' ')}\n")

    with open(os.path.join(out_dir, "case_study_summary.md"), "w") as f:
        f.writelines(lines)

    print(f"Case study figures saved to {out_dir}/")
    print(f"  case_study_sigma.png")
    print(f"  case_study_zscore_heatmap.png")
    print(f"  case_study_summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="MCF7")
    main(parser.parse_args())
