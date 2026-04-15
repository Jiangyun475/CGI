#!/usr/bin/env python3
"""
interp/go_mode_barplot.py
=========================
从已有的 GO enrichment 结果（interaction view）生成一张紧凑的模式标注图：
  - 每个模式取 interaction view 中 FDR 最小的 top-1 term
  - 横向 bar chart，按 -log10(FDR) 显示显著性
  - 右侧标注模式生物学标签

用法:
  python interp/go_mode_barplot.py --cell MCF7
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 手动定义 interaction view 中每个模式最具代表性的 term
# 来源：go_summary.md 中 interaction view 的 FDR 最优 term
MODE_BEST_TERMS = {
    0: ("Regulation Of Protein Kinase Activity",               1.97e-3, "Kinase Signaling"),
    1: ("Positive Regulation Of Mitotic Nuclear Division",     2.54e-2, "Cell Cycle"),
    2: ("Regulation Of Amide Metabolic Process",               4.25e-2, "Metabolic Process"),
    3: ("Retrograde Transport, Vesicle Recycling Within Golgi",6.59e-3, "Golgi / Vesicular"),
    4: ("Mitochondrial Fragmentation In Apoptotic Process",    6.65e-3, "Mito Apoptosis"),
    5: ("Regulation Of I-κB Kinase / NF-κB Signaling",        2.13e-3, "NF-κB / MAPK"),
    6: ("Neg. Reg. Of Protein Tyrosine Kinase Activity",       9.10e-2, "RTK (weak)"),
    7: ("Positive Regulation Of Apoptotic Process",            2.22e-2, "Apoptosis Regulation"),
}

# 显著性阈值
FDR_THRESHOLD = 0.05


def main(args):
    out_dir = os.path.join("interp/results", args.cell, "go")
    os.makedirs(out_dir, exist_ok=True)

    modes    = sorted(MODE_BEST_TERMS.keys())
    go_terms = [MODE_BEST_TERMS[m][0] for m in modes]
    fdrs     = [MODE_BEST_TERMS[m][1] for m in modes]
    labels   = [MODE_BEST_TERMS[m][2] for m in modes]
    neg_log_fdr = [-np.log10(f) for f in fdrs]

    # Color per mode (consistent with case study palette)
    cmap   = plt.cm.Set2(np.linspace(0, 1, len(modes)))
    colors = [cmap[i] for i in range(len(modes))]

    # Combined y-tick labels: "Mode N\nBiological Label"
    ytick_labels = [f"Mode {m}\n{labels[m]}" for m in modes]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars = ax.barh(range(len(modes)), neg_log_fdr, color=colors, edgecolor="white",
                   height=0.7, alpha=0.9)

    # Significance line at FDR=0.05
    ax.axvline(-np.log10(FDR_THRESHOLD), color="red", linewidth=1.0,
               linestyle="--", alpha=0.7, label="FDR = 0.05")

    # Annotate GO term text inside bars
    for i, (bar, term) in enumerate(zip(bars, go_terms)):
        ax.text(0.05, i, term, va="center", ha="left", fontsize=8.5,
                color="black", style="italic")

    # -log10(FDR) value to the right of each bar with enough offset
    x_max_data = max(neg_log_fdr) * 1.15
    ax.set_xlim(0, x_max_data)
    for i, v in enumerate(neg_log_fdr):
        ax.text(v + x_max_data * 0.015, i, f"{v:.2f}", va="center", ha="left",
                fontsize=8.5, color="dimgray")

    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(ytick_labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("−log₁₀(FDR)", fontsize=11)
    ax.set_title(f"Spectral Mode Biological Identity — {args.cell}\n"
                 "(GO Biological Process, Interaction View)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()  # Mode 0 at top
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mode_identity_barplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="MCF7")
    main(parser.parse_args())
