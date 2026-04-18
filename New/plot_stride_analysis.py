#!/usr/bin/env python3
"""
Stride/Length Sweep 结果分析图
==============================
从 logs_stride_sweep/ 读取每个实验的最优 AUC，
生成两张图：
  Fig A: AUC vs 序列覆盖 bp（不同策略对比）
  Fig B: stride vs max_len 热图（完整性分布 + AUC 叠加）
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch

LOG_DIR = Path("logs_stride_sweep")

# ── 实验配置 ──────────────────────────────────────────────────
EXPS = [
    # (tag,           max_len, stride, coverage_bp)
    ("s1_len1000",   1000,    1,      1006),
    ("s2_len1000",   1000,    2,      2006),
    ("s3_len1000",   1000,    3,      3006),
    ("s5_len1000",   1000,    5,      5006),
    ("s1_len2000",   2000,    1,      2006),
    ("s1_len3000",   3000,    1,      3006),
    ("s1_len5000",   5000,    1,      5006),
]

# ── 基因长度分布（用于计算完整覆盖率）────────────────────────
def get_gene_lengths():
    data_path = ("/home/data/jiangyun/cgi_data_pipeline/outputs/"
                 "datasets_classification_test_recommended/MCF7/"
                 "preprocessed_graphs_MCF7.pt")
    data = torch.load(data_path, weights_only=False)
    seqs = data['gene_sequences']
    return sorted(set(len(s) for s in seqs))

def coverage_rate(lengths, bp):
    return sum(1 for l in lengths if l <= bp) / len(lengths) * 100

# ── 从 log 文件提取最优 AUC ──────────────────────────────────
def extract_best_auc(log_file):
    best = 0.0
    pattern = re.compile(r'VAL_AUC:([\d.]+)')
    try:
        with open(log_file) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    val = float(m.group(1))
                    if val > best:
                        best = val
    except FileNotFoundError:
        return None
    return best if best > 0 else None

# ── 主逻辑 ───────────────────────────────────────────────────
def main():
    gene_lengths = get_gene_lengths()

    results = []
    missing = []
    for tag, max_len, stride, coverage_bp in EXPS:
        log_file = LOG_DIR / f"{tag}.log"
        auc = extract_best_auc(log_file)
        cov = coverage_rate(gene_lengths, coverage_bp)
        results.append({
            "tag": tag, "max_len": max_len, "stride": stride,
            "coverage_bp": coverage_bp, "coverage_pct": cov, "auc": auc
        })
        if auc is None:
            missing.append(tag)

    if missing:
        print(f"⚠️  以下实验尚未完成或日志为空: {missing}")
        print("已完成的实验:")
        for r in results:
            if r["auc"] is not None:
                print(f"  {r['tag']:15s}  coverage={r['coverage_pct']:.1f}%  AUC={r['auc']:.4f}")
        # 过滤掉未完成的
        results = [r for r in results if r["auc"] is not None]

    if not results:
        print("没有完成的实验结果，退出。")
        return

    # ── 打印汇总表 ──────────────────────────────────────────
    print("\n{'='*60}")
    print(f"{'Config':<15} {'MaxLen':>7} {'Stride':>6} {'~BP':>6} "
          f"{'FullCov%':>8} {'AUC':>7}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["coverage_bp"]):
        print(f"{r['tag']:<15} {r['max_len']:>7} {r['stride']:>6} "
              f"{r['coverage_bp']:>6} {r['coverage_pct']:>8.1f} "
              f"{r['auc']:>7.4f}")

    # ── 绘图 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gene Sequence Coverage Sweep\n(MCF7 Fold0, no_moe baseline)",
                 fontsize=12, fontweight='bold')

    # 分离两种策略
    fix_len  = [r for r in results if r["max_len"] == 1000]   # fixed tokens, vary stride
    fix_str  = [r for r in results if r["stride"] == 1]       # fixed stride=1, vary tokens

    # ── Fig A: AUC vs 覆盖 bp ──────────────────────────────
    ax = axes[0]
    ax2 = ax.twinx()

    # 背景：完整覆盖率曲线
    bp_range = np.linspace(500, 10000, 200)
    cov_curve = [coverage_rate(gene_lengths, bp) for bp in bp_range]
    ax2.fill_between(bp_range, cov_curve, alpha=0.08, color='gray')
    ax2.plot(bp_range, cov_curve, color='gray', lw=1.2, ls='--', alpha=0.6,
             label='Full coverage %')
    ax2.set_ylabel("Genes fully covered (%)", color='gray', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 105)

    colors = {'fix_len': '#E05C5C', 'fix_str': '#3A85C0'}
    markers = {'fix_len': 'o', 'fix_str': 's'}

    for r in sorted(fix_len, key=lambda x: x["coverage_bp"]):
        ax.scatter(r["coverage_bp"], r["auc"], color=colors['fix_len'],
                   marker=markers['fix_len'], s=80, zorder=5)
        ax.annotate(f"s={r['stride']}", (r["coverage_bp"], r["auc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7.5,
                    color=colors['fix_len'])
    for r in sorted(fix_str, key=lambda x: x["coverage_bp"]):
        ax.scatter(r["coverage_bp"], r["auc"], color=colors['fix_str'],
                   marker=markers['fix_str'], s=80, zorder=5)
        ax.annotate(f"L={r['max_len']}", (r["coverage_bp"], r["auc"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7.5,
                    color=colors['fix_str'])

    # 连线
    if len(fix_len) > 1:
        xs = [r["coverage_bp"] for r in sorted(fix_len, key=lambda x: x["coverage_bp"])]
        ys = [r["auc"] for r in sorted(fix_len, key=lambda x: x["coverage_bp"])]
        ax.plot(xs, ys, color=colors['fix_len'], lw=1.5, alpha=0.7)
    if len(fix_str) > 1:
        xs = [r["coverage_bp"] for r in sorted(fix_str, key=lambda x: x["coverage_bp"])]
        ys = [r["auc"] for r in sorted(fix_str, key=lambda x: x["coverage_bp"])]
        ax.plot(xs, ys, color=colors['fix_str'], lw=1.5, alpha=0.7)

    ax.axhline(y=0.8941, color='black', lw=1, ls=':', alpha=0.5, label='baseline 0.8941')
    ax.set_xlabel("Sequence coverage (bp)", fontsize=10)
    ax.set_ylabel("Best Val AUC", fontsize=10)
    ax.set_title("AUC vs Coverage", fontsize=10)
    patch_len = mpatches.Patch(color=colors['fix_len'], label='Fixed tokens (stride↑)')
    patch_str = mpatches.Patch(color=colors['fix_str'], label='Fixed stride=1 (tokens↑)')
    ax.legend(handles=[patch_len, patch_str], fontsize=8, loc='lower right')

    # ── Fig B: 配置矩阵 ──────────────────────────────────────
    ax = axes[1]

    max_lens = sorted(set(r["max_len"] for r in results))
    strides  = sorted(set(r["stride"]  for r in results))

    auc_grid = np.full((len(strides), len(max_lens)), np.nan)
    for r in results:
        i = strides.index(r["stride"])
        j = max_lens.index(r["max_len"])
        auc_grid[i, j] = r["auc"]

    im = ax.imshow(auc_grid, aspect='auto', cmap='RdYlGn',
                   vmin=max(0.88, np.nanmin(auc_grid) - 0.002),
                   vmax=min(0.93, np.nanmax(auc_grid) + 0.002))
    plt.colorbar(im, ax=ax, label='Val AUC')

    ax.set_xticks(range(len(max_lens)))
    ax.set_xticklabels([str(l) for l in max_lens])
    ax.set_yticks(range(len(strides)))
    ax.set_yticklabels([str(s) for s in strides])
    ax.set_xlabel("max_len (tokens)", fontsize=10)
    ax.set_ylabel("stride", fontsize=10)
    ax.set_title("AUC Heatmap (stride × max_len)", fontsize=10)

    for i in range(len(strides)):
        for j in range(len(max_lens)):
            val = auc_grid[i, j]
            if not np.isnan(val):
                bp = max_lens[j] * strides[i]
                cov = coverage_rate(gene_lengths, bp)
                ax.text(j, i, f"{val:.4f}\n({cov:.0f}%)", ha='center', va='center',
                        fontsize=7.5, color='black')

    plt.tight_layout()
    out = Path("analysis_spectrum_geometry") / "stride_sweep_analysis.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n图已保存: {out}")
    plt.show()

    # ── 结论 ────────────────────────────────────────────────
    best = max(results, key=lambda x: x["auc"])
    print(f"\n最优配置: {best['tag']}  "
          f"(max_len={best['max_len']}, stride={best['stride']}, "
          f"coverage={best['coverage_bp']}bp, "
          f"full_cov={best['coverage_pct']:.1f}%)")
    print(f"最优 AUC: {best['auc']:.4f}  vs 基准: 0.8941")

if __name__ == '__main__':
    main()
