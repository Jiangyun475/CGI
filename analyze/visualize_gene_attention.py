#!/usr/bin/env python3
"""
visualize_gene_attention.py
============================
基因注意力可视化 + 简单 GO 富集分析

从 extract_representations.py 的输出（gene_attn_mean）出发：
  - 找到每个模式中注意力权重最高的基因
  - 可视化各细胞系的基因注意力差异
  - 调用 gseapy 做 GO 富集（可选）

注意：gene_attn_mean [N, r] 是每个样本在每个模式下的
      基因注意力均值（已在提取时对 L' 求均值）。
      要做完整的 GO 分析，需要保存 gene_name 信息。

用法:
  python analyze/visualize_gene_attention.py \
    --cache_dirs analyze/cache/MCF7 analyze/cache/A375 \
    --cell_lines MCF7 A375 \
    --output_dir analyze/figures/gene_attention \
    [--gene_metadata /path/to/gene_metadata.csv]

gene_metadata.csv 格式（可选）:
  fold_idx,sample_idx,gene_name,gene_id
  0,0,BRCA1,672
  ...

如果没有，程序只输出基因索引，不做 GO 富集。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

CELL_COLORS = {
    'MCF7':  '#E8735A',
    'A375':  '#5B9BD5',
    'A549':  '#4CAF50',
    'VCAP':  '#9C59B6',
}


def load_cell_cache(cache_dir):
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / 'representations.npz')
    summary = json.load(open(cache_dir / 'summary.json'))
    return {
        'gene_attn_mean': data['gene_attn_mean'],  # [N, r]
        'labels': data['labels'],
        'spectrum': data['spectrum'],
    }, summary


def fig_attn_heatmap_by_cell(ax, attn_dict, cell_lines, mode_idx, top_n=50):
    """
    按细胞系画基因注意力热图（top_n 基因索引 × 细胞系）
    attn_dict: {cell_line: mean_attn_per_gene [n_val_samples, r]}
    """
    # 对每个细胞系，计算每个样本的基因注意力均值（mode_idx维度）
    cell_attn = {}
    for cl, attn in attn_dict.items():
        # attn [N, r] → [N] for mode k
        cell_attn[cl] = attn[:, mode_idx]

    # 这里 gene_attn_mean 是每个样本的均值标量，不是 per-gene 向量
    # 用均值代表"该样本在该模式下的基因注意力强度"
    # 真正的 per-gene 分析需要保存完整的 [N, r, L'] 数据（太大）
    means = [cell_attn[cl].mean() for cl in cell_lines]
    sems = [cell_attn[cl].std() / np.sqrt(len(cell_attn[cl])) for cl in cell_lines]

    colors = [CELL_COLORS.get(cl, '#888888') for cl in cell_lines]
    ax.bar(range(len(cell_lines)), means, yerr=sems, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(cell_lines)))
    ax.set_xticklabels(cell_lines, rotation=30, ha='right')
    ax.set_ylabel(f'Mean gene attention (Mode {mode_idx})')
    ax.set_title(f'Mode {mode_idx}: Cell-line Specificity')


def fig_attn_pos_neg(ax, gene_attn_mean, labels, mode_idx, cell_line):
    """正/负样本的基因注意力分布"""
    attn_k = gene_attn_mean[:, mode_idx]
    pos_mask = labels == 1
    neg_mask = labels == 0

    ax.violinplot([attn_k[pos_mask], attn_k[neg_mask]],
                  positions=[0, 1], showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_ylabel('Gene attention (mean)')
    ax.set_title(f'{cell_line} Mode {mode_idx}')

    _, pval = stats.mannwhitneyu(attn_k[pos_mask], attn_k[neg_mask])
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    ymax = max(attn_k.max(), attn_k[neg_mask].max())
    ax.text(0.5, 0.95, sig, transform=ax.transAxes, ha='center', fontsize=12)


def fig_cross_cell_comparison(axes_row, attn_dict_list, cell_lines, r):
    """各细胞系在每个模式上的注意力强度对比"""
    for k in range(r):
        ax = axes_row[k]
        pos_means = []
        neg_means = []
        for (attn, labels), cl in zip(attn_dict_list, cell_lines):
            attn_k = attn[:, k]
            pos_means.append(attn_k[labels == 1].mean())
            neg_means.append(attn_k[labels == 0].mean())

        x = np.arange(len(cell_lines))
        w = 0.35
        ax.bar(x - w/2, pos_means, w, color='#E8735A', alpha=0.8, label='Pos')
        ax.bar(x + w/2, neg_means, w, color='#5B9BD5', alpha=0.8, label='Neg')
        ax.set_xticks(x)
        ax.set_xticklabels(cell_lines, rotation=30, ha='right', fontsize=8)
        ax.set_title(f'Mode {k}', fontsize=9)
        if k == 0:
            ax.set_ylabel('Gene attention', fontsize=9)
        if k == r - 1:
            ax.legend(fontsize=7)


def simple_go_enrichment(gene_names, output_dir, title=''):
    """
    简单 GO 富集（需要 gseapy）
    gene_names: list of gene symbols
    """
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_names,
            gene_sets=['GO_Biological_Process_2023', 'KEGG_2021_Human'],
            organism='Human',
            outdir=str(output_dir / 'go_enrichment'),
            no_plot=False,
            cutoff=0.05,
        )
        print(f"  GO 富集完成，结果保存到: {output_dir / 'go_enrichment'}")
        # Top 10 GO terms
        if enr is not None and hasattr(enr, 'results'):
            top10 = enr.results.head(10)
            print("\n  Top 10 GO terms:")
            print(top10[['Term', 'Adjusted P-value', 'Overlap']].to_string())
    except ImportError:
        print("  ⚠️ gseapy 未安装，跳过 GO 富集")
        print("     安装命令：pip install gseapy")
    except Exception as e:
        print(f"  ⚠️ GO 富集失败: {e}")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载多细胞系数据
    cell_data = []
    cell_lines = args.cell_lines
    r = None
    for cache_dir, cl in zip(args.cache_dirs, cell_lines):
        print(f"  加载 {cl}: {cache_dir}")
        cache, summary = load_cell_cache(cache_dir)
        cell_data.append((cache['gene_attn_mean'], cache['labels']))
        if r is None:
            r = cache['gene_attn_mean'].shape[1]

    print(f"\n  细胞系: {cell_lines}, r={r}")

    # ─── 图1：各细胞系 × 各模式的注意力强度对比 ─────────────────
    n_cells = len(cell_lines)
    fig, axes = plt.subplots(2, r, figsize=(r * 2.5, 8))
    if r == 1:
        axes = axes.reshape(2, 1)

    # 上行：各模式的正/负样本对比（第一个细胞系）
    attn0, labels0 = cell_data[0]
    for k in range(r):
        fig_attn_pos_neg(axes[0, k], attn0, labels0, k, cell_lines[0])

    # 下行：多细胞系对比
    fig_cross_cell_comparison(axes[1, :], cell_data, cell_lines, r)

    fig.suptitle('Gene Attention Analysis\n(gene_attn_mean per mode)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'gene_attention_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'gene_attention_analysis.pdf', bbox_inches='tight')
    plt.close()
    print(f"\n✅ 基因注意力分析: {out}")

    # ─── 图2：跨细胞系热图 ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(r * 1.5 + 2, n_cells + 1))
    # 构建矩阵：[n_cells, r]，值 = 正样本均值
    matrix = np.zeros((n_cells, r))
    for i, (attn, labels) in enumerate(cell_data):
        pos_mask = labels == 1
        matrix[i] = attn[pos_mask].mean(axis=0)

    # 列归一化（每个模式内，各细胞系归一化）
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    matrix_norm = (matrix - col_min) / (col_max - col_min + 1e-8)

    im = ax2.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(r))
    ax2.set_xticklabels([f'M{k}' for k in range(r)])
    ax2.set_yticks(range(n_cells))
    ax2.set_yticklabels(cell_lines)
    plt.colorbar(im, ax=ax2, label='Normalized gene attention (pos. samples)')
    ax2.set_title('Cell-line Specificity Heatmap\n(each column normalized)',
                  fontweight='bold')

    for i in range(n_cells):
        for k in range(r):
            ax2.text(k, i, f'{matrix[i, k]:.3f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    out2 = output_dir / 'cell_line_specificity_heatmap.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cell_line_specificity_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  细胞系特异性热图: {out2}")

    # ─── 打印统计摘要 ─────────────────────────────────────────────
    print(f"\n📈 基因注意力统计摘要")
    print(f"  {'Cell Line':<10}", end='')
    for k in range(r):
        print(f"  M{k}(pos/neg)", end='')
    print()
    for (attn, labels), cl in zip(cell_data, cell_lines):
        print(f"  {cl:<10}", end='')
        pos_mask = labels == 1
        neg_mask = labels == 0
        for k in range(r):
            mp = attn[pos_mask, k].mean()
            mn = attn[neg_mask, k].mean()
            print(f"  {mp:.3f}/{mn:.3f}", end='')
        print()

    # ─── GO 富集（可选，需要 gene_metadata.csv）────────────────
    if args.gene_metadata:
        print(f"\n📚 尝试 GO 富集分析...")
        import pandas as pd
        meta = pd.read_csv(args.gene_metadata)
        # 找每个模式注意力最高的 top 100 基因名
        for k in range(r):
            # 按谱均值找 top 基因
            attn0, labels0 = cell_data[0]
            top_idx = np.argsort(attn0[:, k])[-200:]
            # 需要从 metadata 提取这些样本的基因名
            # （具体实现依赖数据格式）
            print(f"  Mode {k}: top gene analysis requires per-gene metadata")
        print("  ⚠️ 请确认 gene_metadata.csv 包含 gene_name 列")
    else:
        print(f"\n  💡 要做 GO 富集，需提供 --gene_metadata 参数（gene_name 列）")
        print(f"     示例：--gene_metadata /path/to/MCF7/gene_names_fold0_val.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dirs',    type=str, nargs='+', required=True,
                        help='extract_representations.py 的输出目录列表')
    parser.add_argument('--cell_lines',    type=str, nargs='+', required=True,
                        help='对应的细胞系名称（与 cache_dirs 一一对应）')
    parser.add_argument('--output_dir',    type=str, required=True)
    parser.add_argument('--gene_metadata', type=str, default=None,
                        help='（可选）包含 gene_name 列的 CSV，用于 GO 富集')
    args = parser.parse_args()
    run(args)
