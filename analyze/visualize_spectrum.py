#!/usr/bin/env python3
"""
visualize_spectrum.py
=====================
交互谱（Interaction Spectrum）完整可视化

从 extract_representations.py 的输出中生成论文级别图表：
  Fig A: 各模式激活幅度（bar + error bar）
  Fig B: 正样本 vs 负样本的谱对比（grouped bar）
  Fig C: 谱稀疏性分布（每个样本激活几个模式）
  Fig D: 交互谱的 t-SNE（正负样本着色）
  Fig E: 模式间相关矩阵（验证正交性）
  Fig F: 谱范数 vs 预测概率（散点图）
  Fig G: 各模式激活的 Violin plot（最直观）

用法:
  python analyze/visualize_spectrum.py \
    --cache_dir analyze/cache/MCF7 \
    --output_dir analyze/figures/MCF7 \
    [--cell_line MCF7]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {
    'pos': '#E8735A',   # 暖红
    'neg': '#5B9BD5',   # 冷蓝
    'mode': '#4CAF50',  # 绿
    'neutral': '#9E9E9E',
}


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / 'representations.npz')
    summary = json.load(open(cache_dir / 'summary.json'))
    return {
        'spectrum': data['spectrum'],   # [N, r]
        'sigma':    data['sigma'],      # [N, r]
        'preds':    data['preds'],      # [N]
        'labels':   data['labels'],     # [N]
    }, summary


def fig_mode_magnitude(ax, spectrum, labels):
    """A: 各模式平均激活幅度"""
    abs_S = np.abs(spectrum)
    r = spectrum.shape[1]
    means = abs_S.mean(axis=0)
    sems = abs_S.std(axis=0) / np.sqrt(len(abs_S))

    bars = ax.bar(range(r), means, yerr=sems, color=COLORS['mode'],
                  alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mode index k', fontsize=11)
    ax.set_ylabel('Mean |spectrum_k|', fontsize=11)
    ax.set_title('(A) Mode Activation Magnitude', fontweight='bold')
    ax.set_xticks(range(r))
    ax.set_xticklabels([f'M{k}' for k in range(r)])

    # 标注主导模式
    top2 = np.argsort(means)[-2:]
    for i in top2:
        bars[i].set_color('#FF6B35')
        ax.text(i, means[i] + sems[i] + 0.005, '★', ha='center', fontsize=9, color='#FF6B35')


def fig_pos_neg_comparison(ax, spectrum, labels):
    """B: 正负样本谱对比 + 统计检验"""
    abs_S = np.abs(spectrum)
    r = spectrum.shape[1]
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_means = abs_S[pos_mask].mean(axis=0)
    neg_means = abs_S[neg_mask].mean(axis=0)
    pos_sems = abs_S[pos_mask].std(axis=0) / np.sqrt(pos_mask.sum())
    neg_sems = abs_S[neg_mask].std(axis=0) / np.sqrt(neg_mask.sum())

    x = np.arange(r)
    w = 0.35
    ax.bar(x - w/2, pos_means, w, yerr=pos_sems, label='Positive (CGI=1)',
           color=COLORS['pos'], alpha=0.85, capsize=3)
    ax.bar(x + w/2, neg_means, w, yerr=neg_sems, label='Negative (CGI=0)',
           color=COLORS['neg'], alpha=0.85, capsize=3)

    # 标注显著差异的模式（Mann-Whitney U test）
    for k in range(r):
        stat, pval = stats.mannwhitneyu(
            abs_S[pos_mask, k], abs_S[neg_mask, k], alternative='two-sided')
        if pval < 0.001:
            ymax = max(pos_means[k] + pos_sems[k], neg_means[k] + neg_sems[k])
            ax.text(k, ymax + 0.01, '***', ha='center', fontsize=9)
        elif pval < 0.01:
            ymax = max(pos_means[k] + pos_sems[k], neg_means[k] + neg_sems[k])
            ax.text(k, ymax + 0.01, '**', ha='center', fontsize=9)

    ax.set_xlabel('Mode index k', fontsize=11)
    ax.set_ylabel('Mean |spectrum_k|', fontsize=11)
    ax.set_title('(B) Positive vs Negative Spectrum', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{k}' for k in range(r)])
    ax.legend(fontsize=9, framealpha=0)


def fig_sparsity(ax, spectrum):
    """C: 每个样本激活的模式数（稀疏性）"""
    abs_S = np.abs(spectrum)
    threshold = abs_S.mean() + abs_S.std()
    active_counts = (abs_S > threshold).sum(axis=1)

    r = spectrum.shape[1]
    bins = range(r + 2)
    counts, edges = np.histogram(active_counts, bins=bins)
    pct = counts / counts.sum() * 100

    ax.bar(edges[:-1], pct, color=COLORS['mode'], alpha=0.85, edgecolor='white')
    ax.axvline(active_counts.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {active_counts.mean():.1f}')
    ax.set_xlabel(f'# Active Modes (|s_k| > μ+σ)', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'(C) Sparsity: Modes per Sample\n(threshold={threshold:.3f})', fontweight='bold')
    ax.set_xticks(range(r + 1))
    ax.legend(fontsize=9)


def fig_tsne(ax, spectrum, labels):
    """D: t-SNE 可视化（降采样）"""
    from sklearn.manifold import TSNE

    N = spectrum.shape[0]
    n_plot = min(5000, N)
    idx = np.random.choice(N, n_plot, replace=False)
    S_sub = spectrum[idx]
    l_sub = labels[idx]

    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    n_iter=500, learning_rate='auto', init='pca')
        S_2d = tsne.fit_transform(S_sub)

        for label, color, name in [(0, COLORS['neg'], 'Negative'),
                                   (1, COLORS['pos'], 'Positive')]:
            mask = l_sub == label
            ax.scatter(S_2d[mask, 0], S_2d[mask, 1],
                       c=color, alpha=0.3, s=4, label=name, rasterized=True)

        ax.set_title('(D) t-SNE of Interaction Spectrum', fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.legend(fontsize=9, markerscale=2, framealpha=0)
    except Exception as e:
        ax.text(0.5, 0.5, f't-SNE failed:\n{str(e)[:80]}',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('(D) t-SNE (Failed)')


def fig_correlation_matrix(ax, spectrum):
    """E: 模式间相关矩阵（验证正交性）"""
    r = spectrum.shape[1]
    corr = np.corrcoef(spectrum.T)  # [r, r]

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(r))
    ax.set_yticks(range(r))
    ax.set_xticklabels([f'M{k}' for k in range(r)], fontsize=9)
    ax.set_yticklabels([f'M{k}' for k in range(r)], fontsize=9)
    ax.set_title('(E) Mode Correlation Matrix\n(orthogonal → near-diagonal)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 标注非对角元素的最大绝对值
    off_diag = corr.copy()
    np.fill_diagonal(off_diag, 0)
    max_corr = np.abs(off_diag).max()
    ax.set_xlabel(f'Max off-diagonal |r| = {max_corr:.3f}', fontsize=10)


def fig_spectrum_norm_vs_pred(ax, spectrum, preds, labels):
    """F: 谱范数 vs 预测概率"""
    norms = np.linalg.norm(spectrum, axis=1)
    pos_mask = labels == 1
    neg_mask = labels == 0

    ax.scatter(norms[neg_mask], preds[neg_mask], c=COLORS['neg'],
               alpha=0.2, s=3, label='Negative', rasterized=True)
    ax.scatter(norms[pos_mask], preds[pos_mask], c=COLORS['pos'],
               alpha=0.2, s=3, label='Positive', rasterized=True)
    ax.set_xlabel('||Spectrum||₂', fontsize=11)
    ax.set_ylabel('Predicted P(CGI=1)', fontsize=11)
    ax.set_title('(F) Spectrum Norm vs Prediction', fontweight='bold')
    ax.legend(fontsize=9, markerscale=3, framealpha=0)

    # 相关系数
    r_pos, _ = stats.pearsonr(norms[pos_mask], preds[pos_mask])
    r_neg, _ = stats.pearsonr(norms[neg_mask], preds[neg_mask])
    ax.text(0.05, 0.95, f'r(pos)={r_pos:.3f}\nr(neg)={r_neg:.3f}',
            transform=ax.transAxes, fontsize=9, va='top')


def fig_violin(ax, spectrum, labels):
    """G: Violin plot — 每个模式的正负样本分布"""
    abs_S = np.abs(spectrum)
    r = spectrum.shape[1]
    pos_mask = labels == 1
    neg_mask = labels == 0

    positions_pos = np.arange(r) * 2.5
    positions_neg = positions_pos + 1.0

    vp_pos = ax.violinplot([abs_S[pos_mask, k] for k in range(r)],
                           positions=positions_pos, widths=0.8,
                           showmedians=True, showextrema=False)
    vp_neg = ax.violinplot([abs_S[neg_mask, k] for k in range(r)],
                           positions=positions_neg, widths=0.8,
                           showmedians=True, showextrema=False)

    for body in vp_pos['bodies']:
        body.set_facecolor(COLORS['pos'])
        body.set_alpha(0.7)
    for body in vp_neg['bodies']:
        body.set_facecolor(COLORS['neg'])
        body.set_alpha(0.7)
    vp_pos['cmedians'].set_color('darkred')
    vp_neg['cmedians'].set_color('darkblue')

    center_pos = (positions_pos + positions_neg) / 2
    ax.set_xticks(center_pos)
    ax.set_xticklabels([f'M{k}' for k in range(r)])
    ax.set_xlabel('Mode index k', fontsize=11)
    ax.set_ylabel('|spectrum_k|', fontsize=11)
    ax.set_title('(G) Per-Mode Distribution: Positive vs Negative', fontweight='bold')
    ax.legend(handles=[
        Patch(color=COLORS['pos'], label='Positive'),
        Patch(color=COLORS['neg'], label='Negative'),
    ], fontsize=9, framealpha=0)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 加载缓存: {args.cache_dir}")
    cache, summary = load_cache(args.cache_dir)
    spectrum = cache['spectrum']   # [N, r]
    labels   = cache['labels']
    preds    = cache['preds']
    cell_line = args.cell_line or summary.get('cell_line', 'Unknown')
    auc       = summary.get('auc', 0.0)

    print(f"  细胞系: {cell_line}, AUC={auc:.4f}")
    print(f"  N={len(labels)}, pos={labels.sum():.0f}, neg={(1-labels).sum():.0f}")
    print(f"  spectrum shape: {spectrum.shape}")

    # ─── 主图：7个子图 ───────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])
    ax_G = fig.add_subplot(gs[2, :])

    fig_mode_magnitude(ax_A, spectrum, labels)
    fig_pos_neg_comparison(ax_B, spectrum, labels)
    fig_sparsity(ax_C, spectrum)
    fig_tsne(ax_D, spectrum, labels)
    fig_correlation_matrix(ax_E, spectrum)
    fig_spectrum_norm_vs_pred(ax_F, spectrum, preds, labels)
    fig_violin(ax_G, spectrum, labels)

    fig.suptitle(
        f'DrugOperatorNet — Interaction Spectrum Analysis\n{cell_line} | AUC={auc:.4f} | N={len(labels)}',
        fontsize=14, fontweight='bold', y=0.98)

    out_png = output_dir / f'spectrum_analysis_{cell_line}.png'
    out_pdf = output_dir / f'spectrum_analysis_{cell_line}.pdf'
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 保存: {out_png}")

    # ─── 单独保存每个子图（投稿用）─────────────────────────────
    single_dir = output_dir / 'single_panels'
    single_dir.mkdir(exist_ok=True)

    panels = [
        ('A_mode_magnitude', fig_mode_magnitude),
        ('B_pos_neg', fig_pos_neg_comparison),
        ('C_sparsity', fig_sparsity),
        ('E_correlation', fig_correlation_matrix),
    ]
    for name, func in panels:
        fig_s, ax_s = plt.subplots(figsize=(6, 5))
        if name == 'A_mode_magnitude':
            func(ax_s, spectrum, labels)
        elif name == 'B_pos_neg':
            func(ax_s, spectrum, labels)
        elif name == 'C_sparsity':
            func(ax_s, spectrum)
        elif name == 'E_correlation':
            func(ax_s, spectrum)
        plt.tight_layout()
        plt.savefig(single_dir / f'{name}_{cell_line}.pdf', bbox_inches='tight')
        plt.close()

    print(f"  单图保存到: {single_dir}/")

    # ─── 打印统计摘要 ─────────────────────────────────────────────
    abs_S = np.abs(spectrum)
    pos_mask = labels == 1
    neg_mask = labels == 0
    print(f"\n📈 统计摘要 ({cell_line})")
    print(f"  {'Mode':<8} {'Mean(pos)':>10} {'Mean(neg)':>10} {'Δ':>8} {'p-val':>10}")
    print(f"  {'-'*50}")
    for k in range(spectrum.shape[1]):
        m_pos = abs_S[pos_mask, k].mean()
        m_neg = abs_S[neg_mask, k].mean()
        _, pval = stats.mannwhitneyu(abs_S[pos_mask, k], abs_S[neg_mask, k])
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        print(f"  M{k:<7} {m_pos:>10.4f} {m_neg:>10.4f} {m_pos-m_neg:>8.4f} {sig:>10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',   type=str, required=True,
                        help='extract_representations.py 的输出目录')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='图像输出目录')
    parser.add_argument('--cell_line',   type=str, default=None,
                        help='细胞系名称（用于图题，可自动从 summary.json 读取）')
    args = parser.parse_args()
    run(args)
