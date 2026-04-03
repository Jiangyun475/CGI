#!/usr/bin/env python3
"""
交互谱分析与可视化 (Interaction Spectrum Analysis)
==================================================
用于生成论文中的可解释性图表:

1. 模式激活热力图: 哪些模式对哪些药物-基因对最活跃
2. 模式稀疏性分布: σ 的分布，验证少数模式主导假设
3. 正/负样本的谱差异: 显著互作 vs 无互作的谱模式区别
4. 消融对比汇总表: 4 种交互方式 × 5 折的 AUC/PRC/F1

用法:
  python analyze_spectrum.py --results_dir results_operator/VCAP
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch


def load_spectrum_files(results_dir):
    """加载所有 spectrum_*.pt 文件"""
    spectra = {}
    for f in sorted(Path(results_dir).glob('spectrum_*.pt')):
        data = torch.load(f, map_location='cpu')
        name = f.stem.replace('spectrum_', '')
        spectra[name] = data
        print(f"  加载: {name} | spectrum shape: {data['spectrum'].shape}")
    return spectra


def plot_spectrum_analysis(spectra, save_dir):
    """生成交互谱分析图"""

    # --- 找到 operator 模型的谱 ---
    op_keys = [k for k in spectra if k.startswith('operator')]
    if not op_keys:
        print("⚠️ 未找到 operator 模型的谱数据")
        return

    key = op_keys[0]
    data = spectra[key]
    S = data['spectrum'].numpy()       # [N_val, r]
    labels = data['labels']             # [N_val]
    preds = data['preds']               # [N_val]

    r = S.shape[1]
    pos_mask = labels == 1
    neg_mask = labels == 0

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ---- 1. 模式幅度分布 (|s_k| by mode) ----
    ax1 = fig.add_subplot(gs[0, 0])
    abs_S = np.abs(S)
    mode_means = abs_S.mean(axis=0)
    mode_stds  = abs_S.std(axis=0)
    ax1.bar(range(r), mode_means, yerr=mode_stds, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Mode Index k')
    ax1.set_ylabel('Mean |spectrum_k|')
    ax1.set_title('(a) Mode Activation Magnitude')
    ax1.set_xticks(range(r))

    # ---- 2. 正/负样本的谱差异 ----
    ax2 = fig.add_subplot(gs[0, 1])
    pos_means = abs_S[pos_mask].mean(axis=0)
    neg_means = abs_S[neg_mask].mean(axis=0)
    x = np.arange(r)
    w = 0.35
    ax2.bar(x - w/2, pos_means, w, label='Positive (|Z|>2)', alpha=0.8, color='coral')
    ax2.bar(x + w/2, neg_means, w, label='Negative', alpha=0.8, color='skyblue')
    ax2.set_xlabel('Mode Index k')
    ax2.set_ylabel('Mean |spectrum_k|')
    ax2.set_title('(b) Spectrum: Positive vs Negative')
    ax2.set_xticks(x)
    ax2.legend(fontsize=9)

    # ---- 3. 模式稀疏性 (每样本有多少模式显著激活) ----
    ax3 = fig.add_subplot(gs[0, 2])
    threshold = abs_S.mean() + abs_S.std()
    active_counts = (abs_S > threshold).sum(axis=1)
    ax3.hist(active_counts, bins=range(r+2), alpha=0.8, color='seagreen', edgecolor='white')
    ax3.set_xlabel('Number of Active Modes (|s_k| > μ+σ)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'(c) Mode Sparsity Distribution (threshold={threshold:.3f})')
    ax3.axvline(active_counts.mean(), color='red', linestyle='--',
                label=f'Mean={active_counts.mean():.1f}')
    ax3.legend()

    # ---- 4. 谱的 t-SNE (如果样本量合理) ----
    ax4 = fig.add_subplot(gs[1, 0])
    n_plot = min(3000, S.shape[0])
    idx = np.random.choice(S.shape[0], n_plot, replace=False)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        S_2d = tsne.fit_transform(S[idx])
        colors = labels[idx]
        scatter = ax4.scatter(S_2d[:, 0], S_2d[:, 1], c=colors,
                              cmap='coolwarm', alpha=0.3, s=5)
        ax4.set_title('(d) t-SNE of Interaction Spectrum')
        plt.colorbar(scatter, ax=ax4, label='Label')
    except Exception as e:
        ax4.text(0.5, 0.5, f't-SNE failed:\n{e}', ha='center', va='center',
                 transform=ax4.transAxes)

    # ---- 5. 模式间相关矩阵 ----
    ax5 = fig.add_subplot(gs[1, 1])
    corr = np.corrcoef(S.T)  # [r, r]
    im = ax5.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_title('(e) Mode Correlation Matrix')
    ax5.set_xlabel('Mode k')
    ax5.set_ylabel('Mode k')
    ax5.set_xticks(range(r))
    ax5.set_yticks(range(r))
    plt.colorbar(im, ax=ax5)

    # ---- 6. 预测分数与谱范数的关系 ----
    ax6 = fig.add_subplot(gs[1, 2])
    spec_norms = np.linalg.norm(S, axis=1)  # L2 norm of spectrum
    ax6.scatter(spec_norms[neg_mask], preds[neg_mask], alpha=0.2, s=3,
                label='Negative', color='skyblue')
    ax6.scatter(spec_norms[pos_mask], preds[pos_mask], alpha=0.2, s=3,
                label='Positive', color='coral')
    ax6.set_xlabel('||Spectrum||₂')
    ax6.set_ylabel('Predicted Probability')
    ax6.set_title('(f) Spectrum Norm vs Prediction')
    ax6.legend(fontsize=9)

    fig.suptitle(f'Interaction Spectrum Analysis — {key}', fontsize=14, y=0.98)
    out_path = save_dir / 'spectrum_analysis.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 谱分析图已保存: {out_path}")


def collect_ablation_results(results_dir):
    """
    从模型文件名和训练日志收集消融对比结果。
    注意: 这里只检查模型文件是否存在，真正的 metrics 需要加载模型重新评估。
    """
    results_dir = Path(results_dir)
    model_files = sorted(results_dir.glob('*.pt'))
    model_files = [f for f in model_files if not f.stem.startswith('spectrum')]

    summary = {}
    for f in model_files:
        name = f.stem
        # 从文件名解析: {interaction_type}_r{rank}_Fold{fold}_{tag}
        parts = name.split('_')
        itype = parts[0]
        if itype == 'ortho':
            itype = 'ortho_concat'
            parts = parts[1:]  # skip 'concat' part
        summary.setdefault(itype, []).append(name)

    print("\n" + "=" * 60)
    print("  消融对比模型汇总")
    print("=" * 60)
    for itype, files in sorted(summary.items()):
        print(f"\n  [{itype}] ({len(files)} 个模型)")
        for f in files:
            print(f"    - {f}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果目录，如 results_operator/VCAP')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    assert results_dir.exists(), f"目录不存在: {results_dir}"

    print(f"\n📊 分析目录: {results_dir}")

    # 1. 汇总消融模型
    collect_ablation_results(results_dir)

    # 2. 分析交互谱
    print(f"\n📈 加载交互谱...")
    spectra = load_spectrum_files(results_dir)
    if spectra:
        plot_spectrum_analysis(spectra, results_dir)
    else:
        print("  未找到谱数据，跳过可视化")

    print("\n✅ 分析完成")
