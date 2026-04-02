#!/usr/bin/env python3
"""
visualize_pharmacophore.py
==========================
药效团热图（Pharmacophore Heatmap）：在分子结构图上，
用颜色标注每个原子对每个交互模式的贡献权重。

依赖：rdkit（conda install -c rdkit rdkit）

用法:
  python analyze/visualize_pharmacophore.py \
    --cache_dir analyze/cache/MCF7 \
    --output_dir analyze/figures/MCF7/pharmacophore \
    [--n_examples 20] \
    [--mode_idx 0 1 2]   # 只画指定模式；默认画所有

输出：
  output_dir/
    mol_{i}_mode{k}.png    每个示例分子在每个模式上的热图
    top_mols_mode{k}.png   每个模式的 top-k 最高激活分子
    summary_grid.png       所有模式 × top 分子的汇总图
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("⚠️ RDKit 未安装，将跳过分子结构图，只输出注意力权重统计")

from io import BytesIO
from PIL import Image


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / 'representations.npz')
    smiles = np.load(cache_dir / 'smiles.npy', allow_pickle=True)
    atom_attn = np.load(cache_dir / 'atom_attn.npy', allow_pickle=True)  # object array
    summary = json.load(open(cache_dir / 'summary.json'))
    return {
        'spectrum': data['spectrum'],
        'labels':   data['labels'],
        'preds':    data['preds'],
        'smiles':   smiles,
        'atom_attn': atom_attn,   # [N] of [n_atoms_i, r]
    }, summary


def draw_mol_with_atom_weights(smiles, atom_weights, mode_idx, title='',
                                size=(400, 300)):
    """
    在分子结构图上，根据 atom_weights 用颜色标注每个原子。
    atom_weights: [n_atoms]，值域 [0, 1]（已 softmax 过）
    返回 PIL Image
    """
    if not HAS_RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)

    # 颜色映射：低→蓝，高→红
    cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
    n_atoms = mol.GetNumAtoms()

    # 如果 atom_weights 长度和分子原子数不匹配（常见于 H 被省略）
    if len(atom_weights) != n_atoms:
        # 截断或填充
        if len(atom_weights) > n_atoms:
            atom_weights = atom_weights[:n_atoms]
        else:
            atom_weights = np.concatenate([
                atom_weights, np.zeros(n_atoms - len(atom_weights))])

    # 归一化到 [0, 1]
    w = atom_weights.copy()
    if w.max() > w.min():
        w = (w - w.min()) / (w.max() - w.min())

    atom_colors = {}
    atom_radii = {}
    highlight_atoms = list(range(n_atoms))
    for i in range(n_atoms):
        rgba = cmap(float(w[i]))
        atom_colors[i] = (rgba[0], rgba[1], rgba[2])
        # 高权重原子画大一些
        atom_radii[i] = 0.3 + 0.4 * float(w[i])

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # SVG → PIL Image
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=size[0], output_height=size[1])
        img = Image.open(BytesIO(png_bytes))
    except ImportError:
        # 如果没有 cairosvg，用 rdkit 的 PNG 接口
        drawer2 = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer2.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
        )
        drawer2.FinishDrawing()
        img = Image.open(BytesIO(drawer2.GetDrawingText()))

    return img


def draw_fallback_bar(smiles, atom_weights, mode_idx, ax, title=''):
    """当无法绘制分子结构时，用 bar chart 显示原子权重"""
    n_atoms = len(atom_weights)
    w = atom_weights.copy()
    if w.max() > w.min():
        w = (w - w.min()) / (w.max() - w.min())

    colors = plt.cm.RdYlBu_r(w)
    ax.bar(range(n_atoms), atom_weights, color=colors, edgecolor='none')
    ax.set_xlabel('Atom index')
    ax.set_ylabel('Attention weight')
    ax.set_title(f'{title}\nMode {mode_idx}', fontsize=8)
    # 显示 SMILES（截短）
    ax.text(0.5, -0.25, smiles[:50] + ('...' if len(smiles) > 50 else ''),
            transform=ax.transAxes, ha='center', fontsize=6)


def plot_top_mols_for_mode(smiles_list, atom_attn_list, spectrum, labels,
                            mode_idx, output_dir, n_top=8):
    """
    对指定模式 mode_idx，找出正/负样本中谱值绝对值最大的 n_top 个分子，
    并在分子图上标注原子注意力。
    """
    abs_spec_k = np.abs(spectrum[:, mode_idx])
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Top positive / negative
    pos_top = np.argsort(abs_spec_k * pos_mask)[-n_top:][::-1]
    neg_top = np.argsort(abs_spec_k * neg_mask)[-n_top:][::-1]

    fig, axes = plt.subplots(2, n_top, figsize=(n_top * 3.5, 7))
    fig.suptitle(f'Top-{n_top} Molecules by |spectrum| — Mode {mode_idx}',
                 fontsize=13, fontweight='bold')

    for row, (indices, row_label) in enumerate([(pos_top, 'Positive'), (neg_top, 'Negative')]):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            smi = smiles_list[idx]
            attn = atom_attn_list[idx][:, mode_idx]   # [n_atoms]
            spec_val = spectrum[idx, mode_idx]

            if HAS_RDKIT:
                img = draw_mol_with_atom_weights(smi, attn, mode_idx,
                                                  size=(320, 240))
                if img is not None:
                    ax.imshow(img)
                    ax.set_title(f'{row_label}\nspec={spec_val:.3f}', fontsize=7)
                    ax.axis('off')
                    continue

            # Fallback: bar chart
            draw_fallback_bar(smi, attn, mode_idx, ax,
                              title=f'{row_label} spec={spec_val:.3f}')

    plt.tight_layout()
    out_path = output_dir / f'top_mols_mode{mode_idx}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Mode {mode_idx} 热图: {out_path}")
    return out_path


def plot_atom_attn_stats(atom_attn_list, labels, output_dir):
    """
    统计每个模式中，不同原子类型的平均注意力权重。
    （不依赖 RDKit 分子结构图）
    """
    # 统计每个分子中，注意力最高的前3个原子的索引分布
    r = atom_attn_list[0].shape[1]
    pos_mask = labels == 1
    neg_mask = labels == 0

    fig, axes = plt.subplots(1, r, figsize=(r * 2.5, 4))
    if r == 1:
        axes = [axes]

    for k in range(r):
        ax = axes[k]
        pos_attn_maxs = []
        neg_attn_maxs = []
        for i, attn in enumerate(atom_attn_list):
            max_w = attn[:, k].max()
            if labels[i] == 1:
                pos_attn_maxs.append(max_w)
            else:
                neg_attn_maxs.append(max_w)

        ax.violinplot([pos_attn_maxs, neg_attn_maxs], positions=[0, 1],
                      showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pos', 'Neg'], fontsize=8)
        ax.set_title(f'M{k}', fontsize=9)
        if k == 0:
            ax.set_ylabel('Max atom attention', fontsize=9)

    fig.suptitle('Max Atom Attention per Mode (Positive vs Negative)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'atom_attn_violin.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  原子注意力分布: {out}")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔬 加载缓存: {args.cache_dir}")
    cache, summary = load_cache(args.cache_dir)
    spectrum    = cache['spectrum']
    labels      = cache['labels']
    smiles_list = list(cache['smiles'])
    atom_attn   = list(cache['atom_attn'])   # list of [n_atoms, r]
    cell_line   = args.cell_line or summary.get('cell_line', 'Unknown')
    r           = spectrum.shape[1]

    print(f"  细胞系: {cell_line}, N={len(labels)}, r={r}")
    if not HAS_RDKIT:
        print("  ⚠️ 无 RDKit，只输出统计图（安装命令：conda install -c rdkit rdkit）")

    # 选择要可视化的模式
    mode_indices = args.mode_idx if args.mode_idx else list(range(r))

    # 1. 各模式 top 分子热图
    print(f"\n[1/2] 生成各模式 top 分子热图...")
    for k in mode_indices:
        plot_top_mols_for_mode(
            smiles_list, atom_attn, spectrum, labels,
            mode_idx=k, output_dir=output_dir, n_top=args.n_top)

    # 2. 原子注意力分布统计（不依赖 RDKit）
    print(f"\n[2/2] 生成原子注意力统计...")
    plot_atom_attn_stats(atom_attn, labels, output_dir)

    # 3. 生成汇总网格图（所有模式 × top 3 分子）
    print(f"\n[3/3] 生成汇总网格图...")
    n_modes = len(mode_indices)
    n_per_mode = min(4, args.n_top)
    fig, axes = plt.subplots(n_modes, n_per_mode,
                              figsize=(n_per_mode * 3.5, n_modes * 3.5))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for row, k in enumerate(mode_indices):
        abs_spec_k = np.abs(spectrum[:, k]) * (labels == 1)
        top_idx = np.argsort(abs_spec_k)[-n_per_mode:][::-1]
        for col, idx in enumerate(top_idx):
            ax = axes[row, col]
            smi = smiles_list[idx]
            attn_k = atom_attn[idx][:, k]
            spec_val = spectrum[idx, k]

            if HAS_RDKIT:
                img = draw_mol_with_atom_weights(smi, attn_k, k, size=(280, 200))
                if img is not None:
                    ax.imshow(img)
                    ax.axis('off')
                    if col == 0:
                        ax.set_ylabel(f'Mode {k}', fontsize=9, rotation=90, labelpad=10)
                    ax.set_title(f'spec={spec_val:.3f}', fontsize=7)
                    continue
            # Fallback
            draw_fallback_bar(smi, attn_k, k, ax, title=f'M{k} s={spec_val:.2f}')

    fig.suptitle(f'Pharmacophore Heatmap — {cell_line}\n(Top pos. samples per mode)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = output_dir / f'summary_grid_{cell_line}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 汇总图: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',   type=str, required=True)
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--cell_line',   type=str, default=None)
    parser.add_argument('--n_top',       type=int, default=8,
                        help='每个模式展示的 top 分子数')
    parser.add_argument('--mode_idx',    type=int, nargs='+', default=None,
                        help='只画指定模式（默认全部）')
    args = parser.parse_args()
    run(args)
