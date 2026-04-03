#!/usr/bin/env python3
"""
双模型注意力热力图对比脚本
=============================
对同一活性分子：
  - Full 模型 (hybrid pool)   → 注意力受 sum/mean 稀释，梯度压力弱
  - TargetOnly 模型 (target)  → 注意力是唯一池化机制，梯度压力强

热力图诊断：
  - 大面积泛红 → over-smoothing 或 softmax 均摊
  - 锁定非活性基团 → 快捷特征（shortcut）
  - 熵比 H/H_uniform 接近 1 → 均匀分布；接近 0 → 强聚焦
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

# 将项目根目录加入 sys.path，使 analysis/ 子目录下也能导入根目录模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_ultimate import PaperModel, encode_kmer_sequence


# ─────────────────────────────────────────────────────────────────
# 从模型文件名自动解析配置
# ─────────────────────────────────────────────────────────────────
def infer_config(model_path: str):
    name = Path(model_path).stem
    # pool_type 从已知合法值中搜索，而非取最后一段（文件名可能带 tag）
    pool_type = 'hybrid'  # 默认
    for pt in ['sum_mean', 'target', 'hybrid']:
        if f'_{pt}' in name or f'_{pt}_' in name:
            pool_type = pt
            break
    use_ortho = "orthoTrue" in name
    return pool_type, use_ortho


# ─────────────────────────────────────────────────────────────────
# 单模型推理：返回 (alpha_np, pred_prob)
# ─────────────────────────────────────────────────────────────────
def run_model(model_path, gene_ids, x, edge_index, edge_attr, num_nodes_list, device):
    pool_type, use_ortho = infer_config(model_path)
    model = PaperModel(hidden_dim=128, dropout=0.0,
                       pool_type=pool_type, use_ortho=use_ortho)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device).eval()

    with torch.no_grad():
        logits, _, _, _, alpha = model(
            gene_ids, x, edge_index, edge_attr, num_nodes_list
        )
    pred_prob = torch.sigmoid(logits).item()
    alpha_np  = alpha.squeeze().cpu().numpy()   # [N_atoms]
    return alpha_np, pred_prob, pool_type


# ─────────────────────────────────────────────────────────────────
# 注意力分布诊断指标
# ─────────────────────────────────────────────────────────────────
def attention_diagnostics(alpha, label=""):
    n = len(alpha)
    entropy     = -np.sum(alpha * np.log(alpha + 1e-12))
    uniform_ent = np.log(n)
    ratio       = entropy / uniform_ent
    max_w       = alpha.max()
    top3_idx    = np.argsort(alpha)[::-1][:3]

    print(f"\n  [{label}]")
    print(f"    原子数 N = {n}")
    print(f"    均匀权重 1/N = {1/n:.4f}")
    print(f"    最大权重 max(α) = {max_w:.4f}  (均匀时 = {1/n:.4f})")
    print(f"    注意力熵 H = {entropy:.4f}  /  均匀熵 log(N) = {uniform_ent:.4f}")
    print(f"    聚焦度 1 - H/H_uniform = {1-ratio:.4f}  (0=均匀, 1=单点聚焦)")
    print(f"    Top-3 原子索引: {top3_idx.tolist()}  权重: {alpha[top3_idx].round(4).tolist()}")

    if ratio > 0.90:
        print(f"    ⚠️  诊断: 注意力近乎均匀 → 可能存在 over-smoothing 或 softmax 稀释")
    elif ratio < 0.50:
        print(f"    ✅  诊断: 注意力高度聚焦")
    else:
        print(f"    ℹ️  诊断: 注意力中等聚焦")
    return ratio


# ─────────────────────────────────────────────────────────────────
# 主对比函数
# ─────────────────────────────────────────────────────────────────
def compare_two_models(smiles, gene_seq, smiles_to_graph_dict, model_configs,
                       out_file="attention_comparison.png", device_str="cpu"):
    """
    model_configs: list of (label, model_path)
    """
    device = torch.device(device_str)

    # 准备输入
    gene_ids = torch.tensor([encode_kmer_sequence(gene_seq)]).to(device)
    # 从预计算的 smiles_to_graph 字典中取图特征（与训练完全一致）
    graph = smiles_to_graph_dict.get(smiles)
    if graph is None:
        raise ValueError(f"SMILES 在 smiles_to_graph 中不存在: {smiles}")

    x              = graph['x'].to(device)
    edge_index     = graph['edge_index'].to(device)
    edge_attr      = graph['edge_attr'].to(device)
    num_nodes_list = [x.shape[0]]

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    n_atoms = mol.GetNumAtoms()

    print(f"\nSMILES : {smiles}")
    print(f"原子数 : {n_atoms}")
    print(f"基因长度: {len(gene_seq)} bp")
    print("=" * 60)

    # ── 推理 ────────────────────────────────────────────────────
    results = []
    for label, model_path in model_configs:
        alpha, prob, pool_type = run_model(
            model_path, gene_ids, x, edge_index, edge_attr, num_nodes_list, device
        )
        # 维度检查
        if len(alpha) != n_atoms:
            print(f"  ⚠️  [{label}] alpha长度 {len(alpha)} ≠ 原子数 {n_atoms}，跳过")
            continue
        results.append((label, pool_type, alpha, prob))
        print(f"\n  [{label}] pool={pool_type}  pred_prob={prob:.4f}")
        attention_diagnostics(alpha, label=label)

    if not results:
        print("没有可用结果，退出。")
        return

    # ── 绘图 ────────────────────────────────────────────────────
    n_models = len(results)
    fig = plt.figure(figsize=(8 * n_models, 10))
    gs  = gridspec.GridSpec(2, n_models, height_ratios=[3, 1], hspace=0.35)

    for col, (label, pool_type, alpha, prob) in enumerate(results):
        # 上：热力图
        ax_map = fig.add_subplot(gs[0, col])
        ax_map.axis('off')
        ax_map.set_title(f"{label}\npool={pool_type}  P(active)={prob:.4f}",
                         fontsize=13, fontweight='bold')

        # 归一化到 [-1, 1] 供 SimilarityMaps 着色（使它正好用红色系）
        w_plot = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-12)
        w_plot = w_plot.tolist()

        # 用新版 RDKit API 绘制热力图
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, w_plot, drawer, colorMap='RdYlGn_r',
            contourLines=5, alpha=0.6, size=(400, 400)
        )
        drawer.FinishDrawing()
        svg_str = drawer.GetDrawingText()

        # SVG → PNG via cairosvg 或 matplotlib patch
        import io, base64
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg_str.encode())
            img = plt.imread(io.BytesIO(png_bytes))
        except ImportError:
            # fallback: 保存 svg 然后用 PIL 读
            svg_path = f"_tmp_heatmap_{col}.svg"
            with open(svg_path, 'w') as f:
                f.write(svg_str)
            from PIL import Image
            import subprocess
            png_path = f"_tmp_heatmap_{col}.png"
            subprocess.run(["rsvg-convert", "-o", png_path, svg_path],
                           check=True, capture_output=True)
            img = plt.imread(png_path)
        ax_map.imshow(img)

        # 下：原子权重条形图
        ax_bar = fig.add_subplot(gs[1, col])
        atom_idx = np.arange(n_atoms)
        colors   = plt.cm.RdYlGn_r(w_plot)
        ax_bar.bar(atom_idx, alpha, color=colors, edgecolor='none')
        ax_bar.axhline(y=1/n_atoms, color='blue', linestyle='--',
                       linewidth=1, label=f'均匀 1/N={1/n_atoms:.3f}')
        ax_bar.set_xlabel("原子索引", fontsize=10)
        ax_bar.set_ylabel("注意力权重 α", fontsize=10)
        ax_bar.legend(fontsize=8)
        ax_bar.set_xlim(-0.5, n_atoms - 0.5)

        # 熵标注
        ent   = -np.sum(alpha * np.log(alpha + 1e-12))
        ratio = ent / np.log(n_atoms)
        ax_bar.set_title(f"H/H_uniform={ratio:.3f}  聚焦度={1-ratio:.3f}", fontsize=9)

    # 总标题
    fig.suptitle(
        f"注意力热力图对比\nSMILES: {smiles[:60]}{'...' if len(smiles)>60 else ''}",
        fontsize=11, y=1.01
    )

    plt.savefig(out_file, bbox_inches='tight', dpi=200)
    print(f"\n✅ 热力图已保存: {out_file}")

    # 清理临时文件
    for col in range(n_models):
        tmp = f"_tmp_heatmap_{col}.png"
        if os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 使用 MCF7 数据集中真实正样本 idx=11（24个原子，结构清晰）
    DATA_DIR = Path("/home/data/jiangyun/cgi_data_pipeline/outputs/"
                    "datasets_classification_test_recommended/MCF7")
    raw = torch.load(DATA_DIR / "preprocessed_graphs_MCF7.pt")

    SAMPLE_IDX = 11   # atoms=24, label=1
    smiles   = raw['graph_indices'][SAMPLE_IDX]
    gene_seq = raw['gene_sequences'][SAMPLE_IDX]
    label    = raw['labels'][SAMPLE_IDX]
    print(f"样本 idx={SAMPLE_IDX}  label={label}")

    MODEL_DIR = ROOT / "results_paper" / "MCF7"
    model_configs = [
        ("JK+Entmax",        str(MODEL_DIR / "model_orthoTrue_clTrue_hybrid_Fold0_JK_Entmax.pt")),
        ("tau0.1+JK+Context",str(MODEL_DIR / "model_orthoTrue_clTrue_hybrid_Fold0_tau_0.1_JK_Context.pt")),
    ]

    smiles_to_graph_dict = raw['smiles_to_graph']

    # 输出图保存在 analysis/ 目录下（即脚本所在目录）
    out_file = Path(__file__).parent / "attention_comparison.png"

    compare_two_models(
        smiles               = smiles,
        gene_seq             = gene_seq,
        smiles_to_graph_dict = smiles_to_graph_dict,
        model_configs        = model_configs,
        out_file             = str(out_file),
        device_str           = "cpu",
    )
