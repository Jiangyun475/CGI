#!/usr/bin/env python3
"""
interp/go_enrich.py
===================
对每个谱模式做 GO 富集分析。

核心思路：
  - 每个模式 j 有两种视角：
    (A) 基因视角：h_g_modes[:,j,:] 的 L2 范数 → 哪些基因在模式 j 的子空间参与度高？
        这是基因的"内在"属性，与药物无关。
    (B) 交互视角：spectrum 中模式 j 的分数最高的正样本涉及哪些基因？
        这是"哪些基因最常通过模式 j 被显著扰动"

  - 两种视角互补：A 是编码器学到的基因空间结构，B 是实际预测时模式 j 的使用情况

输出：
  results/{cell}/go/
    mode{j}_gene_view_GO.csv     # 基因视角 GO 富集
    mode{j}_interaction_view_GO.csv  # 交互视角 GO 富集
    summary_heatmap.png          # 各模式 top GO terms 热图

用法:
  python interp/go_enrich.py --cell MCF7 --top_k 100
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mygene
import gseapy as gp
from tqdm import tqdm

GENE_SETS = ["GO_Biological_Process_2023", "KEGG_2021_Human"]


def entrez_to_symbols(entrez_ids):
    """Entrez ID 列表 → Gene Symbol 列表（去掉转换失败的）"""
    mg = mygene.MyGeneInfo()
    result = mg.querymany(list(set(int(g) for g in entrez_ids)),
                          scopes="entrezgene", fields="symbol",
                          species="human", verbose=False)
    id2sym = {}
    for r in result:
        if "symbol" in r and "query" in r:
            id2sym[int(r["query"])] = r["symbol"]
    return [id2sym.get(int(g), None) for g in entrez_ids], id2sym


def run_enrichr(gene_symbols, label, out_dir):
    """用 gseapy Enrichr 做 GO 富集，保存结果"""
    symbols = [s for s in gene_symbols if s is not None]
    if len(symbols) < 5:
        print(f"  [{label}] 基因数量不足 ({len(symbols)})，跳过")
        return None
    try:
        enr = gp.enrichr(
            gene_list=symbols,
            gene_sets=GENE_SETS,
            outdir=None,
            verbose=False,
        )
        df = enr.results
        df.to_csv(os.path.join(out_dir, f"{label}_GO.csv"), index=False)
        return df
    except Exception as e:
        print(f"  [{label}] Enrichr 失败: {e}")
        return None


def plot_mode_summary(mode_results, out_dir):
    """
    汇总热图：行=模式，列=top GO terms，颜色=-log10(FDR)
    每个模式取排名最高的 5 个 BP GO terms
    """
    rows = []
    for mode_j, df in mode_results.items():
        if df is None:
            continue
        bp = df[df["Gene_set"].str.contains("Biological_Process")].copy()
        bp = bp.sort_values("Adjusted P-value").head(5)
        for _, row in bp.iterrows():
            rows.append({
                "mode": f"Mode {mode_j}",
                "term": row["Term"].split("(")[0].strip()[:50],
                "neg_log_fdr": -np.log10(max(row["Adjusted P-value"], 1e-10))
            })

    if not rows:
        print("没有足够的 GO 结果绘制热图")
        return

    pivot_df = pd.DataFrame(rows).pivot_table(
        index="mode", columns="term", values="neg_log_fdr", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(12, len(pivot_df.columns)*0.8),
                                    max(4, len(pivot_df)*0.6)))
    im = ax.imshow(pivot_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=10)
    plt.colorbar(im, ax=ax, label="-log10(FDR)")
    ax.set_title("GO Enrichment per Spectral Mode (Gene View, BP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mode_GO_heatmap.png"), dpi=150)
    plt.close()
    print(f"  热图保存至 {out_dir}/mode_GO_heatmap.png")


def main(args):
    data_path = os.path.join("interp/results", args.cell, "representations.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} 不存在，请先运行 extract.py")
        sys.exit(1)

    out_dir = os.path.join("interp/results", args.cell, "go")
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    spectrum  = d["spectrum"]     # [N, r]
    h_g_modes = d["h_g_modes"]   # [N, r, H]
    labels    = d["labels"]       # [N]
    gene_ids  = d["gene_ids"]     # [N]

    N, r, H = h_g_modes.shape
    print(f"N={N}, r={r}, H={H}")
    print(f"Unique genes: {len(np.unique(gene_ids))}")

    pos_mask = labels == 1
    print(f"Positive pairs: {pos_mask.sum()}")

    # ── 构建每个 gene 的 h_g_modes（每个基因只有一个表示）────────
    unique_genes = np.unique(gene_ids)
    gene_hg = {}  # gene_id → h_g_modes [r, H]
    for gid in unique_genes:
        mask = gene_ids == gid
        gene_hg[gid] = h_g_modes[mask][0]  # 取第一个（同基因 h_g_modes 相同）

    hg_matrix = np.stack([gene_hg[g] for g in unique_genes])  # [G, r, H]
    # 每个模式 j 的基因参与度 = ||h_g_modes[:,j,:]||
    gene_mode_norm = np.linalg.norm(hg_matrix, axis=-1)  # [G, r]

    # ── Entrez → Symbol 批量转换 ─────────────────────────────────
    print("Converting Entrez IDs to gene symbols...")
    _, id2sym = entrez_to_symbols(unique_genes)
    print(f"  Successfully mapped {len(id2sym)}/{len(unique_genes)} genes")

    mode_results_gene = {}
    mode_results_inter = {}

    for j in tqdm(range(r), desc="Modes"):
        # ── 视角 A：基因视角（h_g_modes 范数最高的 top_k 基因）──
        top_gene_idx = np.argsort(gene_mode_norm[:, j])[-args.top_k:]
        top_gene_ids = unique_genes[top_gene_idx]
        top_symbols  = [id2sym.get(int(g)) for g in top_gene_ids]
        label_a = f"mode{j}_gene_view"
        print(f"\nMode {j} [Gene view] top genes: {[s for s in top_symbols if s][:10]}")
        df_a = run_enrichr(top_symbols, label_a, out_dir)
        mode_results_gene[j] = df_a

        # ── 视角 B：交互视角（正样本中 spectrum[:,j] 最高的 top_k pairs）
        pos_spectrum_j = spectrum[pos_mask, j]
        pos_gene_ids   = gene_ids[pos_mask]
        top_pair_idx   = np.argsort(pos_spectrum_j)[-args.top_k:]
        top_inter_gene_ids = pos_gene_ids[top_pair_idx]
        top_inter_symbols  = [id2sym.get(int(g)) for g in top_inter_gene_ids]
        label_b = f"mode{j}_interaction_view"
        print(f"Mode {j} [Interaction view] top genes: {[s for s in top_inter_symbols if s][:10]}")
        df_b = run_enrichr(top_inter_symbols, label_b, out_dir)
        mode_results_inter[j] = df_b

    # ── 汇总热图 ─────────────────────────────────────────────────
    print("\nGenerating summary heatmap (gene view)...")
    plot_mode_summary(mode_results_gene, out_dir)

    # ── 文本摘要 ─────────────────────────────────────────────────
    summary_lines = ["# GO Enrichment Summary per Spectral Mode\n"]
    for j in range(r):
        summary_lines.append(f"\n## Mode {j}\n")
        for view, results in [("Gene view", mode_results_gene[j]),
                               ("Interaction view", mode_results_inter[j])]:
            summary_lines.append(f"### {view}\n")
            if results is not None:
                bp = results[results["Gene_set"].str.contains("Biological_Process")]
                bp = bp.sort_values("Adjusted P-value").head(5)
                for _, row in bp.iterrows():
                    summary_lines.append(
                        f"  - {row['Term']} (FDR={row['Adjusted P-value']:.2e}, "
                        f"genes={row['Overlap']})\n")
            else:
                summary_lines.append("  No significant enrichment\n")

    with open(os.path.join(out_dir, "go_summary.md"), "w") as f:
        f.writelines(summary_lines)
    print(f"\nSummary written to {out_dir}/go_summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell",  default="MCF7")
    parser.add_argument("--top_k", type=int, default=100,
                        help="每个模式取 top-K 基因做 GO 富集")
    main(parser.parse_args())
