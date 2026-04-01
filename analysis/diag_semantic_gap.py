#!/usr/bin/env python3
"""
语义鸿沟诊断：计算 atom features x 与 gene query q 的余弦相似度分布
=======================================================================
验证猜想：若 x 和 q 处于不同语义空间，cos_sim(x_i, q) ≈ 0，
         scores 趋于 0，softmax 输出均匀 → over-smoothing。

对比：
  1. 随机初始化模型（未训练）
  2. 训练后模型（Full hybrid）
  3. 训练后模型（TargetOnly）

输出：分布统计 + 可视化图 semantic_gap.png
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_ultimate import PaperModel, OptimizedGraphDataset, optimized_collate_fn as collate_fn


# ──────────────────────────────────────────────────────────────────
# Hook：从 ChemEncoder_Ablation.forward 中截取 x 和 q
# ──────────────────────────────────────────────────────────────────
class SemanticGapProbe:
    """注册在 chem_enc 上的 forward hook，截取 GIN 后的 x 和 query q"""

    def __init__(self):
        self.x_after_gin = None   # [N_total_atoms, 128]
        self.q_expanded  = None   # [N_total_atoms, 128]，已按 batch_idx 展开
        self.batch_idx   = None   # 每个原子属于哪个分子

    def hook_fn(self, module, input, output):
        # output = (h_c, alpha)
        # 我们在 hook 里访问 module 的 _probe_data（由 monkey-patch 写入）
        if hasattr(module, '_probe_x'):
            self.x_after_gin = module._probe_x.detach().cpu()
            self.q_expanded  = module._probe_q.detach().cpu()
            self.batch_idx   = module._probe_batch.detach().cpu()


def patch_chem_enc(model):
    """Monkey-patch ChemEncoder_Ablation.forward 以暴露中间变量"""
    import math
    from torch_geometric.utils import softmax as geo_softmax

    original_forward = model.chem_enc.forward

    def patched_forward(x, edge_index, edge_attr, num_nodes_list, h_g):
        device = x.device
        x = model.chem_enc.atom_embed(x)
        for gin, norm in zip(model.chem_enc.gin_layers, model.chem_enc.norms):
            x = F.dropout(F.relu(norm(x + gin(x, edge_index, edge_attr))),
                          p=0.0, training=False)   # eval mode, dropout=0

        num_nodes_tensor = torch.tensor(num_nodes_list, device=device)
        batch_idx = torch.repeat_interleave(
            torch.arange(len(num_nodes_list), device=device), num_nodes_tensor
        )
        sum_pool  = torch.zeros(len(num_nodes_list), x.size(1), device=device).index_add_(0, batch_idx, x)
        mean_pool = sum_pool / num_nodes_tensor.float().unsqueeze(1).clamp(min=1)

        q      = model.chem_enc.attn_proj(h_g)
        scores = (x * q[batch_idx]).sum(dim=-1) / math.sqrt(x.size(-1))
        alpha  = geo_softmax(scores, batch_idx)

        x_weighted  = x * alpha.unsqueeze(-1)
        target_pool = torch.zeros(len(num_nodes_list), x.size(1), device=device).index_add_(0, batch_idx, x_weighted)

        # 暴露中间变量
        model.chem_enc._probe_x     = x                  # [N_atoms, D]
        model.chem_enc._probe_q     = q[batch_idx]        # [N_atoms, D]
        model.chem_enc._probe_batch = batch_idx

        pool = model.chem_enc.pool_type
        if pool == 'hybrid':
            h_c = torch.cat([sum_pool, mean_pool, target_pool], dim=1)
        elif pool == 'sum_mean':
            h_c = torch.cat([sum_pool, mean_pool], dim=1)
        else:
            h_c = target_pool

        return model.chem_enc.readout(h_c), alpha

    model.chem_enc.forward = patched_forward


# ──────────────────────────────────────────────────────────────────
# 计算一个 dataloader 上的余弦相似度分布
# ──────────────────────────────────────────────────────────────────
def compute_cosine_sim_distribution(model, loader, device, max_batches=20):
    model.eval()
    all_cos = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            num_nodes   = batch['num_nodes_list']

            # 触发 patched forward
            model(gene_ids, x, edge_index, edge_attr, num_nodes)

            x_gin = model.chem_enc._probe_x    # [N, D]
            q_exp = model.chem_enc._probe_q    # [N, D]

            cos_sim = F.cosine_similarity(x_gin, q_exp, dim=-1)  # [N]
            all_cos.append(cos_sim.cpu().numpy())

    return np.concatenate(all_cos)  # [total_atoms]


# ──────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────
def main():
    device   = torch.device('cpu')
    data_dir = Path("/home/data/jiangyun/cgi_data_pipeline/outputs/"
                    "datasets_classification_test_recommended/MCF7")

    # 加载验证集 (fold 0 val)
    print("加载数据集 (MCF7 fold=0 val)...")
    dataset = OptimizedGraphDataset(str(data_dir), fold_idx=0, split='val')
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)

    configs = [
        ("随机初始化",           "hybrid", True,  None),
        ("Full (hybrid, 训练后)", "hybrid", True,  str(ROOT / "results_paper/MCF7/model_orthoTrue_clTrue_hybrid.pt")),
        ("TargetOnly (训练后)",   "target", True,  str(ROOT / "results_paper/MCF7/model_orthoTrue_clTrue_target.pt")),
    ]

    results = {}
    for label, pool_type, use_ortho, ckpt in configs:
        print(f"\n{'='*55}")
        print(f"  配置: {label}")
        model = PaperModel(hidden_dim=128, dropout=0.0,
                           pool_type=pool_type, use_ortho=use_ortho)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model = model.to(device).eval()
        patch_chem_enc(model)

        cos_arr = compute_cosine_sim_distribution(model, loader, device, max_batches=20)
        results[label] = cos_arr

        print(f"  样本原子数: {len(cos_arr):,}")
        print(f"  cos_sim 均值:  {cos_arr.mean():.4f}")
        print(f"  cos_sim 标准差: {cos_arr.std():.4f}")
        print(f"  cos_sim 中位数: {np.median(cos_arr):.4f}")
        print(f"  cos_sim [5%, 95%]: [{np.percentile(cos_arr,5):.4f}, {np.percentile(cos_arr,95):.4f}]")
        print(f"  |cos_sim| < 0.1 占比: {(np.abs(cos_arr)<0.1).mean()*100:.1f}%  （越高越像语义鸿沟）")
        print(f"  cos_sim > 0.5 占比:  {(cos_arr>0.5).mean()*100:.1f}%  （高相似，对齐较好）")

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    colors = ['#888888', '#e74c3c', '#3498db']

    for ax, (label, cos_arr), color in zip(axes, results.items(), colors):
        ax.hist(cos_arr, bins=60, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(x=0,            color='black', linestyle='--', linewidth=1, label='cos=0')
        ax.axvline(x=cos_arr.mean(), color='orange', linestyle='-',
                   linewidth=1.5, label=f'mean={cos_arr.mean():.3f}')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("cosine_similarity(x_atom, q_gene)", fontsize=9)
        ax.set_ylabel("原子数", fontsize=9)
        ax.legend(fontsize=8)

        # 标注关键统计
        stats = (f"μ={cos_arr.mean():.3f}  σ={cos_arr.std():.3f}\n"
                 f"|cos|<0.1: {(np.abs(cos_arr)<0.1).mean()*100:.0f}%\n"
                 f"cos>0.5:  {(cos_arr>0.5).mean()*100:.0f}%")
        ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle("语义鸿沟诊断：atom x 与 gene query q 的余弦相似度分布\n"
                 "（接近0=语义不对齐，越右越好）",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = Path(__file__).parent / "semantic_gap.png"
    plt.savefig(str(out_path), bbox_inches='tight', dpi=180)
    print(f"\n✅ 图表已保存: {out_path}")

    # ── 结论输出 ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📊 语义鸿沟诊断结论：")
    r = results["随机初始化"]
    f = results["Full (hybrid, 训练后)"]
    t = results["TargetOnly (训练后)"]
    delta_full   = f.mean() - r.mean()
    delta_target = t.mean() - r.mean()
    print(f"  随机初始化均值:        {r.mean():.4f}")
    print(f"  Full训练后均值:        {f.mean():.4f}  (Δ={delta_full:+.4f})")
    print(f"  TargetOnly训练后均值:  {t.mean():.4f}  (Δ={delta_target:+.4f})")
    if abs(delta_full) < 0.05:
        print("  ⚠️  训练几乎没有拉近 x 和 q 的距离 → 语义鸿沟确实存在")
    else:
        print("  ✅  训练成功拉近了 x 和 q 的距离")


if __name__ == "__main__":
    main()
