#!/usr/bin/env python3
"""
诊断 GCAA 的 S_gene 密度
========================
加载一个 batch 的真实数据，用随机初始化的模型计算 V_g，
检查 batch 内 S_gene 非零元素的比例（density）。

用法：
  python analysis/diag_gcaa_density.py \
      --data_dir /home/data/jiangyun/cgi_data_pipeline5/data/MCF7 \
      --batch_size 512

判断标准：
  density > 0.1  → batch 内有足够多的基因相似对，GCAA 梯度正常
  density < 0.05 → 有效对稀少，GCAA 退化为弱正则化，需增大 batch size
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 把项目根目录加入 path，复用 train_summean 里的类
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_summean import (
    OptimizedGraphDataset, optimized_collate_fn,
    GeneEncoderV1, GINLayer, ChemEncoder, PaperModel
)


def compute_s_gene_stats(V_g: torch.Tensor):
    """给定一批归一化后的 V_g，计算 S_gene 的统计信息。"""
    S_gene_raw = torch.matmul(V_g, V_g.T)          # [-1, 1]
    S_gene     = F.relu(S_gene_raw)                 # [0, 1]，ReLU 截断负值

    batch_size = V_g.size(0)
    mask_no_diag = 1.0 - torch.eye(batch_size, device=V_g.device)

    off_diag_vals = S_gene * mask_no_diag           # 排除对角线

    total_pairs   = mask_no_diag.sum().item()
    nonzero_pairs = (off_diag_vals > 1e-4).float().sum().item()
    density       = nonzero_pairs / total_pairs
    mean_weight   = off_diag_vals.sum().item() / total_pairs
    max_weight    = off_diag_vals.max().item()

    # 原始（未 ReLU）的分布
    raw_off = S_gene_raw * mask_no_diag
    frac_positive = (raw_off > 0).float().sum().item() / total_pairs
    frac_negative = (raw_off < 0).float().sum().item() / total_pairs

    return {
        'batch_size':    batch_size,
        'total_pairs':   int(total_pairs),
        'nonzero_pairs': int(nonzero_pairs),
        'density':       density,
        'mean_weight':   mean_weight,
        'max_weight':    max_weight,
        'frac_positive_raw': frac_positive,
        'frac_negative_raw': frac_negative,
    }


def run(args):
    device = torch.device('cpu')  # 诊断不需要 GPU

    print(f"加载数据: {args.data_dir}  fold=0  split=train")
    dataset = OptimizedGraphDataset(args.data_dir, fold_idx=0, split='train')
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=optimized_collate_fn, num_workers=0)

    model = PaperModel(hidden_dim=args.hidden_dim, dropout=0.0).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt)
        ckpt_label = f"训练权重: {args.ckpt}"
    else:
        ckpt_label = "随机初始化（无训练权重）"
    model.eval()

    print(f"\n{'='*55}")
    print(f"  batch_size={args.batch_size}  hidden_dim={args.hidden_dim}")
    print(f"  模型：{ckpt_label}")
    print(f"{'='*55}\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x          = batch['x'].to(device)
            edge_index  = batch['edge_index'].to(device)
            edge_attr   = batch['edge_attr'].to(device)
            gene_ids    = batch['gene_ids'].to(device)
            labels      = batch['label'].to(device)

            _, V_g, _, _ = model(gene_ids, x, edge_index, edge_attr,
                                 batch['num_nodes_list'])
            # V_g 已经是 F.normalize 后的结果（见 PaperModel.forward）

            stats = compute_s_gene_stats(V_g)

            print(f"Batch {batch_idx+1}")
            print(f"  样本数:            {stats['batch_size']}")
            print(f"  总对数 (非对角线): {stats['total_pairs']}")
            print(f"  S_gene 非零对数:   {stats['nonzero_pairs']}")
            print(f"  density:           {stats['density']:.4f}  "
                  f"({'⚠️  偏低，考虑加大 batch_size' if stats['density'] < 0.05 else '✅ 正常'})")
            print(f"  mean_weight:       {stats['mean_weight']:.4f}")
            print(f"  max_weight:        {stats['max_weight']:.4f}")
            print(f"  原始余弦 >0 比例:  {stats['frac_positive_raw']:.4f}")
            print(f"  原始余弦 <0 比例:  {stats['frac_negative_raw']:.4f}")

            if batch_idx + 1 >= args.num_batches:
                break

    print(f"\n{'='*55}")
    print("结论判断：")
    print("  density > 0.10  → GCAA 有足够的有效对，可以正常训练")
    print("  density 0.05~0.10 → 边界，可尝试 batch_size=1024")
    print("  density < 0.05  → 有效对过少，GCAA 退化，需增大 batch_size 或换 CL 策略")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,
                        default='/home/data/jiangyun/cgi_data_pipeline5/data/MCF7')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ckpt',       type=str, default='',
                        help='训练好的模型权重路径，不填则随机初始化')
    parser.add_argument('--num_batches',type=int, default=3,
                        help='检查前 N 个 batch，默认 3')
    args = parser.parse_args()
    run(args)
