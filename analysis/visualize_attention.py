#!/usr/bin/env python3
"""
Attention 可解释性可视化脚本
==================================
给定一个化学分子的 SMILES 和一条基因序列，加载我们训练好的最终 SOTA 模型，
提取 Cross-Attention 权重 (alpha)，并在二维分子结构上绘制热力图。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 导入你训练脚本里的依赖
from train_ultimate import PaperModel, encode_kmer_sequence

def visualize_drug_gene_attention(smiles, gene_seq, model_path, device_str='cuda:0'):
    device = torch.device(device_str)
    
    # 1. 准备基因特征 [1, L]
    gene_ids = torch.tensor([encode_kmer_sequence(gene_seq)]).to(device)
    
    # 2. 准备化学图特征 [N_atoms, ...]
    # 使用训练时一样的 MolGraphBuilder
    graph_data = MolGraphBuilder.build(smiles)
    if graph_data is None:
        raise ValueError(f"SMILES 解析失败: {smiles}")
        
    x = graph_data['x'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_attr = graph_data['edge_attr'].to(device) # 新增
    num_nodes_list = [x.shape[0]] # batch_size = 1
    
    # 3. 加载模型
    print(f"Loading model from {model_path}...")
    model = PaperModel(hidden_dim=128, dropout=0.0, pool_type='hybrid', use_ortho=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    # 4. 前向传播提取 Alpha (Attention Weights)
    with torch.no_grad():
        logits, _, _, _, alpha = model(gene_ids, x, edge_index, edge_attr, num_nodes_list)
        pred_prob = torch.sigmoid(logits).item()
    
    # 取出 alpha 并转为 numpy (大小正好等于原子数)
    weights = alpha.cpu().numpy()
    print(f"预测结合概率 (Probability): {pred_prob:.4f}")
    
    # 5. RDKit 可视化
    mol = Chem.MolFromSmiles(smiles)
    # 因为构建特征时如果加过氢，这里也应该一致。通常可视化二维不需要强行加氢，
    # 但必须保证 weights 数组的长度和 mol 的原子数完全一致。
    if len(weights) != mol.GetNumAtoms():
        print(f"警告：权重数 {len(weights)} 与分子原子数 {mol.GetNumAtoms()} 不一致，尝试补齐...")
    
    fig = plt.figure(figsize=(8, 8))
    # 使用红白绿色谱，红色越深代表权重（重要性）越高
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, 
        weights.tolist(), 
        colorMap='Reds', 
        contourLines=5, 
        alpha=0.5,
        fig=fig
    )
    
    # 保存图片（保存在 analysis/ 目录下）
    safe_smiles = "".join([c if c.isalnum() else "_" for c in smiles])[:20]
    out_file = Path(__file__).parent / f"attn_heatmap_{safe_smiles}.png"
    plt.savefig(str(out_file), bbox_inches='tight', dpi=300)
    print(f"✅ 热力图已保存至: {out_file}")

if __name__ == "__main__":
    # ==========================
    # 在此配置测试数据
    # ==========================
    # 举例1: 他莫昔芬 (Tamoxifen) - 经典的 MCF7 (乳腺癌) 正样本药物
    test_smiles = "CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3"
    
    # 提供一个你的数据集中真实导致高表达的基因序列 (字符串)
    # 这里用假序列代替，你需要换成真实的基因 string
    test_gene = "ATGC" * 250 
    
    # 你的最优模型路径（相对于项目根目录）
    model_checkpoint = ROOT / "results_paper" / "MCF7" / "model_orthoTrue_clTrue_hybrid.pt"

    if model_checkpoint.exists():
        visualize_drug_gene_attention(test_smiles, test_gene, str(model_checkpoint))
    else:
        print(f"模型文件 {model_checkpoint} 不存在，请先使用 train_ultimate.py 完成训练。")