#!/usr/bin/env python3
"""
图数据预处理脚本
================

功能：
1. 提取数据集中所有独特的SMILES
2. 使用RDKit将SMILES转换为图数据
3. 序列化保存到硬盘，训练时直接加载

优势：
- 训练时无需RDKit解析，速度提升100x+
- 图数据全部加载到内存，O(1)访问
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("❌ RDKit未安装")
    sys.exit(1)


# ================================================================
# 分子图构建器
# ================================================================

class MolGraphBuilder:
    """
    分子图构建器
    
    输出：
        x: [N_atoms, atom_feat_dim] 原子特征矩阵
        edge_index: [2, N_edges] 边索引（COO格式）
        edge_attr: [N_edges, edge_feat_dim] 边特征矩阵
    """
    
    ATOM_FEATURE_DIM = 31
    EDGE_FEATURE_DIM = 4
    
    @staticmethod
    def get_atom_features(atom) -> np.ndarray:
        """提取原子特征（31维）"""
        features = []
        
        # 1. 原子类型 (12维: B, C, N, O, F, Si, P, S, Cl, Br, I, 其他)
        atom_num = atom.GetAtomicNum()
        atom_type_map = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
        atom_type = atom_type_map.get(atom_num, 11)
        features.extend(F.one_hot(torch.tensor(atom_type), num_classes=12).tolist())
        
        # 2. 度数 (6维: 0-5)
        degree = atom.GetDegree()
        features.extend(F.one_hot(torch.tensor(min(degree, 5)), num_classes=6).tolist())
        
        # 3. 杂化类型 (6维: SP, SP2, SP3, SP3D, SP3D2, 其他)
        hybridization = atom.GetHybridization()
        hyb_map = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1,
                   Chem.HybridizationType.SP3: 2, Chem.HybridizationType.SP3D: 3,
                   Chem.HybridizationType.SP3D2: 4}
        hyb_type = hyb_map.get(hybridization, 5)
        features.extend(F.one_hot(torch.tensor(hyb_type), num_classes=6).tolist())
        
        # 4. 芳香性 (1维)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # 5. 形式电荷 (1维，归一化到[-1, 1])
        charge = max(-3, min(3, atom.GetFormalCharge()))
        features.append(charge / 3.0)
        
        # 6. 氢原子数 (5维: 0, 1, 2, 3, 4+)
        num_hs = atom.GetTotalNumHs()
        features.extend(F.one_hot(torch.tensor(min(num_hs, 4)), num_classes=5).tolist())
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def get_bond_features(bond) -> np.ndarray:
        """提取键特征（4维one-hot向量）"""
        bond_type = bond.GetBondType()
        
        if bond_type == Chem.BondType.SINGLE:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        elif bond_type == Chem.BondType.DOUBLE:
            return np.array([0, 1, 0, 0], dtype=np.float32)
        elif bond_type == Chem.BondType.TRIPLE:
            return np.array([0, 0, 1, 0], dtype=np.float32)
        elif bond_type == Chem.BondType.AROMATIC:
            return np.array([0, 0, 0, 1], dtype=np.float32)
        else:
            return np.array([0, 0, 0, 0], dtype=np.float32)
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
        """
        将SMILES字符串转换为图数据
        
        Returns:
            graph_data: Dict {
                'x': [N_atoms, atom_feat_dim] 原子特征,
                'edge_index': [2, N_edges] 边索引,
                'edge_attr': [N_edges, edge_feat_dim] 边特征
            }
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 返回空图
            return {
                'x': torch.zeros(1, MolGraphBuilder.ATOM_FEATURE_DIM),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, MolGraphBuilder.EDGE_FEATURE_DIM)
            }
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(MolGraphBuilder.get_atom_features(atom))
        x = torch.tensor(np.array(atom_features), dtype=torch.float32)
        
        # 边特征和边索引
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 双向边
            edge_indices.extend([[i, j], [j, i]])
            bond_feat = MolGraphBuilder.get_bond_features(bond)
            edge_attrs.extend([bond_feat, bond_feat])
        
        if len(edge_indices) == 0:
            # 孤立分子
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, MolGraphBuilder.EDGE_FEATURE_DIM)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }


# ================================================================
# 预处理主函数
# ================================================================

def preprocess_dataset(data_dir: str, output_path: str):
    """
    预处理数据集
    
    Args:
        data_dir: 数据集目录
        output_path: 输出文件路径
    """
    print(f"预处理数据集: {data_dir}")
    
    # 加载数据
    data_file = Path(data_dir) / 'full_data.pkl'
    with open(data_file, 'rb') as f:
        full_data = pickle.load(f)
    
    print(f"  总样本数: {len(full_data)}")
    
    # 提取所有独特的SMILES
    all_smiles = full_data['smiles'].tolist()
    unique_smiles = list(set(all_smiles))
    
    print(f"  独特SMILES数: {len(unique_smiles)}")
    print(f"  去重率: {(1 - len(unique_smiles) / len(all_smiles)) * 100:.1f}%")
    
    # 构建SMILES到图数据的映射
    print("\n开始转换SMILES为图...")
    smiles_to_graph = {}
    
    for smiles in tqdm(unique_smiles, desc="转换进度"):
        graph = MolGraphBuilder.smiles_to_graph(smiles)
        smiles_to_graph[smiles] = graph
    
    # 为所有样本创建图数据索引
    print("\n创建样本索引...")
    graph_indices = []
    for smiles in tqdm(all_smiles, desc="索引进度"):
        graph_indices.append(smiles)
    
    # 保存预处理数据
    print(f"\n保存预处理数据到: {output_path}")
    preprocess_data = {
        'smiles_to_graph': smiles_to_graph,
        'graph_indices': graph_indices,
        'labels': full_data['label'].values,
        'gene_sequences': full_data['gene_sequence'].tolist(),
        'metadata': {
            'num_samples': len(full_data),
            'num_unique_smiles': len(unique_smiles),
            'atom_feat_dim': MolGraphBuilder.ATOM_FEATURE_DIM,
            'edge_feat_dim': MolGraphBuilder.EDGE_FEATURE_DIM
        }
    }
    
    torch.save(preprocess_data, output_path)
    
    # 显示文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.2f} MB")
    
    print("\n✅ 预处理完成！")
    print(f"\n使用方法:")
    print(f"  data = torch.load('{output_path}')")
    print(f"  graph_data = data['smiles_to_graph'][smiles]")


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='图数据预处理脚本')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (默认: data_dir/preprocessed_graphs.pt)')
    
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        data_dir = Path(args.data_dir)
        cell_line = data_dir.name
        args.output = str(data_dir / f'preprocessed_graphs_{cell_line}.pt')
    
    # 预处理
    preprocess_dataset(args.data_dir, args.output)


if __name__ == '__main__':
    main()
