#!/usr/bin/env python3
"""
批量图数据预处理脚本 (基于第三份脚本的 MolGraphBuilder)
=======================================================
用法：
    python batch_preprocess.py --data_root /path/to/data_root

逻辑：
    遍历 data_root 下所有子文件夹，每个子文件夹视为一个细胞系。
    若已存在 preprocessed_graphs_{cell_line}.pt，则自动跳过。
    原始数据文件为每个子文件夹下的 full_data.pkl。
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
except ImportError:
    print("❌ RDKit 未安装，请先执行: pip install rdkit")
    sys.exit(1)


# ================================================================
# 分子图构建器（完全来自第三份脚本，原子特征 31 维）
# ================================================================

class MolGraphBuilder:
    """
    分子图构建器

    原子特征（31 维）：
        12维  原子类型 one-hot  (按原子序数映射: B/C/N/O/F/Si/P/S/Cl/Br/I/其他)
         6维  度数 one-hot      (0~5)
         6维  杂化类型 one-hot  (SP/SP2/SP3/SP3D/SP3D2/其他)
         1维  芳香性
         1维  形式电荷          (归一化到 [-1, 1])
         5维  氢原子数 one-hot  (0~4+)

    边特征（4 维）：
        单键 / 双键 / 三键 / 芳香键，双向存储。
    """

    ATOM_FEATURE_DIM = 31
    EDGE_FEATURE_DIM = 4

    @staticmethod
    def get_atom_features(atom) -> np.ndarray:
        features = []

        # 1. 原子类型 (12维)
        atom_num = atom.GetAtomicNum()
        atom_type_map = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4,
                         14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
        atom_type = atom_type_map.get(atom_num, 11)
        features.extend(F.one_hot(torch.tensor(atom_type), num_classes=12).tolist())

        # 2. 度数 (6维: 0-5)
        degree = atom.GetDegree()
        features.extend(F.one_hot(torch.tensor(min(degree, 5)), num_classes=6).tolist())

        # 3. 杂化类型 (6维)
        hybridization = atom.GetHybridization()
        hyb_map = {
            Chem.HybridizationType.SP:    0,
            Chem.HybridizationType.SP2:   1,
            Chem.HybridizationType.SP3:   2,
            Chem.HybridizationType.SP3D:  3,
            Chem.HybridizationType.SP3D2: 4,
        }
        hyb_type = hyb_map.get(hybridization, 5)
        features.extend(F.one_hot(torch.tensor(hyb_type), num_classes=6).tolist())

        # 4. 芳香性 (1维)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)

        # 5. 形式电荷 (1维，归一化到 [-1, 1])
        charge = max(-3, min(3, atom.GetFormalCharge()))
        features.append(charge / 3.0)

        # 6. 氢原子数 (5维: 0, 1, 2, 3, 4+)
        num_hs = atom.GetTotalNumHs()
        features.extend(F.one_hot(torch.tensor(min(num_hs, 4)), num_classes=5).tolist())

        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_bond_features(bond) -> np.ndarray:
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
    def smiles_to_graph(smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'x':          torch.zeros(1, MolGraphBuilder.ATOM_FEATURE_DIM),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr':  torch.zeros(0, MolGraphBuilder.EDGE_FEATURE_DIM),
            }

        atom_features = [MolGraphBuilder.get_atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_features), dtype=torch.float32)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = MolGraphBuilder.get_bond_features(bond)
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([feat, feat])

        if len(edge_indices) == 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr  = torch.zeros(0, MolGraphBuilder.EDGE_FEATURE_DIM)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(np.array(edge_attrs), dtype=torch.float32)

        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


# ================================================================
# 单个细胞系预处理
# ================================================================

def process_cell_line(cell_dir: Path, skip_existing: bool = True) -> bool:
    """
    处理一个细胞系目录。

    期望目录结构：
        cell_dir/
            full_data.pkl          ← 原始 DataFrame，含 smiles / label / gene_sequence 列
            chemical_cold_splits.pkl
            preprocessed_graphs_{cell_line}.pt  ← 本脚本生成

    Returns:
        True  处理成功（或已跳过）
        False 处理失败
    """
    cell_line = cell_dir.name
    output_path = cell_dir / f'preprocessed_graphs_{cell_line}.pt'

    # 跳过已处理
    if skip_existing and output_path.exists():
        print(f"⏭️  [{cell_line}] 已存在，跳过。")
        return True

    # 检查原始数据
    data_file = cell_dir / 'full_data.pkl'
    if not data_file.exists():
        print(f"⚠️  [{cell_line}] 未找到 full_data.pkl，跳过。")
        return False

    print(f"\n🔬 开始处理: {cell_line}")
    try:
        with open(data_file, 'rb') as f:
            full_data = pickle.load(f)

        all_smiles    = full_data['smiles'].tolist()
        unique_smiles = list(set(all_smiles))

        print(f"   总样本数: {len(full_data)} | 独特 SMILES: {len(unique_smiles)} "
              f"(去重率 {(1 - len(unique_smiles)/len(all_smiles))*100:.1f}%)")

        # SMILES → 图
        smiles_to_graph = {}
        for smi in tqdm(unique_smiles, desc=f"   [{cell_line}] 构建分子图", leave=False):
            smiles_to_graph[smi] = MolGraphBuilder.smiles_to_graph(smi)

        preprocess_data = {
            'smiles_to_graph': smiles_to_graph,
            'graph_indices':   all_smiles,          # 每条样本对应的 SMILES key
            'labels':          full_data['label'].values,
            'gene_sequences':  full_data['gene_sequence'].tolist(),
            'metadata': {
                'cell_line':          cell_line,
                'num_samples':        len(full_data),
                'num_unique_smiles':  len(unique_smiles),
                'atom_feat_dim':      MolGraphBuilder.ATOM_FEATURE_DIM,
                'edge_feat_dim':      MolGraphBuilder.EDGE_FEATURE_DIM,
            }
        }

        torch.save(preprocess_data, output_path)
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"   ✅ 保存至 {output_path.name}  ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"   ❌ [{cell_line}] 处理失败: {e}")
        return False


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='批量图数据预处理（使用第三份脚本的 MolGraphBuilder）')
    parser.add_argument('--data_root', type=str, required=True,
                        help='包含多个细胞系子文件夹的根目录')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理，忽略已存在的 .pt 文件')
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"❌ 目录不存在: {root}")
        sys.exit(1)

    # 找出所有子目录（按名称排序，方便查看进度）
    cell_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not cell_dirs:
        print(f"⚠️  {root} 下没有找到任何子文件夹。")
        sys.exit(0)

    print(f"📂 根目录: {root}")
    print(f"🔢 发现 {len(cell_dirs)} 个子文件夹\n")

    results = {'success': [], 'skipped': [], 'failed': []}

    for cell_dir in cell_dirs:
        output_path = cell_dir / f'preprocessed_graphs_{cell_dir.name}.pt'
        already_exists = output_path.exists()

        ok = process_cell_line(cell_dir, skip_existing=not args.force)

        if ok and already_exists and not args.force:
            results['skipped'].append(cell_dir.name)
        elif ok:
            results['success'].append(cell_dir.name)
        else:
            results['failed'].append(cell_dir.name)

    # 汇总报告
    print("\n" + "=" * 50)
    print("📊 处理汇总")
    print("=" * 50)
    print(f"  ✅ 成功处理: {len(results['success'])} 个")
    for name in results['success']:
        print(f"     - {name}")
    print(f"  ⏭️  已跳过:   {len(results['skipped'])} 个")
    print(f"  ❌ 失败:     {len(results['failed'])} 个")
    for name in results['failed']:
        print(f"     - {name}")
    print("=" * 50)


if __name__ == '__main__':
    main()
