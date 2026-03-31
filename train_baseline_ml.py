#!/usr/bin/env python3
"""
传统机器学习基准 (Random Forest & XGBoost)
==========================================
设计原则：
1. 严格使用 chemical_cold_splits.pkl，与主模型切分完全一致
2. 化学特征：Morgan Fingerprint (1024-bit, ECFP4)
3. 基因特征：6-mer 频率向量 (4096-dim)，与深度学习模型 k=6 对齐
4. 特征缓存：全量提取后缓存，所有 fold 共用，避免重复计算
5. 特征维度：1024 + 4096 = 5120

用法：
    # 单 fold
    python train_baseline_ml.py --data_dir /path/to/MCF7 --fold 0 --model rf
    # XGBoost
    python train_baseline_ml.py --data_dir /path/to/MCF7 --fold 0 --model xgb
"""

import os
import sys
import pickle
import argparse
import itertools
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    print("❌ RDKit 未安装")
    sys.exit(1)


# ================================================================
# 1. 特征提取
# ================================================================

def get_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    SMILES → Morgan Fingerprint (ECFP4, 1024-bit)
    关键：ConvertToNumpyArray 要求预先分配 n_bits 长度的数组
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)   # ← 必须是 n_bits 而非 0 或 1
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# 预构建 6-mer 词表（全局，只建一次）
_BASES_ML   = ['A', 'C', 'G', 'T']
_KMER6_LIST = [''.join(p) for p in itertools.product(_BASES_ML, repeat=6)]  # 4096 个
_KMER6_IDX  = {km: i for i, km in enumerate(_KMER6_LIST)}


def get_kmer6_frequencies(sequence: str) -> np.ndarray:
    """
    基因序列 → 6-mer 频率向量 (4096-dim)
    与深度学习模型 k=6 保持一致，保证特征表达能力对等。
    """
    freqs = np.zeros(len(_KMER6_LIST), dtype=np.float32)
    seq   = sequence.upper()
    k     = 6
    for i in range(len(seq) - k + 1):
        idx = _KMER6_IDX.get(seq[i:i+k])
        if idx is not None:
            freqs[idx] += 1.0
    total = freqs.sum()
    if total > 0:
        freqs /= total   # L1 归一化 → 频率
    return freqs


# ================================================================
# 2. 全量特征缓存（与 fold 无关，只算一次）
# ================================================================

def build_or_load_full_cache(data_dir: Path) -> dict:
    """
    提取全量特征并缓存到 {data_dir}/ml_features_full.pkl
    包含：
        chem_features : np.ndarray [N, 1024]  float32
        gene_features : np.ndarray [N, 4096]  float32
        labels        : np.ndarray [N]        int8
    """
    cache_path = data_dir / 'ml_features_full.pkl'

    if cache_path.exists():
        print(f"⚡ 加载全量 ML 特征缓存: {cache_path.name}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("首次运行，提取全量特征（后续直接加载缓存）...")

    cell_line        = data_dir.name
    preprocessed_file = data_dir / f'preprocessed_graphs_{cell_line}.pt'
    if not preprocessed_file.exists():
        raise FileNotFoundError(f"找不到预处理文件: {preprocessed_file}")

    raw = torch.load(preprocessed_file)
    all_smiles    = raw['graph_indices']     # List[str], N 条
    all_gene_seqs = raw['gene_sequences']    # List[str], N 条
    all_labels    = np.array(raw['labels'], dtype=np.int8)
    N = len(all_labels)
    print(f"  总样本数: {N:,}")

    # ── 化学特征：SMILES 去重后计算 ──
    print("  提取 Morgan Fingerprints (1024-bit, ECFP4)...")
    unique_smiles = list(set(all_smiles))
    print(f"  独特 SMILES: {len(unique_smiles):,}（去重率 {(1-len(unique_smiles)/N)*100:.1f}%）")
    smiles_to_fp = {}
    for smi in tqdm(unique_smiles, desc="  Morgan FP"):
        smiles_to_fp[smi] = get_morgan_fingerprint(smi)
    chem_features = np.array([smiles_to_fp[s] for s in all_smiles], dtype=np.float32)
    print(f"  化学特征矩阵: {chem_features.shape}")  # [N, 1024]

    # ── 基因特征 ──
    print("  提取 6-mer 频率向量 (4096-dim)...")
    gene_features = np.array(
        [get_kmer6_frequencies(seq) for seq in tqdm(all_gene_seqs, desc="  6-mer Freq")],
        dtype=np.float32
    )
    print(f"  基因特征矩阵: {gene_features.shape}")  # [N, 4096]

    payload = {
        'chem_features': chem_features,
        'gene_features': gene_features,
        'labels':        all_labels,
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"✅ 全量特征已保存: {cache_path.name}\n")
    return payload


# ================================================================
# 3. 按 fold 切分
# ================================================================

def get_fold_data(data_dir: Path, fold_idx: int):
    """
    返回 (X_train, y_train, X_val, y_val)
    X 维度: [N, 5120] = concat([1024, 4096])
    """
    payload = build_or_load_full_cache(data_dir)

    chem   = payload['chem_features']   # [N, 1024]
    gene   = payload['gene_features']   # [N, 4096]
    labels = payload['labels']          # [N]

    X_all  = np.hstack([chem, gene])    # [N, 5120]

    split_file = data_dir / 'chemical_cold_splits.pkl'
    if not split_file.exists():
        raise FileNotFoundError(f"找不到切分文件: {split_file}")
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)

    train_idx, val_idx = splits[fold_idx][0], splits[fold_idx][1]

    X_train, y_train = X_all[train_idx], labels[train_idx]
    X_val,   y_val   = X_all[val_idx],   labels[val_idx]

    print(f"\nFold {fold_idx} 数据切分完成:")
    print(f"  X_train: {X_train.shape} | 正样本率: {y_train.mean():.3f}")
    print(f"  X_val:   {X_val.shape}   | 正样本率: {y_val.mean():.3f}")
    return X_train, y_train, X_val, y_val


# ================================================================
# 4. 评估
# ================================================================

def evaluate(y_true, y_prob, model_name: str, fold_idx: int) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    f1     = f1_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"📊 [{model_name} | Fold {fold_idx}]")
    print(f"   AUROC : {auroc:.4f}")
    print(f"   AUPRC : {auprc:.4f}")
    print(f"   F1    : {f1:.4f}")
    print("=" * 50)
    return {'model': model_name, 'fold': fold_idx,
            'auroc': auroc, 'auprc': auprc, 'f1': f1}


# ================================================================
# 5. 主流程
# ================================================================

def run(args):
    data_dir = Path(args.data_dir)
    print(f"\n{'='*60}")
    print(f"🚀 ML 基准 | 细胞系: {data_dir.name} | "
          f"Fold: {args.fold} | 模型: {args.model.upper()}")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val = get_fold_data(data_dir, args.fold)

    if args.model == 'rf':
        print(f"\n🌲 训练 Random Forest ...")
        clf = RandomForestClassifier(
            n_estimators = args.n_estimators,
            max_depth    = args.max_depth,
            n_jobs       = -1,
            random_state = 42,
            verbose      = 1,
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_val)[:, 1]
        evaluate(y_val, y_prob, 'Random Forest', args.fold)

    elif args.model == 'xgb':
        if not HAS_XGB:
            print("❌ 请先安装: pip install xgboost")
            return
        print(f"\n⚡ 训练 XGBoost ...")
        clf = xgb.XGBClassifier(
            n_estimators  = args.n_estimators,
            max_depth     = args.max_depth,
            learning_rate = 0.1,
            tree_method   = 'hist',
            device        = 'cuda' if torch.cuda.is_available() else 'cpu',
            random_state  = 42,
            eval_metric   = 'logloss',
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50)
        y_prob = clf.predict_proba(X_val)[:, 1]
        evaluate(y_val, y_prob, 'XGBoost', args.fold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     type=str, required=True)
    parser.add_argument('--fold',         type=int, default=0)
    parser.add_argument('--model',        type=str, default='rf',
                        choices=['rf', 'xgb'])
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth',    type=int, default=20)
    run(parser.parse_args())
