#!/usr/bin/env python3
"""
传统机器学习基准测试 (Random Forest & XGBoost)
=================================================
核心逻辑：
1. 严格读取 chemical_cold_splits.pkl 保证化合物冷启动切分。
2. SMILES -> 1024-bit Morgan Fingerprint
3. Gene Sequence -> 4-mer 频率统计向量 (256维)
4. 特征拼接后训练 RF 和 XGBoost。
"""

import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb

# 隐藏 RDKit 警告
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits)

def get_kmer_frequencies(sequence, k=4):
    """提取 k-mer 频率作为传统 ML 的基因特征 (4^4 = 256 维)"""
    bases = ['A', 'C', 'G', 'T']
    import itertools
    vocab = {''.join(combo): i for i, combo in enumerate(itertools.product(bases, repeat=k))}
    
    freqs = np.zeros(len(vocab))
    valid_kmers = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if kmer in vocab:
            freqs[vocab[kmer]] += 1
            valid_kmers += 1
    
    if valid_kmers > 0:
        freqs = freqs / valid_kmers # 归一化频率
    return freqs

def prepare_ml_data(data_dir, fold_idx=0):
    print(f"Loading data from {data_dir}, Fold {fold_idx}...")
    with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    train_indices, val_indices = splits[fold_idx][0], splits[fold_idx][1]
    cell_line = Path(data_dir).name
    preprocessed_file = Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt'
    
    data = torch.load(preprocessed_file)
    smiles_list = data['graph_indices']
    gene_sequences = data['gene_sequences']
    labels = np.array(data['labels'])
    
    print("Generating Features (Morgan FP + K-mer Freqs)...")
    # 缓存机制，避免重复计算指纹
    X_all = []
    for i in tqdm(range(len(labels)), desc="Extracting ML Features"):
        fp = get_morgan_fingerprint(smiles_list[i])
        gene_freq = get_kmer_frequencies(gene_sequences[i])
        # 物理拼接：[1024维化学 + 256维基因]
        x_concat = np.concatenate([fp, gene_freq])
        X_all.append(x_concat)
    
    X_all = np.array(X_all)
    
    X_train, y_train = X_all[train_indices], labels[train_indices]
    X_val, y_val = X_all[val_indices], labels[val_indices]
    
    return X_train, y_train, X_val, y_val

def run_ml_baselines(args):
    import torch # 仅用于加载 .pt 文件
    X_train, y_train, X_val, y_val = prepare_ml_data(args.data_dir, args.fold)
    
    print(f"\n--- Fold {args.fold} Data Shape ---")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}\n")
    
    # ==========================================
    # Model 1: Random Forest
    # ==========================================
    print("🚀 Training Random Forest (n_estimators=200, n_jobs=-1)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    rf_preds = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_preds)
    rf_prc = average_precision_score(y_val, rf_preds)
    rf_f1 = f1_score(y_val, (rf_preds > 0.5).astype(int))
    print(f"📊 [Random Forest] AUROC: {rf_auc:.4f} | AUPRC: {rf_prc:.4f} | F1: {rf_f1:.4f}\n")
    
    # ==========================================
    # Model 2: XGBoost
    # ==========================================
    print("🚀 Training XGBoost (tree_method='hist')...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, 
        tree_method='hist', n_jobs=-1, random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_preds)
    xgb_prc = average_precision_score(y_val, xgb_preds)
    xgb_f1 = f1_score(y_val, (xgb_preds > 0.5).astype(int))
    print(f"📊 [XGBoost] AUROC: {xgb_auc:.4f} | AUPRC: {xgb_prc:.4f} | F1: {xgb_f1:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    run_ml_baselines(args)