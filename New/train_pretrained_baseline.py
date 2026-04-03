#!/usr/bin/env python3
"""
预训练嵌入基线（Pretrained Drug Encoder Baseline）
====================================================

对比目标：证明端到端 GIN 优于预训练分子嵌入。

设计：
  - 药物特征：固定的预计算嵌入（e.g. ChemBERTa, ECFP4 2048-bit, MolBERT）
             直接替换 GIN，不参与梯度优化
  - 基因特征：与 DrugOperatorNet 完全相同的 GeneMultiHeadReader（可优化）
  - 交互：与 DrugOperatorNet 完全相同的 PerturbationOperator

这样控制了除"药物编码器"以外的所有变量：
  - 端到端模型：GIN（可学习） + GeneMultiHeadReader + Operator
  - 预训练基线：固定嵌入（冻结）  + GeneMultiHeadReader + Operator

如果端到端更好 → 证明端到端学习优于黑盒预训练嵌入，体现本文核心价值。

预训练嵌入来源（选其一）：
  A. ECFP4 2048-bit（已有，train_morgan_baseline.py，但那个换掉了GIN+Operator）
  B. ChemBERTa-v2（需要 pip install transformers + 下载模型）
  C. MolBERT / Uni-Mol（同上）

当前实现：
  - 接受 --emb_path（.npy 或 .pt 文件，预计算的药物嵌入，按 smiles 索引）
  - 如果没有，自动用 ECFP4 2048-bit（已内置，无需额外下载）

用法（ECFP4 版本，无需额外依赖）：
  python New/train_pretrained_baseline.py \
    --data_dir /path/to/MCF7 --fold 0 --device cuda:0 \
    --drug_emb_type ecfp4 \
    --epochs 80 --batch_size 512 --lr 2e-4 \
    --lam_ortho_modes 0.1 --lam_sparse 0.01 \
    --patience 10 --use_amp --run_tag pretrained_ecfp4

用法（ChemBERTa，需先提取嵌入）：
  # 1. 先运行 extract_chemberta.py 生成 drug_embeddings.npy
  python New/train_pretrained_baseline.py \
    --data_dir /path/to/MCF7 --fold 0 --device cuda:0 \
    --drug_emb_type external --emb_path drug_embeddings.npy \
    ...
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 从 train_operator_moe.py 复用基因编码器和算子
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ================================================================
# 1. 预训练药物嵌入提取
# ================================================================

def get_ecfp4(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """ECFP4 2048-bit 固定指纹（无需下载模型）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_drug_emb_cache(data_dir: Path, emb_type: str = 'ecfp4',
                          emb_path: str = None, emb_dim: int = 2048) -> dict:
    """
    构建 smiles → 固定嵌入向量 的字典。
    emb_type: 'ecfp4' | 'external'
    emb_path: external 时指向 .npy 文件 (shape: [N_smiles, D])，附带 smiles_order.txt
    """
    cache_file = data_dir / f'pretrained_emb_cache_{emb_type}.pt'
    if cache_file.exists():
        print(f"  ⚡ 加载预训练嵌入缓存: {cache_file.name}")
        return torch.load(cache_file)

    print(f"  生成预训练嵌入缓存 (type={emb_type})...")

    # 获取所有 SMILES
    df = pickle.load(open(data_dir / 'full_data.pkl', 'rb'))
    all_smiles = df['smiles'].unique().tolist()

    if emb_type == 'ecfp4':
        assert HAS_RDKIT, "需要 RDKit: pip install rdkit"
        emb_dict = {}
        for smi in tqdm(all_smiles, desc='ECFP4'):
            emb_dict[smi] = torch.tensor(get_ecfp4(smi), dtype=torch.float32)
        actual_dim = 2048
    elif emb_type == 'external':
        assert emb_path is not None, "--emb_path 必须指定"
        embs = np.load(emb_path)  # [N, D]
        smiles_order = open(emb_path.replace('.npy', '_smiles.txt')).read().splitlines()
        emb_dict = {smi: torch.tensor(embs[i], dtype=torch.float32)
                    for i, smi in enumerate(smiles_order)}
        actual_dim = embs.shape[1]
    else:
        raise ValueError(f"Unknown emb_type: {emb_type}")

    cache = {'emb_dict': emb_dict, 'dim': actual_dim}
    torch.save(cache, cache_file)
    return cache


# ================================================================
# 2. 数据集（复用 OptimizedGraphDataset 的基因部分）
# ================================================================

def encode_kmer_sequence(sequence: str, k: int = 6, max_len: int = 1000):
    vocab_size = 4 ** k
    kmer_ids = []
    base2idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 1,
                'a': 0, 't': 1, 'g': 2, 'c': 3, 'u': 1}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        idx = 0
        valid = True
        for ch in kmer:
            if ch not in base2idx:
                valid = False; break
            idx = idx * 4 + base2idx[ch]
        kmer_ids.append(idx if valid else 0)
    if len(kmer_ids) > max_len:
        kmer_ids = kmer_ids[:max_len]
    else:
        kmer_ids += [0] * (max_len - len(kmer_ids))
    return kmer_ids


class PretrainedDrugDataset(Dataset):
    """使用预训练固定嵌入替代 GIN 图特征"""
    def __init__(self, data_dir, fold_idx, split, gene_max_len, emb_dict):
        with open(Path(data_dir) / 'chemical_cold_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        self.indices = splits[fold_idx][0] if split == 'train' else splits[fold_idx][1]
        cell_line = Path(data_dir).name
        self.data = torch.load(Path(data_dir) / f'preprocessed_graphs_{cell_line}.pt')
        self.labels = torch.tensor(
            [self.data['labels'][i] for i in self.indices], dtype=torch.float32)

        # 预训练药物嵌入（固定，不反向传播）
        # graph_indices 存的就是 smiles 字符串（作为 smiles_to_graph 的 key）
        self.drug_embs = []
        smiles_list = [self.data['graph_indices'][i] for i in self.indices]
        for smi in smiles_list:
            if smi in emb_dict:
                self.drug_embs.append(emb_dict[smi])
            else:
                # 未见 SMILES：用零向量（不应出现）
                self.drug_embs.append(torch.zeros(list(emb_dict.values())[0].shape))
        self.drug_embs = torch.stack(self.drug_embs)  # [N, D_drug]

        # 基因 k-mer（与 DrugOperatorNet 完全相同）
        gene_sequences = [self.data['gene_sequences'][i] for i in self.indices]
        cache_file = Path(data_dir) / f'kmer_cache_fold{fold_idx}_{split}.pt'
        if cache_file.exists():
            self.gene_ids = torch.load(cache_file)
        else:
            self.gene_ids = torch.tensor(
                [encode_kmer_sequence(seq, max_len=gene_max_len)
                 for seq in tqdm(gene_sequences)], dtype=torch.long)
            torch.save(self.gene_ids, cache_file)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'drug_emb': self.drug_embs[idx],
                'gene_ids': self.gene_ids[idx],
                'label':    self.labels[idx]}


def pretrained_collate(batch):
    return {
        'drug_emb': torch.stack([b['drug_emb'] for b in batch]),
        'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
        'label':    torch.stack([b['label']    for b in batch]),
    }


# ================================================================
# 3. 模型：固定药物嵌入 + GeneMultiHeadReader + Operator
#    直接从 train_operator_moe 导入基因编码器和算子
# ================================================================

class PretrainedDrugOperator(nn.Module):
    """
    药物用预训练固定嵌入（冻结），基因用 GeneMultiHeadReader，
    交互用 PerturbationOperator。
    与 DrugOperatorNet 的唯一区别：药物侧无 GIN，换成线性投影层。
    """
    def __init__(self, drug_emb_dim: int, hidden_dim: int = 128,
                 operator_rank: int = 8, dropout: float = 0.3,
                 gene_vocab_size: int = 4096, gene_max_len: int = 1000):
        super().__init__()
        # 药物：固定嵌入 → 线性投影到 hidden_dim（唯一可学习的药物参数）
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_emb_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # U 矩阵生成（药物方向矩阵）
        self.U_head = nn.Linear(hidden_dim, operator_rank * hidden_dim)

        # 基因：与 DrugOperatorNet 完全相同（复制）
        self.gene_emb = nn.Embedding(gene_vocab_size + 1, hidden_dim, padding_idx=0)
        self.gene_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, k, padding=k // 2)
            for k in [6, 8, 10, 12]
        ])
        self.gene_query = nn.Parameter(torch.zeros(operator_rank, hidden_dim))
        nn.init.trunc_normal_(self.gene_query, std=0.02)
        self.gene_attn_proj = nn.Linear(hidden_dim, hidden_dim)

        # Sigma（耦合强度）
        self.sigma_head = nn.Linear(hidden_dim, operator_rank)

        # 分类头
        self.r = operator_rank
        self.H = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_emb, gene_ids):
        B, H, r = drug_emb.shape[0], self.H, self.r

        # ── 药物侧（固定嵌入 → 投影） ────────────────────────────
        h_drug = self.drug_proj(drug_emb)                      # [B, H]
        U = self.U_head(h_drug).view(B, r, H)                  # [B, r, H]
        sigma = torch.sigmoid(self.sigma_head(h_drug))         # [B, r]

        # ── 基因侧（GeneMultiHeadReader，与 DrugOperatorNet 一致）─
        x = self.gene_emb(gene_ids).permute(0, 2, 1)          # [B, H, L]
        feats = [F.relu(conv(x)) for conv in self.gene_convs]
        gene_feat = torch.stack(feats, dim=0).mean(0)          # [B, H, L]
        gene_feat = gene_feat.permute(0, 2, 1)                 # [B, L, H]

        # MultiHead cross-attention
        Q = self.gene_query.unsqueeze(0).expand(B, -1, -1)    # [B, r, H]
        K = self.gene_attn_proj(gene_feat)                     # [B, L, H]
        attn = torch.bmm(Q, K.transpose(1, 2)) / (H ** 0.5)   # [B, r, L]
        attn = torch.softmax(attn, dim=-1)
        V_modes = torch.bmm(attn, gene_feat)                   # [B, r, H]

        h_gene = gene_feat.mean(dim=1)                         # [B, H]

        # ── PerturbationOperator ─────────────────────────────────
        coupling = (U * V_modes).sum(-1)                       # [B, r]
        spectrum = sigma * coupling                             # [B, r]
        delta_h = (spectrum.unsqueeze(-1) * U).sum(1)         # [B, H]

        # ── 分类 ─────────────────────────────────────────────────
        feat = self.dropout(torch.cat([h_gene, delta_h], dim=-1))
        logits = self.classifier(feat).squeeze(-1)             # [B]

        # 正交正则用的 U
        return logits, spectrum, sigma, U


# ================================================================
# 4. 训练
# ================================================================

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # 预训练嵌入
    emb_cache = build_drug_emb_cache(
        Path(args.data_dir), args.drug_emb_type, args.emb_path)
    emb_dict = emb_cache['emb_dict']
    drug_emb_dim = emb_cache['dim']
    print(f"  药物嵌入维度: {drug_emb_dim}（固定，不参与梯度）")

    train_ds = PretrainedDrugDataset(
        args.data_dir, args.fold, 'train', args.gene_max_len, emb_dict)
    val_ds   = PretrainedDrugDataset(
        args.data_dir, args.fold, 'val',   args.gene_max_len, emb_dict)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pretrained_collate, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=pretrained_collate, num_workers=4)

    model = PretrainedDrugOperator(
        drug_emb_dim=drug_emb_dim,
        hidden_dim=args.hidden_dim,
        operator_rank=args.operator_rank,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  预训练药物嵌入基线 | emb_type={args.drug_emb_type} | dim={drug_emb_dim}")
    print(f"  可学习参数: {n_params:,}（不含固定药物嵌入）")
    print(f"  Fold: {args.fold} | Device: {args.device}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    save_dir = Path(f'results_pretrained_baseline/{Path(args.data_dir).name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"pretrained_{args.drug_emb_type}_r{args.operator_rank}_Fold{args.fold}{tag}.pt"

    best_auroc, patience_cnt = 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = total_bce = total_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            drug_emb = batch['drug_emb'].to(device)
            gene_ids = batch['gene_ids'].to(device)
            labels   = batch['label'].to(device)

            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U = model(drug_emb, gene_ids)

                loss_bce = criterion(logits, labels)

                # 算子正则（与 DrugOperatorNet 完全相同）
                loss_sparse = sigma.abs().mean()
                U_n   = F.normalize(U, dim=-1)
                gram  = torch.bmm(U_n, U_n.transpose(1, 2))
                eye   = torch.eye(args.operator_rank, device=device).unsqueeze(0)
                loss_ortho = (gram - eye).pow(2).mean()
                loss_reg   = args.lam_sparse * loss_sparse + args.lam_ortho_modes * loss_ortho

                loss = loss_bce + loss_reg

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += loss_bce.item()
            total_reg  += loss_reg.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                drug_emb = batch['drug_emb'].to(device)
                gene_ids = batch['gene_ids'].to(device)
                with autocast(enabled=args.use_amp):
                    logits, _, _, _ = model(drug_emb, gene_ids)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1    = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        scheduler.step(auroc)

        n = len(train_loader)
        print(f"Ep {epoch+1:02d} | L:{total_loss/n:.3f} "
              f"(BCE:{total_bce/n:.3f} REG:{total_reg/n:.4f}) | "
              f"VAL_AUC:{auroc:.4f} PRC:{auprc:.4f} F1:{f1:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    print(f"\n最优 AUC: {best_auroc:.4f}  模型: {save_dir / model_name}")
    return best_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预训练药物嵌入基线（对照端到端GIN）')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--device',         type=str, default='cuda:0')
    parser.add_argument('--fold',           type=int, default=0)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--epochs',         type=int, default=80)
    parser.add_argument('--batch_size',     type=int, default=512)
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--hidden_dim',     type=int, default=128)
    parser.add_argument('--dropout',        type=float, default=0.3)
    parser.add_argument('--patience',       type=int, default=10)
    parser.add_argument('--use_amp',        action='store_true')
    parser.add_argument('--gene_max_len',   type=int, default=1000)
    parser.add_argument('--operator_rank',  type=int, default=8)
    parser.add_argument('--lam_sparse',     type=float, default=0.01)
    parser.add_argument('--lam_ortho_modes',type=float, default=0.1)
    # 药物嵌入来源
    parser.add_argument('--drug_emb_type',  type=str, default='ecfp4',
                        choices=['ecfp4', 'external'],
                        help='ecfp4: 内置2048-bit固定指纹; external: 外部预训练嵌入')
    parser.add_argument('--emb_path',       type=str, default=None,
                        help='external 时指向 .npy 文件（附带同名 _smiles.txt）')
    parser.add_argument('--run_tag',        type=str, default='')

    train(parser.parse_args())
