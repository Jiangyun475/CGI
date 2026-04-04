#!/usr/bin/env python3
"""
预训练编码器基线 V2（Pretrained Encoder Baseline V2）
====================================================

【设计起点：现有对比的局限】
  train_pretrained_baseline.py 仅对比了固定 ECFP4 2048-bit 指纹。
  ECFP4 是传统化学信息学方法，不是"预训练神经网络模型"。
  它无法回答：
    "相比大规模预训练模型（ChemBERTa/DNABERT-2），
     我们的端到端 GIN+CNN 在 CGI 任务上有多大优势或劣势？"

【真正的预训练对比】
  药物端：ChemBERTa（在 ZINC/ChEMBL 上预训练的 SMILES Transformer，77M 参数）
  基因端：DNABERT-2（在人类基因组上预训练，117M 参数）
  两者均冻结（frozen），作为特征提取器。

  关键设计：保持 PerturbationOperator + 分类头完全不变，
  仅替换编码器。这是严格单变量对照：
    A: [ChemBERTa frozen] + [DNABERT-2 frozen] + Operator
    B: [端到端 GIN]       + [端到端 CNN]      + Operator（主模型）

  通过比较 A 和 B，量化：
    "任务特定端到端训练" vs "通用域预训练知识" 的相对贡献

【对 Chemical Cold Split 的理论预期】
  预训练模型在 ZINC（药物发现）或人类基因组（广泛转录）上训练，
  学到的是通用化学/基因组语言。
  我们的端到端模型在 CGI 任务上直接优化，特征更任务相关。
  直觉：在 cold split 下端到端应该更好，因为预训练特征对"差异表达"缺乏直接监督。
  但如果预训练覆盖了大量多样化分子，它可能在新分子的泛化上有优势。
  实验结果将量化这个权衡。

【工程细节：两阶段训练】
  阶段 1（precompute）：运行预训练模型提取所有样本的 embeddings，保存到磁盘缓存
  阶段 2（train）：从缓存加载 embeddings，不再运行预训练模型（快速）

  为什么：ChemBERTa/DNABERT-2 的前向推断每 batch 约 200ms（对比 GIN 的 5ms），
  若每 step 运行会使训练慢 40x。预计算只需一次（~30 分钟），之后训练速度与主模型相当。

【支持的编码器组合（--drug_encoder × --gene_encoder）】
  drug:  ecfp4（传统，已有对比）| chemberta（ChemBERTa-77M，推荐）
  gene:  kmer（端到端 CNN，基准）| dnabert2（DNABERT-2-117M，推荐）

  完整对比矩阵（MCF7 Fold0 上运行）：
    ecfp4   + kmer     → 传统指纹对照（已有：0.8687）
    chemberta + kmer   → 仅药物预训练
    ecfp4   + dnabert2 → 仅基因预训练
    chemberta + dnabert2 → 双端预训练

【使用方法】
  # 步骤 1：预计算嵌入（只需一次，约 30 分钟）
  python New/train_pretrained_v2.py \\
    --data_dir /path/to/MCF7 --precompute \\
    --drug_encoder chemberta --gene_encoder dnabert2 \\
    --cache_dir .pretrained_cache --device cuda:0

  # 步骤 2：训练
  python New/train_pretrained_v2.py \\
    --data_dir /path/to/MCF7 --fold 0 \\
    --drug_encoder chemberta --gene_encoder dnabert2 \\
    --cache_dir .pretrained_cache --device cuda:0 \\
    --epochs 80 --batch_size 512 --lr 2e-4 --use_amp
"""

import argparse
import hashlib
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_operator_moe import (
    set_seed, _KMER_VOCAB, encode_kmer_sequence,
    GINLayer, AtomEncoder, PharmacophoreExtractor, GeneMultiHeadReader,
    PerturbationOperator, collate_fn, OptimizedGraphDataset,
)
from torch_geometric.utils import scatter_softmax

# ── 预训练模型配置 ─────────────────────────────────────────────────
DRUG_PRETRAINED = {
    'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',   # 77M, 384-dim hidden
}
GENE_PRETRAINED = {
    'dnabert2':  'zhihan1996/DNABERT-2-117M',         # 117M, 768-dim hidden
}


# ================================================================
# 1. Slot Attention Extractor（通用：将 token 序列 → r 个 slot 表示）
# ================================================================

class SlotAttentionExtractor(nn.Module):
    """
    从预训练模型输出的 token 序列中提取 r 个结构化 slot 表示。

    设计逻辑：
      预训练模型输出 [B, L, d_model]（token 序列），其中 L 可变。
      我们需要 [B, r, H]（r 个固定维度 slot，与 PerturbationOperator 接口一致）。

      解决方案：learned slot attention（与 PharmacophoreExtractor 结构一致）
        - r 个可学习 query slot，每个代表一类"语义关注点"
        - 对 token 序列做注意力加权求和
        - 输出维度投影到 H

      这保持了与主模型相同的接口，使 PerturbationOperator 可以直接复用。

    参数：
      input_dim:  预训练模型的 hidden size（384 for ChemBERTa，768 for DNABERT-2）
      hidden_dim: 输出维度（与主模型一致，128）
      num_slots:  r（与 operator_rank 一致，8）
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int):
        super().__init__()
        H = hidden_dim
        self.num_slots = num_slots
        # 将预训练 token 投影到工作维度（可训练，Operator 侧的适配层）
        self.input_proj = nn.Linear(input_dim, H)
        # r 个可学习 query slot
        self.queries    = nn.Parameter(torch.randn(num_slots, H) * 0.02)
        self.key_proj   = nn.Linear(H, H)
        self.val_proj   = nn.Linear(H, H)
        self.norm       = nn.LayerNorm(H)

    def forward(self, token_emb, padding_mask=None):
        """
        Args:
          token_emb:    [B, L, input_dim]  预训练模型 token 嵌入（已冻结）
          padding_mask: [B, L] bool，True = 有效 token（可选）

        Returns:
          slots: [B, r, H]
        """
        B, L, _ = token_emb.shape
        h = F.relu(self.input_proj(token_emb))   # [B, L, H]  可训练适配
        K = self.key_proj(h)                      # [B, L, H]
        V = self.val_proj(h)                      # [B, L, H]

        # 所有 token 与所有 slot 的相似度
        scores = torch.einsum('blh,rh->brl', K, self.queries) / math.sqrt(K.size(-1))  # [B, r, L]

        if padding_mask is not None:
            # padding 位置设为 -inf，确保 softmax 后权重为 0
            pad_mask = ~padding_mask  # True = padding
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        attn  = F.softmax(scores, dim=-1)                        # [B, r, L]
        slots = torch.einsum('brl,blh->brh', attn, V)           # [B, r, H]
        return self.norm(slots)


# ================================================================
# 2. 预计算嵌入缓存
# ================================================================

def seq_hash(seq: str) -> str:
    """对序列字符串计算 MD5 hash，用作缓存 key。"""
    return hashlib.md5(seq.encode()).hexdigest()


def build_drug_cache(data_dir: Path, encoder_name: str,
                     cache_dir: Path, device: torch.device) -> dict:
    """
    预计算所有唯一 SMILES 的 ChemBERTa token 嵌入，保存到磁盘。

    Returns:
      dict: {smiles_str: tensor [L, d_model]}
    """
    cache_path = cache_dir / f'drug_{encoder_name}.pkl'
    if cache_path.exists():
        print(f"  [Cache] 加载药物嵌入缓存: {cache_path}")
        return pickle.load(open(cache_path, 'rb'))

    from transformers import AutoTokenizer, AutoModel
    model_id = DRUG_PRETRAINED[encoder_name]
    print(f"  [Pretrain] 加载 {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pt_model  = AutoModel.from_pretrained(model_id).to(device)
    pt_model.eval()

    # 加载所有唯一 SMILES
    full_data    = pickle.load(open(data_dir / 'full_data.pkl', 'rb'))
    unique_smiles = full_data['smiles'].unique().tolist()
    print(f"  [Pretrain] 计算 {len(unique_smiles)} 个唯一分子的嵌入...")

    cache = {}
    BATCH = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_smiles), BATCH), desc='药物嵌入'):
            batch_smiles = unique_smiles[i: i + BATCH]
            enc = tokenizer(batch_smiles, return_tensors='pt',
                            padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = pt_model(**enc)
            # last_hidden_state: [B, L, d_model]
            hidden   = out.last_hidden_state.cpu()
            attn_mask = enc['attention_mask'].cpu().bool()
            for j, smi in enumerate(batch_smiles):
                L = attn_mask[j].sum().item()
                cache[smi] = hidden[j, :L]   # [L, d_model]，去掉 padding

    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(cache_path, 'wb'))
    print(f"  [Cache] 药物嵌入已保存: {cache_path}")
    return cache


def build_gene_cache(data_dir: Path, encoder_name: str,
                     cache_dir: Path, device: torch.device) -> dict:
    """
    预计算所有唯一基因序列的 DNABERT-2 token 嵌入，保存到磁盘。

    Returns:
      dict: {seq_hash: tensor [L', d_model]}
    """
    cache_path = cache_dir / f'gene_{encoder_name}.pkl'
    if cache_path.exists():
        print(f"  [Cache] 加载基因嵌入缓存: {cache_path}")
        return pickle.load(open(cache_path, 'rb'))

    from transformers import AutoTokenizer, AutoModel
    model_id = GENE_PRETRAINED[encoder_name]
    print(f"  [Pretrain] 加载 {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    pt_model  = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    pt_model.eval()

    full_data      = pickle.load(open(data_dir / 'full_data.pkl', 'rb'))
    unique_seqs    = full_data['gene_sequence'].unique().tolist()
    print(f"  [Pretrain] 计算 {len(unique_seqs)} 个唯一基因的嵌入...")

    cache = {}
    BATCH = 8   # 基因序列较长，batch 更小
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_seqs), BATCH), desc='基因嵌入'):
            batch_seqs = unique_seqs[i: i + BATCH]
            # DNABERT-2 使用 BPE tokenizer，输入是原始 DNA 字符串（ACGT）
            # 截断到前 1000 bp（与主模型一致）
            batch_seqs_trunc = [s[:1000] for s in batch_seqs]
            enc = tokenizer(batch_seqs_trunc, return_tensors='pt',
                            padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = pt_model(**enc)
            hidden    = out.last_hidden_state.cpu()
            attn_mask = enc['attention_mask'].cpu().bool()
            for j, seq in enumerate(batch_seqs):
                h_key = seq_hash(seq)
                L     = attn_mask[j].sum().item()
                cache[h_key] = hidden[j, :L]   # [L', d_model]

    cache_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, open(cache_path, 'wb'))
    print(f"  [Cache] 基因嵌入已保存: {cache_path}")
    return cache


# ================================================================
# 3. 预训练缓存数据集
# ================================================================

class PretrainedCachedDataset(Dataset):
    """
    使用预计算 embeddings 替代原始分子图/序列的数据集。

    支持以下组合：
      drug: ecfp4（PyG 图，走 GIN）| chemberta（cached tensor，走 SlotAttn）
      gene: kmer（6-mer 序列，走 CNN）| dnabert2（cached tensor，走 SlotAttn）
    """
    def __init__(self, data_dir: Path, fold: int, split: str, gene_max_len: int,
                 drug_encoder: str, gene_encoder: str,
                 drug_cache: dict = None, gene_cache: dict = None):
        splits   = pickle.load(open(data_dir / 'chemical_cold_splits.pkl', 'rb'))
        full_df  = pickle.load(open(data_dir / 'full_data.pkl', 'rb'))
        indices  = splits[fold][split]
        self.df  = full_df.iloc[indices].reset_index(drop=True)

        self.drug_encoder = drug_encoder
        self.gene_encoder = gene_encoder
        self.drug_cache   = drug_cache
        self.gene_cache   = gene_cache
        self.gene_max_len = gene_max_len

        # 若使用 GIN 或 kmer，需要预处理图和 6-mer
        if drug_encoder == 'ecfp4':
            # 加载图，按 full_data 索引
            graphs_path = data_dir / f'preprocessed_graphs_{data_dir.name}.pt'
            self.graphs = torch.load(graphs_path, map_location='cpu')
            # full_data 中每行对应 graphs 中相同位置
            self.graph_indices = self.graphs['graph_indices']
            self._build_smiles_to_graph_idx()

        if gene_encoder == 'kmer':
            seqs = self.df['gene_sequence'].tolist()
            self.kmer_seqs = [
                torch.tensor(encode_kmer_sequence(s, k=6, max_len=gene_max_len),
                             dtype=torch.long)
                for s in seqs
            ]

    def _build_smiles_to_graph_idx(self):
        self._smiles_to_gidx = {}
        for i, smi in enumerate(self.graph_indices):
            self._smiles_to_gidx[smi] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        item  = {'label': label}

        # ── 药物特征 ─────────────────────────────────────────────
        if self.drug_encoder in DRUG_PRETRAINED:
            emb = self.drug_cache.get(row['smiles'])
            if emb is None:
                raise KeyError(f"SMILES not in drug cache: {row['smiles'][:30]}...")
            item['drug_token_emb'] = emb       # [L, d_model]
        else:
            # 走原 PyG 图路径（供 GIN 使用，与原数据集一致）
            gidx = self._smiles_to_gidx[row['smiles']]
            item['x']          = self.graphs['atom_features'][gidx]
            item['edge_index'] = self.graphs['edge_indices'][gidx]
            item['edge_attr']  = self.graphs['edge_features'][gidx]

        # ── 基因特征 ─────────────────────────────────────────────
        if self.gene_encoder in GENE_PRETRAINED:
            h_key = seq_hash(row['gene_sequence'])
            emb   = self.gene_cache.get(h_key)
            if emb is None:
                raise KeyError(f"gene_sequence hash not in cache: {h_key[:16]}")
            item['gene_token_emb'] = emb       # [L', d_model]
        else:
            item['gene_ids'] = self.kmer_seqs[idx]

        return item


def pretrained_collate(batch):
    """自定义 collate，同时处理可变长度 token 序列和 PyG 图。"""
    labels = torch.stack([b['label'] for b in batch])
    out    = {'labels': labels}
    B      = len(batch)

    # 药物
    if 'drug_token_emb' in batch[0]:
        # 可变长度 token 序列 → pad 到最大 L
        seqs    = [b['drug_token_emb'] for b in batch]
        max_L   = max(s.size(0) for s in seqs)
        d_model = seqs[0].size(1)
        padded  = torch.zeros(B, max_L, d_model)
        mask    = torch.zeros(B, max_L, dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = s.size(0)
            padded[i, :L] = s
            mask[i, :L]   = True
        out['drug_token_emb']  = padded   # [B, max_L, d_model]
        out['drug_attn_mask']  = mask     # [B, max_L]
    else:
        # PyG 图打包（与原 collate_fn 一致）
        from torch_geometric.data import Batch
        import torch_geometric
        graphs = []
        for b in batch:
            d = torch_geometric.data.Data(
                x=b['x'], edge_index=b['edge_index'], edge_attr=b['edge_attr'])
            graphs.append(d)
        batched = Batch.from_data_list(graphs)
        out['x']          = batched.x
        out['edge_index'] = batched.edge_index
        out['edge_attr']  = batched.edge_attr
        out['num_nodes_list'] = [b['x'].size(0) for b in batch]

    # 基因
    if 'gene_token_emb' in batch[0]:
        seqs    = [b['gene_token_emb'] for b in batch]
        max_L   = max(s.size(0) for s in seqs)
        d_model = seqs[0].size(1)
        padded  = torch.zeros(B, max_L, d_model)
        mask    = torch.zeros(B, max_L, dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = s.size(0)
            padded[i, :L] = s
            mask[i, :L]   = True
        out['gene_token_emb'] = padded
        out['gene_attn_mask'] = mask
    else:
        out['gene_ids'] = torch.stack([b['gene_ids'] for b in batch])

    return out


# ================================================================
# 4. 预训练模型
# ================================================================

class PretrainedDrugEncoder(nn.Module):
    """从缓存的预训练 token 嵌入中提取 r 个 slot 表示（可训练 slot attention）。"""
    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int):
        super().__init__()
        self.slot_attn = SlotAttentionExtractor(input_dim, hidden_dim, num_slots)

    def forward(self, token_emb, attn_mask=None):
        return self.slot_attn(token_emb, attn_mask)   # [B, r, H]


class PretrainedGeneEncoder(nn.Module):
    """从缓存的预训练 token 嵌入中提取 r 个 slot 表示。"""
    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int):
        super().__init__()
        self.slot_attn = SlotAttentionExtractor(input_dim, hidden_dim, num_slots)

    def forward(self, token_emb, attn_mask=None):
        slots      = self.slot_attn(token_emb, attn_mask)   # [B, r, H]
        h_g_global = slots.mean(1)                           # [B, H]
        return slots, h_g_global


class PretrainedCGIModel(nn.Module):
    """
    预训练编码器 + DrugOperatorNet 算子的完整模型。

    支持 4 种组合（drug_encoder × gene_encoder）：
      chemberta × dnabert2  → 双端预训练
      chemberta × kmer      → 仅药物预训练
      ecfp4     × dnabert2  → 仅基因预训练
      ecfp4     × kmer      → 传统指纹（参考，已有 0.8687）

    核心：PerturbationOperator 完全复用主模型，确保对照严格。
    """
    def __init__(self, hidden_dim=128, operator_rank=8, dropout=0.3,
                 drug_encoder='chemberta', gene_encoder='dnabert2',
                 drug_input_dim=384, gene_input_dim=768):
        super().__init__()
        H, r = hidden_dim, operator_rank
        self.drug_encoder_type = drug_encoder
        self.gene_encoder_type = gene_encoder

        # ── 药物编码器 ────────────────────────────────────────────
        if drug_encoder in DRUG_PRETRAINED:
            self.drug_enc = PretrainedDrugEncoder(drug_input_dim, H, r)
        else:
            # ECFP4：Linear(2048 → H) + r 个复制（简化版，与原 pretrained_baseline.py 一致）
            self.drug_proj = nn.Sequential(
                nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, H))
            self.drug_slot = nn.Parameter(torch.randn(r, H) * 0.02)
            self.drug_enc  = None  # 用 _ecfp4_encode 处理

        # ── 基因编码器 ────────────────────────────────────────────
        if gene_encoder in GENE_PRETRAINED:
            self.gene_enc = PretrainedGeneEncoder(gene_input_dim, H, r)
        else:
            self.gene_enc = GeneMultiHeadReader(4097, H, r, dropout)

        # ── Operator + 分类头（与主模型完全一致）─────────────────
        self.perturb_op = PerturbationOperator(H)
        self.classifier = nn.Sequential(
            nn.Linear(H * 2, H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(H, 1))
        self.hidden_dim = H
        self.r          = r

    def _ecfp4_encode(self, ecfp4_vec):
        """ECFP4 [B, 2048] → pharma_emb [B, r, H]"""
        h = self.drug_proj(ecfp4_vec)                                # [B, H]
        # 用可学习 slot 权重把 h 分配到 r 个 slot
        slot_w = F.softmax(h @ self.drug_slot.T, dim=-1)             # [B, r]
        return slot_w.unsqueeze(-1) * h.unsqueeze(1)                 # [B, r, H]

    def forward(self, batch):
        """batch 来自 pretrained_collate，内容取决于 encoder 组合。"""
        # ── 药物 ─────────────────────────────────────────────────
        if self.drug_encoder_type in DRUG_PRETRAINED:
            pharma_emb = self.drug_enc(
                batch['drug_token_emb'], batch.get('drug_attn_mask'))    # [B, r, H]
        else:
            pharma_emb = self._ecfp4_encode(batch['ecfp4'])             # [B, r, H]

        # ── 基因 ─────────────────────────────────────────────────
        if self.gene_encoder_type in GENE_PRETRAINED:
            h_g_modes, h_g_global = self.gene_enc(
                batch['gene_token_emb'], batch.get('gene_attn_mask'))    # [B, r, H], [B, H]
        else:
            h_g_modes, h_g_global, _ = self.gene_enc(batch['gene_ids'])

        # ── Operator（与主模型完全一致）──────────────────────────
        delta_h, spectrum, sigma, U = self.perturb_op(pharma_emb, h_g_modes)
        logits = self.classifier(torch.cat([h_g_global, delta_h], dim=-1))
        return logits.squeeze(-1), spectrum, sigma, U


# ================================================================
# 5. 训练循环
# ================================================================

def train(args):
    set_seed(args.seed)
    device    = torch.device(args.device)
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    save_dir  = Path('results_new_models') / data_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    tag        = f"_{args.run_tag}" if args.run_tag else ""
    model_name = f"pretrained_{args.drug_encoder}_{args.gene_encoder}_Fold{args.fold}{tag}.pt"

    # ── 加载/构建缓存 ────────────────────────────────────────────
    drug_cache = gene_cache = None
    if args.drug_encoder in DRUG_PRETRAINED:
        drug_cache = build_drug_cache(data_dir, args.drug_encoder, cache_dir, device)
    if args.gene_encoder in GENE_PRETRAINED:
        gene_cache = build_gene_cache(data_dir, args.gene_encoder, cache_dir, device)

    # ── 数据集 ───────────────────────────────────────────────────
    train_ds = PretrainedCachedDataset(
        data_dir, args.fold, 'train', args.gene_max_len,
        args.drug_encoder, args.gene_encoder, drug_cache, gene_cache)
    val_ds   = PretrainedCachedDataset(
        data_dir, args.fold, 'val', args.gene_max_len,
        args.drug_encoder, args.gene_encoder, drug_cache, gene_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pretrained_collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              collate_fn=pretrained_collate, num_workers=2, pin_memory=True)

    # 获取预训练模型的 d_model 维度
    drug_d  = drug_cache[next(iter(drug_cache))].size(-1) if drug_cache else 0
    gene_d  = gene_cache[next(iter(gene_cache))].size(-1) if gene_cache else 0

    model = PretrainedCGIModel(
        hidden_dim=args.hidden_dim,
        operator_rank=args.operator_rank,
        dropout=args.dropout,
        drug_encoder=args.drug_encoder,
        gene_encoder=args.gene_encoder,
        drug_input_dim=drug_d if drug_d else 384,
        gene_input_dim=gene_d if gene_d else 768,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if args.use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PretrainedCGI] drug={args.drug_encoder}, gene={args.gene_encoder}, "
          f"可训练参数: {n_params:,}")
    print(f"  注：预训练模型权重已冻结（特征提取器），仅 SlotAttn + Operator 可训练。")

    best_auc, patience_cnt, base_lr = 0.0, 0, args.lr
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                lr = base_lr * global_step / warmup_steps
                for pg in optimizer.param_groups: pg['lr'] = lr

            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = batch['labels']

            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits, spectrum, sigma, U = model(batch)
                loss_bce   = criterion(logits, labels)
                loss_sp    = sigma.abs().mean()
                U_n        = F.normalize(U, dim=-1)
                gram       = torch.bmm(U_n, U_n.transpose(1, 2))
                eye        = torch.eye(U.size(1), device=device).unsqueeze(0)
                loss_ortho = (gram - eye).pow(2).mean()
                loss = (loss_bce
                        + args.lam_sparse * loss_sp
                        + args.lam_ortho_modes * loss_ortho)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ep_loss += loss.item()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with autocast(enabled=args.use_amp):
                    logits, _, _, _ = model(batch)
                all_p.append(torch.sigmoid(logits).cpu())
                all_l.append(batch['labels'].cpu())

        probs   = torch.cat(all_p).numpy()
        lbls    = torch.cat(all_l).numpy()
        val_auc = roc_auc_score(lbls, probs)
        val_prc = average_precision_score(lbls, probs)
        val_f1  = f1_score(lbls, probs > 0.5)

        scheduler.step(val_auc)
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:3d} [lr={lr_cur:.2e}] | L:{ep_loss/len(train_loader):.3f} | "
              f"VAL_AUC:{val_auc:.4f} PRC:{val_prc:.4f} F1:{val_f1:.4f}")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save({'model': model.state_dict(), 'args': args},
                       save_dir / model_name)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop @ ep{epoch}，最优 AUC: {best_auc:.4f}")
                break

    print(f"最优 AUC: {best_auc:.4f}  模型: {save_dir / model_name}")
    return best_auc


# ================================================================
# 入口
# ================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pretrained Encoder Baseline V2')
    p.add_argument('--data_dir',        required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--fold',            type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=80)
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--hidden_dim',      type=int,   default=128)
    p.add_argument('--dropout',         type=float, default=0.3)
    p.add_argument('--operator_rank',   type=int,   default=8)
    p.add_argument('--gene_max_len',    type=int,   default=1000)
    p.add_argument('--warmup_epochs',   type=int,   default=5)
    p.add_argument('--lam_sparse',      type=float, default=0.01)
    p.add_argument('--lam_ortho_modes', type=float, default=0.1)
    p.add_argument('--patience',        type=int,   default=10)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--use_amp',         action='store_true')
    p.add_argument('--run_tag',         default='pretrained_v2')
    p.add_argument('--drug_encoder',    default='chemberta',
                   choices=['ecfp4', 'chemberta'],
                   help='药物编码器：ecfp4=固定指纹（传统），chemberta=预训练Transformer')
    p.add_argument('--gene_encoder',    default='dnabert2',
                   choices=['kmer', 'dnabert2'],
                   help='基因编码器：kmer=端到端CNN（主模型），dnabert2=预训练DNA模型')
    p.add_argument('--cache_dir',       default='.pretrained_cache',
                   help='预训练嵌入缓存目录（首次运行自动计算，后续直接加载）')
    p.add_argument('--precompute',      action='store_true',
                   help='仅计算并保存嵌入缓存，不训练（首次运行时使用）')
    args = p.parse_args()

    if args.precompute:
        data_dir  = Path(args.data_dir)
        cache_dir = Path(args.cache_dir)
        device    = torch.device(args.device)
        if args.drug_encoder in DRUG_PRETRAINED:
            build_drug_cache(data_dir, args.drug_encoder, cache_dir, device)
        if args.gene_encoder in GENE_PRETRAINED:
            build_gene_cache(data_dir, args.gene_encoder, cache_dir, device)
        print("预计算完成。使用相同参数（去掉 --precompute）开始训练。")
    else:
        train(args)
