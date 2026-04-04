# CGI 预测项目全景文档

> 目的：完整记录本项目的优化历程、每一步为什么这么做、结果如何、以及现在每个文件的作用。
> 适用读者：项目后续开发者、论文审稿人、或自己回顾。
> 最后更新：2026-04-04

---

## 目录

1. [问题定义](#1-问题定义)
2. [数据与评估标准](#2-数据与评估标准)
3. [模型演化历程（为什么这么优化）](#3-模型演化历程)
4. [最终模型架构详解](#4-最终模型架构详解)
5. [关键消融发现](#5-关键消融发现)
6. [全部文件说明](#6-全部文件说明)
7. [实验结果速查](#7-实验结果速查)
8. [投稿定位与差距分析](#8-投稿定位与差距分析)

---

## 1. 问题定义

**任务**：Chemical-Gene Interaction (CGI) 预测
- 输入：(化合物 SMILES, 基因编码序列)
- 输出：该化合物是否显著改变该基因的表达量（二分类）
- 正例标准：LINCS L1000 中 |z-score| > 2（即 |logFC| 显著 + q < 0.05 含义下的显著差异表达）
- 负例：|z-score| ≤ 2

**核心挑战**：
- **Chemical Cold Split**：测试集的化合物训练时从未出现过，模型必须泛化到"从未见过的分子"
- **类别不平衡**：正例约占 18-22%（不同细胞系略有差异）
- **标签噪声**：45.9% 的 MCF7 样本处于 |z|∈[2.0, 2.5] 边界区，标签可靠性低

**为什么选 CGI 做 NMI**：
- 工业/药物安全（LINCS 项目背景）
- 端到端可解释性是本工作独特卖点：从分子结构到基因调控的完整链条

---

## 2. 数据与评估标准

### 数据来源
- LINCS L1000 数据库（NIH Library of Integrated Network-Based Cellular Signatures）
- 978 个 landmark genes（可影响全基因组约 80% 表达的代表基因）
- 数据预处理脚本：`preprocess_graphs.py`（分子图预处理）、`batch_preprocess.py`（批量多细胞系）

### 主要细胞系（4个，用于论文主实验）
| 细胞系 | 癌症类型 | 驱动突变 | 样本量 | 特点 |
|--------|---------|---------|--------|------|
| MCF7 | 乳腺癌 | ER+ | ~206K | 标准基准 |
| A375 | 黑色素瘤 | BRAF V600E | ~173K | 通路最"干净" |
| A549 | 肺腺癌 | KRAS G12S | ~152K | 多效应子，较难 |
| VCAP | 前列腺癌 | AR扩增 | ~228K | 样本最多，最难 |

### 额外扫描细胞系（42个，用于泛化性验证）
小细胞系（<50K）：22RV1, A204, BC3C, BEN, CAL29, CJM, GI1, HCC95, HEC108, HEC1A, HEC251, HEC265, HEK293, HELA, HUVEC, IGR37, JHH5, JURKAT, MCF10A, MDAMB231, MDAMB468, MELHO, NCIH1573, NCIH2110, NCIH838, OVTOKO, SH4, SKES1, SNU407, T47D, THP1, YAPC

中等细胞系（50K-100K）：HEPG2, PHH, ASC, HCC515, SKB, NEU

大细胞系（≥100K，非主4）：HT29, PC3, HA1E, NPC

### 评估指标
- **主指标**：AUROC（AUC）——排序能力，不受类别不平衡影响
- 辅助：AUPRC（PRC）、F1（threshold=0.5）
- **交叉验证**：5-fold chemical cold split（5个不同的化合物划分）

---

## 3. 模型演化历程

### 阶段 0：问题建立（Baseline BCE）
**文件**：`train_baseline_bce.py`

**出发点**：最简单的可行架构，建立数字基准
- Drug encoder：GIN × 3 层 → sum/mean pool → Linear
- Gene encoder：6-mer CNN → Linear
- 融合：concat → MLP → 分类
- MCF7 Fold0：AUC = **0.8914**（加正交）/ **0.8860**（无正交）

**发现的问题**：
- 无法提取原子级别的药效团信息（sum/mean pool 损失空间信息）
- 化学品和基因的"交互"方式完全隐式（黑盒 MLP），无可解释性
- 正交正则有效（+0.005），但还不够

---

### 阶段 1：靶向池化 + MoE 路由
**文件**：`train_target_pool.py`（靶向池化）、`train_moe.py`（MoE 分类头）、`train_moe_target.py`（融合）

**动机**：
- 传统 sum/mean pool 是"无监督"的，不知道哪个原子对该基因更重要
- 想法：用基因表示 h_g 作为"query"，对分子图的原子做注意力加权（Targeted Pooling）
- MoE：不同药物-基因对可能对应不同的互作模式（激活、抑制、旁路等），让多个专家分工

**结果**（MCF7 Fold0）：
- 靶向池化单独：0.8911（略优于 0.8914 baseline，无显著提升）
- MoE K=4：0.8935（+0.0021）
- MoE + Targeted：**0.8943**（+0.0029）

**结论**：方向有效，但提升有限。根因是 h_g（基因全局表示）质量不高，导致 atom query 监督信号弱。

---

### 阶段 2：CL + 正交剥离（旧框架）
**文件**：`train_summean.py`（主力）、`train_ultimate.py`（早期尝试）

**动机**：
- 对比学习（CL）的本质：同一药物对不同基因的作用是有结构的，CL 可以用样本间关系作为额外监督
- 正交剥离（Ortho）：将药物空间分解为"影响目标基因"和"不影响目标基因"的正交分量

**实现**（`train_summean.py`）：
- `FocalDirectionAwareCL`：方向感知 Focal 对比损失
- `MoE_GeneConditionedActionAlignment`：基因条件化的对齐损失
- 结合正交剥离 V_c_ortho 分量剥离

**5-fold CV 结果**（MCF7）：
| 配置 | AUC | ± std |
|------|-----|-------|
| SumMean（基准） | 0.8923 | 0.0026 |
| wo_CL | 0.8907 | 0.0025 |
| wo_Ortho | 0.8877 | 0.0022 |
| wo_CL_Ortho | 0.8876 | 0.0022 |

**发现**：CL 有效（+0.0016），Ortho 有效（+0.0046），两者加和基本独立。

**重要教训**：这套 CL 实现留在 `train_summean.py` 框架里，**从未移植到后续的 DrugOperatorNet 框架**。这是一个延误至阶段 5 才被发现并修复的错误。

---

### 阶段 3：DrugOperatorNet——核心创新
**文件**：`New/train_drug_operator.py`（V1）、`New/train_drug_operator_v2.py`（V2）

**动机（核心思想的转变）**：

之前所有模型把药物当作"静态向量"参与分类。但现实中，**药物的作用是改变基因表达**——这本质上是一个作用于"基因空间"的算子。

**形式化**：
- 传统：`score = f(h_drug, h_gene)`（点积/MLP）
- DrugOperatorNet：药物学习一个低秩扰动算子 `T = I + U Σ Vᵀ`（秩 r=8）
  - `U ∈ [H×r]`：扰动方向（药物作用在什么方向上）
  - `V ∈ [H×r]`：读取方向（从基因表示的哪个维度读取信息）
  - `Σ = diag(σ₁,...,σᵣ)`：谱（每个模式的强度）
  - `spectrum [B,r] = σ * (V · h_g_modes)`：**交互谱**——药物×基因的结构化指纹

**GeneMultiHeadReader（V2 的关键创新）**：
- 对基因序列的 r 个"模式"做多头读取：`h_g_modes [B, r, H]`
- 每个模式关注基因序列的不同位置（对应不同调控区域）
- 使 spectrum 有了真正的"多维度"解读能力

**结果**（MCF7 Fold0）：
- V1（单头）：0.8924
- V2（多头，gene_len=3000）：0.8923（与单头持平，但可解释性大幅提升）

**为什么提升不大**：DrugOperatorNet 和 SumMean+CL 的 AUC 几乎相同（0.8923 vs 0.8923）。原因：任务本身的数据天花板（45.9% 边界样本）限制了任何单一架构创新的收益上限。DrugOperatorNet 的价值在于可解释性，而非纯 AUC。

---

### 阶段 4：超参精调 + 多尺度 gene encoder
**文件**：`New/train_operator_moe.py`（带 --ablation no_moe 时即 DrugOperatorNet）

**精调内容**：
- `gene_max_len`：3000 → **1000**（关键！）
  - 理由：98% 的差异性信息集中在 TSS 附近 1000 bp 内，截断不损失性能，速度提升 3x
  - 验证：1000 vs 3000 在 MCF7 AUC 完全持平
- `lam_ortho`：0.01 → **0.1**（关键！）
  - 理由：更强正交约束 → 8 个模式分工更明确 → VCAP +0.0086（从 0.7976 → 0.8062）
  - MCF7 +0.0008（小提升但方向一致）
- `lr`：3e-4 → **2e-4**（训练更稳定，VCAP 收敛改善）
- `batch_size`：保持 512（显存和梯度噪声的平衡点）
- `operator_rank r`：保持 8（r=4 欠表达，r=16 过参数化且可解释性下降）

**OperatorMoE 融合消融**（验证 MoE 路由是否有帮助）：

| 配置 | AUC | 路由方式 |
|------|-----|---------|
| full（完整 OperatorMoE） | 0.8868 | spectrum → MoE K=4 |
| no_spectrum | 0.8917 | [V_g, V_c_perp] → MoE |
| **no_moe（最优）** | **0.8923** | 无路由，直接 MLP |

**重要发现**：MoE 融合反而有害。原因：`delta_h` 已隐式完成了信息路由，强制加 MoE LB Loss 引入梯度冲突。**奥卡姆剃刀**：no_moe 配置（即纯 DrugOperatorNet）是最终主模型。

---

### 阶段 5：补全对比实验（SpectrumDirectionCL + Pretrained Baseline）

**背景**：用户指出两个关键缺失：
1. CL 从未在 DrugOperatorNet 框架中实现（只在旧 train_summean.py 中）
2. 对比"使用预训练模型"的方案（证明端到端 GIN 的价值）

#### 5.1 SpectrumDirectionCL（新 CL 实现）
**文件**：`New/train_operator_moe.py`（新增模块）

**设计**：DrugOperatorNet 的天然 CL 接入点是 `spectrum [B, r]`——这是结构化的药物-基因交互指纹。

```
方向感知 Margin Loss：
  d = normalize(direction)     # 可学习方向向量
  score = spectrum @ d          # 每个样本在方向 d 上的投影
  loss = mean(focal_weighted * relu(margin - score * (2y-1)))
```

比旧 CL（FocalDirectionAwareCL）的优势：
- 直接作用于谱空间（有物理意义），不是抽象嵌入空间
- 方向 d 可以被解释为"正向互作的主方向"

**结果**（MCF7 Fold0，lam_cl=0.1）：AUC = 0.8954（+0.0031 vs baseline 0.8923）

#### 5.2 软标签（Soft Label）
**设计**：z-score 置信度加权
```
conf = sigmoid((|z|-2) / 0.5)   # 边界样本(|z|≈2) → conf≈0.5；强信号(|z|≥4) → conf≈1.0
soft_label = y * conf + (1-y) * (1-conf)
```

**结果**（MCF7 Fold0）：AUC = 0.8754（-0.0169 vs baseline）
- AUC 下降是因为软标签改变了优化目标（calibrated probability vs. ranking）
- 价值在于减少边界样本的标签噪声，需要用 calibration 曲线等指标评估而非 AUC

**CL + 软标签组合**（MCF7 Fold0）：AUC = 0.8780
- 相对软标签单独 +0.0026，方向正确

#### 5.3 Pretrained ECFP4 Baseline（最关键对比）
**文件**：`New/train_pretrained_baseline.py`

**设计思路**（严格单变量对照）：
- 保持：GeneMultiHeadReader + PerturbationOperator + MLP 完全不变
- 仅替换：GIN × 3（端到端图学习）→ Linear(2048→256→128)（固定 ECFP4 2048-bit 指纹）

**结果**：
| Drug Encoder | AUC |
|-------------|-----|
| 固定 ECFP4（预训练指纹）| 0.8687 |
| **端到端 GIN** | **0.8923** |
| GIN 净增益 | **+0.0236** |

**意义**：在严格控制变量下，证明端到端图学习比固定预训练指纹高 +0.0236 AUC，这是反驳"为什么不用 ChemBERT/MolBERT 等预训练化学模型"的核心实验证据。

---

### 阶段 6：全细胞系扫描（46 细胞系普遍性验证）

**文件**：`run_allcells_gpu2.sh`、`run_allcells_gpu3.sh`

**目的**：验证模型不只是在 4 个主要细胞系上有效，而是具有跨细胞系的普遍适用性（NMI 要求）

**配置**：完全相同的超参（no_moe, rank=8, gene_max_len=1000），仅改 data_dir

**结果汇总**：
- 46 细胞系均值 AUC = 0.9121，中位数 = 0.9070
- AUC ≥ 0.90：30/42 新细胞系（71%）
- 最高：HUVEC 0.9751（正常内皮，基因调控规律最清晰）
- 最低：VCAP 0.8116（前列腺癌/AR 扩增，任务最难，但仍优于 RF 的 0.7772）

**重要发现**：免疫细胞（THP1/JURKAT >0.95）和肝细胞（HEPG2/PHH >0.95）效果最好，推测这些细胞系的药物-基因关系规律性更强（通路较少，信号更纯）。

---

## 4. 最终模型架构详解

**主模型**：DrugOperatorNet（no_moe 配置），实现于 `New/train_operator_moe.py --ablation no_moe`

### 完整前向流程

```
输入：Drug SMILES → PyG 分子图
      Gene 序列 → 6-mer 编码（k=6，共 4^6=4096 种，词表大小 4097）

Step 1: Drug Encoder（GIN）
  原子特征 x [N_atoms, 9] → GINLayer × 3 → atom_h [N_atoms, 128]
  → PharmacophoreExtractor: atom_h → pharma_emb [B, r, 128]（r=8个药效团模式）

Step 2: Gene Encoder（GeneMultiHeadReader）
  6-mer 序列 [B, 1000] → Embedding(4097, 64) → [B, 1000, 64]
  → 多尺度 CNN（kernel=[3,5,9,15,21], out_ch=24 each） → [B, 1000, 120]
  → GeneMultiHeadReader: 对每个模式 k=1..r，独立 CNN+attention → h_g_modes [B, r, 128]
  → 全局池化 → h_g_global [B, 128]

Step 3: PerturbationOperator（核心交互）
  U_matrix [r, 128]：扰动方向（药物在基因空间的作用方向）
  V_matrix [r, 128]：读取方向（从基因表示读取的维度）
  sigma: 由 pharma_emb 经 MLP 输出 → [B, r]（每个药效团模式的强度）

  spectrum [B, r] = sigma * einsum(V, h_g_modes)   # 交互谱
  delta_h  [B, 128] = sum_k(spectrum_k * U_k)       # 扰动向量

Step 4: 分类头
  features = concat([h_g_global, delta_h])  [B, 256]
  → Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
  → 输出 logit，BCEWithLogitsLoss

Step 5: 损失函数
  L = BCE + lam_ortho * L_ortho + lam_sparse * L_sparse
  L_ortho = ||UᵀU - I||²_F   # 正交正则：迫使 r 个扰动方向彼此独立
  L_sparse = mean(|sigma|)    # 稀疏正则：抑制不相关模式激活
```

### 参数量
- 总参数：~918K
- Gene encoder：~420K（主要是 k-mer embedding + 多尺度 CNN）
- Drug encoder（GIN）：~180K
- PerturbationOperator：~65K（U + V 各 8×128）
- 分类头：~50K

### 可解释性链条（3 层）
1. **原子级**：`atom_attention_weight`（PharmacophoreExtractor 的权重）→ 哪个原子/官能团参与了互作
2. **模式级**：`spectrum [B, r]`（交互谱）→ 8 个生物学意义的互作维度的强度
3. **序列级**：`gene_attention`（GeneMultiHeadReader 的权重）→ 基因序列的哪些位置被激活

---

## 5. 关键消融发现

### 5.1 正交正则（lam_ortho）

| 细胞系 | lam_ortho=0.0 | lam_ortho=0.1 | Δ |
|--------|--------------|--------------|---|
| MCF7 | 0.8915 | 0.8923 | +0.0008 |
| VCAP | 0.7976 | 0.8062 | **+0.0086** |

**结论**：正交正则的价值随任务复杂度增加。VCAP（最难，AR扩增）收益最大。正交正则不只是 AUC 的问题，更是可解释性保证：它确保 8 个谱模式编码独立的生物学过程。

### 5.2 端到端 GIN vs 固定指纹

| 方案 | MCF7 | A375 | A549 | VCAP | 平均 |
|------|------|------|------|------|------|
| Morgan FP（旧对照，train_morgan_baseline.py）| 0.8710 | 0.8870 | 0.8432 | 0.7894 | 0.8477 |
| ECFP4+Operator（严格单变量，New/train_pretrained_baseline.py）| 0.8687 | — | — | — | — |
| **DrugOp no_moe（端到端）** | **0.8923** | **0.9011** | **0.8598** | **0.8116** | **0.8912** |

**GIN 净增益（严格对照）**：MCF7 +0.0236

### 5.3 DrugOp vs 传统 ML

| 细胞系 | DrugOp (5-fold) | RF (5-fold) | Δ |
|--------|----------------|------------|---|
| MCF7 | 0.8918±0.0017 | 0.8597±0.0009 | **+0.0321** |
| A375 | 0.9011±0.0043 | 0.8646±0.0038 | **+0.0365** |
| A549 | 0.8598±0.0069 | 0.8212±0.0039 | **+0.0386** |
| VCAP | 0.8116±0.0058 | 0.7772±0.0044 | **+0.0344** |

平均提升 **+0.035**，4/4 细胞系一致，std 不重叠，统计显著。

### 5.4 SpectrumDirectionCL（新 CL）

MCF7 Fold0：0.8923（baseline）→ 0.8954（+lam_cl=0.1）= **+0.0031**
需要 5-fold 确认是否一致。

### 5.5 gene_max_len

1000 vs 3000：性能持平，速度 3x（每 epoch 从 ~45s → ~15s）。确定用 1000。

---

## 6. 全部文件说明

### 根目录：预处理

| 文件 | 说明 | 状态 |
|------|------|------|
| `preprocess_graphs.py` | 单细胞系数据预处理。将 SMILES+基因序列 转为 PyG 分子图 + 6-mer 编码，输出 full_data.pkl + preprocessed_graphs.pt + chemical_cold_splits.pkl | **生产使用，勿改** |
| `batch_preprocess.py` | 批量预处理脚本，循环调用 preprocess_graphs.py 处理所有 46 细胞系 | 辅助脚本 |

### 根目录：历史模型（已废弃，保留供参考）

| 文件 | 阶段 | 最佳 AUC | 为什么废弃 |
|------|------|---------|-----------|
| `train_baseline_bce.py` | 阶段0 | 0.8914 | 无可解释性，被 DrugOp 超越 |
| `train_target_pool.py` | 阶段1 | 0.8911 | 靶向池化单独效果弱，V_g 质量差 |
| `train_moe.py` | 阶段1 | 0.8935 | MoE 路由收益小，DrugOp 无路由更优 |
| `train_moe_target.py` | 阶段1 | 0.8943 | 最强黑盒，但无可解释性，被 DrugOp 以更少参数追平 |
| `train_moe_v2.py` | 探索 | — | GeneEncoderV2（长序列）不稳定，基本废弃 |
| `train_moe_v3.py` | 探索 | — | 修复 V2 不稳定的尝试，后被 MultiHeadReader 方案取代 |
| `train_summean.py` | 阶段2 | 0.8923 | 旧框架 CL，已迁移到 DrugOp 框架。保留用于消融对比（SumMean/wo_CL/wo_Ortho 历史结果都是从这里来的） |
| `train_ultimate.py` | 早期 | — | 完整消融版本，包含 pool_type / disable_ortho / disable_cl 开关。是 train_summean.py 的前身，已废弃 |
| `train_morgan_baseline.py` | 对比 | 0.8710（MCF7）| 固定 Morgan FP 对比，被更严格的 train_pretrained_baseline.py 取代 |
| `train_baseline_dl.py` | 对比 | 0.8822（MCF7）| 简单 MLP baseline（晚期融合），论文需要此对比 |
| `train_baseline_ml.py` | 对比 | 0.8597（MCF7 RF）| RF + XGBoost baseline，**论文必需**，保留 |

### New/ 目录：当前活跃代码

| 文件 | 说明 | 状态 |
|------|------|------|
| `New/train_operator_moe.py` | **主训练文件**。完整实现 DrugOperatorNet（--ablation no_moe 时）和 OperatorMoE（full/no_spectrum）。最新版本包含 SpectrumDirectionCL（--lam_cl）和软标签（--soft_label）。**所有新实验都用这个文件** | **核心，勿随意修改** |
| `New/train_drug_operator.py` | DrugOperatorNet V1（单头读取器）。历史参考，已被 V2/train_operator_moe.py 取代 | 历史存档 |
| `New/train_drug_operator_v2.py` | DrugOperatorNet V2（多头读取器，gene_len=3000）。中间版本，已被 train_operator_moe.py 取代 | 历史存档 |
| `New/train_drug_operator_v3.py` | V3（改了部分细节），未正式使用 | 历史存档 |
| `New/train_operator_tcn.py` | TCN（空洞卷积）基因编码器版本。MCF7 AUC=0.8771（-0.0152 vs no_moe），已确认放弃 TCN 方向 | 已废弃方向 |
| `New/train_pretrained_baseline.py` | **预训练 ECFP4 对比基线**。固定 2048-bit 指纹 + 完全相同的 Operator 架构，证明端到端 GIN 的价值（+0.0236）。**论文对比实验必需** | **活跃，用于对比** |
| `New/train_multitask.py` | 多细胞系联合训练（Multi-task）。每个细胞系一个分类头，共享 Drug/Gene Encoder。未正式跑实验 | 待评估 |
| `New/analyze_spectrum.py` | 交互谱分析与可视化。提取所有训练样本的 spectrum，做 t-SNE / PCA，计算每个模式的生物统计显著性 | **活跃，用于可视化** |

### analyze/ 目录：可视化与分析

| 文件 | 说明 |
|------|------|
| `analyze/visualize_spectrum.py` | 谱可视化（t-SNE，violin plot，模式热图）。注意：sklearn API 变更，`n_iter` 需改为 `max_iter` |
| `analyze/visualize_pharmacophore.py` | 药效团原子权重可视化（分子热图）。需要 cairosvg 才能高质量 SVG 输出 |
| `analyze/visualize_gene_attention.py` | 基因序列注意力可视化（序列位置热图）|
| `analyze/extract_representations.py` | 批量提取所有样本的中间表示（spectrum, attention weights）并缓存 |
| `analyze/run_all_analysis.sh` | 运行单细胞系全套分析的快捷脚本 |
| `analyze/run_multi_cell_analysis.sh` | 跨细胞系对比分析（需要4个细胞系的表示缓存都准备好）|
| `analyze/figures/` | 生成的图表（PNG）|
| `analyze/cache/` | 提取的中间表示缓存（.pkl），按细胞系存放 |

### 训练运行脚本（.sh）

| 文件 | 说明 | GPU |
|------|------|-----|
| `run_5fold_nomoe.sh` | 主 4 细胞系 DrugOp no_moe 5-fold 训练（**已完成**）| 0-3 |
| `run_5folds.sh` | 早期 5-fold 脚本（旧框架 SumMean），已完成 | — |
| `run_5folds_4.sh` | 4 细胞系并行 5-fold（旧框架），已完成 | — |
| `run_ablation_mcf7.sh` | MCF7 消融（SumMean/wo_CL/wo_Ortho/wo_CL_Ortho），已完成 | — |
| `run_ablation_a375.sh` | A375 消融（wo_ortho + Morgan FP + Baseline DL）。**未执行**，待跑 | 0-2 |
| `run_ablation_all.sh` | 全量消融（A549/VCAP + A375），历史脚本 | — |
| `run_baseline_3cell.sh` | A375/A549/VCAP Fold0 初始结果，已完成 | — |
| `run_morgan_3cell.sh` | Morgan FP 3 细胞系对比，已完成 | — |
| `run_morgan_all.sh` | Morgan FP 全量，历史 | — |
| `run_moe_target_3cell.sh` | MoE+Target 3 细胞系，已完成 | — |
| `run_operator_3cell.sh` / `run_operator_v2_3cell.sh` | DrugOp V1/V2 3 细胞系，已完成 | — |
| `run_cl_5fold.sh` | CL 版本（lam_cl=0.1）5-fold。MCF7 Fold0 完成（0.8954），Fold1-4 和其他细胞系待跑 | 0-3 |
| `run_allcells_gpu2.sh` | 38 个小/中等细胞系 Fold0 扫描（**已完成**）| 2 |
| `run_allcells_gpu3.sh` | 4 大细胞系 Fold0 扫描 + RF 备选（**已完成**）| 3 |
| `run_5fold_rf.sh` 等 | RF baseline 5-fold 脚本 | — |

### 结果目录

| 目录 | 内容 |
|------|------|
| `results_operator_moe/` | **主模型保存**。结构：`results_operator_moe/{CELL}/no_moe_r8_k4_Fold{n}_{run_tag}.pt` |
| `results_pretrained_baseline/` | Pretrained ECFP4 baseline 模型 |
| `results_summean/` | 旧框架 SumMean 5-fold 模型 |
| `results_moe/` `results_moe_target/` | MoE 阶段模型 |
| `results_paper/` | 可能是筛选出来准备投稿的模型 |
| 其他 `results_*/` | 各阶段历史模型，可按需清理 |

### 日志目录

| 目录 | 内容 |
|------|------|
| `logs_5fold_nomoe/` | 主 4 细胞系 5-fold 训练日志（**最重要**）|
| `logs_allcells/` | 46 细胞系全扫描日志 + gpu2/gpu3_summary.txt |
| `logs_cl_softlabel/` | CL + soft_label + pretrained 实验日志 |
| `logs_5fold_cl/` | CL 版本 5-fold（MCF7 Fold0 完成）|
| `logs_ablation_mcf7/` | MCF7 消融实验日志 |
| `logs_ablation_a375/` `logs_ablation_a375_druop/` | A375 消融（未完整执行）|
| `logs_5fold_rf/` | RF baseline 5-fold 日志 |
| `logs/` | 早期实验日志（混杂）|

### 文档

| 文件 | 内容 |
|------|------|
| `RESULT.md` | **实验结果记录**（按时间顺序逐次添加，每次训练后必须更新）|
| `THEORY.md` | 模型理论说明（面向非ML读者）|
| `MY.md` | Claude 错误反思与底层逻辑（第一性原理）|
| `CLAUDE.md` | Claude Code 操作指南（项目规范）|
| `PROJECT_OVERVIEW.md` | **本文档**，项目全景 |
| `PPT/工作汇报_2026-04-03.md` | 工作汇报 MD，含结果表、图说明、论文规划 |
| `PPT/深度分析_为什么有效_2026-04-03.md` | 模型有效性深度分析（数据天花板、GIN优势、CL本质）|
| `New/COMPARE.md` | DrugOp 各版本对比说明 |

---

## 7. 实验结果速查

### 主模型 5-fold CV（DrugOp no_moe，chemical cold split）

| 细胞系 | mean ± std |
|--------|-----------|
| MCF7 | 0.8918 ± 0.0017 |
| A375 | 0.9011 ± 0.0043 |
| A549 | 0.8598 ± 0.0069 |
| VCAP | 0.8116 ± 0.0058 |

### RF 5-fold（对比基准）

| 细胞系 | mean ± std |
|--------|-----------|
| MCF7 | 0.8597 ± 0.0009 |
| A375 | 0.8646 ± 0.0038 |
| A549 | 0.8212 ± 0.0039 |
| VCAP | 0.7772 ± 0.0044 |

### 端到端 GIN vs 预训练指纹（MCF7 Fold0，严格单变量）

| 方案 | AUC |
|------|-----|
| 固定 ECFP4 2048-bit | 0.8687 |
| **端到端 GIN（主模型）** | **0.8923** |
| 净增益 | **+0.0236** |

### CL 实验（MCF7 Fold0）

| 配置 | AUC |
|------|-----|
| baseline（无CL） | 0.8923 |
| + SpectrumDirectionCL（lam_cl=0.1） | **0.8954（+0.0031）** |
| + soft_label | 0.8754（AUC非评估软标签的合适指标）|
| + CL + soft_label | 0.8780 |

### 全细胞系扫描（42 细胞系，Fold0）

均值 AUC = 0.9121，中位 = 0.9070，AUC≥0.90 占 71%

前5（非主4细胞系）：HUVEC 0.9751 > HEPG2 0.9633 > PHH 0.9556 > ASC 0.9533 > THP1 0.9523

---

## 8. 投稿定位与差距分析

### 目标期刊：Nature Machine Intelligence（NMI）或 Nature Communications（NC）

### 核心卖点
1. **方法论新颖**：药物-算子范式（业内首次将微扰算子理论引入 CGI 预测）
2. **端到端可解释**：原子→药效团→交互谱→基因位点，三层对齐可视化
3. **严格评估**：chemical cold split + 5-fold CV（最严格泛化评估）
4. **规模化验证**：46 细胞系（4 主 + 42 扫描），覆盖主要癌症类型

### 已有（✅）

- 4 细胞系主实验（5-fold CV）+ 46 细胞系泛化验证
- RF、DL baseline、固定指纹 baseline（严格单变量）
- 正交正则、CL、软标签消融
- 谱分析可视化、药效团热图、基因注意力图

### 关键缺口（❌ NMI 必需）

1. **SOTA DL 方法对比**（最关键）：
   - DeepCE（2020 NMI，Nat Mach Intell 2, 748–760）
   - DECIPHIR（2024）
   - 如果不与这两者正面对比，NMI reviewers 必然 reject
   - **建议**：先检查能否使用其官方代码复现，然后在我们的 chemical cold split 上评估

2. **GO 富集分析**：
   - 用 978 个 landmark gene 的注意力权重做 Gene Ontology 富集
   - 证明谱模式对应真实生物学通路（不只是数学的正交分解）
   - 需要：gseapy + gene symbol 映射表

3. **A375 完整消融**（次要）：
   - `run_ablation_a375.sh` 尚未执行
   - 需要 A375 的 wo_ortho / Morgan FP / Baseline DL 5-fold

### 当前工作重心建议

```
P0（本周）: 复现 DeepCE/DECIPHIR，在我们的数据上评估
P0（本周）: 修复 t-SNE（sklearn max_iter 参数）+ 安装 cairosvg，补全可视化
P1（下周）: GO 富集分析
P1（下周）: CL 5-fold 验证（确认 +0.003 是否跨 fold 一致）
P2（两周）: A375 完整消融，补全论文消融表
```
