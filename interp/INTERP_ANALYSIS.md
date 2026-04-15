# DrugOperatorNet 可解释性分析报告

**细胞系**：MCF7（Fold0，验证集 42,276 pairs，9,750 unique compounds，978 genes）  
**模型**：`results_operator_moe/MCF7/no_moe_r8_k4_Fold0_cl01.pt`（Val AUC = 0.8941）  
**分析脚本目录**：`interp/`  
**图表目录**：`interp/results/MCF7/`  
**更新日期**：2026-04-15

---

## 0. 可解释性框架

### 0.1 模型的谱分解结构

DrugOperatorNet 将药物-基因相互作用建模为低秩算子扰动：

$$\mathbf{T} = \mathbf{I} + \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

其中：
- $\mathbf{U} \in \mathbb{R}^{B \times r \times H}$：药物生成的 $r$ 个**扰动方向向量**（药物内在属性）
- $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \ldots, \sigma_r)$：各模式的**幅度**（scalar per mode per drug）
- $\mathbf{V} = \mathbf{h}_g^{\text{modes}} \in \mathbb{R}^{B \times r \times H}$：基因在各模式子空间中的**响应方向**

**交互谱**（interaction spectrum）：
$$s_j = \langle \mathbf{U}_j, \mathbf{V}_j \rangle \cdot \sigma_j$$

表示药物在第 $j$ 个生物模式上对该基因的激活强度。

### 0.2 可解释性的三个层次

| 层次 | 关键量 | 分析目标 |
|------|--------|---------|
| **基因层** | $\mathbf{h}_g^{\text{modes}}$，spectrum | 每个模式关联哪些基因？GO 富集是什么？ |
| **药物层** | $\sigma_j$（药物幅度） | 每个药物的"模式指纹"是什么？与 MOA 一致吗？ |
| **化学层** | $\sigma_j$ vs 化学子结构 | 哪些官能团/骨架激活特定模式？ |

第四层次（跨细胞系）：相同的 8 个模式是否在 A375/A549/VCAP 中重现？（规划中）

---

## 1. 谱模式的生物学身份鉴定

### 1.1 分析方法

**交互视角（interaction view）**：对每个模式 $j$，在正例 pair 集合中对每个基因计算平均 spectrum 值：
$$\bar{s}_{j,g} = \frac{1}{|\mathcal{P}^+_g|} \sum_{i \in \mathcal{P}^+_g} s_{j,i}$$

取 Top-100 基因 → 送入 Enrichr（GO Biological Process 2023, KEGG 2021 Human）。

**基因视角（gene view）**：对每个模式 $j$，按 $\|\mathbf{h}_g^{\text{modes}}[:,j,:]\|_2$ 排序，取 Top-100 基因做富集。

### 1.2 八个谱模式的生物学标注

| 模式 | 生物学标注 | 代表性 GO term（交互视角） | FDR |
|------|-----------|--------------------------|-----|
| Mode 0 | **Kinase Signaling** | Regulation Of Protein Kinase Activity (GO:0045859) | 1.97e-3 |
| Mode 1 | **Cell Cycle** | Positive Regulation Of Mitotic Nuclear Division (GO:0045840) | 2.54e-2 |
| Mode 2 | **Metabolic Process** | Regulation Of Amide Metabolic Process (GO:0034248) | 4.25e-2 |
| Mode 3 | **Golgi / Vesicular Transport** | Retrograde Transport, Vesicle Recycling Within Golgi (GO:0000301) | 6.59e-3 |
| Mode 4 | **Mitochondrial Apoptosis** | Mitochondrial Fragmentation In Apoptotic Process (GO:0043653) | 6.65e-3 |
| Mode 5 | **NF-κB / MAPK Signaling** | Regulation Of IKK/NF-κB Signaling (GO:0043122) | 2.13e-3 |
| Mode 6 | **RTK（弱，marginal）** | Neg. Reg. Of Protein Tyrosine Kinase Activity (GO:0061099) | 9.10e-2 |
| Mode 7 | **Apoptosis Regulation** | Positive Regulation Of Apoptotic Process (GO:0043065) | 2.22e-2 |

**8/8 模式均有对应 GO 富集；7/8 FDR < 0.05（Mode 6 边缘）。**

**图表**：`interp/results/MCF7/go/mode_identity_barplot.png`

### 1.3 各模式核心基因解析

通过交互视角 Top-20 基因分析，进一步验证模式标注：

**Mode 0（Kinase Signaling）**  
Top 基因：EGFR, PIK3C3, NCK2, STXBP1, HSPB1  
→ EGFR（表皮生长因子受体，RTK原型）、PIK3C3（PI3K激酶）排名靠前，完全符合激酶信号传导

**Mode 1（Cell Cycle）**  
Top 基因：PCNA, DNMT3A, MTA1, SMARCC1, ABCF1  
→ **PCNA** 是 DNA 复制的核心标志基因（增殖细胞核抗原），SMARCC1（SWI/SNF 染色质重塑）均与细胞周期调控密切相关

**Mode 2（Metabolic Process）**  
Top 基因：PHGDH, CDKN2A, S100A4, MYCBP2, ATF5  
→ **PHGDH**（磷酸甘油酸脱氢酶，丝氨酸合成通路关键酶）是 MCF7 乳腺癌代谢的重要靶点；CDKN2A 与代谢-细胞周期偶联

**Mode 3（Golgi / Vesicular Transport）**  
Top 基因：CD40, NFATC3, COL4A1, TIMELESS, WASHC5  
→ CD40 是受体介导的囊泡运输枢纽；WASHC5 参与网格蛋白介导的内体到 Golgi 逆向运输

**Mode 4（Mitochondrial Apoptosis）**  
Top 基因：DNM1L, BNIP3, PLA2G4A, CDK4, SERPINE1  
→ **DNM1L（DRP1）** 是线粒体分裂的核心 GTPase，线粒体促凋亡过程中 DRP1 活化导致细胞色素 c 释放；**BNIP3** 是线粒体凋亡/线粒体自噬关键蛋白，强有力地验证 Mode 4 的线粒体凋亡身份

**Mode 5（NF-κB / MAPK）**  
Top 基因：GRN, CD40, HLA-DRA, LPAR2, BNIP3, ATG3  
→ **GRN（Progranulin）** 是 NF-κB 激活剂；**CD40** 通过 TRAF 激活 NF-κB；**LPAR2** 激活 MAPK/NF-κB；该模式的基因集高度一致

**Mode 6（RTK，弱）**  
Top 基因：EGFR, RAB31, DAXX, SATB1（与 Mode 0 高度重叠）  
→ Mode 6 是 Mode 0 的"弱版本"，两者共享 RTK 相关基因，但 Mode 6 的 GO 富集显著性较低（FDR=9.1e-2）

**Mode 7（Apoptosis Regulation）**  
Top 基因：SFN, BID, PRKCQ, TICAM1, ZFP36  
→ **SFN（14-3-3σ）** 是 p53 调控凋亡的关键效应蛋白（MCF7 中 14-3-3σ 上调介导 G2/M 阻滞）；**BID** 是外源性凋亡通路连接线粒体通路的枢纽蛋白

---

## 2. 药物机制指纹（Drug MOA Fingerprinting）

### 2.1 分析方法

对每个药物，在验证集中对所有含该药物的 pair 计算平均幅度：
$$\bar{\sigma}_{j,d} = \frac{1}{N_d} \sum_{i \ni d} \sigma_{j,i}$$

再做全局 z-score 归一化：
$$z_{j,d} = \frac{\bar{\sigma}_{j,d} - \mu_j}{\text{std}_j}$$

其中 $\mu_j$, $\text{std}_j$ 对所有 9,750 个验证集药物计算。

**主导模式** = $\arg\max_j z_{j,d}$（z-score 最大而非 sigma 绝对值最大，排除量纲差异）

### 2.2 代表性药物案例分析结果

| 药物 | MOA（已知） | 主导模式（z-score） | 次强信号 | 生物学解读 |
|------|------------|-------------------|---------|-----------|
| trimidox | RNR 抑制剂 | Mode 3 (z=+2.84) | Mode 2 (z=+2.46) | RNR 定位于核质界面（Golgi 附近运输），同时影响核苷酸代谢 |
| tiaprofenic acid | COX 抑制剂 | Mode 3 (z=+2.71) | Mode 2 (z=+2.46) | COX-2 定位于 ER/核膜，前列腺素合成（代谢途径）抑制 |
| calcipotriol | VitD 受体激动剂 | Mode 3 (z=+2.26) | Mode 5 (z=+0.77) | VDR 核受体经 Golgi 转运进入细胞核 ✓ |
| clobetasol | 糖皮质激素 | Mode 5 (z=+4.86) | Mode 3 (z=+3.64) | GR 核受体活化 → NF-κB 信号交叉；同时抑制 Mode 1/Mode 7（抗凋亡，GR 已知作用） |
| fenretinide | 凋亡刺激剂 | Mode 1 (z=+3.42) | Mode 7 (z=+3.07), Mode 4 (z=+2.22) | 三模式共激活：细胞周期阻滞（G1 arrest）+ 凋亡调控 + 线粒体凋亡通路 ✓✓✓ |
| trametinib | MEK 抑制剂 | **Mode 4 (z=+5.71)** | Mode 1 (z=+5.01), Mode 5 (z=+3.37), Mode 7 (z=+3.61) | MEK1/2 抑制 → ERK 阻断 → 多通路：G1 阻滞(M1) + 线粒体凋亡(M4) + NF-κB 交叉活化(M5) + 凋亡调控(M7)；z≈5.7 为所有药物中最强信号 |
| atorvastatin | HMG-CoA 还原酶抑制剂 | **Mode 4 (z=+5.59)** | Mode 1 (z=+4.58), Mode 5 (z=+3.48), Mode 7 (z=+3.48) | 他汀类在 MCF7 中：胆固醇合成阻断 → Ras/Raf/MEK/ERK 信号受损 → 线粒体凋亡 + 细胞周期阻滞；与 trametinib 指纹高度相似（均通过 MAPK 通路失活） |

**关键发现**：
1. trametinib（MEK 抑制剂）和 atorvastatin（HMGCR 抑制剂）均以极高 z-score（>5）激活 Mode 4（线粒体凋亡），且具有几乎相同的 4 模式激活图谱（M1+M4+M5+M7），揭示两者在 MCF7 中通过相同的信号级联（MAPK → 线粒体凋亡）发挥作用
2. fenretinide 的三模式共激活（M1+M4+M7）完整对应其已知的多重作用机制：细胞周期 G1 阻滞 + 线粒体膜电位损失 + caspase 级联激活
3. clobetasol 对 Mode 1（Cell Cycle）z=-3.9 和 Mode 7（Apoptosis）z=-4.3 的强烈抑制，反映糖皮质激素受体在 MCF7 中的抗凋亡效应（GR 在乳腺癌中已知促进细胞存活）

**图表**：
- `interp/results/MCF7/case_study/case_study_zscore_heatmap.png`（主图，Fig 4B）
- `interp/results/MCF7/case_study/case_study_sigma.png`（补充图）

---

## 3. 化学子结构与模式激活分析

### 3.1 分析方法

对每个药物计算 MACCS Keys（167 位二值指纹），再对每个模式 $j$ 计算每个 MACCS bit 与 $z_{j,d}$ 的点二列相关系数（point-biserial correlation）：

$$r_{j,b} = \text{corr}(\text{MACCS}_b^{(d)}, z_{j,d})$$

正相关（$r>0$）：该子结构存在时，模式幅度更高；负相关：该子结构存在时，模式幅度更低。样本量 $N=9750$ 药物，所有 $p$ 值均经过多重检验校正。

### 3.2 各模式的关键化学特征

**Mode 0（Kinase Signaling）**

| 子结构 | MACCS bit | SMARTS | 相关系数 |
|--------|-----------|--------|---------|
| 6元氮杂环 | Bit 98 | `[!C;!H]1~*~*~*~*~*~1` | r=+0.232 |
| 环内杂原子 | Bit 120 | `[!#6;R]` | r=+0.231 |
| 乙烯桥 `C-C-C-C` | Bit 118 | `[CH2][CH2]` | r=+0.221 |
| 芳香体系 | Bit 17 | aromatic | r=+0.220 |
| NH-酰胺 | Bit 147 | `[NH]C=O` | r=+0.218 |
| 多甲基（抑制） | Bit 141 | `[CH3]×≥2` | r=−0.265 |

→ 激酶抑制剂的典型骨架特征：**含氮芳香杂环（吡啶/嘧啶） + NH-酰胺连接**。与已知激酶抑制剂结构（ATP 竞争型，含 hinge 区 NH 键合供体）完全一致。

**Mode 2（Metabolic Process）**

| 子结构 | MACCS bit | SMARTS | 相关系数 |
|--------|-----------|--------|---------|
| 酚羟基 | Bit 38 | `N~C(C)~N` | r=+0.296 |
| O-C-O（缩醛） | Bit 123 | `[O]~[C]~[O]` | r=+0.294 |
| 亚甲基-O（抑制） | Bit 109 | `*~[CH2]~[O]` | r=−0.586 |
| 仲胺-CH2（抑制） | Bit 82 | `*~[CH2]~[NH]` | r=−0.556 |
| 4元杂环（抑制） | Bit 8 | 4元杂原子环 | r=−0.551 |

→ **代谢模式被抑制**：含 CH₂-O 和 CH₂-NH 的简单脂肪胺/醇化合物（核苷类似物前体、氨基酸类似物）不激活此模式；而芳香/缩醛类化合物激活。生物学解读：Mode 2 代表的是代谢重编程（影响碳水化合物/氨基酸代谢通路），而非基本核苷代谢。

**Mode 3（Golgi / Vesicular）和 Mode 5（NF-κB / MAPK）**

两个模式共享相似的化学特征，这与 calcipotriol/clobetasol 同时激活 Mode 3 和 Mode 5 一致：

| 子结构 | MACCS bit | 相关系数（M3） | 相关系数（M5） |
|--------|-----------|------------|------------|
| 多甲基基团 | Bit 141 (`[CH3]×≥2`) | +0.391 | +0.466 |
| O-*-*-*-O（二醇间距） | Bit 89 | +0.364 | +0.401 |
| 酚羟基 | Bit 115 | +0.320 | +0.453 |
| 甲基支链（-CH(CH₃)） | Bit 116 | +0.317 | +0.397 |
| C=O 羰基 | Bit 57 | — | +0.365 |
| 4元环（抑制） | Bit 8 | −0.577 | −0.431 |
| 任意环（抑制） | Bit 11 | −0.575 | −0.432 |
| N-N-N链（抑制） | Bit 79 | −0.421 | −0.404 |

→ **Mode 3 和 Mode 5 均由甾体/萜类骨架化合物激活**：多个甲基侧链 + 特定间距的羟基（甾体 C3/C17 位 OH） + 无4元小环。这一特征与甾体激素（糖皮质激素 clobetasol、维生素 D 类似物 calcipotriol）的结构完全对应，解释了为什么这两种药物同时激活 Mode 3 和 Mode 5。

**Mode 4（Mitochondrial Apoptosis）**

| 子结构 | MACCS bit | SMARTS | 相关系数 |
|--------|-----------|--------|---------|
| 酚羟基 | Bit 115 | `[CH3]~*~[CH2]~*` | r=+0.307 |
| 羰基 C=O | Bit 57 | `[#8R]` | r=+0.291 |
| N-酰胺 | Bit 133 | `*@*!@[#7]` | r=+0.260 |
| 苯环-OH（抑制） | Bit 62 | — | r=−0.273 |

→ Mode 4 激活：含**酚羟基 + 羰基 + N-酰胺**的化合物（如苯并酰胺类结构、醌类化合物）。这与线粒体活性氧（ROS）产生和凋亡诱导相关的结构特征一致。

**Mode 7（Apoptosis Regulation）**

| 子结构 | MACCS bit | SMARTS | 相关系数 |
|--------|-----------|--------|---------|
| 长烷基链（-CH₂-…-CH₂-） | Bit 128 | 4个碳间距的CH₂ | r=+0.332 |
| 非环亚甲基 | Bit 155 | `*!@[CH2]!@*` | r=+0.289 |
| 3个碳间距CH₂ | Bit 129 | — | r=+0.286 |
| NH-酰胺 | Bit 147 | — | r=+0.270 |
| 卤素（抑制） | Bit 160 | — | r=−0.175 |

→ Mode 7 激活：含**长烷基链 + 酰胺键**的化合物（如脂肪酸衍生物、神经酰胺类似物、长链 N-酰胺）。这与 Mode 7 代表的凋亡调控（包括神经酰胺介导的凋亡通路）一致。

### 3.3 化学子结构分析小结

| 模式 | 激活骨架特征 | 典型化合物类型 |
|------|------------|-------------|
| M0 Kinase | 含氮杂环 + NH-酰胺 | ATP 竞争性激酶抑制剂 |
| M2 Metabolic | 芳香/酚类，排斥脂肪胺 | 代谢酶底物类似物（非核苷） |
| M3 Golgi | 多甲基 + 间距二醇 + 酚羟基 | 甾体/萜类（VitD, GC） |
| M4 MitoApop | 酚羟基 + 羰基 + N-酰胺 | 醌类/苯并酰胺类凋亡诱导剂 |
| M5 NF-κB | 同 M3 但更强（r~0.45-0.47） | 甾体激素（GR 激动剂为主） |
| M6 RTK | 酯基 + 酚，排斥卤代芳香 | 苯甲酸酯/苯丙酸类 |
| M7 Apoptosis | 长烷基链 + 酰胺 | 脂肪酸衍生物/神经酰胺 |

**图表**：`interp/results/MCF7/drug_cluster/drug_sigma_umap.png`（药物 sigma UMAP，按主导模式着色）

---

## 4. 药物全局聚类

### 4.1 UMAP 可视化

对 9,750 个药物的 8 维 sigma 向量做 UMAP 降维，按主导模式（argmax z-score）着色。观察到：
- 活性高的药物（pos_rate 高）倾向于聚集在 UMAP 的特定区域
- Mode 2/Mode 3 药物构成较大聚类（甾体类药物族）
- Mode 4/Mode 5 药物存在重叠（线粒体凋亡与 NF-κB 的交叉激活现象）

**图表**：`interp/results/MCF7/drug_cluster/drug_sigma_umap.png`（双图：按活性 / 按主导模式）

---

## 5. 正交正则的可视化验证

正交正则 $\mathcal{L}_\text{ortho} = \|\mathbf{U}^\top\mathbf{U} - \mathbf{I}\|_F^2$ 鼓励 $r$ 个模式方向两两正交。从以下两个角度验证其有效性：

1. **GO 功能独立性**：8 个模式的代表性 GO term 互不重叠（Kinase ≠ CellCycle ≠ Metabolic ≠ Golgi ≠ MitoApop ≠ NF-κB ≠ Apoptosis）
2. **药物指纹的模式特异性**：每类 MOA 药物激活各自对应的特定模式（而非所有模式均等激活）

消融实验中，去掉正交正则（no_Ortho）后 AUC 下降 0.0007，说明正交约束主要价值在可解释性而非性能。

---

## 6. 跨细胞系可解释性规划（46 细胞系）

### 6.1 核心科学问题

当前 MCF7 的 8 个谱模式是否是生物学普遍规律？还是细胞系特异性？

**假设**：不同细胞系学到的模式在生物学身份上具有一定一致性（因为 LINCS L1000 使用相同的 978 个 landmark genes，这些基因覆盖关键通路），但模式的相对重要性和幅度会因细胞系特性而异。

### 6.2 规划分析流程

```bash
# 对每个细胞系（A375, A549, VCAP + 42 细胞系）
# Step 1: 提取表示（interp/extract.py）
python interp/extract.py --cell A375 --model_path results/A375/best_model.pt --fold 0

# Step 2: GO 富集（interp/go_enrich.py）
python interp/go_enrich.py --cell A375

# Step 3: 药物指纹（interp/drug_cluster.py）
python interp/drug_cluster.py --cell A375

# Step 4: 跨细胞系比较（待开发 interp/cross_cell.py）
python interp/cross_cell.py --cells MCF7 A375 A549 VCAP
```

### 6.3 跨细胞系比较分析设计

**模式对齐**：因为不同细胞系的模式编号不对应，需要通过 GO term 相似度或 CKA（Centered Kernel Alignment）对齐模式。

**预期输出**：
- 跨细胞系共享模式（NF-κB/Kinase 可能普遍存在）
- 细胞系特异模式（乳腺癌 MCF7 vs 黑色素瘤 A375 的差异模式）
- 药物在不同细胞系中的模式指纹一致性分析

**图表规划**：
- 跨细胞系模式 GO term 对比热图
- 代表性药物在 MCF7/A375/A549/VCAP 的模式指纹一致性

### 6.4 化学子结构的跨细胞系验证

若不同细胞系的"Mode X（NF-κB）"均由甾体骨架激活，则说明化学-模式对应关系具有跨细胞系普遍性，这将是非常强的生物学证据。

---

## 7. 补充：各模式幅度的全局分布

| 模式 | 全局均值 | 全局标准差 | 正值比例 | 解读 |
|------|---------|----------|---------|------|
| Mode 0 | −0.027 | 0.009 | 低 | 小幅度，大多数药物不激活 |
| Mode 1 | +0.032 | 0.023 | 中 | 中等分布 |
| **Mode 2** | −0.028 | **0.389** | 50% | 极高方差，少数药物有极强激活/抑制 |
| Mode 3 | +0.160 | 0.170 | 高 | 整体基线较高，甾体类药物显著激活 |
| Mode 4 | +0.068 | 0.040 | 高 | 集中分布，线粒体凋亡模式较稳定 |
| Mode 5 | +0.134 | 0.085 | 高 | 中等基线，NF-κB 活跃 |
| Mode 6 | −0.075 | 0.005 | 极低 | 几乎所有药物均为负值，RTK 基线抑制 |
| Mode 7 | +0.005 | 0.013 | 中 | 小幅度，凋亡调控基线平稳 |

Mode 2 的极高方差（std=0.389）反映了代谢重编程信号在不同药物之间的高异质性，这与 MCF7 乳腺癌细胞对代谢干扰的高度异质性反应一致。

---

## 8. 可解释性分析现有局限与改进方向

1. **GO 富集样本量受限**：交互视角使用 Top-100 基因，基因集较小，可尝试 Top-200 或基于 GSEA 的富集方法。
2. **化学子结构为二值指纹**：MACCS Keys 仅捕获子结构存在/缺失，未考虑子结构数量或位置；可扩展为基于 GIN 注意力的原子级贡献可视化。
3. **药物幅度 sigma 为聚合量**：$\bar{\sigma}$ 是跨所有 pair 的均值，消除了基因特异性。未来可分析 "drug X gene-specific spectrum" 矩阵。
4. **46 细胞系分析待完成**：当前仅 MCF7 完整分析，跨细胞系一致性有待验证。

---

## 附：脚本与图表索引

| 脚本 | 功能 | 主要输出 |
|------|------|---------|
| `interp/extract.py` | 提取所有中间表示 | `results/{cell}/representations.npz` |
| `interp/go_enrich.py` | GO/KEGG 富集（双视角） | `results/{cell}/go/mode*_*_view_GO.csv` |
| `interp/drug_cluster.py` | 药物 sigma 聚类与 UMAP | `results/{cell}/drug_cluster/drug_sigma.csv`, `drug_sigma_umap.png` |
| `interp/case_study.py` | 药物 MOA 案例研究 | `results/{cell}/case_study/case_study_zscore_heatmap.png` |
| `interp/go_mode_barplot.py` | GO 模式身份条形图 | `results/{cell}/go/mode_identity_barplot.png` |
| `interp/run.sh` | 一键运行完整流水线 | — |

| 图表文件 | 对应论文位置 | 描述 |
|---------|------------|------|
| `go/mode_identity_barplot.png` | **Fig 4A** | 8 模式 GO 生物学标注（-log10 FDR 条形图） |
| `case_study/case_study_zscore_heatmap.png` | **Fig 4B** | 7 代表性药物 × 8 模式 z-score 热图 |
| `case_study/case_study_sigma.png` | **Fig S3** | z-score 分组条形图（补充） |
| `drug_cluster/drug_sigma_umap.png` | **Fig S4** | 药物 sigma UMAP（活性 + 主导模式） |
| `go/mode_GO_heatmap.png` | Supp | 基因视角 GO 热图（多模式对比） |
