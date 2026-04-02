# 两种 CGI 建模思路对比

> 背景：L1000 化学-基因互作预测，二分类（上调/下调），Chemical Cold Split（5-fold）

---

## 一、我们现在做的东西

### 核心架构（train_moe.py，当前最优 AUC=0.8935）

```
Gene → k-mer CNN → V_g
Chem → GIN×3 → sum/mean pool → V_c
Ortho: V_c_perp = V_c - (V_c·V_g)*V_g
Router([V_g, V_c_perp]) → route_weights [B, K]
K expert classifiers → expert_logits [B, K]
final_logit = (route_weights * expert_logits).sum(-1)
```

**设计哲学**：  
药物向量和基因向量分别编码，用正交剥离去除基因信息中已有的化学偏置，再用 MoE 路由器学习"这对药-基因交互属于哪种机制类型"。整个设计是**表示层面**的交互——两个静态向量的代数组合。

### 演化历程

| 版本 | 思路 | AUC |
|------|------|-----|
| Baseline BCE | V_c⊥ ⊕ V_g → MLP | 0.8914 |
| no Ortho | V_c ⊕ V_g → MLP | 0.8860 |
| CL/GCAA | V_g 语义对比学习 | 0.8903（有害）|
| Targeted Pooling (hybrid) | h_g 驱动原子注意力 | 0.8911 |
| **MoE** | 路由器直连 BCE | **0.8935** |
| MoE+Target（进行中）| 路由器条件化靶向池化 | 待出 |

---

## 二、他做的东西（DrugOperatorNet）

### 核心架构（train_drug_operator.py）

```
Gene → k-mer CNN → h_g [B, d]
Chem → GIN×3 → atom features H_atoms [N, d]
PharmacophoreExtractor:
  r 个可学习 query 向量 × 原子交叉注意力 → pharma_emb [B, r, d]
PerturbationOperator:
  pharma_emb → U[B,r,d], V[B,r,d], σ[B,r]
  coupling_k = v_k^T · h_g      （基因对第k模式的耦合强度）
  spectrum_k = σ_k · coupling_k  （第k模式的激活强度）
  Δh = Σ_k spectrum_k · u_k     （总扰动向量）
classify([h_g, Δh])
```

**设计哲学**：  
药物不是一个静态向量，而是一个**线性算子** T = I + Σ_k σ_k·u_k⊗v_k^T，作用于基因状态空间，输出扰动向量 Δh。分类的输入是"基因原始状态 + 被药物扰动后的变化量"。

---

## 三、相同点

| 维度 | 共同之处 |
|------|---------|
| 数据 | 完全相同：L1000，k-mer 编码基因序列，GIN 编码化学图 |
| 编码器 | GeneEncoderV1（k-mer CNN + TopK）完全一致 |
| GIN 骨架 | 3层 GIN + LayerNorm，结构相同 |
| 训练设置 | AdamW + ReduceLROnPlateau + early stop + AMP |
| 目标 | 预测药物→基因上调/下调（二分类） |
| 问题意识 | 都意识到简单拼接不够，需要显式建模化学-基因**交互** |

---

## 四、核心差异

### 4.1 交互的本质定义不同

| | 我们（MoE） | 他们（Operator） |
|--|------------|----------------|
| 药物是什么 | 静态特征向量 V_c | 作用于基因状态的**线性算子** T |
| 交互怎么建模 | 两向量代数组合（正交剥离 + 路由） | 药物作用于基因：Δh = T(h_g) - h_g |
| 分类输入 | [V_g, V_c_perp] | [h_g, Δh] |
| 输出语义 | 隐式（MoE 路由权重） | 显式（交互谱 S = [s_1,...,s_r]） |

### 4.2 化学编码策略不同

| | 我们 | 他们 |
|--|------|------|
| 池化 | sum/mean（全局）± 靶向 | **不池化**：保留原子级特征，用药效团注意力提取 r 个模式 |
| 化学表示 | 单一向量 V_c [B, d] | r 个药效团嵌入 pharma_emb [B, r, d] |
| 基因条件化时机 | 池化**之后**（Targeted Pooling / MoE 路由） | 交互计算时（v_k^T · h_g，基因条件化的耦合） |

### 4.3 可解释性机制不同

| | 我们 | 他们 |
|--|------|------|
| 可解释性来源 | MoE 路由权重（哪个专家激活） | 交互谱 S（哪种模式激活）+ 药效团注意力（哪些原子） |
| 解释粒度 | 分子级（K 个机制类型） | 原子级（哪个原子属于哪种药效团）+ 模式级 |
| 论文图表 | 路由分布热力图 | 谱热力图 + 药效团可视化 + t-SNE |

### 4.4 正则化策略不同

| | 我们 | 他们 |
|--|------|------|
| 主要正则 | Load Balancing（路由均匀）+ Entropy（路由稀疏） | L_sparse（σ 稀疏）+ L_ortho（U 列正交） |
| 目的 | 避免专家坍缩 | 鼓励模式独立可解释 |

---

## 五、各自的核心优势

### 我们的优势

1. **梯度路径更直接**：路由器直连 BCE，MoE 的监督信号不依赖中间表示的质量
2. **已有实验验证**：0.8935 有实测支撑，知道哪些设计有效（Ortho +0.005，MoE +0.002）
3. **轻量**：参数少，无 torch_geometric 依赖，运行稳定

### 他们的优势

1. **与生物学更对齐**：L1000 本身就是测药物对基因的扰动，药物算子天然对应这个概念
2. **可解释性更强**：交互谱是一个结构化的、每维度独立的指纹，可直接用于下游分析
3. **消融框架完整**：4 种交互方式共享编码器，论文消融表直接生成
4. **原子级可视化**：药效团注意力权重可以高亮分子中驱动每种作用模式的原子

---

## 六、潜在融合方向

两个设计不是竞争关系，可以组合：

```
方案A：用 MoE 路由作为 Operator 的"初始化信号"
  route_weights [B, K] → 引导 r 个药效团查询的初始权重

方案B：Operator 的交互谱作为 MoE 的路由输入
  spectrum [B, r] → router → route_weights [B, K]
  （谱比拼接向量更有意义，路由器学习"哪个专家处理这种谱"）

方案C：Operator 直接替代 MoE 分类头
  Δh 作为交互表示 → 直接分类（更干净，当前 DrugOperatorNet 的做法）
```

**最有价值的融合**（方案B）：先跑 Operator，如果 AUC 有提升，把 spectrum 作为更语义化的路由输入接入 MoE。

---

## 七、运行 DrugOperatorNet 需要解决的问题

1. **torch_geometric 依赖**（`from torch_geometric.utils import softmax`）  
   → 替换为纯 PyTorch 实现（参考 train_target_pool.py 中的 `scatter_softmax`）

2. **PharmacophoreExtractor 效率**（Python for 循环 r 次）  
   → 向量化：`scores = (K @ self.queries.T) / sqrt(d)` → [N, r]，并行计算所有 slot

3. **`ortho_concat` 的 h_g vs V_g**  
   → 与我们的 Baseline 对齐，入分类头的应是 V_g（归一化）而非 h_g

---

## 八、建议

| 优先级 | 行动 | 理由 |
|--------|------|------|
| 高 | 修复 torch_geometric 依赖，跑 `operator` vs `ortho_concat` 的 Fold0 对比 | 验证算子思路是否有 AUC 提升 |
| 高 | 关注 `operator` 的交互谱 S 的分布（稀疏性、正负样本差异） | 可解释性是投 NC 的关键 |
| 中 | 如果 `operator` > MoE，考虑 spectrum → MoE router 融合 | 两个创新点合并 |
| 低 | PharmacophoreExtractor 向量化 | 性能优化，不影响结果 |

---

## 九、已做的修复（2026-04-02）

原始代码有两个问题，已修复并提交（commit `23acabf`）：

### 修复 1：移除 torch_geometric 依赖

**原代码**（第 62 行）：
```python
from torch_geometric.utils import softmax
```

**修复后**：删除该 import，在文件顶部添加纯 PyTorch 实现：
```python
def scatter_softmax(scores, batch_idx):
    """图内原子级 softmax，纯 PyTorch 实现。"""
    max_scores = torch.zeros(...).index_reduce_(0, batch_idx, scores, 'amax', ...)
    exp_scores = torch.exp(scores - max_scores[batch_idx])
    exp_sum = torch.zeros(...).index_add_(0, batch_idx, exp_scores)
    return exp_scores / (exp_sum[batch_idx] + 1e-8)
```

`PharmacophoreExtractor.forward()` 中的调用同步替换：
```python
# 原：alpha = softmax(scores, batch_idx)
alpha = scatter_softmax(scores_all[:, s], batch_idx)
```

### 修复 2：PharmacophoreExtractor 循环优化

**原代码**：在 Python for 循环内对每个 slot 重复计算 `K = key_proj(atom_h)`，每次循环都做一次线性层前向。

**修复后**：K、V 在循环外统一计算，所有 slot 的 scores 用矩阵乘法一次得到：
```python
K = self.key_proj(atom_h)               # [N_total, d]，计算一次
V = self.val_proj(atom_h)               # [N_total, d]，计算一次
scores_all = (K @ self.queries.T) / math.sqrt(d)  # [N_total, r]，一次矩阵乘

pharma = torch.zeros(num_graphs, self.num_slots, d, ...)
for s in range(self.num_slots):
    alpha = scatter_softmax(scores_all[:, s], batch_idx)
    pharma[:, s].index_add_(0, batch_idx, V * alpha.unsqueeze(-1))
```

修复后可直接运行，无需安装 `torch_geometric` 或 `torch_scatter`。

---

*最后更新：2026-04-02*
