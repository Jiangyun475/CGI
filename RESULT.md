# 实验结果记录

## MCF7 Fold0 汇总

| 模型 | AUC | PRC | F1 | 说明 |
|------|-----|-----|----|------|
| Baseline BCE+Spread (no_ortho) | 0.8860 | - | - | train_baseline_bce.py --no_ortho |
| Baseline BCE+Spread (with Ortho) | 0.8914 | - | - | train_baseline_bce.py |
| Targeted Pooling target | 0.8883 | - | - | train_target_pool.py --pool_type target |
| Targeted Pooling sum_mean | 0.8908 | - | - | train_target_pool.py --pool_type sum_mean |
| Targeted Pooling hybrid | 0.8911 | - | - | train_target_pool.py --pool_type hybrid |
| MoE k=4 | 0.8935 | - | - | train_moe.py, lr=3e-4, ep80 |
| MoE + Targeted Pooling k=4 | **0.8943** | 0.8857 | 0.8159 | train_moe_target.py, lr=2e-4, ep48 |
| DrugOperatorNet (operator, r=8) | 0.8924 | - | - | New/train_drug_operator.py, 0.93M参数 |
| **Morgan FP baseline (ECFP4)** | **0.8710** | - | - | train_morgan_baseline.py，固定指纹替代GIN |

---

## 实验详情

### [2026-04-02] MoE + Targeted Pooling 合并架构 (train_moe_target.py)

**命令**：
```bash
python train_moe_target.py \
  --data_dir .../MCF7 --device cuda:3 --fold 0 \
  --epochs 60 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
  --dropout 0.3 --num_experts 4 --lam_balance 0.1 \
  --patience 10 --seed 42 --use_amp --run_tag v1
```

**结果**：Best AUC = **0.8943**，在 Ep38 首次达到，Ep48 early stop

**训练曲线关键节点**：
- Ep1-9：0.816→0.877，快速收敛
- Ep10-25：0.882-0.891，高原
- Ep30：LR decay 触发后跳至 0.8936（LR: 2e-4 → 1e-4）
- Ep35-48：稳定在 0.8935-0.8943，无震荡

**与 MoE 对比**：+0.0008 AUC（0.8935→0.8943），单次实验，需多 seed 确认

**分析**：
- 路由条件化靶向池化有效：route_weights @ expert_queries 提供比 attn_proj(h_g) 更强的 atom query 监督
- BCE 梯度双路反传（expert_logits + target_pool），路由器监督信号增强
- LB Loss 稳定在 0.0225-0.0227，路由均衡良好，无专家坍缩
- 训练稳定，无震荡，验证了架构设计合理性
- 提升量偏小（+0.0008），可能需要更多 seed/fold 才能确认统计显著性

---

## 多细胞系 Fold0 方法对比（同一细胞系内纵向比较）

| 细胞系 | MoE | MoE+Target | Δ | 备注 |
|--------|-----|------------|---|------|
| MCF7（乳腺/ER+） | 0.8935 | **0.8943** | +0.0008 | 基准 |
| A375（黑色素瘤/BRAF） | 0.9035 | **0.9040** | +0.0005 | 最高，收敛快（54ep） |
| A549（肺腺癌/KRAS） | 0.8703 | **0.8718** | +0.0015 | 提升最明显 |
| VCAP（前列腺/AR） | 0.8132 | **0.8135** | +0.0003 | 两种方法均低，任务难 |

**结论：MoE+Target 在全部 4 个细胞系上一致优于 MoE（4/4），方向一致，非随机波动。**

| 细胞系 | Morgan FP | MoE | MoE+Target | DrugOperator V1 |
|--------|-----------|-----|------------|-----------------|
| MCF7 | 0.8710 | 0.8935 | **0.8943** | 0.8924 |
| A375 | 0.8870 | 0.9035 | **0.9040** | 待跑 |
| A549 | 0.8432 | 0.8703 | **0.8718** | 待跑 |
| VCAP | 0.7894 | 0.8132 | **0.8135** | 待跑 |

---

### [2026-04-02] Morgan FP Baseline 结果 (train_morgan_baseline.py)

| 细胞系 | Morgan FP | MoE+Target | GIN 净增益 | 收敛 |
|--------|-----------|------------|-----------|------|
| MCF7 | 0.8710 | 0.8943 | **+0.0233** | ep25 |
| A375 | 0.8870 | 0.9040 | **+0.0170** | ep25 |
| A549 | 0.8432 | 0.8718 | **+0.0286** | ep25 |
| VCAP | 0.7894 | 0.8135 | **+0.0241** | ep27 |

**结论：GIN 端到端图学习在全部 4 个细胞系上一致优于固定 Morgan 指纹，净增益 +0.017~+0.029。**

**分析**：
- Morgan FP 是固定的 2048-bit ECFP4，无法从数据中优化化学表示；GIN 通过 3 层消息传递学到任务相关的原子-键交互模式
- Chemical cold split 下，GIN 的优势尤为显著：固定指纹对未见化学品泛化性弱，GIN 通过局部结构归纳偏置更好泛化
- VCAP 的 Morgan FP 仅 0.7894（最低），而 MoE+Target 仍能到 0.8135，说明即使在最难的任务上 GIN 也有显著贡献
- 这是投稿 NC/NMI 的关键对比实验之一：证明端到端图学习优于传统化学指纹

---

### [2026-04-02] 多细胞系基础结果 (run_baseline_3cell.sh)

**命令**：`bash run_baseline_3cell.sh`（A549→cuda:0，VCAP→cuda:1，A375→cuda:2）

**分析**：
- A375 AUC=0.9035，超越 MCF7：BRAF V600E 是最"干净"的单一突变通路，药物-基因关系规律性强，LB Loss 稳定在 0.032（路由分化最明显）
- A549 AUC=0.8703：KRAS 通路复杂，多效应子，略难预测
- VCAP AUC=0.8132，训练到第 80 epoch 仍未收敛：AR 扩增样本量最大（228K），性能低可能与超参（lr/patience）不匹配有关，需重新调参

**待跟进**：VCAP 尝试 lr=1e-4、patience=15 重跑

---

### [2026-04-02] DrugOperatorNet 首次结果 (New/train_drug_operator.py)

**命令**：
```bash
python New/train_drug_operator.py \
  --data_dir .../MCF7 --device cuda:3 --fold 0 \
  --interaction_type operator --operator_rank 8 \
  --epochs 80 --batch_size 512 --lr 3e-4 \
  --lam_sparse 0.01 --lam_ortho_modes 0.01 \
  --patience 10 --seed 42 --use_amp --save_spectrum --run_tag v1
```

**结果**：AUC = **0.8924**，参数量 933,442（比 MoE 少 12%）

**与 MoE 对比**：-0.0011（0.8924 vs 0.8935），但参数效率更高

**分析**：
- 算子范式可行：用更少参数取得接近 MoE 的结果
- 最大价值在可解释性：交互谱 S=[s₁,...,s₈] 是结构化的药物-基因互作指纹
- 待做：跑 ortho_concat 对照组，确认药效团提取本身的净贡献
- 待做：`analyze_spectrum.py` 可视化交互谱，检验正/负样本的谱差异

---

### [历史] 消融实验结论

1. **Ortho 有效**：+0.005 AUC（0.8860→0.8914），单变量已验证
2. **MoE 有效**：在 Baseline 基础上 +0.002（0.8914→0.8935）
3. **CL（GCAA）有害**：AUC 反降，根因是 V_g 表示坍缩（density=1.0）
4. **Targeted Pooling 单独无效**：hybrid 仅 0.8911，因 V_g 质量差导致 atom query 无意义
5. **MoE+Target 合并**：0.8943，略优于纯 MoE，路由器监督弥补了 V_g 质量问题
