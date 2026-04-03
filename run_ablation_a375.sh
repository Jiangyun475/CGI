#!/bin/bash
# ==============================================================================
# A375 消融实验 5-fold
# 与 MCF7/A549/VCAP 保持完全一致的消融配置，补全论文消融表第4行
#
# 消融对象：train_operator_moe.py（DrugOperatorNet框架）
#   1. no_moe (baseline)       ← 已完成，5-fold mean=0.9011
#   2. no_moe + CL             ← 等MCF7 CL收敛后启动
#   3. no_moe 无正交正则       ← lam_ortho_modes=0.0
#   4. Morgan FP 5-fold        ← 固定指纹替代GIN
#   5. RF 5-fold               ← 已完成 0.8646±0.0038
#   6. Baseline DL 5-fold      ← 简单MLP（train_baseline_dl.py）
#
# 旧框架消融（SumMean/wo_CL/wo_Ortho，来自train_summean.py）在A375上没有跑过
# 决策：不补跑旧框架，统一用DrugOperatorNet框架的消融，保持架构一致性
#
# GPU分配（等当前实验结束后启动）：
#   cuda:0 → DrugOp 无正交 5-fold（串行）
#   cuda:1 → Morgan FP 5-fold（串行）
#   cuda:2 → Baseline DL 5-fold（串行）
# ==============================================================================

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
CELL="A375"
LOG_DIR="logs_ablation_a375_druop"
mkdir -p "$LOG_DIR"

COMMON_OPERATOR="--epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
  --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
  --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 \
  --ablation no_moe --patience 10 --seed 42 --use_amp"

# ── 1. DrugOp 无正交正则（lam_ortho=0）──────────────────────────────
echo "[A375] 启动 no_ortho 消融（5-fold, cuda:0）"
for fold in 0 1 2 3 4; do
  python New/train_operator_moe.py \
    --data_dir "$DATA_ROOT/$CELL" --device cuda:0 --fold $fold \
    $COMMON_OPERATOR \
    --lam_ortho_modes 0.0 \
    --run_tag "wo_ortho" \
    2>&1 | tee "$LOG_DIR/${CELL}_wo_ortho_Fold${fold}.log"
  echo "  Fold${fold} 完成"
done &
echo "  PID=$!"

sleep 5

# ── 2. Morgan FP 5-fold（固定ECFP4指纹替代GIN）────────────────────
echo "[A375] 启动 Morgan FP 消融（5-fold, cuda:1）"
for fold in 0 1 2 3 4; do
  python New/train_pretrained_baseline.py \
    --data_dir "$DATA_ROOT/$CELL" --device cuda:1 --fold $fold \
    --epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
    --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
    --lam_sparse 0.01 --lam_ortho_modes 0.1 \
    --patience 10 --use_amp --drug_emb_type ecfp4 \
    --run_tag "5fold" \
    2>&1 | tee "$LOG_DIR/${CELL}_morgan_Fold${fold}.log"
  echo "  Fold${fold} 完成"
done &
echo "  PID=$!"

sleep 5

# ── 3. Baseline DL（简单MLP，无算子结构）──────────────────────────
echo "[A375] 启动 Baseline DL 消融（5-fold, cuda:2）"
for fold in 0 1 2 3 4; do
  python train_baseline_dl.py \
    --data_dir "$DATA_ROOT/$CELL" --device cuda:2 --fold $fold \
    --epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
    --dropout 0.3 --patience 10 --seed 42 --use_amp \
    2>&1 | tee "$LOG_DIR/${CELL}_baseline_dl_Fold${fold}.log"
  echo "  Fold${fold} 完成"
done &
echo "  PID=$!"

wait
echo "=== A375 消融实验全部完成 ==="
python3 - << 'PYEOF'
import re, glob, numpy as np
log_dir = 'logs_ablation_a375_druop'
configs = {
    'wo_ortho': f'{log_dir}/A375_wo_ortho_Fold*.log',
    'morgan_fp': f'{log_dir}/A375_morgan_Fold*.log',
    'baseline_dl': f'{log_dir}/A375_baseline_dl_Fold*.log',
}
for name, pat in configs.items():
    aucs = []
    for f in sorted(glob.glob(pat)):
        ms = re.findall(r'VAL_AUC:([\d.]+)', open(f).read())
        if ms: aucs.append(max(float(x) for x in ms))
    if aucs:
        print(f'A375 {name}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (n={len(aucs)})')
PYEOF
