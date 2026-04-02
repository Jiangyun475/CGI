#!/bin/bash
# ============================================================
# run_5fold_nomoe.sh
# DrugOperatorNet (no_moe) 4细胞系 5-fold 完整交叉验证
# ============================================================
#
# 用法：bash run_5fold_nomoe.sh [CELL_LINE] [START_FOLD] [END_FOLD]
#
# 示例：
#   bash run_5fold_nomoe.sh MCF7         # 运行 MCF7 全部 5 fold
#   bash run_5fold_nomoe.sh A375 0 2     # 运行 A375 fold 0-2
#
# 全部 4 细胞系（建议分 4 个 GPU 并行）：
#   nohup bash run_5fold_nomoe.sh MCF7 > logs_5fold_nomoe/MCF7.log 2>&1 &
#   nohup bash run_5fold_nomoe.sh A375 > logs_5fold_nomoe/A375.log 2>&1 &
#   nohup bash run_5fold_nomoe.sh A549 > logs_5fold_nomoe/A549.log 2>&1 &
#   nohup bash run_5fold_nomoe.sh VCAP > logs_5fold_nomoe/VCAP.log 2>&1 &
# ============================================================

CELL=${1:-MCF7}
START_FOLD=${2:-0}
END_FOLD=${3:-4}

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_5fold_nomoe"
mkdir -p "$LOG_DIR"

# GPU 分配（根据当前实验情况调整）
declare -A CELL_GPU
CELL_GPU[MCF7]="cuda:0"
CELL_GPU[A375]="cuda:1"
CELL_GPU[A549]="cuda:2"
CELL_GPU[VCAP]="cuda:3"

DEVICE="${CELL_GPU[$CELL]:-cuda:0}"

COMMON="--epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
        --dropout 0.3 --operator_rank 8 \
        --gene_max_len 1000 --warmup_epochs 5 \
        --lam_sparse 0.01 --lam_ortho_modes 0.1 \
        --lam_balance 0.1 --num_experts 4 \
        --ablation no_moe \
        --patience 10 --seed 42 --use_amp --save_spectrum"

echo "================================================"
echo "  DrugOperatorNet (no_moe) 5-fold CV"
echo "  细胞系: ${CELL} | GPU: ${DEVICE}"
echo "  Fold 范围: ${START_FOLD} - ${END_FOLD}"
echo "  数据: ${DATA_ROOT}/${CELL}"
echo "================================================"

# ─── 串行训练每个 fold（在同一 GPU 上） ─────────────────────────
for FOLD in $(seq $START_FOLD $END_FOLD); do
    echo ""
    echo "─────────────────────────────────────────"
    echo "  [$(date '+%H:%M:%S')] Fold ${FOLD} / 4"
    echo "─────────────────────────────────────────"

    python New/train_operator_moe.py \
        --data_dir "${DATA_ROOT}/${CELL}" \
        --device "${DEVICE}" \
        --fold ${FOLD} \
        ${COMMON} \
        --run_tag "5fold" \
        2>&1 | tee "${LOG_DIR}/${CELL}_fold${FOLD}.log"

    echo "  Fold ${FOLD} 完成"
done

echo ""
echo "================================================"
echo "  ${CELL} 全部 Fold 训练完成！"
echo "  日志: ${LOG_DIR}/${CELL}_fold*.log"
echo "================================================"

# ─── 汇总 AUC ────────────────────────────────────────────────────
echo ""
echo "📊 汇总各 Fold AUC:"
echo "─────────────────────────────────────────"
for FOLD in $(seq $START_FOLD $END_FOLD); do
    LOG="${LOG_DIR}/${CELL}_fold${FOLD}.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep "最优 AUC" "$LOG" | tail -1)
        echo "  Fold ${FOLD}: $BEST"
    fi
done

echo ""
echo "计算均值和标准差（Python）："
python3 - << 'PYEOF'
import re, glob, sys, os
import numpy as np

cell = os.environ.get('CELL', 'MCF7')
log_dir = 'logs_5fold_nomoe'
pattern = f'{log_dir}/{cell}_fold*.log'
aucs = []
for log_path in sorted(glob.glob(pattern)):
    with open(log_path) as f:
        content = f.read()
    matches = re.findall(r'最优 AUC: ([\d.]+)', content)
    if matches:
        auc = float(matches[-1])
        aucs.append(auc)
        print(f"  {os.path.basename(log_path)}: AUC={auc:.4f}")

if aucs:
    print(f"\n  {cell}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (n={len(aucs)})")
else:
    print("  未找到 AUC 记录")
PYEOF
