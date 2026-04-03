#!/bin/bash
# ==============================================================================
# CL 5-fold 全细胞系训练
# 在 MCF7 CL Fold0 确认有效后运行（lam_cl=0.1，其余超参与 no_moe 一致）
#
# 用法：
#   bash run_cl_5fold.sh MCF7   # 只跑MCF7（Fold1-4，Fold0已跑）
#   bash run_cl_5fold.sh ALL    # 全4细胞系
# ==============================================================================

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_5fold_cl"
mkdir -p "$LOG_DIR"

COMMON="--epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
  --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
  --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 \
  --lam_cl 0.1 \
  --ablation no_moe --patience 10 --seed 42 --use_amp"

declare -A CELL_GPU=( [MCF7]="cuda:0" [A375]="cuda:1" [A549]="cuda:2" [VCAP]="cuda:3" )

run_cell() {
  local CELL=$1
  local GPU="${CELL_GPU[$CELL]}"
  local START=${2:-0}  # 默认从Fold0开始（全5折）

  echo "=== $CELL CL 5-fold | GPU=$GPU | 从Fold$START开始 ==="
  for fold in $(seq $START 4); do
    echo "  [$(date '+%H:%M:%S')] Fold $fold"
    python New/train_operator_moe.py \
      --data_dir "$DATA_ROOT/$CELL" --device $GPU --fold $fold \
      $COMMON --run_tag "cl_5fold" \
      2>&1 | tee "$LOG_DIR/${CELL}_fold${fold}.log"
    echo "  Fold$fold 完成"
  done

  echo "--- $CELL CL 汇总 ---"
  python3 - << PYEOF
import re, glob, numpy as np
aucs = []
for f in sorted(glob.glob('$LOG_DIR/${CELL}_fold*.log')):
    ms = re.findall(r'最优 AUC: ([\d.]+)', open(f).read())
    if ms: aucs.append(float(ms[-1]))
if aucs:
    print(f'$CELL CL: {[round(a,4) for a in aucs]}')
    print(f'  mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
PYEOF
}

TARGET=${1:-MCF7}

if [ "$TARGET" = "ALL" ]; then
  # MCF7 Fold0已跑，从Fold1开始
  nohup bash -c "$(declare -f run_cell); run_cell MCF7 1" > $LOG_DIR/MCF7_full.log 2>&1 &
  echo "[MCF7 CL Fold1-4] PID=$!"
  sleep 5
  nohup bash -c "$(declare -f run_cell); run_cell A375 0" > $LOG_DIR/A375_full.log 2>&1 &
  echo "[A375 CL 5-fold] PID=$!"
  sleep 5
  nohup bash -c "$(declare -f run_cell); run_cell A549 0" > $LOG_DIR/A549_full.log 2>&1 &
  echo "[A549 CL 5-fold] PID=$!"
  sleep 5
  nohup bash -c "$(declare -f run_cell); run_cell VCAP 0" > $LOG_DIR/VCAP_full.log 2>&1 &
  echo "[VCAP CL 5-fold] PID=$!"
else
  START_FOLD=0
  [ "$TARGET" = "MCF7" ] && START_FOLD=1  # MCF7 Fold0已完成
  run_cell "$TARGET" $START_FOLD
fi
