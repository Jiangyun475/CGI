#!/bin/bash
# Morgan FP Baseline 四细胞系 Fold0（一键启动）
# MCF7 已完成（AUC=0.8710），此脚本重跑其余三个
# A549 → cuda:0, VCAP → cuda:1, A375 → cuda:2
# MCF7 已有结果，不重复跑

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_morgan_3cell"
mkdir -p "$LOG_DIR"

COMMON="--epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
        --dropout 0.3 --lam_spread 0.1 \
        --morgan_radius 2 --morgan_bits 2048 \
        --patience 10 --seed 42 --use_amp --fold 0 --run_tag v1"

echo "========================================"
echo "  Morgan FP Baseline 重跑"
echo "  A549  → cuda:0"
echo "  VCAP  → cuda:1"
echo "  A375  → cuda:2"
echo "  MCF7  已完成，跳过"
echo "========================================"

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/A549" \
    --device cuda:0 \
    $COMMON \
    > "$LOG_DIR/A549_fold0.log" 2>&1 &
echo "[A549]  PID=$! → $LOG_DIR/A549_fold0.log"

sleep 3

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/VCAP" \
    --device cuda:1 \
    $COMMON \
    > "$LOG_DIR/VCAP_fold0.log" 2>&1 &
echo "[VCAP]  PID=$! → $LOG_DIR/VCAP_fold0.log"

sleep 3

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/A375" \
    --device cuda:3 \
    $COMMON \
    > "$LOG_DIR/A375_fold0.log" 2>&1 &
echo "[A375]  PID=$! → $LOG_DIR/A375_fold0.log"

echo ""
echo "查看日志："
echo "  tail -f $LOG_DIR/A549_fold0.log"
echo "  tail -f $LOG_DIR/VCAP_fold0.log"
echo "  tail -f $LOG_DIR/A375_fold0.log"
