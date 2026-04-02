#!/bin/bash
# Morgan FP Baseline（ECFP4）三细胞系 Fold0
# 验证 GIN 图学习的有效性：同架构仅替换化学编码器
# A549 → cuda:0, VCAP → cuda:1, A375 → cuda:2
# cuda:3 保留给模型调优

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_morgan_3cell"
mkdir -p "$LOG_DIR"

COMMON="--epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
        --dropout 0.3 --lam_spread 0.1 \
        --morgan_radius 2 --morgan_bits 2048 \
        --patience 10 --seed 42 --use_amp --fold 0"

echo "========================================"
echo "  Morgan FP Baseline 三细胞系 Fold0"
echo "  A549  → cuda:0"
echo "  VCAP  → cuda:1"
echo "  A375  → cuda:2"
echo "========================================"

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/A549" \
    --device cuda:0 \
    $COMMON --run_tag v1 \
    > "$LOG_DIR/A549_fold0.log" 2>&1 &
echo "[A549]  PID=$! → $LOG_DIR/A549_fold0.log"

sleep 5

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/VCAP" \
    --device cuda:1 \
    $COMMON --run_tag v1 \
    > "$LOG_DIR/VCAP_fold0.log" 2>&1 &
echo "[VCAP]  PID=$! → $LOG_DIR/VCAP_fold0.log"

sleep 5

nohup python train_morgan_baseline.py \
    --data_dir "$DATA_ROOT/A375" \
    --device cuda:2 \
    $COMMON --run_tag v1 \
    > "$LOG_DIR/A375_fold0.log" 2>&1 &
echo "[A375]  PID=$! → $LOG_DIR/A375_fold0.log"

echo ""
echo "查看日志："
echo "  tail -f $LOG_DIR/A549_fold0.log"
echo "  tail -f $LOG_DIR/VCAP_fold0.log"
echo "  tail -f $LOG_DIR/A375_fold0.log"
