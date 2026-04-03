#!/bin/bash
# MoE+Target 三细胞系并行训练
# A549 → cuda:0, VCAP → cuda:1, A375 → cuda:2
# 显存占用小，与现有任务共享 GPU

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_moe_target"
mkdir -p "$LOG_DIR"

COMMON="--epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
        --dropout 0.3 --num_experts 4 --lam_balance 0.1 \
        --patience 10 --seed 42 --use_amp --run_tag v1"

echo "========================================"
echo "  MoE+Target 三细胞系训练"
echo "  A549  → cuda:0"
echo "  VCAP  → cuda:2  (样本量最大，独占空闲卡)"
echo "  A375  → cuda:1"
echo "========================================"

nohup python train_moe_target.py \
    --data_dir "$DATA_ROOT/A549" \
    --device cuda:0 --fold 0 \
    $COMMON \
    > "$LOG_DIR/A549_fold0.log" 2>&1 &
echo "[A549]  PID=$! → $LOG_DIR/A549_fold0.log"

sleep 5

nohup python train_moe_target.py \
    --data_dir "$DATA_ROOT/VCAP" \
    --device cuda:2 --fold 0 \
    $COMMON \
    > "$LOG_DIR/VCAP_fold0.log" 2>&1 &
echo "[VCAP]  PID=$! → $LOG_DIR/VCAP_fold0.log"

sleep 5

nohup python train_moe_target.py \
    --data_dir "$DATA_ROOT/A375" \
    --device cuda:1 --fold 0 \
    $COMMON \
    > "$LOG_DIR/A375_fold0.log" 2>&1 &
echo "[A375]  PID=$! → $LOG_DIR/A375_fold0.log"

echo ""
echo "启动完成。查看日志："
echo "  tail -f $LOG_DIR/A549_fold0.log"
echo "  tail -f $LOG_DIR/VCAP_fold0.log"
echo "  tail -f $LOG_DIR/A375_fold0.log"
