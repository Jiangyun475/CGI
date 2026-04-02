#!/bin/bash
# DrugOperatorNet V2（GeneMultiHeadReader + 模式对齐耦合）三细胞系 Fold0
# A549 → cuda:0, VCAP → cuda:1, A375 → cuda:2
# cuda:3 保留给模型调优

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_operator_v2_3cell"
mkdir -p "$LOG_DIR"

COMMON="--epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
        --dropout 0.3 --operator_rank 8 \
        --gene_max_len 3000 --warmup_epochs 5 \
        --lam_sparse 0.01 --lam_ortho_modes 0.01 \
        --patience 10 --seed 42 --use_amp \
        --save_spectrum --save_gene_attn --fold 0"

echo "========================================"
echo "  DrugOperatorNet V2 三细胞系 Fold0"
echo "  A549  → cuda:0"
echo "  VCAP  → cuda:1"
echo "  A375  → cuda:2"
echo "========================================"

nohup python New/train_drug_operator_v2.py \
    --data_dir "$DATA_ROOT/A549" \
    --device cuda:0 \
    $COMMON --run_tag v1 \
    > "$LOG_DIR/A549_fold0.log" 2>&1 &
echo "[A549]  PID=$! → $LOG_DIR/A549_fold0.log"

sleep 5

nohup python New/train_drug_operator_v2.py \
    --data_dir "$DATA_ROOT/VCAP" \
    --device cuda:1 \
    $COMMON --run_tag v1 \
    > "$LOG_DIR/VCAP_fold0.log" 2>&1 &
echo "[VCAP]  PID=$! → $LOG_DIR/VCAP_fold0.log"

sleep 5

nohup python New/train_drug_operator_v2.py \
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
