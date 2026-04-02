#!/bin/bash
# 三细胞系基础结果训练脚本
# 细胞系选择理由：
#   A549  - 肺腺癌，KRAS G12S，与 MCF7(乳腺/ER+) 组织和驱动基因均不同
#   VCAP  - 前列腺癌，AR 扩增，激素依赖型，样本量最大(228K)
#   A375  - 黑色素瘤，BRAF V600E，MAPK 通路，靶向治疗背景清晰
# GPU 分配：cuda:0/1/2 各跑一个细胞系，cuda:3 保留给模型调优

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_baseline_3cell"
mkdir -p "$LOG_DIR"

# 公共超参（与 MCF7 最优实验一致）
COMMON="--epochs 80 --batch_size 512 --lr 3e-4 --hidden_dim 128 \
        --dropout 0.3 --num_experts 4 --lam_balance 0.1 \
        --patience 10 --seed 42 --use_amp"

echo "========================================"
echo "  启动三细胞系 Fold0 基础训练"
echo "  A549  → cuda:0"
echo "  VCAP  → cuda:1"
echo "  A375  → cuda:2"
echo "  cuda:3 保留给模型调优"
echo "========================================"

# A549 - 肺腺癌 KRAS
nohup python train_moe.py \
    --data_dir "$DATA_ROOT/A549" \
    --device cuda:0 --fold 0 \
    $COMMON --run_tag baseline_fold0 \
    > "$LOG_DIR/A549_fold0.log" 2>&1 &
echo "[A549]  PID=$! → $LOG_DIR/A549_fold0.log"

sleep 5   # 错开启动，避免缓存文件竞争

# VCAP - 前列腺癌 AR
nohup python train_moe.py \
    --data_dir "$DATA_ROOT/VCAP" \
    --device cuda:1 --fold 0 \
    $COMMON --run_tag baseline_fold0 \
    > "$LOG_DIR/VCAP_fold0.log" 2>&1 &
echo "[VCAP]  PID=$! → $LOG_DIR/VCAP_fold0.log"

sleep 5

# A375 - 黑色素瘤 BRAF V600E
nohup python train_moe.py \
    --data_dir "$DATA_ROOT/A375" \
    --device cuda:2 --fold 0 \
    $COMMON --run_tag baseline_fold0 \
    > "$LOG_DIR/A375_fold0.log" 2>&1 &
echo "[A375]  PID=$! → $LOG_DIR/A375_fold0.log"

echo ""
echo "全部后台启动完成。实时查看日志："
echo "  tail -f $LOG_DIR/A549_fold0.log"
echo "  tail -f $LOG_DIR/VCAP_fold0.log"
echo "  tail -f $LOG_DIR/A375_fold0.log"
