#!/bin/bash
# ============================================================
# run_allcells_gpu3.sh
# 所有细胞系 DrugOp no_moe Fold0 扫描 — cuda:3
#
# 负责：大细胞系(≥100K，不含MCF7/A375/A549/VCAP已完成的)，共4个
#       + 备选：如果 GPU2 的任务全部完成后可以接着跑 GPU2 剩余
# 预计总时长：~2小时
#
# 用法：nohup bash run_allcells_gpu3.sh > logs_allcells/gpu3_all.log 2>&1 &
# ============================================================

DEVICE="cuda:3"
DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_allcells"
RESULT_DIR="results_operator_moe"
mkdir -p "$LOG_DIR"

COMMON="--fold 0 --epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
        --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
        --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 \
        --ablation no_moe --patience 10 --seed 42 --use_amp --run_tag allcells"

# 等待 GPU3 当前任务结束
echo "[$(date '+%H:%M:%S')] 等待 $DEVICE 释放..."
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null)
    if [ "$used" -lt 1000 ] 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] $DEVICE 已释放 (${used}MiB)，开始训练"
        break
    fi
    sleep 60
done

# ── 大细胞系（≥100K，MCF7/A375/A549/VCAP已有5-fold，跳过）──────
# HT29(198K), PC3(209K): 每个~25-30 min
# HA1E(103K), NPC(119K): 每个~13-16 min
# JURKAT在GPU2跑，这里不重复
BIG_CELLS=(
    HT29
    PC3
    HA1E
    NPC
)

RESULTS_SUMMARY="$LOG_DIR/gpu3_summary.txt"
echo "Cell | AUC | PRC | F1 | Time" > "$RESULTS_SUMMARY"

total=${#BIG_CELLS[@]}
idx=0
for CELL in "${BIG_CELLS[@]}"; do
    idx=$((idx + 1))
    DATA_DIR="$DATA_ROOT/$CELL"

    if [ ! -d "$DATA_DIR" ]; then
        echo "[SKIP] $CELL: 目录不存在"
        continue
    fi

    LOG_FILE="$LOG_DIR/${CELL}_fold0.log"

    if [ -f "$RESULT_DIR/$CELL/no_moe_r8_k4_Fold0_allcells.pt" ]; then
        echo "[SKIP] $CELL: 已有结果"
        continue
    fi

    echo ""
    echo "════════════════════════════════════════"
    echo "  [$idx/$total] $(date '+%H:%M:%S') 训练 $CELL"
    echo "════════════════════════════════════════"

    python New/train_operator_moe.py \
        --data_dir "$DATA_DIR" \
        --device "$DEVICE" \
        $COMMON \
        2>&1 | tee "$LOG_FILE"

    best_auc=$(grep "VAL_AUC" "$LOG_FILE" | awk -F'VAL_AUC:' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    best_prc=$(grep "VAL_AUC" "$LOG_FILE" | grep "$best_auc" | awk -F'PRC:' '{print $2}' | awk '{print $1}')
    best_f1=$(grep  "VAL_AUC" "$LOG_FILE" | grep "$best_auc" | awk -F'F1:' '{print $2}' | awk '{print $1}')

    echo "$CELL | $best_auc | $best_prc | $best_f1 | $(date '+%H:%M:%S')" >> "$RESULTS_SUMMARY"
    echo "  → $CELL 最优 AUC: $best_auc"
done

echo ""
echo "════════════════════════════════════════"
echo "  GPU3 大细胞系完成！$(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════"

# GPU3 大细胞系完成后，检查是否可以接 RF 和 DL baseline 对比
echo ""
echo "  开始 GPU3 备选任务：HT29/PC3 的 RF baseline..."

for CELL in HT29 PC3 HA1E NPC; do
    DATA_DIR="$DATA_ROOT/$CELL"
    for fold in 0 1 2 3 4; do
        rf_log="logs_allcells/${CELL}_RF_Fold${fold}.log"
        if [ -f "$rf_log" ]; then continue; fi
        echo "  RF $CELL Fold$fold..."
        python train_baseline_ml.py \
            --data_dir "$DATA_DIR" --fold $fold --model rf \
            > "$rf_log" 2>&1
    done
done

echo ""
echo "GPU3 全部完成！$(date '+%Y-%m-%d %H:%M:%S')"
cat "$LOG_DIR/gpu3_summary.txt"
