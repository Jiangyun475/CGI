#!/bin/bash
# ============================================================
# run_allcells_gpu2.sh
# 所有细胞系 DrugOp no_moe Fold0 扫描 — cuda:2
#
# 负责：小细胞系(<50K) + 中等细胞系(50K-100K)，共38个
# 预计总时长：~2.5小时（等当前实验结束后自动启动）
#
# 用法：nohup bash run_allcells_gpu2.sh > logs_allcells/gpu2_all.log 2>&1 &
# ============================================================

DEVICE="cuda:2"
DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
LOG_DIR="logs_allcells"
RESULT_DIR="results_operator_moe"
mkdir -p "$LOG_DIR"

COMMON="--fold 0 --epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
        --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
        --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 \
        --ablation no_moe --patience 10 --seed 42 --use_amp --run_tag allcells"

# 等待 GPU2 当前任务结束（检查显存占用）
echo "[$(date '+%H:%M:%S')] 等待 $DEVICE 释放..."
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null)
    if [ "$used" -lt 1000 ] 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] $DEVICE 已释放 (${used}MiB)，开始训练"
        break
    fi
    sleep 60
done

# ── 小细胞系（< 50K，~2-5 min/cell）────────────────────────────
SMALL_CELLS=(
    22RV1 A204 BC3C BEN CAL29 CJM GI1
    HCC95 HEC108 HEC1A HEC251 HEC265
    HEK293 HELA HUVEC
    IGR37 JHH5 JURKAT MCF10A
    MDAMB231 MDAMB468 MELHO
    NCIH1573 NCIH2110 NCIH838
    OVTOKO SH4 SKES1 SNU407
    T47D THP1 YAPC
)

# ── 中等细胞系（50K-100K，~8-15 min/cell）──────────────────────
MEDIUM_CELLS=(
    HEPG2 PHH ASC HCC515 SKB NEU
)

ALL_CELLS=("${SMALL_CELLS[@]}" "${MEDIUM_CELLS[@]}")

RESULTS_SUMMARY="$LOG_DIR/gpu2_summary.txt"
echo "Cell | AUC | PRC | F1 | Time" > "$RESULTS_SUMMARY"

total=${#ALL_CELLS[@]}
idx=0
for CELL in "${ALL_CELLS[@]}"; do
    idx=$((idx + 1))
    DATA_DIR="$DATA_ROOT/$CELL"

    if [ ! -d "$DATA_DIR" ]; then
        echo "[SKIP] $CELL: 目录不存在"
        continue
    fi

    LOG_FILE="$LOG_DIR/${CELL}_fold0.log"

    # 如果已有结果且AUC合理，跳过
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

    # 提取最优 AUC
    best_auc=$(grep "最优 AUC" "$LOG_FILE" | tail -1 | awk '{print $NF}' | sed 's|.*Fold.*||')
    best_auc=$(grep "VAL_AUC" "$LOG_FILE" | awk -F'VAL_AUC:' '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    best_prc=$(grep "VAL_AUC" "$LOG_FILE" | grep "$best_auc" | awk -F'PRC:' '{print $2}' | awk '{print $1}')
    best_f1=$(grep  "VAL_AUC" "$LOG_FILE" | grep "$best_auc" | awk -F'F1:' '{print $2}' | awk '{print $1}')

    echo "$CELL | $best_auc | $best_prc | $best_f1 | $(date '+%H:%M:%S')" >> "$RESULTS_SUMMARY"
    echo "  → $CELL 最优 AUC: $best_auc"
done

echo ""
echo "════════════════════════════════════════"
echo "  GPU2 全部完成！$(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════"
echo ""
echo "汇总结果："
cat "$RESULTS_SUMMARY"
