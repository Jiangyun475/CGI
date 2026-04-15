#!/bin/bash
# interp/run.sh
# 在 GPU3 上运行完整的可解释性分析流水线（MCF7, Fold0, cl01 模型）

set -e
cd /home/data/jiangyun/cgi_data_pipeline5

CELL="MCF7"
MODEL="results_operator_moe/MCF7/no_moe_r8_k4_Fold0_cl01.pt"
DEVICE="cuda:3"
LOG_DIR="interp/results/MCF7"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Step 1: Extract representations"
echo "========================================"
python interp/extract.py \
    --cell   $CELL \
    --model_path $MODEL \
    --fold   0 \
    --rank   8 \
    --device $DEVICE \
    2>&1 | tee "$LOG_DIR/extract.log"

echo ""
echo "========================================"
echo "Step 2: GO enrichment per mode"
echo "========================================"
python interp/go_enrich.py \
    --cell  $CELL \
    --top_k 100 \
    2>&1 | tee "$LOG_DIR/go_enrich.log"

echo ""
echo "========================================"
echo "Step 3: Drug sigma clustering"
echo "========================================"
python interp/drug_cluster.py \
    --cell $CELL \
    2>&1 | tee "$LOG_DIR/drug_cluster.log"

echo ""
echo "========================================"
echo "All done. Results in interp/results/$CELL/"
echo "========================================"
