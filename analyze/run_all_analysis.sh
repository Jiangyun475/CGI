#!/bin/bash
# ============================================================
# run_all_analysis.sh
# DrugOperatorNet 完整可视化分析流水线
# ============================================================
#
# 用法：
#   bash analyze/run_all_analysis.sh [CELL_LINE] [FOLD] [DEVICE]
#
# 参数（可选，有默认值）：
#   CELL_LINE : MCF7（默认）
#   FOLD      : 0（默认）
#   DEVICE    : cuda:0（默认）
#
# 示例：
#   bash analyze/run_all_analysis.sh MCF7 0 cuda:0
#   bash analyze/run_all_analysis.sh A375 0 cuda:1
#
# 完整 4 细胞系并行（建议分别运行）：
#   bash analyze/run_all_analysis.sh MCF7 0 cuda:0
#   bash analyze/run_all_analysis.sh A375 0 cuda:1
#   bash analyze/run_all_analysis.sh A549 0 cuda:2
#   bash analyze/run_all_analysis.sh VCAP 0 cuda:3
# ============================================================

set -e

CELL=${1:-MCF7}
FOLD=${2:-0}
DEVICE=${3:-cuda:0}

DATA_ROOT="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
MODEL_DIR="results_operator_moe"
CACHE_DIR="analyze/cache/${CELL}"
FIG_DIR="analyze/figures/${CELL}"

# 自动找模型文件（no_moe，指定 fold）
MODEL_PATH="${MODEL_DIR}/${CELL}/no_moe_r8_k4_Fold${FOLD}_v1.pt"
if [ ! -f "$MODEL_PATH" ]; then
    # 如果没有特定 tag 的文件，列出可用文件
    echo "⚠️ 模型文件不存在: $MODEL_PATH"
    echo "   ${MODEL_DIR}/${CELL}/ 下可用模型："
    ls "${MODEL_DIR}/${CELL}/"*.pt 2>/dev/null | grep "no_moe" || echo "   无 no_moe 模型"
    exit 1
fi

echo "================================================"
echo "  DrugOperatorNet 可视化分析流水线"
echo "  细胞系: ${CELL} | Fold: ${FOLD} | Device: ${DEVICE}"
echo "  模型: ${MODEL_PATH}"
echo "  缓存: ${CACHE_DIR}"
echo "  图像: ${FIG_DIR}"
echo "================================================"

# ── Step 1: 提取中间表示 ─────────────────────────────────────────
echo ""
echo "[Step 1/3] 提取模型中间表示..."
echo "  (spectrum, atom_attn, gene_attn → ${CACHE_DIR}/)"

python analyze/extract_representations.py \
    --data_dir "${DATA_ROOT}/${CELL}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${CACHE_DIR}" \
    --fold ${FOLD} \
    --device ${DEVICE} \
    --batch_size 256 \
    --hidden_dim 128 \
    --operator_rank 8 \
    --gene_max_len 1000

echo "  ✅ 提取完成"

# ── Step 2: 交互谱可视化 ─────────────────────────────────────────
echo ""
echo "[Step 2/3] 交互谱可视化..."
echo "  → ${FIG_DIR}/spectrum_analysis_${CELL}.{png,pdf}"

mkdir -p "${FIG_DIR}"
python analyze/visualize_spectrum.py \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${FIG_DIR}" \
    --cell_line "${CELL}"

echo "  ✅ 谱分析完成"

# ── Step 3: 药效团热图 ──────────────────────────────────────────
echo ""
echo "[Step 3/3] 药效团热图..."
echo "  → ${FIG_DIR}/pharmacophore/"

mkdir -p "${FIG_DIR}/pharmacophore"
python analyze/visualize_pharmacophore.py \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${FIG_DIR}/pharmacophore" \
    --cell_line "${CELL}" \
    --n_top 8

echo "  ✅ 药效团热图完成"

echo ""
echo "================================================"
echo "  全部分析完成！"
echo "  查看图像: ls ${FIG_DIR}/"
echo "================================================"
