#!/bin/bash
# ============================================================
# run_multi_cell_analysis.sh
# 4 细胞系基因注意力跨细胞系对比分析
# 前提：4 个细胞系的 run_all_analysis.sh 均已完成
# ============================================================

CACHE_MCF7="analyze/cache/MCF7"
CACHE_A375="analyze/cache/A375"
CACHE_A549="analyze/cache/A549"
CACHE_VCAP="analyze/cache/VCAP"

FIG_DIR="analyze/figures/cross_cell"
mkdir -p "$FIG_DIR"

echo "================================================"
echo "  跨细胞系基因注意力对比分析"
echo "================================================"

# 检查哪些细胞系的缓存已就绪
AVAILABLE_CACHE=()
AVAILABLE_CELLS=()

for CL in MCF7 A375 A549 VCAP; do
    CACHE="analyze/cache/${CL}"
    if [ -f "${CACHE}/representations.npz" ]; then
        AVAILABLE_CACHE+=("$CACHE")
        AVAILABLE_CELLS+=("$CL")
        echo "  ✓ ${CL}: 缓存就绪"
    else
        echo "  ✗ ${CL}: 缓存未就绪（先运行 run_all_analysis.sh ${CL}）"
    fi
done

if [ ${#AVAILABLE_CELLS[@]} -lt 2 ]; then
    echo ""
    echo "⚠️ 至少需要 2 个细胞系的缓存，当前只有 ${#AVAILABLE_CELLS[@]} 个"
    exit 1
fi

echo ""
echo "分析 ${#AVAILABLE_CELLS[@]} 个细胞系: ${AVAILABLE_CELLS[*]}"
echo "→ 输出目录: ${FIG_DIR}"

python analyze/visualize_gene_attention.py \
    --cache_dirs "${AVAILABLE_CACHE[@]}" \
    --cell_lines "${AVAILABLE_CELLS[@]}" \
    --output_dir "$FIG_DIR"

echo ""
echo "✅ 跨细胞系分析完成: ${FIG_DIR}/"
