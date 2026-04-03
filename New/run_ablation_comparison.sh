#!/bin/bash
# ================================================================
# run_ablation_comparison.sh
# 公平消融对比: 4 种交互建模方式 × 5 折交叉验证
# ================================================================
#
# 用法:
#   bash run_ablation_comparison.sh <DATA_DIR> <GPU_ID> [EXTRA_ARGS]
#
# 示例:
#   bash run_ablation_comparison.sh /path/to/VCAP cuda:0 --use_amp
#   bash run_ablation_comparison.sh /path/to/A549 cuda:1 --use_amp --epochs 100
#
# 输出结构:
#   results_operator/<cell_line>/
#     operator_r8_Fold0.pt
#     concat_r8_Fold0.pt
#     ortho_concat_r8_Fold0.pt
#     hadamard_r8_Fold0.pt
#     ...
# ================================================================

set -e

DATA_DIR=${1:?"用法: bash $0 <DATA_DIR> <GPU_ID> [EXTRA_ARGS]"}
GPU=${2:-"cuda:0"}
shift 2
EXTRA_ARGS="$@"

CELL_LINE=$(basename "$DATA_DIR")
echo "============================================================"
echo "  细胞系: $CELL_LINE  |  GPU: $GPU"
echo "  附加参数: $EXTRA_ARGS"
echo "============================================================"

INTERACTION_TYPES=("operator" "concat" "ortho_concat" "hadamard")
FOLDS=(0 1 2 3 4)
RANK=8

for itype in "${INTERACTION_TYPES[@]}"; do
    for fold in "${FOLDS[@]}"; do
        echo ""
        echo ">>> [$CELL_LINE] interaction=$itype  fold=$fold"
        echo "-----------------------------------------------------------"

        python train_drug_operator.py \
            --data_dir "$DATA_DIR" \
            --device "$GPU" \
            --fold "$fold" \
            --interaction_type "$itype" \
            --operator_rank "$RANK" \
            --save_spectrum \
            $EXTRA_ARGS

        echo "<<< 完成: $itype fold=$fold"
    done
done

echo ""
echo "============================================================"
echo "  全部完成: $CELL_LINE"
echo "  结果目录: results_operator/$CELL_LINE/"
echo "============================================================"
