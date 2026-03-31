#!/bin/bash
# ==============================================================================
# CGI Paper Edition - 多细胞系 5-Fold 交叉验证批量训练
# ==============================================================================
# GPU 分配策略（按数据量均衡）:
#   cuda:2 → MCF7 (20万+) → HT29 (20万) → THP1 (2.2万)  [2大1小]
#   cuda:3 → VCAP (20万+) → A549 (20万+) → HELA (1.7万)  [2大1小]
# ==============================================================================

# ──────────────────────────────────────────
# 路径配置（按实际情况修改）
# ──────────────────────────────────────────
BASE_DIR="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
SCRIPT_NAME="train_ultimate.py"
LOG_DIR="logs_5folds_all"

mkdir -p "${LOG_DIR}"

# ──────────────────────────────────────────
# 超参配置
# 大数据集 (20万+): batch=512, lam_var=0.2, lam_cl=0.2
# 小数据集 (<3万):  batch=128, lam_var=0.05, lam_cl=0.05
# ──────────────────────────────────────────

declare -A BATCH_SIZE
# BATCH_SIZE["MCF7"]=512
BATCH_SIZE["VCAP"]=512
BATCH_SIZE["A549"]=512
BATCH_SIZE["HT29"]=512
BATCH_SIZE["HELA"]=128
BATCH_SIZE["THP1"]=128

declare -A LAM_VAR
# LAM_VAR["MCF7"]=0.2
LAM_VAR["VCAP"]=0.2
LAM_VAR["A549"]=0.2
LAM_VAR["HT29"]=0.2
LAM_VAR["HELA"]=0.05
LAM_VAR["THP1"]=0.05

declare -A LAM_CL
# LAM_CL["MCF7"]=0.2
LAM_CL["VCAP"]=0.2
LAM_CL["A549"]=0.2
LAM_CL["HT29"]=0.2
LAM_CL["HELA"]=0.05
LAM_CL["THP1"]=0.05

# ──────────────────────────────────────────
# 核心训练函数：在指定 GPU 上顺序跑完多个细胞系 × 5 Fold
# 参数: GPU_ID  DATASET_NAME1  DATASET_NAME2  ...
# ──────────────────────────────────────────
run_queue() {
    local GPU_ID=$1
    shift
    local DATASETS=("$@")

    for DATASET in "${DATASETS[@]}"; do
        echo "================================================================"
        echo "🔬 [GPU ${GPU_ID}] 开始处理细胞系: ${DATASET}"
        echo "   Batch=${BATCH_SIZE[$DATASET]}  lam_var=${LAM_VAR[$DATASET]}  lam_cl=${LAM_CL[$DATASET]}"
        echo "================================================================"

        for FOLD in 0 1 2 3 4; do
            LOG_FILE="${LOG_DIR}/${DATASET}_Fold${FOLD}.log"
            echo "  ▶ [${DATASET} | Fold ${FOLD}] 启动 → ${LOG_FILE}"

            python ${SCRIPT_NAME} \
                --data_dir  "${BASE_DIR}/${DATASET}" \
                --device    "${GPU_ID}" \
                --fold      ${FOLD} \
                --epochs    50 \
                --batch_size ${BATCH_SIZE[$DATASET]} \
                --lr        2e-4 \
                --hidden_dim 128 \
                --dropout   0.3 \
                --lam_var   ${LAM_VAR[$DATASET]} \
                --lam_cl    ${LAM_CL[$DATASET]} \
                --patience  10 \
                --seed      42 \
                --use_amp \
                --pool_type hybrid \
                > "${LOG_FILE}" 2>&1

            echo "  ✅ [${DATASET} | Fold ${FOLD}] 完成"
            sleep 2
        done

        echo "🎉 [GPU ${GPU_ID}] ${DATASET} 全部 5 Fold 完成！"
        echo ""
    done

    echo "🏁 [GPU ${GPU_ID}] 队列全部执行完毕：${DATASETS[*]}"
}

# ──────────────────────────────────────────
# 派发任务
# ──────────────────────────────────────────
echo "=========================================================="
echo "🚀 启动多细胞系 5-Fold 批量训练"
echo "   cuda:2 队列: MCF7 → HT29 → THP1"
echo "   cuda:3 队列: VCAP → A549 → HELA"
echo "=========================================================="
echo ""

# GPU 2: 大 → 大 → 小
run_queue "cuda:2" "HT29" "THP1" &
PID_GPU2=$!

# GPU 3: 大 → 大 → 小
run_queue "cuda:3" "VCAP" "A549" "HELA" &
PID_GPU3=$!

echo "🔥 两条队列已派发至后台"
echo "   cuda:2 PID: ${PID_GPU2}"
echo "   cuda:3 PID: ${PID_GPU3}"
echo ""
echo "实时查看日志示例："
echo "  tail -f ${LOG_DIR}/MCF7_Fold0.log"
echo "  tail -f ${LOG_DIR}/VCAP_Fold0.log"
echo ""
echo "查看所有细胞系最新进度："
echo "  tail -n 3 ${LOG_DIR}/*.log"
echo "=========================================================="

# 等待两条队列全部结束
wait ${PID_GPU2}
wait ${PID_GPU3}

echo ""
echo "=========================================================="
echo "🏆 全部细胞系 × 5 Fold 训练完美收官！"
echo "结果目录: results_paper/{MCF7,VCAP,A549,HT29,HELA,THP1}/"
echo "=========================================================="