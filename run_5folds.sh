#!/bin/bash
# ==============================================================================
# CGI Ultimate - 5 Folds Cross Validation 异步调度脚本
# GPU 2: 专注跑 MCF7 (大数据集，耗时长)
# GPU 3: 专注跑 HUVEC (小数据集，耗时短)
# ==============================================================================

# 基础路径配置
BASE_DIR="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
SCRIPT_NAME="train_ultimate.py"
LOG_DIR="logs_5folds"

mkdir -p ${LOG_DIR}

echo "=========================================================="
echo "🚀 启动 5-Fold 交叉验证批量训练"
echo "=========================================================="

# ==========================================
# 任务函数: 在指定GPU上连续跑完5个Fold
# ==========================================
run_dataset() {
    local DATASET_NAME=$1
    local GPU_ID=$2
    local BATCH_SIZE=$3
    local LAM_VAR=$4
    local LAM_CL=$5

    echo ">>> 开始分配 ${DATASET_NAME} 任务队列到 GPU ${GPU_ID} ..."

    for FOLD in 0 1 2 3 4; do
        LOG_FILE="${LOG_DIR}/${DATASET_NAME}_Fold${FOLD}.log"
        echo "[${DATASET_NAME} | Fold ${FOLD}] 启动训练 -> 正在写入 ${LOG_FILE}"
        
        # 执行 Python 训练代码
        python ${SCRIPT_NAME} \
            --data_dir ${BASE_DIR}/${DATASET_NAME} \
            --device ${GPU_ID} \
            --fold ${FOLD} \
            --epochs 50 \
            --batch_size ${BATCH_SIZE} \
            --lr 2e-4 \
            --hidden_dim 128 \
            --dropout 0.3 \
            --lam_var ${LAM_VAR} \
            --lam_cl ${LAM_CL} \
            --patience 10 \
            --use_amp \
            --pool_type hybrid \
            > ${LOG_FILE} 2>&1
            
        echo "[${DATASET_NAME} | Fold ${FOLD}] ✅ 训练完成！"
        sleep 2 # 缓冲时间，防止显存释放延迟
    done
    
    echo "🎉 ${DATASET_NAME} 所有 5 个 Fold 已经全部在 GPU ${GPU_ID} 上执行完毕！"
}

# ==========================================
# 异步派发任务 (利用 & 放入后台并发执行)
# ==========================================

# 派发 MCF7 到 GPU 2 
# 参数: 数据集名称, GPU编号, Batch Size, lam_var, lam_cl
run_dataset "MCF7" "cuda:2" 512 0.2 0.2 &
PID_MCF7=$!

# 派发 HUVEC 到 GPU 3
# 注意：HUVEC数据少，为了流形稳定，必须减小BatchSize和约束权重（基于之前的实验经验）
run_dataset "HUVEC" "cuda:3" 128 0.05 0.05 &
PID_HUVEC=$!

echo ""
echo "🔥 任务已全部派发至后台！"
echo "MCF7 进程 PID: ${PID_MCF7} (运行于 cuda:2)"
echo "HUVEC 进程 PID: ${PID_HUVEC} (运行于 cuda:3)"
echo ""
echo "你可以使用以下命令随时查看进度："
echo "tail -f ${LOG_DIR}/MCF7_Fold0.log"
echo "tail -f ${LOG_DIR}/HUVEC_Fold0.log"
echo "=========================================================="

# 等待所有后台任务完成
wait
echo "🏆 所有交叉验证训练任务完美收官！"