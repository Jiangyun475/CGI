#!/bin/bash
# ==============================================================================
# MCF7 消融实验 + 对比实验
# ==============================================================================
#
# 实验设计：
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ 消融实验（train_ultimate.py，已有 Full 结果，共 6 组新实验）      │
#   │   Full(Ours)     hybrid  + ortho + CL     → 已完成(logs_5folds)  │
#   │   wo_CL          hybrid  + ortho          → 证明 CL 的贡献       │
#   │   wo_Ortho       hybrid  + CL             → 证明正交剥离的贡献    │
#   │   wo_CL_Ortho    hybrid  (仅本文池化)      → 池化单独的贡献       │
#   │   SumMean        sum_mean + ortho + CL    → 证明靶向池化的贡献    │
#   │   TargetOnly     target  + ortho + CL    → 靶向池化单独效果      │
#   │   Baseline_Ult   sum_mean (全关)          → 最弱 DL 基线         │
#   ├──────────────────────────────────────────────────────────────────┤
#   │ 对比实验（独立基线）                                               │
#   │   Baseline_DL    晚期融合 GNN（GraphDTA 变体，train_baseline_dl） │
#   │   Baseline_RF    Random Forest（Morgan FP + 6-mer, train_ml）     │
#   │   Baseline_XGB   XGBoost（Morgan FP + 6-mer, train_ml）          │
#   └──────────────────────────────────────────────────────────────────┘
#
# GPU 分配：
#   cuda:2 → wo_CL → wo_Ortho → wo_CL_Ortho → Baseline_Ult
#   cuda:3 → SumMean → TargetOnly → Baseline_DL
#   CPU    → Baseline_RF (后台) + Baseline_XGB (后台)
#
# 注意：Full 模型已在 logs_5folds/MCF7_Fold*.log 中完成，本脚本不重跑。
#       若需要统一到本目录，取消注释 GPU2 队列中的 Full 部分。
# ==============================================================================

BASE_DIR="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
MCF7_DIR="${BASE_DIR}/MCF7"
SCRIPT_ULT="train_ultimate.py"
SCRIPT_DL="train_baseline_dl.py"
SCRIPT_ML="train_baseline_ml.py"
LOG_DIR="logs_ablation_mcf7"

mkdir -p "${LOG_DIR}"

# ==============================================================================
# 消融实验核心函数
# 用法：run_ablation GPU CONFIG_NAME POOL_TYPE [extra_flags...]
# ==============================================================================
run_ablation() {
    local GPU=$1
    local NAME=$2
    local POOL=$3
    shift 3
    # "$@" 接收剩余 flag，如 --disable_cl --disable_ortho

    echo "================================================================"
    echo "▶  [GPU ${GPU}]  ${NAME}  (pool=${POOL}  extra: $*)"
    echo "================================================================"

    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/${NAME}_Fold${FOLD}.log"
        echo "   Fold ${FOLD} → ${LOG}"

        python "${SCRIPT_ULT}" \
            --data_dir   "${MCF7_DIR}" \
            --device     "${GPU}" \
            --fold       "${FOLD}" \
            --epochs     50 \
            --batch_size 512 \
            --lr         2e-4 \
            --hidden_dim 128 \
            --dropout    0.3 \
            --lam_var    0.2 \
            --lam_cl     0.2 \
            --patience   10 \
            --seed       42 \
            --use_amp \
            --pool_type  "${POOL}" \
            "$@" \
            > "${LOG}" 2>&1

        echo "   ✅ ${NAME} Fold ${FOLD} 完成"
        sleep 2
    done

    echo "🎉 ${NAME} 全部 5 Fold 完成"
    echo ""
}

# ==============================================================================
# DL 基线函数
# ==============================================================================
run_dl_baseline() {
    local GPU=$1

    echo "================================================================"
    echo "▶  [GPU ${GPU}]  Baseline_DL（晚期融合 GraphDTA 变体）"
    echo "================================================================"

    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/Baseline_DL_Fold${FOLD}.log"
        echo "   Fold ${FOLD} → ${LOG}"

        python "${SCRIPT_DL}" \
            --data_dir   "${MCF7_DIR}" \
            --device     "${GPU}" \
            --fold       "${FOLD}" \
            --epochs     50 \
            --batch_size 512 \
            --lr         2e-4 \
            --hidden_dim 128 \
            --dropout    0.3 \
            --patience   10 \
            --seed       42 \
            --use_amp \
            > "${LOG}" 2>&1

        echo "   ✅ Baseline_DL Fold ${FOLD} 完成"
        sleep 2
    done

    echo "🎉 Baseline_DL 全部 5 Fold 完成"
    echo ""
}

# ==============================================================================
# ML 基线函数（CPU，缓存在第一个 fold 自动构建后后续复用）
# ==============================================================================
run_ml_baseline() {
    local MODEL=$1
    local NAME="Baseline_${MODEL^^}"

    echo "================================================================"
    echo "▶  [CPU]  ${NAME}（Morgan FP 1024-bit + 6-mer 4096-dim）"
    echo "================================================================"

    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/${NAME}_Fold${FOLD}.log"
        echo "   Fold ${FOLD} → ${LOG}"

        python "${SCRIPT_ML}" \
            --data_dir     "${MCF7_DIR}" \
            --fold         "${FOLD}" \
            --model        "${MODEL}" \
            --n_estimators 200 \
            --max_depth    20 \
            > "${LOG}" 2>&1

        echo "   ✅ ${NAME} Fold ${FOLD} 完成"
    done

    echo "🎉 ${NAME} 全部 5 Fold 完成"
    echo ""
}

# ==============================================================================
# 派发任务
# ==============================================================================
echo ""
echo "=========================================================="
echo "🚀 MCF7 消融实验 + 对比实验启动"
echo "   cuda:2 队列: wo_CL → wo_Ortho → wo_CL_Ortho → Baseline_Ult"
echo "   cuda:3 队列: SumMean → TargetOnly → Baseline_DL"
echo "   CPU   队列: Baseline_RF + Baseline_XGB（后台并行）"
echo "   Full(Ours) 结果已存于 logs_5folds/MCF7_Fold*.log"
echo "=========================================================="
echo ""

# ── GPU cuda:2 ──────────────────────────────────────────────────
(
    # 可选：重跑 Full 以统一日志目录
    # run_ablation "cuda:2" "Full"        "hybrid"

    run_ablation "cuda:2" "wo_CL"       "hybrid"   --disable_cl
    run_ablation "cuda:2" "wo_Ortho"    "hybrid"   --disable_ortho
    run_ablation "cuda:2" "wo_CL_Ortho" "hybrid"   --disable_cl --disable_ortho
    run_ablation "cuda:2" "Baseline_Ult" "sum_mean" --disable_cl --disable_ortho
    echo "🏁 [cuda:2] 队列全部完成"
) &
PID_GPU2=$!

# ── GPU cuda:3 ──────────────────────────────────────────────────
(
    run_ablation "cuda:3" "SumMean"    "sum_mean"
    run_ablation "cuda:3" "TargetOnly" "target"
    run_dl_baseline "cuda:3"
    echo "🏁 [cuda:3] 队列全部完成"
) &
PID_GPU3=$!

# ── CPU 后台（与 GPU 并行，互不干扰）────────────────────────────
run_ml_baseline "rf" &
PID_RF=$!

run_ml_baseline "xgb" &
PID_XGB=$!

echo "🔥 所有任务已派发至后台"
echo "   cuda:2  PID: ${PID_GPU2}"
echo "   cuda:3  PID: ${PID_GPU3}"
echo "   RF CPU  PID: ${PID_RF}"
echo "   XGB CPU PID: ${PID_XGB}"
echo ""
echo "实时查看进度示例："
echo "  tail -f ${LOG_DIR}/wo_CL_Fold0.log"
echo "  tail -f ${LOG_DIR}/SumMean_Fold0.log"
echo "  tail -f ${LOG_DIR}/Baseline_DL_Fold0.log"
echo "  tail -f ${LOG_DIR}/Baseline_RF_Fold0.log"
echo "=========================================================="

# 等待所有任务完成
wait ${PID_GPU2}
wait ${PID_GPU3}
wait ${PID_RF}
wait ${PID_XGB}

# ==============================================================================
# 自动汇总结果
# ==============================================================================
echo ""
echo "=========================================================="
echo "📊 结果汇总"
echo "=========================================================="

python3 - << 'PYEOF'
import os, re
import numpy as np

LOG_DIR = "logs_ablation_mcf7"
FULL_LOG_DIR = "logs_5folds"   # Full 模型日志目录

configs = [
    # (显示名,          日志前缀,      日志目录)
    ("Full (Ours)",     "MCF7",        FULL_LOG_DIR),
    ("wo_CL",          "wo_CL",       LOG_DIR),
    ("wo_Ortho",       "wo_Ortho",    LOG_DIR),
    ("wo_CL_Ortho",    "wo_CL_Ortho", LOG_DIR),
    ("SumMean",        "SumMean",     LOG_DIR),
    ("TargetOnly",     "TargetOnly",  LOG_DIR),
    ("Baseline_Ult",   "Baseline_Ult",LOG_DIR),
    ("Baseline_DL",    "Baseline_DL", LOG_DIR),
    ("Baseline_RF",    "Baseline_RF", LOG_DIR),
    ("Baseline_XGB",   "Baseline_XGB",LOG_DIR),
]

pat_dl  = re.compile(r"VAL_AUC:\s*([\d.]+)\s*\|\s*PRC:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)")
pat_ml  = re.compile(r"AUROC\s*:\s*([\d.]+)")
pat_prc = re.compile(r"AUPRC\s*:\s*([\d.]+)")
pat_f1  = re.compile(r"F1\s*:\s*([\d.]+)")

print(f"\n{'配置':<16s} | {'AUC mean±std':>18s} | {'PRC mean±std':>18s} | {'F1 mean±std':>18s}")
print("-" * 78)

for display, prefix, logdir in configs:
    aucs, prcs, f1s = [], [], []
    for fold in range(5):
        logfile = os.path.join(logdir, f"{prefix}_Fold{fold}.log")
        if not os.path.exists(logfile):
            continue
        with open(logfile) as f:
            content = f.read()

        if "VAL_AUC" in content:          # DL 类日志
            matches = pat_dl.findall(content)
            if matches:
                best = max(matches, key=lambda x: float(x[0]))
                aucs.append(float(best[0]))
                prcs.append(float(best[1]))
                f1s.append(float(best[2]))
        else:                              # ML 类日志
            m_a = pat_ml.search(content)
            m_p = pat_prc.search(content)
            m_f = pat_f1.search(content)
            if m_a:
                aucs.append(float(m_a.group(1)))
                prcs.append(float(m_p.group(1)) if m_p else 0.0)
                f1s.append(float(m_f.group(1)) if m_f else 0.0)

    if not aucs:
        print(f"{display:<16s} | {'(未完成)':>18s}")
        continue

    a = np.array(aucs); p = np.array(prcs); f = np.array(f1s)
    print(f"{display:<16s} | {a.mean():.4f} ± {a.std():.4f}    "
          f"| {p.mean():.4f} ± {p.std():.4f}    "
          f"| {f.mean():.4f} ± {f.std():.4f}")

print()
PYEOF

echo "=========================================================="
echo "🏆 全部实验完成！详细日志: ${LOG_DIR}/"
echo "=========================================================="
