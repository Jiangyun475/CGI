#!/bin/bash
# ==============================================================================
# 全细胞系消融实验 + 对比实验（MCF7 已单独跑，本脚本覆盖其余 6 个细胞系）
# ==============================================================================
#
# 细胞系 & GPU 分配：
#   cuda:1 (立即训练) → VCAP(大) → HELA(小)
#   cuda:3 (立即训练) → A549(大) → THP1(小)
#   cuda:0 (排队)     → HT29(大)
#   cuda:2 (排队)     → HUVEC(小)  ← MCF7 ablation 结束后自动接续
#
# Full 模型结果来源（不重跑）：
#   MCF7   → logs_5folds/MCF7_Fold*.log
#   VCAP/A549/HT29/HELA/THP1 → logs_5folds_all/{CELL}_Fold*.log
#   HUVEC  → logs_5folds/HUVEC_Fold*.log
#
# 消融配置（每个细胞系 × 8 组 × 5 Fold = 40 runs/细胞系）：
#   wo_CL         hybrid  + ortho           --disable_cl
#   wo_Ortho      hybrid  + CL              --disable_ortho
#   wo_CL_Ortho   hybrid  only              --disable_cl --disable_ortho
#   SumMean       sum_mean + ortho + CL
#   TargetOnly    target  + ortho + CL
#   Baseline_Ult  sum_mean (全关)            --disable_cl --disable_ortho
#   Baseline_DL   晚期融合 GNN（独立脚本）
#   Baseline_RF + Baseline_XGB（CPU，串行保证缓存不竞争）
# ==============================================================================

BASE_DIR="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended"
SCRIPT_ULT="train_ultimate.py"
SCRIPT_DL="train_baseline_dl.py"
SCRIPT_ML="train_baseline_ml.py"
LOG_DIR="logs_ablation_all"

mkdir -p "${LOG_DIR}"

# ==============================================================================
# 超参数表（大/小数据集分档）
# ==============================================================================
declare -A BATCH_SIZE=( [VCAP]=512  [A549]=512  [HT29]=512  [MCF7]=512
                        [HELA]=128  [THP1]=128  [HUVEC]=128 )
declare -A LAM_VAR=(    [VCAP]=0.2  [A549]=0.2  [HT29]=0.2  [MCF7]=0.2
                        [HELA]=0.05 [THP1]=0.05 [HUVEC]=0.05 )
declare -A LAM_CL=(     [VCAP]=0.2  [A549]=0.2  [HT29]=0.2  [MCF7]=0.2
                        [HELA]=0.05 [THP1]=0.05 [HUVEC]=0.05 )

# ==============================================================================
# 工具函数：单 fold 消融训练
# run_one_fold GPU DATASET CONFIG_NAME POOL_TYPE FOLD [extra_flags...]
# ==============================================================================
run_one_fold() {
    local GPU=$1 DATASET=$2 NAME=$3 POOL=$4 FOLD=$5
    shift 5
    local LOG="${LOG_DIR}/${DATASET}_${NAME}_Fold${FOLD}.log"
    local DATA_DIR="${BASE_DIR}/${DATASET}"

    python "${SCRIPT_ULT}" \
        --data_dir   "${DATA_DIR}" \
        --device     "${GPU}" \
        --fold       "${FOLD}" \
        --epochs     50 \
        --batch_size "${BATCH_SIZE[$DATASET]}" \
        --lr         2e-4 \
        --hidden_dim 128 \
        --dropout    0.3 \
        --lam_var    "${LAM_VAR[$DATASET]}" \
        --lam_cl     "${LAM_CL[$DATASET]}" \
        --patience   10 \
        --seed       42 \
        --use_amp \
        --pool_type  "${POOL}" \
        "$@" \
        > "${LOG}" 2>&1
}

# ==============================================================================
# 消融：对某数据集在某 GPU 上跑完 1 个配置的 5 个 Fold
# run_ablation GPU DATASET CONFIG_NAME POOL_TYPE [extra_flags...]
# ==============================================================================
run_ablation() {
    local GPU=$1 DATASET=$2 NAME=$3 POOL=$4
    shift 4

    echo "  ▶ [GPU ${GPU}] ${DATASET} / ${NAME} (pool=${POOL} $*)"
    for FOLD in 0 1 2 3 4; do
        run_one_fold "${GPU}" "${DATASET}" "${NAME}" "${POOL}" "${FOLD}" "$@"
        echo "    ✅ ${DATASET} ${NAME} Fold${FOLD}"
        sleep 1
    done
    echo "  🎉 ${DATASET} / ${NAME} 完成"
}

# ==============================================================================
# 对某数据集运行全部 6 个 DL 消融配置
# run_all_ablations GPU DATASET
# ==============================================================================
run_all_ablations() {
    local GPU=$1 DATASET=$2

    echo "========================================================"
    echo "🔬 [GPU ${GPU}] ${DATASET} 消融实验启动"
    echo "   batch=${BATCH_SIZE[$DATASET]}  lam_var=${LAM_VAR[$DATASET]}  lam_cl=${LAM_CL[$DATASET]}"
    echo "========================================================"

    run_ablation "${GPU}" "${DATASET}" "wo_CL"        "hybrid"   --disable_cl
    run_ablation "${GPU}" "${DATASET}" "wo_Ortho"     "hybrid"   --disable_ortho
    run_ablation "${GPU}" "${DATASET}" "wo_CL_Ortho"  "hybrid"   --disable_cl --disable_ortho
    run_ablation "${GPU}" "${DATASET}" "SumMean"      "sum_mean"
    run_ablation "${GPU}" "${DATASET}" "TargetOnly"   "target"
    run_ablation "${GPU}" "${DATASET}" "Baseline_Ult" "sum_mean" --disable_cl --disable_ortho

    echo "🏆 [GPU ${GPU}] ${DATASET} 全部消融完成"
    echo ""
}

# ==============================================================================
# DL 基线：对某数据集在某 GPU 上跑 5 Fold
# run_dl_baseline GPU DATASET
# ==============================================================================
run_dl_baseline() {
    local GPU=$1 DATASET=$2
    local DATA_DIR="${BASE_DIR}/${DATASET}"

    echo "  ▶ [GPU ${GPU}] ${DATASET} / Baseline_DL"
    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/${DATASET}_Baseline_DL_Fold${FOLD}.log"
        python "${SCRIPT_DL}" \
            --data_dir   "${DATA_DIR}" \
            --device     "${GPU}" \
            --fold       "${FOLD}" \
            --epochs     50 \
            --batch_size "${BATCH_SIZE[$DATASET]}" \
            --lr         2e-4 \
            --hidden_dim 128 \
            --dropout    0.3 \
            --patience   10 \
            --seed       42 \
            --use_amp \
            > "${LOG}" 2>&1
        echo "    ✅ ${DATASET} Baseline_DL Fold${FOLD}"
        sleep 1
    done
    echo "  🎉 ${DATASET} / Baseline_DL 完成"
}

# ==============================================================================
# ML 基线：RF 和 XGB 串行（保证缓存不竞争写入）
# run_ml_baselines DATASET
# ==============================================================================
run_ml_baselines() {
    local DATASET=$1
    local DATA_DIR="${BASE_DIR}/${DATASET}"

    echo "  ▶ [CPU] ${DATASET} / Baseline_RF + Baseline_XGB（串行）"

    # RF：Fold 0 会自动构建并保存 ml_features_full.pkl，后续 fold 直接读缓存
    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/${DATASET}_Baseline_RF_Fold${FOLD}.log"
        python "${SCRIPT_ML}" \
            --data_dir     "${DATA_DIR}" \
            --fold         "${FOLD}" \
            --model        rf \
            --n_estimators 200 \
            --max_depth    20 \
            > "${LOG}" 2>&1
        echo "    ✅ ${DATASET} Baseline_RF Fold${FOLD}"
    done

    # XGB：缓存已存在，直接复用
    for FOLD in 0 1 2 3 4; do
        LOG="${LOG_DIR}/${DATASET}_Baseline_XGB_Fold${FOLD}.log"
        python "${SCRIPT_ML}" \
            --data_dir     "${DATA_DIR}" \
            --fold         "${FOLD}" \
            --model        xgb \
            --n_estimators 200 \
            --max_depth    20 \
            > "${LOG}" 2>&1
        echo "    ✅ ${DATASET} Baseline_XGB Fold${FOLD}"
    done

    echo "  🎉 ${DATASET} / Baseline_RF + XGB 完成"
}

# ==============================================================================
# 派发任务
# ==============================================================================
echo ""
echo "=========================================================="
echo "🚀 全细胞系消融 + 对比实验"
echo "   cuda:1 → VCAP → HELA"
echo "   cuda:3 → A549 → THP1"
echo "   cuda:0 → HT29"
echo "   cuda:2 → HUVEC（MCF7 ablation 完成后自动接续）"
echo "   CPU    → 6 细胞系 × (RF + XGB)，各自独立后台"
echo "=========================================================="
echo ""

# ── GPU cuda:1：VCAP(大) → HELA(小) ───────────────────────────
(
    run_all_ablations "cuda:1" "VCAP"
    run_dl_baseline   "cuda:1" "VCAP"
    run_all_ablations "cuda:1" "HELA"
    run_dl_baseline   "cuda:1" "HELA"
    echo "🏁 [cuda:1] 队列完成: VCAP HELA"
) &
PID1=$!

# ── GPU cuda:3：A549(大) → THP1(小) ───────────────────────────
(
    run_all_ablations "cuda:3" "A549"
    run_dl_baseline   "cuda:3" "A549"
    run_all_ablations "cuda:3" "THP1"
    run_dl_baseline   "cuda:3" "THP1"
    echo "🏁 [cuda:3] 队列完成: A549 THP1"
) &
PID3=$!

# ── GPU cuda:0：HT29(大) ──────────────────────────────────────
(
    run_all_ablations "cuda:0" "HT29"
    run_dl_baseline   "cuda:0" "HT29"
    echo "🏁 [cuda:0] 队列完成: HT29"
) &
PID0=$!

# ── GPU cuda:2：HUVEC(小)，MCF7 ablation 结束后接续 ───────────
(
    run_all_ablations "cuda:2" "HUVEC"
    run_dl_baseline   "cuda:2" "HUVEC"
    echo "🏁 [cuda:2] 队列完成: HUVEC"
) &
PID2=$!

# ── CPU：各细胞系 ML 基线，并行但每个数据集内部串行 ─────────────
for CELL in VCAP A549 HT29 HELA THP1 HUVEC; do
    run_ml_baselines "${CELL}" &
done

echo "🔥 所有任务已派发"
echo "   cuda:1 PID: ${PID1}"
echo "   cuda:3 PID: ${PID3}"
echo "   cuda:0 PID: ${PID0}"
echo "   cuda:2 PID: ${PID2}"
echo ""
echo "查看进度："
echo "  tail -f ${LOG_DIR}/VCAP_wo_CL_Fold0.log"
echo "  tail -f ${LOG_DIR}/A549_SumMean_Fold0.log"
echo "=========================================================="

wait ${PID1} ${PID3} ${PID0} ${PID2}
wait  # 等待所有 CPU 后台

# ==============================================================================
# 汇总所有细胞系结果（含 Full 历史结果）
# ==============================================================================
echo ""
echo "=========================================================="
echo "📊 全细胞系消融结果汇总"
echo "=========================================================="

python3 - << 'PYEOF'
import os, re
import numpy as np

LOG_ABLATION   = "logs_ablation_all"
LOG_MCF7_ABL   = "logs_ablation_mcf7"   # MCF7 消融日志
LOG_FULL_SMALL = "logs_5folds"           # MCF7/HUVEC Full
LOG_FULL_LARGE = "logs_5folds_all"       # VCAP/A549/HT29/HELA/THP1 Full

CELLS = ["MCF7", "VCAP", "A549", "HT29", "HELA", "THP1", "HUVEC"]

ABLATION_CONFIGS = [
    "Full",
    "wo_CL",
    "wo_Ortho",
    "wo_CL_Ortho",
    "SumMean",
    "TargetOnly",
    "Baseline_Ult",
    "Baseline_DL",
    "Baseline_RF",
    "Baseline_XGB",
]

def full_log_dir(cell):
    return LOG_FULL_SMALL if cell in ("MCF7", "HUVEC") else LOG_FULL_LARGE

def get_log_path(cell, config, fold):
    if config == "Full":
        d = full_log_dir(cell)
        return os.path.join(d, f"{cell}_Fold{fold}.log")
    elif cell == "MCF7":
        return os.path.join(LOG_MCF7_ABL, f"{config}_Fold{fold}.log")
    else:
        return os.path.join(LOG_ABLATION, f"{cell}_{config}_Fold{fold}.log")

pat_dl  = re.compile(r"VAL_AUC:\s*([\d.]+)\s*\|\s*PRC:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)")
pat_auc = re.compile(r"AUROC\s*:\s*([\d.]+)")
pat_prc = re.compile(r"AUPRC\s*:\s*([\d.]+)")
pat_f1  = re.compile(r"F1\s*:\s*([\d.]+)")

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    if "VAL_AUC" in content:
        matches = pat_dl.findall(content)
        if not matches: return None
        best = max(matches, key=lambda x: float(x[0]))
        return float(best[0]), float(best[1]), float(best[2])
    else:
        m_a = pat_auc.search(content)
        if not m_a: return None
        m_p = pat_prc.search(content)
        m_f = pat_f1.search(content)
        return (float(m_a.group(1)),
                float(m_p.group(1)) if m_p else 0.0,
                float(m_f.group(1)) if m_f else 0.0)

for cell in CELLS:
    print(f"\n── {cell} ─────────────────────────────────────────────")
    print(f"  {'配置':<16s} | {'AUC':>8s}±{'std':>6s} | {'PRC':>8s}±{'std':>6s} | {'F1':>8s}±{'std':>6s} | Folds")
    print("  " + "-" * 72)
    for cfg in ABLATION_CONFIGS:
        results = [parse_log(get_log_path(cell, cfg, f)) for f in range(5)]
        done = [r for r in results if r is not None]
        if not done:
            print(f"  {cfg:<16s} | {'(未完成)':>8s}")
            continue
        aucs = np.array([r[0] for r in done])
        prcs = np.array([r[1] for r in done])
        f1s  = np.array([r[2] for r in done])
        n = len(done)
        std_a = aucs.std() if n > 1 else 0.0
        std_p = prcs.std() if n > 1 else 0.0
        std_f = f1s.std()  if n > 1 else 0.0
        mark = "✅" if n == 5 else f"({n}/5)"
        print(f"  {cfg:<16s} | {aucs.mean():.4f}±{std_a:.4f} | "
              f"{prcs.mean():.4f}±{std_p:.4f} | "
              f"{f1s.mean():.4f}±{std_f:.4f} | {mark}")

print()
PYEOF

echo "=========================================================="
echo "🏆 全部实验完成！日志目录: ${LOG_DIR}/"
echo "=========================================================="
