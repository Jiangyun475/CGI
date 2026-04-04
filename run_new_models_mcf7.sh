#!/bin/bash
# ==============================================================================
# 四个新模型在 MCF7 Fold0 上的验证实验
#
# 实验目的：量化各设计改进在 MCF7 Fold0 上的增益
#   1. Bilateral Sigma（双侧 sigma，最小变动）
#   2. Cross-Modal（全跨模态条件化 + cross-attention + 双侧 sigma）
#   3. Multi-Scale Spectrum（层次化谱，粗粒度 r_c=2 + 细粒度 r_f=6）
#   4. Pretrained V2（ChemBERTa 药物 + DNABERT-2 基因，冻结特征提取器）
#
# 基准：DrugOp no_moe MCF7 Fold0 AUC = 0.8923
#
# GPU 分配：
#   cuda:0 → Bilateral Sigma（快，约 25 min）
#   cuda:1 → Cross-Modal（稍慢，参数多约 10%，约 30 min）
#   cuda:2 → Multi-Scale Spectrum（与基准相当，约 28 min）
#   cuda:3 → Pretrained V2（需先预计算 ~30 min，再训练 ~30 min）
#
# 用法：
#   nohup bash run_new_models_mcf7.sh > logs_new_models/all.log 2>&1 &
# ==============================================================================

DATA_DIR="/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended/MCF7"
LOG_DIR="logs_new_models"
mkdir -p "$LOG_DIR"

COMMON="--fold 0 --epochs 80 --batch_size 512 --lr 2e-4 --hidden_dim 128 \
        --dropout 0.3 --operator_rank 8 --gene_max_len 1000 \
        --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 \
        --patience 10 --seed 42 --use_amp"

echo "============================================================"
echo "  新架构对比实验 — MCF7 Fold0"
echo "  基准：DrugOp no_moe AUC = 0.8923"
echo "============================================================"

# ── 1. Bilateral Sigma（仅修改 sigma 的计算方式）─────────────────
echo ""
echo "[1/4] 启动 Bilateral Sigma（cuda:0）"
python New/train_bilateral_sigma.py \
    --data_dir "$DATA_DIR" --device cuda:0 \
    $COMMON --run_tag "mcf7_v1" \
    > "$LOG_DIR/bilateral_mcf7_fold0.log" 2>&1 &
echo "  PID=$!"

sleep 3

# ── 2. Cross-Modal Conditional Interaction（完整互相条件化）────────
echo "[2/4] 启动 Cross-Modal（cuda:1）"
python New/train_cross_modal.py \
    --data_dir "$DATA_DIR" --device cuda:1 \
    $COMMON --run_tag "mcf7_v1" \
    > "$LOG_DIR/cross_modal_mcf7_fold0.log" 2>&1 &
echo "  PID=$!"

sleep 3

# ── 3. Multi-Scale Spectrum（层次化谱）───────────────────────────
echo "[3/4] 启动 Multi-Scale Spectrum（cuda:2）"
python New/train_multiscale_spectrum.py \
    --data_dir "$DATA_DIR" --device cuda:2 \
    --r_coarse 2 --r_fine 6 \
    $COMMON --run_tag "mcf7_v1" \
    > "$LOG_DIR/multiscale_mcf7_fold0.log" 2>&1 &
echo "  PID=$!"

sleep 3

# ── 4. Pretrained V2（需先预计算，再训练）───────────────────────
echo "[4/4] 启动 Pretrained V2 预计算阶段（cuda:3）"
# 步骤 4a：预计算 ChemBERTa + DNABERT-2 嵌入（需安装 transformers）
# 如果 transformers 未安装：pip install transformers
python New/train_pretrained_v2.py \
    --data_dir "$DATA_DIR" --device cuda:3 \
    --drug_encoder chemberta --gene_encoder dnabert2 \
    --cache_dir .pretrained_cache \
    --precompute \
    > "$LOG_DIR/pretrained_precompute.log" 2>&1

if [ $? -eq 0 ]; then
    echo "  预计算完成，开始训练..."
    # 步骤 4b：正式训练
    python New/train_pretrained_v2.py \
        --data_dir "$DATA_DIR" --device cuda:3 \
        --drug_encoder chemberta --gene_encoder dnabert2 \
        --cache_dir .pretrained_cache \
        $COMMON --batch_size 256 --run_tag "mcf7_v1" \
        > "$LOG_DIR/pretrained_mcf7_fold0.log" 2>&1 &
    echo "  训练 PID=$!"
else
    echo "  [WARNING] 预计算失败（可能 transformers 未安装或网络无法访问 HuggingFace）"
    echo "  跳过 Pretrained V2，改为运行 chemberta+kmer 组合（仅药物预训练）..."
    # 降级方案：ecfp4 + kmer（与原 pretrained_baseline.py 等价，用于验证新框架）
    python New/train_pretrained_v2.py \
        --data_dir "$DATA_DIR" --device cuda:3 \
        --drug_encoder ecfp4 --gene_encoder kmer \
        --cache_dir .pretrained_cache \
        $COMMON --batch_size 256 --run_tag "ecfp4_kmer" \
        > "$LOG_DIR/pretrained_ecfp4_kmer_mcf7_fold0.log" 2>&1 &
    echo "  降级方案 PID=$!"
fi

echo ""
echo "============================================================"
echo "  所有实验已提交，等待完成..."
echo "============================================================"

wait

# ── 汇总结果 ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MCF7 Fold0 实验结果汇总"
echo "============================================================"
echo "  基准（no_moe）:      0.8923"
echo ""

for name in bilateral cross_modal multiscale; do
    log="$LOG_DIR/${name}_mcf7_fold0.log"
    if [ -f "$log" ]; then
        best=$(grep "最优 AUC" "$log" | tail -1 | grep -o '[0-9]\.[0-9]*')
        echo "  $name:  $best"
    fi
done

log="$LOG_DIR/pretrained_mcf7_fold0.log"
[ -f "$log" ] && grep "最优 AUC" "$log" | tail -1 | \
    xargs -I{} echo "  pretrained(chemberta+dnabert2): {}"

echo ""
echo "  详细日志: $LOG_DIR/"
echo "============================================================"
