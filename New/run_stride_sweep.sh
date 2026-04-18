#!/bin/bash
# ============================================================
# Stride/Length Sweep：确定最优基因序列覆盖范围
# MCF7 Fold0，no_moe 官方配置，仅改变 gene_max_len 和 gene_stride
#
# 实验设计：
#   固定 token 数 max_len=1000，改变 stride → 改变覆盖 bp 数
#   + 固定 stride=1，增大 max_len → 直接更多 token
#
# 覆盖范围对照：
#   s1  len1000  → ~1006  bp  (5.6% 完整覆盖，基准)
#   s2  len1000  → ~2006  bp  (30.1%)
#   s3  len1000  → ~3006  bp  (~43%)
#   s5  len1000  → ~5006  bp  (~79%)
#   s1  len2000  → ~2006  bp  (30.1%，对比 s2_len1000)
#   s1  len3000  → ~3006  bp  (~43%，对比 s3_len1000)
#   s1  len5000  → ~5006  bp  (~79%，对比 s5_len1000)
# ============================================================

DATA=/home/data/jiangyun/cgi_data_pipeline/outputs/datasets_classification_test_recommended/MCF7
COMMON="--data_dir $DATA --fold 0 --ablation no_moe \
  --epochs 80 --batch_size 512 --lr 2e-4 \
  --hidden_dim 128 --operator_rank 8 --dropout 0.3 \
  --warmup_epochs 5 --lam_sparse 0.01 --lam_ortho_modes 0.1 --lam_cl 0.0 \
  --patience 10 --seed 42 --use_amp"

LOG=logs_stride_sweep
mkdir -p $LOG

echo "=== Stride/Length Sweep ==="
echo "结果保存至 $LOG/"

# ── 固定 max_len=1000，变 stride ──────────────────────────────
echo "[1/7] baseline: stride=1, len=1000 (~1006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 1000 --gene_stride 1 \
  --device cuda:0 --run_tag stride_s1_l1000 \
  2>&1 | tee $LOG/s1_len1000.log &

echo "[2/7] stride=2, len=1000 (~2006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 1000 --gene_stride 2 \
  --device cuda:1 --run_tag stride_s2_l1000 \
  2>&1 | tee $LOG/s2_len1000.log &

echo "[3/7] stride=3, len=1000 (~3006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 1000 --gene_stride 3 \
  --device cuda:2 --run_tag stride_s3_l1000 \
  2>&1 | tee $LOG/s3_len1000.log &

echo "[4/7] stride=5, len=1000 (~5006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 1000 --gene_stride 5 \
  --device cuda:3 --run_tag stride_s5_l1000 \
  2>&1 | tee $LOG/s5_len1000.log &

wait
echo "第一批完成，启动第二批..."

# ── 固定 stride=1，变 max_len（直接增加 token 数）────────────
echo "[5/7] stride=1, len=2000 (~2006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 2000 --gene_stride 1 \
  --device cuda:0 --run_tag stride_s1_l2000 \
  2>&1 | tee $LOG/s1_len2000.log &

echo "[6/7] stride=1, len=3000 (~3006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 3000 --gene_stride 1 \
  --device cuda:1 --run_tag stride_s1_l3000 \
  2>&1 | tee $LOG/s1_len3000.log &

echo "[7/7] stride=1, len=5000 (~5006bp)"
python New/train_operator_moe.py $COMMON \
  --gene_max_len 5000 --gene_stride 1 \
  --device cuda:2 --run_tag stride_s1_l5000 \
  2>&1 | tee $LOG/s1_len5000.log &

wait
echo "全部完成。运行 python New/plot_stride_analysis.py 生成分析图。"
