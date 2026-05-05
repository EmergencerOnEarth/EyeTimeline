#!/usr/bin/env bash
# ==============================================================================
# OCT MAE 推理可视化脚本
#
# 用法：
#   bash scripts/infer_oct.sh                   # 默认随机抽 10 张，输出至 infer_output/oct
#   bash scripts/infer_oct.sh --n 20            # 抽 20 张
#   bash scripts/infer_oct.sh --out /tmp/vis    # 自定义输出目录
#
# 依赖：checkpoint 必须已存在于 CHECKPOINT 指向的路径
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── 默认参数 ──────────────────────────────────────────────────────────────
# 推理使用原始磁盘路径（非 /dev/shm）
CSV_FILES=(
    "/data1/kechuang/processed_result_new/processed_result_20250710_vol/processing_results_volume_sum_clean_meta.csv"
    "/data1/kechuang/processed_result_new/processed_result_20250720_vol/processing_results_volume_sum_clean_meta.csv"
    "/data1/kechuang/processed_result_new/processed_result_20250901_vol/processing_results_volume_sum_clean_meta.csv"
)
CSV_PATH_COL="output_subdir"

CHECKPOINT="checkpoints/oct_v1/checkpoint_best.pth"
OUTPUT_DIR="infer_output/oct"
N_SAMPLES=10
SEED=42

# ── 解析简单命令行参数 ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)       N_SAMPLES="$2"; shift 2 ;;
        --out)     OUTPUT_DIR="$2"; shift 2 ;;
        --ckpt)    CHECKPOINT="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        *) echo "[Warning] 未知参数: $1"; shift ;;
    esac
done

# ── 激活 conda 环境 ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eyetimeline

# ── 校验 checkpoint ────────────────────────────────────────────────────────
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[Error] Checkpoint 不存在: $CHECKPOINT"
    echo "  请先训练模型，或指定 --ckpt <路径>"
    exit 1
fi

# ── 构建 CSV 参数 ─────────────────────────────────────────────────────────
CSV_ARGS=""
for csv in "${CSV_FILES[@]}"; do
    CSV_ARGS="$CSV_ARGS $csv"
done

echo "================================================================"
echo " OCT MAE 推理可视化"
echo "  Checkpoint:  $CHECKPOINT"
echo "  输出目录:    $OUTPUT_DIR"
echo "  抽样数量:    $N_SAMPLES"
echo "  随机种子:    $SEED"
echo "================================================================"

python infer_mae.py \
    --checkpoint "${CHECKPOINT}" \
    --csv_files   ${CSV_ARGS} \
    --csv_path_col "${CSV_PATH_COL}" \
    --csv_modalities oct oct oct \
    --mask_ratio  0.85 \
    --output_dir  "${OUTPUT_DIR}" \
    --n_samples   "${N_SAMPLES}" \
    --seed        "${SEED}"

echo ""
echo "  ✓ 完成！查看结果: ls ${OUTPUT_DIR}"
echo "  元数据 CSV: ${OUTPUT_DIR}/metadata.csv"
