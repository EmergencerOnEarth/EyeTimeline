#!/usr/bin/env bash
# ==============================================================================
# 眼底（CFP/Fundus）MAE 预训练启动脚本
#
# 用法：
#   bash scripts/train_fundus.sh          # 4 卡，nohup 后台运行
#   bash scripts/train_fundus.sh --dry-run # 仅统计数据量，不启动训练
# ==============================================================================
set -euo pipefail

# ── 项目根目录（自动定位到此脚本所在目录的上级）──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── 元数据 CSV 文件列表 ──────────────────────────────────────────────────
# 注意：CSV 文件路径保留在原始磁盘，图像目录已复制到 /dev/shm 加速 I/O。
# CSV 中的 output_subdir 字段记录的是相对路径，脚本通过 --shm_root 参数
# 将根目录重定向到 /dev/shm 下对应位置。
#
# 若 /dev/shm 中无对应目录（如重启后数据丢失），可将 SHM_ROOT 改回原始路径：
#   SHM_ROOT_20250710="/data1/kechuang/processed_result_new/processed_result_20250710"
SHM_ROOT_20250710="/dev/shm/processed_result_20250710"
SHM_ROOT_20250720="/dev/shm/processed_result_20250720"
SHM_ROOT_20250901="/dev/shm/processed_result_20250901"

CSV_FILES=(
    "${SHM_ROOT_20250710}/processed_result_corrected_clean_corrected_meta.csv"
    "${SHM_ROOT_20250720}/processed_result_clean_meta.csv"
    "${SHM_ROOT_20250901}/processing_results_sum_clean_meta.csv"
)
CSV_PATH_COL="output_subdir"
MODALITY="cfp"

# ── 训练超参 ──────────────────────────────────────────────────────────────
CHECKPOINT_INIT="weights/mae_vit_large_imagenet.bin"
OUTPUT_DIR="checkpoints/fundus_v1"
GPUS=4
BATCH_SIZE=768           # 单卡 batch size
EPOCHS=200
WARMUP_EPOCHS=20
MASK_RATIO_CFP=0.75
MASK_RATIO_OCT=0.85      # 眼底脚本不用，但参数统一保留
SAVE_EVERY=10
MAX_CHECKPOINTS=5
MASTER_PORT=29601        # 避免与 OCT 脚本端口冲突

# ── 日志文件 ──────────────────────────────────────────────────────────────
mkdir -p run_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="run_logs/train_fundus_${TIMESTAMP}.log"

# ── 激活 conda 环境 ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eyetimeline

# ── 校验 CSV 文件 & 统计数据量 ───────────────────────────────────────────
echo "================================================================"
echo " 眼底预训练 — 数据量统计"
echo "================================================================"
TOTAL_ROWS=0
for csv in "${CSV_FILES[@]}"; do
    if [[ ! -f "$csv" ]]; then
        echo "[WARNING] CSV 不存在: $csv"
        continue
    fi
    ROWS=$(( $(wc -l < "$csv") - 1 ))   # 减去表头
    echo "  $(basename "$csv")  →  ${ROWS} 行目录记录"
    TOTAL_ROWS=$(( TOTAL_ROWS + ROWS ))
done
echo "  CSV 合计: ${TOTAL_ROWS} 行目录记录"
echo ""

# 用 Python 精确统计实际 PNG 图像数量
echo "[统计] 正在扫描 PNG 文件总数（可能需要数分钟）..."
python3 - <<PYEOF
import csv
from pathlib import Path

csv_files = [
$(printf '    "%s",\n' "${CSV_FILES[@]}")
]
path_col = "${CSV_PATH_COL}"

total = 0
for csv_path in csv_files:
    p = Path(csv_path)
    if not p.exists():
        print(f"  [skip] {p.name} 不存在")
        continue
    root = p.parent
    count = 0
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get(path_col, "").strip()
            if not rel:
                continue
            d = (root / rel).resolve()
            if d.exists():
                count += len(list(d.rglob("*.png")))
    print(f"  {p.name}: {count} 张 PNG")
    total += count
print(f"\n  ✓ 眼底训练集总计: {total} 张图像")
print(f"  预估 200 epochs 有效样本: {total * 200:,} 张次")
PYEOF
echo ""

# ── dry-run 模式：只统计不训练 ───────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[dry-run] 数据统计完成，未启动训练。"
    exit 0
fi

# ── 构建 CSV 参数字符串 ───────────────────────────────────────────────────
CSV_ARGS=""
for csv in "${CSV_FILES[@]}"; do
    CSV_ARGS="$CSV_ARGS $csv"
done

# ── 启动训练（nohup 后台）────────────────────────────────────────────────
echo "================================================================"
echo " 启动训练"
echo "  输出目录:    $OUTPUT_DIR"
echo "  日志文件:    $LOG_FILE"
echo "  GPU 数量:    $GPUS"
echo "  batch/GPU:   $BATCH_SIZE"
echo "  有效 batch:  $(( BATCH_SIZE * GPUS ))"
echo "  epochs:      $EPOCHS"
echo "  mask_ratio:  cfp=${MASK_RATIO_CFP}"
echo "================================================================"
echo ""
echo "  监控命令: tail -f $LOG_FILE"
echo ""

nohup torchrun \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    pretrain_mae.py \
        --csv_files ${CSV_ARGS} \
        --csv_path_col "${CSV_PATH_COL}" \
        --csv_modalities $(printf "${MODALITY} %.0s" $(seq 1 ${#CSV_FILES[@]})) \
        --mask_ratio_cfp ${MASK_RATIO_CFP} \
        --mask_ratio_oct ${MASK_RATIO_OCT} \
        --checkpoint "${CHECKPOINT_INIT}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --save_every ${SAVE_EVERY} \
        --max_checkpoints ${MAX_CHECKPOINTS} \
        --amp \
        --num_workers 8 \
        --balance_modalities \
> "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "训练进程 PID: ${TRAIN_PID}"
echo "${TRAIN_PID}" > run_logs/train_fundus_latest.pid
echo ""
echo "  ✓ 训练已在后台启动"
echo "  监控日志: tail -f ${LOG_FILE}"
echo "  停止训练: kill \$(cat run_logs/train_fundus_latest.pid)"
