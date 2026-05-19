#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

ASSET_ROOT="${ASSET_ROOT:-/data1/kechuang/EyeTimelineAssets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ASSET_ROOT/benchmark_outputs}"
DATASETS="${DATASETS:-aptos2019 idrid papila glaucoma_fundus octid}"
MODELS="${MODELS:-ours retfound eyeclip}"
MODES="${MODES:-linear}"
BENCHMARK_GPUS="${BENCHMARK_GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-}"

EPOCHS_LINEAR="${EPOCHS_LINEAR:-50}"
EPOCHS_FULL="${EPOCHS_FULL:-50}"
BATCH_SIZE_LINEAR="${BATCH_SIZE_LINEAR:-64}"
BATCH_SIZE_FULL="${BATCH_SIZE_FULL:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a MODEL_ARGS <<< "$MODELS"
read -r -a MODE_ARGS <<< "$MODES"

CMD=(
  python benchmarks/run_matrix.py
  --asset-root "$ASSET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --datasets "${DATASET_ARGS[@]}"
  --models "${MODEL_ARGS[@]}"
  --modes "${MODE_ARGS[@]}"
  --epochs-linear "$EPOCHS_LINEAR"
  --epochs-full "$EPOCHS_FULL"
  --batch-size-linear "$BATCH_SIZE_LINEAR"
  --batch-size-full "$BATCH_SIZE_FULL"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --gpus "$BENCHMARK_GPUS"
)

if [[ -n "$MAX_PARALLEL" ]]; then
  CMD+=(--max-parallel "$MAX_PARALLEL")
fi
if [[ "$FORCE" == "1" ]]; then
  CMD+=(--force)
fi
if [[ "$SKIP_MISSING" == "1" ]]; then
  CMD+=(--skip-missing)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

mkdir -p "$OUTPUT_ROOT/run_logs"
RUN_LOG="$OUTPUT_ROOT/run_logs/benchmark_matrix_$(date +%Y%m%d_%H%M%S).log"

printf '[Benchmark] project=%s\n' "$PROJECT_DIR"
printf '[Benchmark] asset_root=%s\n' "$ASSET_ROOT"
printf '[Benchmark] output_root=%s\n' "$OUTPUT_ROOT"
printf '[Benchmark] command='
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DETACH:-0}" == "1" ]]; then
  nohup "${CMD[@]}" > "$RUN_LOG" 2>&1 &
  PID=$!
  echo "$PID" > "$OUTPUT_ROOT/run_logs/latest.pid"
  echo "$RUN_LOG" > "$OUTPUT_ROOT/run_logs/latest.log"
  printf '[Benchmark] started in background pid=%s\n' "$PID"
  printf '[Benchmark] log=%s\n' "$RUN_LOG"
  printf '[Benchmark] status: bash scripts/benchmark_status.sh\n'
else
  "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
fi
