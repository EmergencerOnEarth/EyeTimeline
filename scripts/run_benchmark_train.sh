#!/usr/bin/env bash
set -euo pipefail

: "${LABELS_CSV:?Set LABELS_CSV=/path/to/labels.csv}"
: "${DATA_ROOT:?Set DATA_ROOT=/path/to/dataset/root}"
: "${BACKBONE:?Set BACKBONE=eyetimeline_mae|retfound_mae|imagenet_mae|random_mae|eyeclip}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR=/path/to/output}"

python benchmarks/train_classifier.py \
  --labels-csv "${LABELS_CSV}" \
  --data-root "${DATA_ROOT}" \
  --backbone "${BACKBONE}" \
  ${CHECKPOINT:+--checkpoint "${CHECKPOINT}"} \
  ${FREEZE_ENCODER:+--freeze-encoder} \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --output-dir "${OUTPUT_DIR}"
