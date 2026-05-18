#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/EyeTimelineAssets}"

mkdir -p "${ASSET_ROOT}"
python benchmarks/download_assets.py \
  --output-root "${ASSET_ROOT}" \
  --continue-on-error \
  --assets \
    retfound_repo \
    eyeclip_repo \
    imagenet_mae_large \
    retfound_cfp \
    retfound_oct \
    eyeclip_visual

echo "[OK] Assets under ${ASSET_ROOT}"
