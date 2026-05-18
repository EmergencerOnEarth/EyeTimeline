#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-eyetimeline-bench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_WHEEL="${CUDA_WHEEL:-cu118}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[Error] conda not found. Install/activate conda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

case "${CUDA_WHEEL}" in
  cu118)
    python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
      --index-url https://download.pytorch.org/whl/cu118
    ;;
  cu121)
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121
    ;;
  cpu)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "[Error] Unsupported CUDA_WHEEL=${CUDA_WHEEL}; use cu118, cu121, or cpu." >&2
    exit 1
    ;;
esac

python -m pip install -r requirements.txt
python -m pip install -r benchmarks/requirements-benchmark.txt

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY
