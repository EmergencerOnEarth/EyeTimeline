#!/usr/bin/env python3
"""
Download pretrained ViT-Large weights for ophthalmic MAE pretraining.

Available models:
  retfound_cfp   - RETFound fundus (CFP) weights, ViT-Large MAE
  retfound_oct   - RETFound OCT weights, ViT-Large MAE
  mae_vit_large  - Meta MAE ViT-Large pretrained on ImageNet-1k

Usage:
  python download_weights.py --model retfound_oct --output_dir weights/
  python download_weights.py --model retfound_cfp --output_dir weights/
  python download_weights.py --model mae_vit_large --output_dir weights/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "mae_vit_large": {
        "hf_repo": "facebook/vit-mae-large",
        "hf_file": "pytorch_model.bin",
        "out_name": "mae_vit_large_imagenet.bin",
        "description": "Meta MAE ViT-Large pretrained on ImageNet-1k (HuggingFace ViTMAE format)",
    },
    # Gated repos — require HF account approval; kept here for reference
    "retfound_oct": {
        "hf_repo": "YukunZhou/RETFound_mae_natureOCT",
        "hf_file": "RETFound_mae_natureOCT.pth",
        "out_name": "RETFound_oct.pth",
        "description": "RETFound ViT-Large pretrained on OCT images (Nature Medicine 2023) [gated]",
    },
    "retfound_cfp": {
        "hf_repo": "YukunZhou/RETFound_mae_natureCFP",
        "hf_file": "RETFound_mae_natureCFP.pth",
        "out_name": "RETFound_cfp.pth",
        "description": "RETFound ViT-Large pretrained on fundus images (Nature Medicine 2023) [gated]",
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _hf_url(repo: str, filename: str, mirror: bool = True) -> str:
    base = "https://hf-mirror.com" if mirror else "https://huggingface.co"
    return f"{base}/{repo}/resolve/main/{filename}"


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  → {dest}")

    def _reporthook(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(100, 100 * downloaded / total)
            mb = downloaded / 1024 ** 2
            total_mb = total / 1024 ** 2
            print(f"\r  {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
        else:
            print(f"\r  {downloaded / 1024**2:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
    print()  # newline after progress


def download_model(model_key: str, output_dir: str, mirror: bool = True) -> Path:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}")

    info = MODELS[model_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = info["hf_file"]
    dest = output_dir / info.get("out_name", filename)

    if dest.exists():
        print(f"[Download] '{dest}' already exists, skipping.")
        return dest

    print(f"\n[Download] {info['description']}")

    url = _hf_url(info["hf_repo"], filename, mirror=mirror)
    try:
        _download_with_progress(url, dest)
    except Exception as e:
        print(f"\n[Download] Mirror failed ({e}), trying official HuggingFace...")
        url = _hf_url(info["hf_repo"], filename, mirror=False)
        _download_with_progress(url, dest)

    print(f"[Download] Done → {dest}  ({dest.stat().st_size / 1024**2:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# Try huggingface_hub as a fallback
# ---------------------------------------------------------------------------

def download_via_hf_hub(model_key: str, output_dir: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    info = MODELS[model_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    path = hf_hub_download(
        repo_id=info["hf_repo"],
        filename=info["hf_file"],
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[Download] Saved via hf_hub → {path}")
    return Path(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download pretrained weights")
    parser.add_argument(
        "--model",
        default="retfound_oct",
        choices=list(MODELS.keys()),
        help="Which pretrained model to download.",
    )
    parser.add_argument(
        "--output_dir", default="weights",
        help="Directory to save the downloaded weights.",
    )
    parser.add_argument(
        "--no_mirror", action="store_true",
        help="Use official HuggingFace instead of hf-mirror.com.",
    )
    parser.add_argument(
        "--use_hf_hub", action="store_true",
        help="Use huggingface_hub library instead of urllib.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available models.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    keys = list(MODELS.keys()) if args.all else [args.model]

    for key in keys:
        try:
            if args.use_hf_hub:
                dest = download_via_hf_hub(key, args.output_dir)
            else:
                dest = download_model(key, args.output_dir, mirror=not args.no_mirror)
            print(f"[OK] {key} → {dest}\n")
        except Exception as e:
            print(f"[ERROR] Failed to download '{key}': {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
