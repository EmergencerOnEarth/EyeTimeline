#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


ASSETS = {
    "retfound_cfp": {
        "type": "hf_snapshot",
        "repo_id": "YukunZhou/RETFound_mae_natureCFP",
        "subdir": "baseline_weights/retfound/RETFound_mae_natureCFP",
    },
    "retfound_oct": {
        "type": "hf_snapshot",
        "repo_id": "YukunZhou/RETFound_mae_natureOCT",
        "subdir": "baseline_weights/retfound/RETFound_mae_natureOCT",
    },
    "imagenet_mae_large": {
        "type": "hf_file",
        "repo_id": "facebook/vit-mae-large",
        "filename": "pytorch_model.bin",
        "subdir": "baseline_weights/imagenet_mae",
        "output_name": "mae_vit_large_imagenet.bin",
    },
    "eyeclip_visual": {
        "type": "gdrive",
        "file_id": "1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr",
        "subdir": "baseline_weights/eyeclip",
        "output_name": "eyeclip_visual.pt",
        "manual_url": "https://drive.google.com/file/d/1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr/view?usp=sharing",
    },
    "retfound_repo": {
        "type": "github_zip",
        "repo": "rmaphoh/RETFound",
        "subdir": "external_repos/RETFound",
    },
    "eyeclip_repo": {
        "type": "github_zip",
        "repo": "Michi-3000/EyeCLIP",
        "subdir": "external_repos/EyeCLIP",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download baseline assets")
    p.add_argument("--assets", nargs="+", default=list(ASSETS))
    p.add_argument("--output-root", default=".")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--continue-on-error", action="store_true",
                   help="Try remaining assets when one download fails.")
    return p.parse_args()


def ensure_empty_or_existing(path: Path, skip_existing: bool) -> bool:
    if path.exists() and skip_existing:
        if path.is_dir() and not any(path.iterdir()):
            print(f"[Retry] Empty directory exists from a previous failed download: {path}")
            return True
        print(f"[Skip] Exists: {path}")
        return False
    path.mkdir(parents=True, exist_ok=True)
    return True


def download_hf_snapshot(asset: dict, root: Path, token: str | None, endpoint: str | None, skip_existing: bool) -> None:
    from huggingface_hub import snapshot_download

    target = root / asset["subdir"]
    if not ensure_empty_or_existing(target, skip_existing):
        return
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    snapshot_download(
        repo_id=asset["repo_id"],
        local_dir=target,
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[OK] {asset['repo_id']} -> {target}")


def download_hf_file(asset: dict, root: Path, token: str | None, endpoint: str | None, skip_existing: bool) -> None:
    from huggingface_hub import hf_hub_download

    target_dir = root / asset["subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / asset["output_name"]
    if out.exists() and skip_existing:
        print(f"[Skip] Exists: {out}")
        return
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    src = hf_hub_download(asset["repo_id"], asset["filename"], token=token)
    shutil.copy2(src, out)
    print(f"[OK] {asset['repo_id']}/{asset['filename']} -> {out}")


def download_gdrive(asset: dict, root: Path, skip_existing: bool) -> None:
    target_dir = root / asset["subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / asset["output_name"]
    if out.exists() and skip_existing:
        print(f"[Skip] Exists: {out}")
        return
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: python -m pip install gdown") from exc
    ok = gdown.download(id=asset["file_id"], output=str(out), quiet=False)
    if ok is None:
        print(f"[Manual] Download failed. URL: {asset['manual_url']}")
    else:
        print(f"[OK] Google Drive {asset['file_id']} -> {out}")


def download_github_zip(asset: dict, root: Path, skip_existing: bool) -> None:
    target = root / asset["subdir"]
    if target.exists() and skip_existing:
        print(f"[Skip] Exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    repo = asset["repo"]
    tmp_zip = target.parent / f"{repo.replace('/', '_')}.zip"
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/main"
    print(f"[Download] {url}")
    urlretrieve(url, tmp_zip)
    extract_root = target.parent / f"{repo.split('/')[-1]}-main"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(target.parent)
    if target.exists():
        shutil.rmtree(target)
    extract_root.rename(target)
    tmp_zip.unlink(missing_ok=True)
    print(f"[OK] {repo} -> {target}")


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    for name in args.assets:
        try:
            if name not in ASSETS:
                raise KeyError(f"Unknown asset '{name}'. Available: {sorted(ASSETS)}")
            asset = ASSETS[name]
            typ = asset["type"]
            if typ == "hf_snapshot":
                download_hf_snapshot(asset, root, args.hf_token, args.hf_endpoint, args.skip_existing)
            elif typ == "hf_file":
                download_hf_file(asset, root, args.hf_token, args.hf_endpoint, args.skip_existing)
            elif typ == "gdrive":
                download_gdrive(asset, root, args.skip_existing)
            elif typ == "github_zip":
                download_github_zip(asset, root, args.skip_existing)
            else:
                raise ValueError(f"Unsupported asset type: {typ}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[Error] {name}: {exc}")


if __name__ == "__main__":
    main()
