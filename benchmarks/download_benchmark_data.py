#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


RETFOUND_SPLITS = {
    "APTOS2019": "162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj",
    "MESSIDOR2": "1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda",
    "IDRID": "1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3",
    "PAPILA": "1JltYs7WRWEU0yyki1CQw5-10HEbqCMBE",
    "Glaucoma_fundus": "18vSazOYDsUGdZ64gGkTg3E6jiNtcrUrI",
    "JSIEC": "1q0GFQb-dYwzIx8AwlaFZenUJItix4s8z",
    "Retina": "1vdmjMRDoUm9yk83HMArLiPcLDk_dm92Q",
    "OCTID": "1I7nAvbkJG4UF29J3HcyIW53rVEFcKRgm",
}

OFFICIAL_URLS = {
    "OCTDL": "https://data.mendeley.com/datasets/sncdhf53xc/1",
    "APTOS2019": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/data",
    "IDRID": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
    "PAPILA": "https://figshare.com/articles/dataset/PAPILA/14798004/1",
    "Glaucoma_fundus": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC",
    "OCTID": "https://borealisdata.ca/dataverse/OCTID",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download RETFound benchmark split packages")
    p.add_argument("--datasets", nargs="+", default=["APTOS2019", "IDRID", "PAPILA", "Glaucoma_fundus", "OCTID"])
    p.add_argument("--output-root", default="benchmark_data")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown first: python -m pip install gdown") from exc

    for name in args.datasets:
        try:
            if name not in RETFOUND_SPLITS:
                print(f"[Manual] No RETFound split package for {name}. Official URL: {OFFICIAL_URLS.get(name, 'unknown')}")
                continue
            out = root / f"{name}.zip"
            if out.exists():
                print(f"[Skip] Exists: {out}")
            else:
                print(f"[Download] {name}")
                gdown.download(id=RETFOUND_SPLITS[name], output=str(out), quiet=False)
            if args.extract and out.exists():
                target = root / name
                target.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(out) as zf:
                        zf.extractall(target)
                    print(f"[OK] Extracted {out} -> {target}")
                except zipfile.BadZipFile:
                    print(f"[Warn] {out} is not a zip file; inspect manually.")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[Error] {name}: {exc}")


if __name__ == "__main__":
    main()
