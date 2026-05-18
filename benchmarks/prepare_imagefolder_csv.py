#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.data import IMAGE_EXTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Create labels.csv from train/val/test image folders")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rows = []
    for split in args.splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[Skip] Missing split directory: {split_dir}")
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                    rows.append({
                        "path": str(image_path.relative_to(root)),
                        "label": label,
                        "split": split,
                    })

    if not rows:
        raise RuntimeError(f"No images found under {root}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
