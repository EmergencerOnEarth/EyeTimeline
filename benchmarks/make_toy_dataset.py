#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Create a tiny synthetic image classification dataset for smoke tests")
    p.add_argument("--output-root", required=True)
    p.add_argument("--classes", nargs="+", default=["normal", "disease"])
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--n-test", type=int, default=2)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_image(label_idx: int, img_size: int, rng: np.random.Generator) -> Image.Image:
    arr = rng.normal(40, 8, size=(img_size, img_size, 3)).clip(0, 255).astype(np.uint8)
    if label_idx == 0:
        arr[img_size // 4: img_size // 2, :, 1] = 180
    else:
        arr[:, img_size // 3: img_size // 2, 0] = 200
    return Image.fromarray(arr)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    rng = np.random.default_rng(args.seed)
    rows = []
    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, count in counts.items():
        for label_idx, label in enumerate(args.classes):
            for i in range(count):
                rel = Path(split) / label / f"{i:04d}.png"
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                make_image(label_idx, args.img_size, rng).save(path)
                rows.append({"path": str(rel), "label": label, "split": split})
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote toy dataset to {root}")


if __name__ == "__main__":
    main()
