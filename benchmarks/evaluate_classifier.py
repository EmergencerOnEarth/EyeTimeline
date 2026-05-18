#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.data import CsvClassificationDataset, build_transform, read_records
from benchmarks.metrics import compute_metrics, write_metrics
from benchmarks.models import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate EyeTimeline benchmark classifier")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits_all, labels_all, paths_all = [], [], []
    for images, labels, paths in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
        paths_all.extend(paths)
    return np.concatenate(labels_all), np.concatenate(logits_all), paths_all


def write_predictions(path: Path, paths: list[str], y_true: np.ndarray, logits: np.ndarray, idx_to_class: dict[int, str]) -> None:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["path", "label", "pred"] + [f"prob_{idx_to_class[i]}" for i in range(len(idx_to_class))]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for img, true_idx, pred_idx, prob in zip(paths, y_true, probs.argmax(axis=1), probs):
            row = {"path": img, "label": idx_to_class[int(true_idx)], "pred": idx_to_class[int(pred_idx)]}
            row.update({f"prob_{idx_to_class[i]}": float(prob[i]) for i in range(len(idx_to_class))})
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**payload["args"])
    class_to_idx = payload["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    norm = train_args.normalization or ("clip" if train_args.backbone == "eyeclip" else "imagenet")

    records = read_records(
        args.labels_csv,
        args.split,
        args.data_root,
        train_args.path_col,
        train_args.label_col,
        train_args.split_col,
    )
    dataset = CsvClassificationDataset(
        records, class_to_idx, build_transform(train_args.img_size, train=False, normalization=norm)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(
        train_args.backbone,
        num_classes=len(class_to_idx),
        checkpoint=None,
        img_size=train_args.img_size,
        mae_variant=getattr(train_args, "mae_variant", "large"),
        dropout=train_args.dropout,
        pool=train_args.pool,
        clip_model_type=train_args.clip_model_type,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)

    y_true, logits, paths = predict(model, loader, device)
    metrics = compute_metrics(y_true, logits)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics, out_dir / "metrics.json")
    write_predictions(out_dir / "predictions.csv", paths, y_true, logits, idx_to_class)
    (out_dir / "eval_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
