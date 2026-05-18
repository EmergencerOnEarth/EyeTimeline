#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.data import (
    CsvClassificationDataset,
    build_class_mapping,
    build_transform,
    read_records,
    save_class_mapping,
)
from benchmarks.metrics import compute_metrics, write_metrics
from benchmarks.models import build_model, set_encoder_trainable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EyeTimeline benchmark classifier training")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--path-col", default="path")
    p.add_argument("--label-col", default="label")
    p.add_argument("--split-col", default="split")
    p.add_argument("--backbone", required=True,
                   choices=["eyetimeline_mae", "retfound_mae", "imagenet_mae", "random_mae", "eyeclip"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--clip-model-type", default="ViT-B/32")
    p.add_argument("--mae-variant", choices=["large", "base"], default="large",
                   help="Use base only for low-memory smoke tests; real benchmarks use large.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--normalization", choices=["imagenet", "clip"], default=None)
    p.add_argument("--pool", choices=["cls", "avg"], default="cls")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--save-head-only",
        action="store_true",
        help="Save only the classifier head. Use this for frozen-encoder linear probes to avoid multi-GB checkpoints.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    paths_all: list[str] = []

    for images, labels, paths in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        paths_all.extend(paths)

    return (
        float(np.mean(losses)),
        np.concatenate(labels_all),
        np.concatenate(logits_all),
        paths_all,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "class_to_idx": class_to_idx,
        "epoch": epoch,
        "metrics": metrics,
    }
    if args.save_head_only:
        if not hasattr(model, "head"):
            raise TypeError(f"Cannot save head-only checkpoint for model type: {type(model)}")
        payload["checkpoint_mode"] = "head_only"
        payload["head"] = model.head.state_dict()
    else:
        payload["checkpoint_mode"] = "full"
        payload["model"] = model.state_dict()
    torch.save(payload, path)


def write_predictions(
    path: Path,
    image_paths: list[str],
    y_true: np.ndarray,
    logits: np.ndarray,
    idx_to_class: dict[int, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "label", "pred"] + [f"prob_{idx_to_class[i]}" for i in range(len(idx_to_class))]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img, true_idx, pred_idx, prob in zip(image_paths, y_true, probs.argmax(axis=1), probs):
            row = {"path": img, "label": idx_to_class[int(true_idx)], "pred": idx_to_class[int(pred_idx)]}
            row.update({f"prob_{idx_to_class[i]}": float(prob[i]) for i in range(len(idx_to_class))})
            writer.writerow(row)


def with_extra_metrics(metrics: dict, **extra) -> dict:
    merged = dict(metrics)
    merged.update(extra)
    return merged


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    norm = args.normalization or ("clip" if args.backbone == "eyeclip" else "imagenet")
    train_records = read_records(args.labels_csv, "train", args.data_root, args.path_col, args.label_col, args.split_col)
    val_records = read_records(args.labels_csv, "val", args.data_root, args.path_col, args.label_col, args.split_col)
    class_to_idx = build_class_mapping(train_records)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    save_class_mapping(class_to_idx, out_dir)

    train_ds = CsvClassificationDataset(
        train_records, class_to_idx, build_transform(args.img_size, train=True, normalization=norm)
    )
    val_ds = CsvClassificationDataset(
        val_records, class_to_idx, build_transform(args.img_size, train=False, normalization=norm)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(
        args.backbone,
        num_classes=len(class_to_idx),
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        mae_variant=args.mae_variant,
        dropout=args.dropout,
        pool=args.pool,
        clip_model_type=args.clip_model_type,
    ).to(device)
    set_encoder_trainable(model, trainable=not args.freeze_encoder)

    lr = args.lr
    if lr is None:
        lr = 1e-3 if args.freeze_encoder else 1e-4
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_y, val_logits, val_paths = run_epoch(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_y, val_logits)
        score = val_metrics.get("auroc") or val_metrics.get("macro_auroc") or val_metrics["accuracy"]
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} score={score}"
        )
        write_metrics(with_extra_metrics(val_metrics, val_loss=val_loss, epoch=epoch), out_dir / "metrics_latest.json")
        save_checkpoint(out_dir / "checkpoint_latest.pth", model, args, class_to_idx, epoch, val_metrics)
        if score is not None and float(score) > best_score:
            best_score = float(score)
            save_checkpoint(out_dir / "checkpoint_best.pth", model, args, class_to_idx, epoch, val_metrics)
            write_predictions(out_dir / "val_predictions_best.csv", val_paths, val_y, val_logits, idx_to_class)
            write_metrics(with_extra_metrics(val_metrics, val_loss=val_loss, epoch=epoch), out_dir / "metrics_best.json")


if __name__ == "__main__":
    main()
