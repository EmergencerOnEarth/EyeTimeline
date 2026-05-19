#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "auroc",
    "macro_auroc",
    "aupr",
    "macro_aupr",
    "quadratic_weighted_kappa",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Summarize EyeTimeline benchmark evaluation results")
    p.add_argument("--output-root", required=True)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--summary-md", default=None)
    return p.parse_args()


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Warn] Failed to parse {path}: {exc}", file=sys.stderr)
        return {}


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def collect_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("*/*/*/*/eval_test/metrics.json")):
        rel = metrics_path.relative_to(output_root)
        mode, modality, dataset, model = rel.parts[:4]
        metrics = _safe_load_json(metrics_path)
        train_dir = metrics_path.parents[1]
        best_metrics = _safe_load_json(train_dir / "metrics_best.json")
        args_json = _safe_load_json(train_dir / "args.json")
        row: dict[str, Any] = {
            "mode": mode,
            "modality": modality,
            "dataset": dataset,
            "model": model,
            "epochs": args_json.get("epochs"),
            "batch_size": args_json.get("batch_size"),
            "freeze_encoder": args_json.get("freeze_encoder"),
            "eval_dir": str(metrics_path.parent),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        row["best_val_score"] = (
            best_metrics.get("auroc")
            or best_metrics.get("macro_auroc")
            or best_metrics.get("accuracy")
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "modality",
        "dataset",
        "model",
        "epochs",
        "batch_size",
        "freeze_encoder",
        *METRIC_KEYS,
        "best_val_score",
        "eval_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "modality",
        "dataset",
        "model",
        "accuracy",
        "macro_f1",
        "auroc",
        "macro_auroc",
        "aupr",
        "macro_aupr",
        "quadratic_weighted_kappa",
    ]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(f)) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    rows = collect_rows(output_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"
    summary_md = Path(args.summary_md) if args.summary_md else output_root / "summary.md"
    write_csv(summary_csv, rows)
    write_markdown(summary_md, rows)
    print(f"[Summary] Found {len(rows)} evaluated runs")
    print(f"[Summary] CSV: {summary_csv}")
    print(f"[Summary] Markdown: {summary_md}")


if __name__ == "__main__":
    main()
