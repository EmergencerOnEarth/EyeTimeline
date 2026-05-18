from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float | None]:
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }

    if n_classes == 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            metrics["auroc"] = None
        try:
            metrics["aupr"] = float(average_precision_score(y_true, probs[:, 1]))
        except ValueError:
            metrics["aupr"] = None
    else:
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        try:
            metrics["macro_auroc"] = float(roc_auc_score(
                y_bin, probs, average="macro", multi_class="ovr"
            ))
        except ValueError:
            metrics["macro_auroc"] = None
        try:
            metrics["macro_aupr"] = float(average_precision_score(
                y_bin, probs, average="macro"
            ))
        except ValueError:
            metrics["macro_aupr"] = None

    return metrics


def write_metrics(metrics: dict[str, float | None], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
