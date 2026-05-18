from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass(frozen=True)
class ClassificationRecord:
    path: Path
    label: str
    split: str


def build_transform(
    img_size: int,
    train: bool,
    normalization: str = "imagenet",
) -> transforms.Compose:
    if normalization == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    elif normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143), interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _resolve_path(raw: str, csv_path: Path, data_root: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    root = data_root if data_root is not None else csv_path.parent
    return (root / p).resolve()


def read_records(
    labels_csv: str | Path,
    split: str | None = None,
    data_root: str | Path | None = None,
    path_col: str = "path",
    label_col: str = "label",
    split_col: str = "split",
) -> list[ClassificationRecord]:
    csv_path = Path(labels_csv)
    root = Path(data_root).resolve() if data_root else None
    records: list[ClassificationRecord] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in (path_col, label_col, split_col) if c not in cols]
        if missing:
            raise ValueError(f"{csv_path} missing columns {missing}; available columns: {cols}")
        for row in reader:
            row_split = row[split_col].strip()
            if split is not None and row_split != split:
                continue
            records.append(ClassificationRecord(
                path=_resolve_path(row[path_col].strip(), csv_path, root),
                label=row[label_col].strip(),
                split=row_split,
            ))

    if not records:
        where = f" split={split}" if split else ""
        raise RuntimeError(f"No records found in {csv_path}{where}")
    return records


def build_class_mapping(records: Iterable[ClassificationRecord]) -> dict[str, int]:
    labels = sorted({r.label for r in records}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    return {label: i for i, label in enumerate(labels)}


def save_class_mapping(class_to_idx: dict[str, int], output_dir: str | Path) -> None:
    path = Path(output_dir) / "class_to_idx.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")


def load_class_mapping(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CsvClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[ClassificationRecord],
        class_to_idx: dict[str, int],
        transform: Callable,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[index]
        try:
            image = Image.open(rec.path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {rec.path}") from exc
        x = self.transform(image)
        y = torch.tensor(self.class_to_idx[rec.label], dtype=torch.long)
        return x, y, str(rec.path)
