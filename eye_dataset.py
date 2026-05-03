"""
Ophthalmic image dataset for MAE pretraining.

两种使用方式：

1. CsvOphthalmicDataset（推荐）：通过 CSV 元数据文件定位图像目录
   每个 CSV 文件的每行通过指定列给出一个相对路径目录，
   该目录下所有 PNG 文件均纳入训练集。
   __getitem__ 返回 (image_tensor, mask_ratio)

2. OphthalmicDataset（兼容旧接口）：递归扫描目录
   __getitem__ 同样返回 (image_tensor, mask_ratio)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

MASK_RATIO_CFP = 0.75   # RETFound: fundus (Color Fundus Photography)
MASK_RATIO_OCT = 0.85   # RETFound: OCT B-scan

_FUND_KEYWORDS = {"fund", "cfp", "colour", "color", "retina"}
_OCT_KEYWORDS  = {"oct", "bscan", "b_scan", "volume", "vol"}


def _infer_modality(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if parts & _OCT_KEYWORDS:
        return "oct"
    if parts & _FUND_KEYWORDS:
        return "cfp"
    return "unknown"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_pretrain_transform(img_size: int, grayscale: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406] if not grayscale else [0.449],
        std =[0.229, 0.224, 0.225] if not grayscale else [0.226],
    )
    aug = [
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if not grayscale:
        aug.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))
    aug += [transforms.ToTensor(), normalize]
    return transforms.Compose(aug)


def build_eval_transform(img_size: int, grayscale: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406] if not grayscale else [0.449],
        std =[0.229, 0.224, 0.225] if not grayscale else [0.226],
    )
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143), interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# CSV-driven dataset
# ---------------------------------------------------------------------------

class CsvOphthalmicDataset(Dataset):
    """CSV 元数据驱动的眼科图像数据集。

    CSV 格式示例（有表头，逗号分隔）：
        relative_dir,patient_id
        patients/00123/fundus,123
        patients/00456/oct,456

    path_col 列的值是相对于该 CSV 文件所在目录的路径，
    指向存放 PNG 图像的目录（递归扫描子目录）。

    Args:
        csv_files:          CSV 文件路径列表。
        path_col:           CSV 中指向图像目录的列名。
        csv_modalities:     与 csv_files 一一对应的模态列表（"cfp"/"oct"/"unknown"）。
                            为 None 时对每行自动从路径推断。
        modality_col:       CSV 中记录每行模态的列名（优先级高于 csv_modalities）。
        mask_ratio_cfp:     CFP/Fundus 掩码率，默认 0.75。
        mask_ratio_oct:     OCT 掩码率，默认 0.85。
        default_mask_ratio: 无法识别模态时的后备掩码率。
        img_size:           图像目标边长。
        grayscale:          是否转为单通道。
        transform:          自定义 transform。
        is_train:           True 用训练增强，False 用中心裁剪。
    """

    def __init__(
        self,
        csv_files: Sequence[str | Path],
        path_col: str,
        csv_modalities: Optional[Sequence[str]] = None,
        modality_col: Optional[str] = None,
        mask_ratio_cfp: float = MASK_RATIO_CFP,
        mask_ratio_oct: float = MASK_RATIO_OCT,
        default_mask_ratio: float = MASK_RATIO_CFP,
        img_size: int = 224,
        grayscale: bool = False,
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.grayscale = grayscale
        self._mask_map: Dict[str, float] = {
            "cfp":     mask_ratio_cfp,
            "fundus":  mask_ratio_cfp,
            "oct":     mask_ratio_oct,
            "unknown": default_mask_ratio,
        }
        self.default_mask_ratio = default_mask_ratio

        if csv_modalities is not None and len(csv_modalities) != len(csv_files):
            raise ValueError(
                f"csv_modalities 长度 ({len(csv_modalities)}) 必须与 "
                f"csv_files 长度 ({len(csv_files)}) 一致"
            )

        records: List[Dict] = []
        for i, csv_path in enumerate(csv_files):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                print(f"[Dataset] Warning: CSV '{csv_path}' 不存在，跳过。")
                continue

            file_modality = csv_modalities[i] if csv_modalities else None
            new_recs = self._parse_csv(csv_path, path_col, modality_col, file_modality)
            records.extend(new_recs)
            print(
                f"[Dataset] {csv_path.name}: {len(new_recs)} 张图像  "
                f"(modality={file_modality or modality_col or 'auto-infer'})"
            )

        if not records:
            raise RuntimeError(
                f"未从 CSV 文件中找到任何图像: {list(csv_files)}\n"
                "请检查 --csv_files 和 --csv_path_col 是否正确。"
            )

        self.records = records

        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = build_pretrain_transform(img_size, grayscale)
        else:
            self.transform = build_eval_transform(img_size, grayscale)

        self._print_summary()

    def _parse_csv(
        self,
        csv_path: Path,
        path_col: str,
        modality_col: Optional[str],
        file_modality: Optional[str],
    ) -> List[Dict]:
        csv_root = csv_path.parent
        records: List[Dict] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if path_col not in cols:
                raise ValueError(
                    f"列 '{path_col}' 在 '{csv_path}' 中不存在。"
                    f"可用列: {cols}"
                )

            for row in reader:
                rel_dir = row[path_col].strip()
                if not rel_dir:
                    continue

                img_dir = (csv_root / rel_dir).resolve()
                if not img_dir.exists():
                    continue  # 数据还在上传中，静默跳过

                # 模态优先级: 行内列 > 文件级指定 > 路径推断
                if modality_col and modality_col in row and row[modality_col].strip():
                    modality = row[modality_col].strip().lower()
                elif file_modality:
                    modality = file_modality.lower()
                else:
                    modality = _infer_modality(img_dir)

                mask_ratio = self._mask_map.get(modality, self.default_mask_ratio)

                for img_path in sorted(img_dir.rglob("*.png")):
                    if img_path.is_file():
                        records.append({
                            "path":       img_path,
                            "modality":   modality,
                            "mask_ratio": mask_ratio,
                        })

        return records

    def _print_summary(self) -> None:
        counts = Counter(r["modality"] for r in self.records)
        print(f"[Dataset] 合计: {len(self.records)} 张图像")
        for mod, cnt in sorted(counts.items()):
            mr = self._mask_map.get(mod, self.default_mask_ratio)
            print(f"  {mod}: {cnt} 张  mask_ratio={mr}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """返回 (image_tensor, mask_ratio)。"""
        record = self.records[idx]
        try:
            img = Image.open(record["path"])
            img = img.convert("L" if self.grayscale else "RGB")
        except Exception as e:
            print(f"[Dataset] Warning: 加载失败 {record['path']}: {e}")
            c = 1 if self.grayscale else 3
            return torch.zeros(c, self.img_size, self.img_size), record["mask_ratio"]

        if self.transform:
            img = self.transform(img)
        return img, record["mask_ratio"]


# ---------------------------------------------------------------------------
# Legacy directory-scan dataset（保持向后兼容）
# ---------------------------------------------------------------------------

class OphthalmicDataset(Dataset):
    """递归扫描目录的旧式数据集，__getitem__ 同样返回 (image, mask_ratio)。"""

    def __init__(
        self,
        data_dirs: Sequence[str | Path],
        img_size: int = 224,
        grayscale: bool = False,
        modality: str = "all",
        mask_ratio_cfp: float = MASK_RATIO_CFP,
        mask_ratio_oct: float = MASK_RATIO_OCT,
        default_mask_ratio: float = MASK_RATIO_CFP,
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.grayscale = grayscale
        self._mask_map = {
            "cfp": mask_ratio_cfp, "fundus": mask_ratio_cfp,
            "oct": mask_ratio_oct, "unknown": default_mask_ratio,
        }
        self.default_mask_ratio = default_mask_ratio

        records: List[Dict] = []
        for root in data_dirs:
            root = Path(root)
            if not root.exists():
                print(f"[Dataset] Warning: '{root}' 不存在，跳过。")
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    mod = _infer_modality(p)
                    records.append({
                        "path":       p,
                        "modality":   mod,
                        "mask_ratio": self._mask_map.get(mod, default_mask_ratio),
                    })

        if modality != "all":
            records = [r for r in records if r["modality"] == modality]

        if not records:
            raise RuntimeError(f"在 {list(data_dirs)} 中未找到图像（modality={modality}）。")

        self.records = records

        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = build_pretrain_transform(img_size, grayscale)
        else:
            self.transform = build_eval_transform(img_size, grayscale)

        counts = Counter(r["modality"] for r in records)
        print(f"[Dataset] 找到 {len(records)} 张图像（modality={modality}）")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v} 张  mask_ratio={self._mask_map.get(k, default_mask_ratio)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        record = self.records[idx]
        try:
            img = Image.open(record["path"])
            img = img.convert("L" if self.grayscale else "RGB")
        except Exception as e:
            print(f"[Dataset] Warning: {record['path']}: {e}")
            c = 1 if self.grayscale else 3
            return torch.zeros(c, self.img_size, self.img_size), record["mask_ratio"]
        if self.transform:
            img = self.transform(img)
        return img, record["mask_ratio"]


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(
    dataset: CsvOphthalmicDataset | OphthalmicDataset,
) -> torch.utils.data.WeightedRandomSampler:
    """各模态等权重采样，避免数量少的模态被淹没。"""
    modalities = [r["modality"] for r in dataset.records]
    counts = Counter(modalities)
    total = len(modalities)
    w = {m: total / (len(counts) * c) for m, c in counts.items()}
    weights = [w.get(m, 1.0) for m in modalities]
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
