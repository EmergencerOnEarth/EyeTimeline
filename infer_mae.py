#!/usr/bin/env python3
"""
MAE 推理可视化脚本

对指定 checkpoint 随机抽取若干图像，输出三张图：
  - original.png  : 原图
  - masked.png    : 按 mask_ratio 遮挡后的图（被遮挡 patch 显示为灰色）
  - reconstructed.png : 模型重建结果

同时在输出目录写入 metadata.csv，记录原图路径和输出路径。

用法：
  # 从 CSV 数据源抽样
  python infer_mae.py \
      --checkpoint checkpoints/fundus_v1/checkpoint_best.pth \
      --csv_files /data1/kechuang/.../meta.csv \
      --csv_path_col output_subdir \
      --csv_modalities cfp \
      --output_dir infer_output/fundus \
      --n_samples 10

  # 从目录抽样
  python infer_mae.py \
      --checkpoint checkpoints/oct_v1/checkpoint_best.pth \
      --data_dirs /data1/kechuang/.../hai_oct \
      --mask_ratio 0.85 \
      --output_dir infer_output/oct
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from models_mae import mae_vit_large_patch16, mae_vit_base_patch16
from eye_dataset import (
    CsvOphthalmicDataset,
    OphthalmicDataset,
    MASK_RATIO_CFP,
    MASK_RATIO_OCT,
    IMAGE_EXTS,
    _infer_modality,
)

# ImageNet 归一化参数
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------

def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """(C,H,W) normalized → (C,H,W) [0,1]"""
    return (tensor * _STD.to(tensor.device) + _MEAN.to(tensor.device)).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(C,H,W) [0,1] float → PIL RGB"""
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# 推理核心：生成 original / masked / reconstructed
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_one(
    model,
    img_tensor: torch.Tensor,      # (1, C, H, W) on device, normalized
    mask_ratio: float,
    patch_size: int = 16,
    norm_pix_loss: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    返回 (original_pil, masked_pil, reconstructed_pil)
    """
    model.eval()
    device = img_tensor.device
    C = img_tensor.shape[1]

    # ── 1. 编码（随机掩码）────────────────────────────────────────────────
    latent, mask, ids_restore = model.module.forward_encoder(img_tensor, mask_ratio) \
        if hasattr(model, "module") else \
        model.forward_encoder(img_tensor, mask_ratio)

    # ── 2. 解码（重建）───────────────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    pred = raw_model.forward_decoder(latent, ids_restore)
    # pred: (1, N, patch_size^2 * C)  —— 在归一化像素空间

    # ── 3. 把预测还原为像素图像 ──────────────────────────────────────────
    # 先将 patch 级别的归一化还原（norm_pix_loss 时模型预测的是 patch-normalized）
    target = raw_model.patchify(img_tensor)  # (1, N, P*P*C) 真实 patch
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var  = target.var( dim=-1, keepdim=True)
        # pred 对应的是 (x - mean) / std 空间，还原回像素空间：
        pred_pixel = pred * (var + 1e-6).sqrt() + mean
    else:
        pred_pixel = pred

    # 把 reconstructed patches 合并回完整图（仅 masked 位置用预测值，visible 用原值）
    target_mod = target.clone()
    # mask: (1, N)，1=masked
    target_mod[mask.bool().unsqueeze(-1).expand_as(target_mod)] = \
        pred_pixel[mask.bool().unsqueeze(-1).expand_as(pred_pixel)]

    recon_img = raw_model.unpatchify(target_mod)  # (1, C, H, W) 仍是 ImageNet 归一化空间

    # ── 4. masked 可视化 ─────────────────────────────────────────────────
    # 将 masked patches 设为中性灰（ImageNet 归一化空间下 0.5→约 0.0）
    masked_target = target.clone()
    gray_val = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / \
               torch.tensor([0.229, 0.224, 0.225])  # (3,)
    # 把每个 patch 的像素全部设为对应通道的灰色值
    P2C = patch_size * patch_size * C
    gray_patch = gray_val.repeat(patch_size * patch_size).to(device)  # (P*P*C,)
    masked_target[mask.bool().unsqueeze(-1).expand_as(masked_target)] = \
        gray_patch.repeat(int(mask.sum().item()))[:masked_target[mask.bool().unsqueeze(-1).expand_as(masked_target)].shape[0]]
    masked_img = raw_model.unpatchify(masked_target)  # (1, C, H, W)

    # ── 5. 转换为 PIL ────────────────────────────────────────────────────
    orig_pil  = tensor_to_pil(denormalize(img_tensor[0]))
    mask_pil  = tensor_to_pil(denormalize(masked_img[0]))
    recon_pil = tensor_to_pil(denormalize(recon_img[0]))

    return orig_pil, mask_pil, recon_pil


# ---------------------------------------------------------------------------
# 图像文件收集
# ---------------------------------------------------------------------------

def collect_image_paths(args) -> List[Path]:
    """从 CSV 或目录收集所有图像路径，返回 Path 列表。"""
    paths: List[Path] = []

    if args.csv_files:
        for i, csv_path in enumerate(args.csv_files):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                print(f"[Warning] CSV 不存在: {csv_path}")
                continue
            root = csv_path.parent
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel = row.get(args.csv_path_col, "").strip()
                    if not rel:
                        continue
                    d = (root / rel).resolve()
                    if d.exists():
                        paths.extend(p for p in sorted(d.rglob("*.png")) if p.is_file())
    elif args.data_dirs:
        for d in args.data_dirs:
            d = Path(d)
            if d.exists():
                paths.extend(
                    p for p in sorted(d.rglob("*"))
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MAE 推理可视化")

    # 数据源
    g = p.add_argument_group("数据源（csv_files 或 data_dirs 二选一）")
    g.add_argument("--csv_files",      nargs="+", default=None)
    g.add_argument("--csv_path_col",   default="output_subdir")
    g.add_argument("--csv_modalities", nargs="+", default=None)
    g.add_argument("--data_dirs",      nargs="+", default=None)

    # 模型
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径（.pth）")
    p.add_argument("--model", default="mae_vit_large_patch16",
                   choices=["mae_vit_large_patch16", "mae_vit_base_patch16"])
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="掩码率。不指定时从模态自动推断（cfp=0.75, oct=0.85）")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--in_chans", type=int, default=3)

    # 抽样
    p.add_argument("--n_samples", type=int, default=10, help="随机抽取图像数量")
    p.add_argument("--seed",      type=int, default=42)

    # 输出
    p.add_argument("--output_dir", required=True, help="结果输出目录")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    # ── 收集图像路径 ──────────────────────────────────────────────────────
    all_paths = collect_image_paths(args)
    if not all_paths:
        raise RuntimeError("未找到任何图像，请检查 --csv_files 或 --data_dirs")
    print(f"[Infer] 数据集共 {len(all_paths)} 张图像")

    # 随机抽样
    n = min(args.n_samples, len(all_paths))
    sampled = random.sample(all_paths, n)
    print(f"[Infer] 抽取 {n} 张进行推理")

    # ── 模型 ──────────────────────────────────────────────────────────────
    model_fn = {
        "mae_vit_large_patch16": mae_vit_large_patch16,
        "mae_vit_base_patch16":  mae_vit_base_patch16,
    }[args.model]

    model = model_fn(img_size=args.img_size, in_chans=args.in_chans, norm_pix_loss=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  "?")
    print(f"[Infer] Loaded checkpoint: epoch={epoch}, loss={loss:.4f}" if isinstance(loss, float) else
          f"[Infer] Loaded checkpoint: epoch={epoch}")

    # ── 输出目录 ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.img_size)

    # ── 推理并保存 ────────────────────────────────────────────────────────
    meta_rows = []

    for idx, img_path in enumerate(sampled):
        sample_dir = out_dir / f"{idx+1:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 确定 mask_ratio
        if args.mask_ratio is not None:
            mask_ratio = args.mask_ratio
        else:
            mod = _infer_modality(img_path)
            mask_ratio = MASK_RATIO_OCT if mod == "oct" else MASK_RATIO_CFP

        # 加载图像
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] 加载失败 {img_path}: {e}, 跳过")
            continue

        img_tensor = transform(pil).unsqueeze(0).to(device)

        # 推理
        orig_pil, masked_pil, recon_pil = infer_one(
            model, img_tensor, mask_ratio,
            patch_size=model.patch_size,
            norm_pix_loss=True,
        )

        # 保存
        orig_out  = sample_dir / "original.png"
        mask_out  = sample_dir / "masked.png"
        recon_out = sample_dir / "reconstructed.png"

        orig_pil.save(orig_out)
        masked_pil.save(mask_out)
        recon_pil.save(recon_out)

        meta_rows.append({
            "sample_id":          idx + 1,
            "original_src_path":  str(img_path),          # 原始磁盘路径（非 shm）
            "mask_ratio":         mask_ratio,
            "output_dir":         str(sample_dir),
            "original_out":       str(orig_out),
            "masked_out":         str(mask_out),
            "reconstructed_out":  str(recon_out),
        })

        print(f"  [{idx+1:2d}/{n}] {img_path.name}  mask_ratio={mask_ratio}  → {sample_dir.name}/")

    # ── 元数据 CSV ────────────────────────────────────────────────────────
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\n[Infer] 完成！结果保存至: {out_dir}")
    print(f"[Infer] 元数据: {meta_path}")
    print(f"[Infer] 目录结构:")
    print(f"  {out_dir}/")
    print(f"  ├── metadata.csv")
    print(f"  ├── 0001/  original.png  masked.png  reconstructed.png")
    print(f"  └── {n:04d}/  ...")


if __name__ == "__main__":
    main()
