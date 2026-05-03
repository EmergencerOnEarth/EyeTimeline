#!/usr/bin/env python3
"""
MAE Pretraining on Ophthalmic Images (ViT-Large).

── 单卡（CSV 模式，推荐）──────────────────────────────────────────────────
python pretrain_mae.py \
    --csv_files data/oct/meta.csv data/fundus/meta.csv \
    --csv_path_col relative_dir \
    --csv_modalities oct cfp \
    --checkpoint weights/mae_vit_large_imagenet.bin \
    --output_dir checkpoints/v1

── 单卡（目录模式，兼容旧接口）──────────────────────────────────────────
python pretrain_mae.py \
    --data_dirs data/hai_oct data/hai_fund \
    --checkpoint weights/mae_vit_large_imagenet.bin \
    --output_dir checkpoints/v1

── 4 卡（torchrun）──────────────────────────────────────────────────────
torchrun --nproc_per_node=4 --master_port=29600 pretrain_mae.py \
    --csv_files data/oct/meta.csv data/fundus/meta.csv \
    --csv_path_col relative_dir \
    --csv_modalities oct cfp \
    --checkpoint weights/mae_vit_large_imagenet.bin \
    --output_dir checkpoints/v1 \
    --batch_size 768 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from eye_dataset import (
    CsvOphthalmicDataset,
    OphthalmicDataset,
    build_weighted_sampler,
    MASK_RATIO_CFP,
    MASK_RATIO_OCT,
)
from models_mae import mae_vit_large_patch16, mae_vit_base_patch16


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MAE Ophthalmic Pretraining", add_help=True)

    # ── 数据输入：CSV 模式（推荐）─────────────────────────────────────────
    g = p.add_argument_group("CSV 数据源（推荐，与 --data_dirs 二选一）")
    g.add_argument(
        "--csv_files", nargs="+", default=None,
        metavar="PATH",
        help="一个或多个 CSV 元数据文件路径。"
    )
    g.add_argument(
        "--csv_path_col", default="relative_dir",
        metavar="COL",
        help="CSV 中指向图像目录的列名（相对路径，相对于 CSV 文件所在目录）。"
    )
    g.add_argument(
        "--csv_modalities", nargs="+", default=None,
        metavar="MOD",
        help="与 --csv_files 一一对应的模态标签，如 'oct cfp'。"
             "未指定则从路径自动推断。"
    )
    g.add_argument(
        "--csv_modality_col", default=None,
        metavar="COL",
        help="CSV 中记录每行模态的列名（优先级高于 --csv_modalities）。"
    )

    # ── 数据输入：目录模式（兼容旧接口）──────────────────────────────────
    g2 = p.add_argument_group("目录数据源（兼容旧接口，与 --csv_files 二选一）")
    g2.add_argument("--data_dirs", nargs="+", default=None, metavar="DIR")
    g2.add_argument(
        "--modality", default="all", choices=["all", "fundus", "cfp", "oct"],
        help="仅在目录模式下生效，过滤模态。"
    )

    # ── 通用数据参数 ──────────────────────────────────────────────────────
    p.add_argument("--img_size",   type=int,  default=224)
    p.add_argument("--grayscale",  action="store_true")
    p.add_argument(
        "--mask_ratio_cfp", type=float, default=MASK_RATIO_CFP,
        help=f"CFP/Fundus 掩码率（默认 {MASK_RATIO_CFP}，来自 RETFound）。"
    )
    p.add_argument(
        "--mask_ratio_oct", type=float, default=MASK_RATIO_OCT,
        help=f"OCT 掩码率（默认 {MASK_RATIO_OCT}，来自 RETFound）。"
    )
    p.add_argument("--balance_modalities", action="store_true", default=False)

    # ── 模型 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--model", default="mae_vit_large_patch16",
        choices=["mae_vit_large_patch16", "mae_vit_base_patch16"],
    )
    p.add_argument("--norm_pix_loss", action="store_true", default=True)

    # ── Checkpoint ────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",  default=None,
                   help="预训练权重路径（如 ImageNet MAE）。")
    p.add_argument("--resume",      default=None,
                   help="恢复训练的 checkpoint 路径。")
    p.add_argument("--output_dir",     default="checkpoints")
    p.add_argument("--save_every",     type=int, default=10)
    p.add_argument("--max_checkpoints",type=int, default=5,
                   help="周期性 checkpoint 最多保留数量，超出时删除最旧的。")

    # ── 训练超参 ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64,
                   help="单卡 batch size。")
    p.add_argument("--lr",           type=float, default=None,
                   help="基础 LR，默认 1.5e-4 × (bs × world × accum) / 256。")
    p.add_argument("--min_lr",       type=float, default=0.0)
    p.add_argument("--warmup_epochs",type=int,   default=10)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--accum_steps",  type=int,   default=1)

    # ── 精度 / 数据加载 ───────────────────────────────────────────────────
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no_amp",      dest="amp", action="store_false")
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--pin_memory",  action="store_true", default=True)

    # ── 日志 ─────────────────────────────────────────────────────────────
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * max(epoch, 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    loss: float,
    tag: str = "",
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint_{tag}.pth" if tag else f"checkpoint_ep{epoch:04d}.pth"
    path = out_dir / name

    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "epoch": epoch,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "loss": loss,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)
    print(f"[Checkpoint] Saved → {path}")

    # 更新 latest 软链接
    latest = out_dir / "checkpoint_latest.pth"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(name)

    # 周期性 checkpoint 只保留最新 max_checkpoints 个
    if not tag:
        _rotate_checkpoints(out_dir, args.max_checkpoints)


def _rotate_checkpoints(out_dir: Path, max_keep: int) -> None:
    """删除最旧的周期性 checkpoint，只保留最新的 max_keep 个。"""
    periodic = sorted(out_dir.glob("checkpoint_ep*.pth"))
    to_delete = periodic[:-max_keep] if len(periodic) > max_keep else []
    for old in to_delete:
        old.unlink(missing_ok=True)
        print(f"[Checkpoint] Removed old checkpoint: {old.name}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0) + 1
    print(f"[Checkpoint] Resumed from '{path}' at epoch {epoch}")
    return epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _forward_by_mask_ratio(
    model: nn.Module,
    imgs: torch.Tensor,
    mask_ratios: torch.Tensor,
    amp: bool,
) -> torch.Tensor:
    """按 mask_ratio 分组做 forward，加权平均 loss。

    同一个 batch 中 CFP 和 OCT 样本可能有不同 mask_ratio，
    逐组 forward 保证每组用正确的掩码率。
    """
    unique_ratios = mask_ratios.unique()

    if len(unique_ratios) == 1:
        # 全部相同，走单次 forward（最常见情况）
        with autocast(enabled=amp):
            loss, _, _ = model(imgs, mask_ratio=unique_ratios[0].item())
        return loss

    # 混合 batch：按 ratio 分组
    total_loss = torch.tensor(0.0, device=imgs.device)
    for ratio in unique_ratios:
        idx = (mask_ratios == ratio).nonzero(as_tuple=True)[0]
        sub = imgs[idx]
        with autocast(enabled=amp):
            loss, _, _ = model(sub, mask_ratio=ratio.item())
        total_loss = total_loss + loss * len(idx)

    return total_loss / len(imgs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    writer: SummaryWriter | None,
    global_step: int,
    world_size: int,
    rank: int,
) -> Tuple[float, int]:
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n_batches  = 0
    t0 = time.perf_counter()

    for step, (imgs, mask_ratios) in enumerate(loader):
        imgs        = imgs.to(device, non_blocking=True)
        mask_ratios = mask_ratios.float().to(device, non_blocking=True)

        loss = _forward_by_mask_ratio(model, imgs, mask_ratios, args.amp)
        loss = loss / args.accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accum_steps == 0:
            if scaler is not None:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            optimizer.zero_grad()

            step_loss   = loss.item() * args.accum_steps
            total_loss += step_loss
            n_batches  += 1
            global_step += 1

            if is_main(rank) and global_step % args.log_freq == 0:
                elapsed = time.perf_counter() - t0
                ips = args.batch_size * world_size * args.log_freq * args.accum_steps / elapsed
                # 显示本 batch 实际使用的 mask ratio 分布
                ratio_info = " ".join(
                    f"{r:.2f}×{int((mask_ratios == r).sum())}"
                    for r in mask_ratios.unique()
                )
                print(
                    f"  Ep {epoch:4d} | step {global_step:7d} | "
                    f"loss {step_loss:.4f} | {ips:.0f} img/s | "
                    f"mask[{ratio_info}]"
                )
                if writer:
                    writer.add_scalar("train/loss_step", step_loss, global_step)
                    writer.add_scalar("train/imgs_per_sec", ips, global_step)
                t0 = time.perf_counter()

    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        t = torch.tensor(avg_loss, device=device)
        avg_loss = reduce_mean(t, world_size).item()
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(args: argparse.Namespace) -> Dataset:
    in_chans = 1 if args.grayscale else 3  # noqa: F841

    if args.csv_files:
        return CsvOphthalmicDataset(
            csv_files=args.csv_files,
            path_col=args.csv_path_col,
            csv_modalities=args.csv_modalities,
            modality_col=args.csv_modality_col,
            mask_ratio_cfp=args.mask_ratio_cfp,
            mask_ratio_oct=args.mask_ratio_oct,
            img_size=args.img_size,
            grayscale=args.grayscale,
            is_train=True,
        )
    elif args.data_dirs:
        return OphthalmicDataset(
            data_dirs=args.data_dirs,
            img_size=args.img_size,
            grayscale=args.grayscale,
            modality=args.modality,
            mask_ratio_cfp=args.mask_ratio_cfp,
            mask_ratio_oct=args.mask_ratio_oct,
            is_train=True,
        )
    else:
        raise ValueError("必须提供 --csv_files 或 --data_dirs 之一。")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()

    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    if is_main(rank):
        print(f"[Main] world_size={world_size} | AMP={args.amp}")
        print(f"[Main] mask_ratio_cfp={args.mask_ratio_cfp}  mask_ratio_oct={args.mask_ratio_oct}")

    # ── 输出目录 & TensorBoard ────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    writer  = None
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = build_dataset(args)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    elif args.balance_modalities:
        sampler = build_weighted_sampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    if is_main(rank):
        print(f"[Main] Dataset: {len(dataset)} 张 | Steps/epoch: {len(loader)} | GPUs: {world_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    model_fn = {
        "mae_vit_large_patch16": mae_vit_large_patch16,
        "mae_vit_base_patch16":  mae_vit_base_patch16,
    }[args.model]

    in_chans = 1 if args.grayscale else 3
    model = model_fn(
        img_size=args.img_size,
        in_chans=in_chans,
        norm_pix_loss=args.norm_pix_loss,
    )

    if args.checkpoint is not None:
        model.load_pretrained(args.checkpoint, strict=False)

    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main(rank):
        raw = model.module if isinstance(model, DDP) else model
        print(f"[Main] 参数量: {sum(p.numel() for p in raw.parameters())/1e6:.1f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────
    raw_model = model.module if isinstance(model, DDP) else model
    decay, no_decay = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(k in name for k in ("pos_embed", "cls_token", "mask_token")):
            no_decay.append(p)
        else:
            decay.append(p)

    eff_batch = args.batch_size * world_size * args.accum_steps
    base_lr   = args.lr if args.lr is not None else 1.5e-4 * eff_batch / 256
    if is_main(rank):
        print(f"[Main] 有效 batch={eff_batch} | Base LR={base_lr:.2e}")

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.95),
    )
    scaler = GradScaler() if args.amp else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = (start_epoch - 1) * len(loader)
    best_loss   = float("inf")

    if is_main(rank):
        print(f"\n[Main] 训练 epoch {start_epoch}–{args.epochs}")
        print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        lr = cosine_lr_schedule(
            optimizer, epoch - 1, args.epochs, args.warmup_epochs, base_lr, args.min_lr
        )

        t_ep = time.perf_counter()
        avg_loss, global_step = train_one_epoch(
            model, loader, optimizer, scaler,
            device, epoch, args, writer, global_step,
            world_size, rank,
        )
        elapsed = time.perf_counter() - t_ep

        if is_main(rank):
            print(
                f"[Ep {epoch:4d}/{args.epochs}] loss={avg_loss:.4f} | "
                f"lr={lr:.2e} | time={elapsed/60:.1f}min"
            )
            if writer:
                writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                writer.add_scalar("train/lr", lr, epoch)

            if epoch % args.save_every == 0:
                save_checkpoint(args, epoch, model, optimizer, scaler, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(args, epoch, model, optimizer, scaler, avg_loss, tag="best")

        barrier(world_size)

    if is_main(rank):
        save_checkpoint(args, args.epochs, model, optimizer, scaler, avg_loss, tag="final")
        if writer:
            writer.close()
        print("\n[Main] 训练完成。")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
