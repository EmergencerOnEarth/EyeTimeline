from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models_mae import mae_vit_base_patch16, mae_vit_large_patch16


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "teacher", "student"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def _convert_meta_mae_to_eyetimeline(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("head."):
            continue
        new_key = key
        if key == "pos_embed":
            new_key = "encoder_pos_embed"
        elif key.startswith("blocks."):
            new_key = "encoder_blocks." + key[len("blocks."):]
        elif key.startswith("norm."):
            new_key = "encoder_norm." + key[len("norm."):]
        converted[new_key] = value
    return converted


class MAEEncoderClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        variant: str = "large",
        dropout: float = 0.0,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        if variant == "large":
            self.backbone = mae_vit_large_patch16(img_size=img_size, in_chans=3)
            feature_dim = 1024
        elif variant == "base":
            self.backbone = mae_vit_base_patch16(img_size=img_size, in_chans=3)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown MAE variant: {variant}")
        self.variant = variant
        self.pool = pool
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self._freeze_unused_decoder()

    def _freeze_unused_decoder(self) -> None:
        for name, param in self.backbone.named_parameters():
            if name.startswith(("decoder_", "mask_token")):
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.backbone.patch_embed(x)
        x = x + self.backbone.encoder_pos_embed[:, 1:, :]
        cls = self.backbone.cls_token + self.backbone.encoder_pos_embed[:, :1, :]
        x = torch.cat([cls.expand(bsz, -1, -1), x], dim=1)
        for block in self.backbone.encoder_blocks:
            x = block(x)
        x = self.backbone.encoder_norm(x)
        if self.pool == "avg":
            return x[:, 1:, :].mean(dim=1)
        if self.pool == "cls":
            return x[:, 0]
        raise ValueError(f"Unknown pool: {self.pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def load_backbone(self, checkpoint: str | Path, strict: bool = False) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            candidates = [
                checkpoint / "pytorch_model.bin",
                checkpoint / "checkpoint.pth",
                checkpoint / "RETFound_mae_natureCFP.pth",
                checkpoint / "RETFound_mae_natureOCT.pth",
            ]
            existing = [p for p in candidates if p.exists()]
            if not existing:
                existing = sorted(checkpoint.glob("*.pth")) + sorted(checkpoint.glob("*.bin"))
            if not existing:
                raise FileNotFoundError(f"No checkpoint file found under {checkpoint}")
            checkpoint = existing[0]

        payload = torch.load(checkpoint, map_location="cpu")
        state = _extract_state_dict(payload)
        state = _strip_prefix(state, "module.")
        state = _strip_prefix(state, "backbone.")

        if any(k.startswith("vit.embeddings") for k in state):
            state = self.backbone._convert_hf_weights(state)
        elif any(k.startswith("blocks.") for k in state) or "pos_embed" in state:
            state = _convert_meta_mae_to_eyetimeline(state)

        msg = self.backbone.load_state_dict(state, strict=strict)
        print(
            f"[Model] Loaded backbone from {checkpoint} | "
            f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
        )
        if msg.missing_keys:
            print(f"[Model] First missing keys: {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"[Model] First unexpected keys: {msg.unexpected_keys[:5]}")


class EyeCLIPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        checkpoint: str | Path | None = None,
        clip_model_type: str = "ViT-B/32",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "EyeCLIP baseline requires openai/CLIP. Install benchmarks/requirements-benchmark.txt"
            ) from exc

        model, _ = clip.load(clip_model_type, device="cpu", jit=False)
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu")
            state = _extract_state_dict(payload)
            state = _strip_prefix(state, "module.")
            msg = model.load_state_dict(state, strict=False)
            print(
                f"[EyeCLIP] Loaded {checkpoint} | "
                f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
            )
        self.clip_model = model.float()
        embed_dim = int(model.visual.output_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(x).float()
        return self.head(features)


def build_model(
    backbone: str,
    num_classes: int,
        checkpoint: str | Path | None = None,
        img_size: int = 224,
        mae_variant: str = "large",
        dropout: float = 0.0,
        pool: str = "cls",
        clip_model_type: str = "ViT-B/32",
) -> nn.Module:
    if backbone in {"eyetimeline_mae", "retfound_mae", "imagenet_mae", "random_mae"}:
        model = MAEEncoderClassifier(
            num_classes=num_classes,
            img_size=img_size,
            variant=mae_variant,
            dropout=dropout,
            pool=pool,
        )
        if checkpoint:
            model.load_backbone(checkpoint, strict=False)
        return model

    if backbone == "eyeclip":
        return EyeCLIPClassifier(
            num_classes=num_classes,
            checkpoint=checkpoint,
            clip_model_type=clip_model_type,
            dropout=dropout,
        )

    raise ValueError(f"Unknown backbone: {backbone}")


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if isinstance(model, EyeCLIPClassifier):
        for param in model.clip_model.visual.parameters():
            param.requires_grad = trainable
        for name, param in model.clip_model.named_parameters():
            if not name.startswith("visual."):
                param.requires_grad = False
        return

    if isinstance(model, MAEEncoderClassifier):
        for name, param in model.backbone.named_parameters():
            if name.startswith(("decoder_", "mask_token")):
                param.requires_grad = False
            else:
                param.requires_grad = trainable
        return

    raise TypeError(f"Unsupported model type: {type(model)}")
