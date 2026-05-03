"""
MAE (Masked Autoencoder) with ViT-Large backbone for ophthalmic image pretraining.

Architecture:
  Encoder: ViT-Large (patch_size=16, embed_dim=1024, depth=24, heads=16)
  Decoder: Lightweight transformer (embed_dim=512, depth=8, heads=16)
  Loss: MSE on normalized pixel values of masked patches

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """Generate 2D sin-cos positional embedding."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (2, H, W)
    grid = torch.stack(grid, dim=0).reshape(2, -1)         # (2, H*W)

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    half = embed_dim // 4
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))

    emb_h = grid[0, :, None] * omega[None, :]   # (N, half)
    emb_w = grid[1, :, None] * omega[None, :]   # (N, half)
    emb = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)  # (N, embed_dim)

    if cls_token:
        emb = torch.cat([torch.zeros(1, embed_dim), emb], dim=0)
    return emb.float()


# ---------------------------------------------------------------------------
# MAE ViT-Large
# ---------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with ViT-Large encoder.

    Args:
        img_size:      Input image size (square assumed).
        patch_size:    Patch size for tokenisation.
        in_chans:      Number of input channels (1 for grayscale OCT, 3 for fundus).
        encoder_embed_dim: ViT-Large hidden dim (1024).
        encoder_depth: Number of encoder transformer blocks (24 for ViT-L).
        encoder_num_heads: Number of encoder attention heads (16).
        decoder_embed_dim: Decoder hidden dim (512).
        decoder_depth: Number of decoder transformer blocks (8).
        decoder_num_heads: Number of decoder attention heads (16).
        mlp_ratio:     MLP expansion ratio.
        norm_pix_loss: Normalise target pixels per patch before computing loss.
        mask_ratio:    Fraction of patches to mask during training.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss: bool = True,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()

        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans

        # ---- Encoder -------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim), requires_grad=False
        )

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ---- Decoder -------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Sin-cos positional embeddings (frozen)
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        enc_pos = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.encoder_pos_embed.data.copy_(enc_pos.unsqueeze(0))

        dec_pos = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(dec_pos.unsqueeze(0))

        # Patch embedding: Xavier uniform
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        # CLS / mask tokens: normal
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # All other layers
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch_size^2 * C)"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p * p * self.in_chans)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_size^2 * C) -> (B, C, H, W)"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_chans, h * p, w * p)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches, keeping (1 - mask_ratio) fraction.

        Returns:
            x_masked:    Visible tokens only, shape (B, N_vis, D).
            mask:        Binary mask (1 = masked), shape (B, N).
            ids_restore: Permutation to undo the shuffle, shape (B, N).
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder / Decoder forward passes
    # ------------------------------------------------------------------

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.encoder_pos_embed[:, 1:, :]  # skip CLS pos

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        x = self.decoder_embed(x)

        # Append mask tokens for all masked positions
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # strip CLS
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # re-prepend CLS

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]  # remove CLS token output
        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss on masked patches only."""
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)        # (B, N) mean over patch pixels
        loss = (loss * mask).sum() / mask.sum()   # mean over masked patches
        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_pretrained(self, ckpt_path: str, strict: bool = False) -> None:
        """Load weights from a checkpoint file.

        Automatically detects and converts:
          - HuggingFace ViTMAE format  (facebook/vit-mae-large pytorch_model.bin)
          - Original Meta MAE format   (model_key = 'model')
          - Plain state dict
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt  # HF format: flat state dict

        # Detect HuggingFace format by checking for a known HF key
        if any(k.startswith("vit.embeddings") for k in state_dict):
            print(f"[MAE] Detected HuggingFace ViTMAE format — converting keys...")
            state_dict = self._convert_hf_weights(state_dict)

        msg = self.load_state_dict(state_dict, strict=strict)
        n_loaded = len(state_dict) - len(msg.missing_keys)
        print(
            f"[MAE] Loaded '{ckpt_path}'\n"
            f"      matched={n_loaded}, "
            f"missing={len(msg.missing_keys)}, "
            f"unexpected={len(msg.unexpected_keys)}"
        )
        if msg.missing_keys:
            print(f"      First missing: {msg.missing_keys[:5]}")

    def _convert_hf_weights(self, hf_sd: dict) -> dict:
        """Convert HuggingFace ViTMAE state-dict keys to this model's timm-based keys.

        HF encoder key pattern:
          vit.embeddings.patch_embeddings.projection.{weight,bias}
          vit.embeddings.cls_token
          vit.encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}
          vit.encoder.layer.{i}.attention.output.dense.{weight,bias}
          vit.encoder.layer.{i}.layernorm_before.{weight,bias}
          vit.encoder.layer.{i}.layernorm_after.{weight,bias}
          vit.encoder.layer.{i}.intermediate.dense.{weight,bias}
          vit.encoder.layer.{i}.output.dense.{weight,bias}
          vit.layernorm.{weight,bias}

        HF decoder key pattern:
          decoder.decoder_embed.{weight,bias}
          decoder.mask_token
          decoder.decoder_layers.{i}...   (same sub-key pattern as encoder)
          decoder.decoder_norm.{weight,bias}
          decoder.decoder_pred.{weight,bias}
        """
        new_sd: dict = {}

        # ---- Encoder: patch embed & cls token ---------------------------
        _map = {
            "vit.embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
            "vit.embeddings.patch_embeddings.projection.bias":   "patch_embed.proj.bias",
            "vit.embeddings.cls_token":                          "cls_token",
            "vit.layernorm.weight":                              "encoder_norm.weight",
            "vit.layernorm.bias":                                "encoder_norm.bias",
            # Decoder misc
            "decoder.decoder_embed.weight":  "decoder_embed.weight",
            "decoder.decoder_embed.bias":    "decoder_embed.bias",
            "decoder.mask_token":            "mask_token",
            "decoder.decoder_norm.weight":   "decoder_norm.weight",
            "decoder.decoder_norm.bias":     "decoder_norm.bias",
            "decoder.decoder_pred.weight":   "decoder_pred.weight",
            "decoder.decoder_pred.bias":     "decoder_pred.bias",
        }
        for hf_key, my_key in _map.items():
            if hf_key in hf_sd:
                new_sd[my_key] = hf_sd[hf_key]

        # We do NOT copy position embeddings — ours are fixed sin-cos and
        # will be re-initialised; HF uses learnable pos embeds with a
        # different initialisation.

        # ---- Encoder blocks ---------------------------------------------
        for i in range(24):  # ViT-Large has 24 blocks
            pfx = f"vit.encoder.layer.{i}"
            tgt = f"encoder_blocks.{i}"

            # LayerNorm
            for ln_hf, ln_my in [("layernorm_before", "norm1"), ("layernorm_after", "norm2")]:
                for p in ("weight", "bias"):
                    src = f"{pfx}.{ln_hf}.{p}"
                    if src in hf_sd:
                        new_sd[f"{tgt}.{ln_my}.{p}"] = hf_sd[src]

            # Attention: merge Q/K/V → QKV
            q_w = hf_sd.get(f"{pfx}.attention.attention.query.weight")
            k_w = hf_sd.get(f"{pfx}.attention.attention.key.weight")
            v_w = hf_sd.get(f"{pfx}.attention.attention.value.weight")
            if q_w is not None and k_w is not None and v_w is not None:
                new_sd[f"{tgt}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_sd.get(f"{pfx}.attention.attention.query.bias")
            k_b = hf_sd.get(f"{pfx}.attention.attention.key.bias")
            v_b = hf_sd.get(f"{pfx}.attention.attention.value.bias")
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{tgt}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            # Attention output projection
            for p in ("weight", "bias"):
                src = f"{pfx}.attention.output.dense.{p}"
                if src in hf_sd:
                    new_sd[f"{tgt}.attn.proj.{p}"] = hf_sd[src]

            # MLP
            for p in ("weight", "bias"):
                src_fc1 = f"{pfx}.intermediate.dense.{p}"
                src_fc2 = f"{pfx}.output.dense.{p}"
                if src_fc1 in hf_sd:
                    new_sd[f"{tgt}.mlp.fc1.{p}"] = hf_sd[src_fc1]
                if src_fc2 in hf_sd:
                    new_sd[f"{tgt}.mlp.fc2.{p}"] = hf_sd[src_fc2]

        # ---- Decoder blocks ---------------------------------------------
        for i in range(8):  # MAE decoder has 8 blocks
            pfx = f"decoder.decoder_layers.{i}"
            tgt = f"decoder_blocks.{i}"

            for ln_hf, ln_my in [("layernorm_before", "norm1"), ("layernorm_after", "norm2")]:
                for p in ("weight", "bias"):
                    src = f"{pfx}.{ln_hf}.{p}"
                    if src in hf_sd:
                        new_sd[f"{tgt}.{ln_my}.{p}"] = hf_sd[src]

            q_w = hf_sd.get(f"{pfx}.attention.attention.query.weight")
            k_w = hf_sd.get(f"{pfx}.attention.attention.key.weight")
            v_w = hf_sd.get(f"{pfx}.attention.attention.value.weight")
            if q_w is not None and k_w is not None and v_w is not None:
                new_sd[f"{tgt}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_sd.get(f"{pfx}.attention.attention.query.bias")
            k_b = hf_sd.get(f"{pfx}.attention.attention.key.bias")
            v_b = hf_sd.get(f"{pfx}.attention.attention.value.bias")
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{tgt}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            for p in ("weight", "bias"):
                src = f"{pfx}.attention.output.dense.{p}"
                if src in hf_sd:
                    new_sd[f"{tgt}.attn.proj.{p}"] = hf_sd[src]

            for p in ("weight", "bias"):
                src_fc1 = f"{pfx}.intermediate.dense.{p}"
                src_fc2 = f"{pfx}.output.dense.{p}"
                if src_fc1 in hf_sd:
                    new_sd[f"{tgt}.mlp.fc1.{p}"] = hf_sd[src_fc1]
                if src_fc2 in hf_sd:
                    new_sd[f"{tgt}.mlp.fc2.{p}"] = hf_sd[src_fc2]

        return new_sd


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoderViT:
    """ViT-Large: 24 blocks, embed_dim=1024, heads=16."""
    return MaskedAutoencoderViT(
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoderViT:
    """ViT-Base: 12 blocks, embed_dim=768, heads=12."""
    return MaskedAutoencoderViT(
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
