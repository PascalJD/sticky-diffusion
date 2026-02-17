from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn

from .image import UNet5DBackbone
from .sequence import TransformerBackbone


def build_sequence_backbone(
    *,
    name: str,
    feature_dim: int,
    num_heads: int,
    n_layers: int,
    vocab_size: int,
    dropout_rate: float,
    use_attn_dropout: bool,
    mlp_type: str,
    depth_scaled_init: bool,
    cond_type: str,
    model_sharding: bool,
) -> nn.Module:
    """Create sequence backbone from a Hydra-selected name."""
    key = str(name).lower()
    if key == "auto":
        key = "sharded_transformer" if model_sharding else "transformer"

    if key == "transformer":
        return TransformerBackbone(
            dim=int(feature_dim) * int(num_heads),
            n_layers=int(n_layers),
            n_heads=int(num_heads),
            output_channels=int(vocab_size),
            dropout_rate=float(dropout_rate),
            use_attn_dropout=bool(use_attn_dropout),
            mlp_type=str(mlp_type),
            depth_scaled_init=bool(depth_scaled_init),
            cond_type=str(cond_type),
            embed_input=False,
            n_embed_classes=1,
            sharded=False,
        )

    if key == "sharded_transformer":
        return TransformerBackbone(
            dim=int(feature_dim) * int(num_heads),
            n_layers=int(n_layers),
            n_heads=int(num_heads),
            output_channels=int(vocab_size),
            dropout_rate=float(dropout_rate),
            use_attn_dropout=bool(use_attn_dropout),
            mlp_type=str(mlp_type),
            depth_scaled_init=bool(depth_scaled_init),
            cond_type=str(cond_type),
            embed_input=False,
            n_embed_classes=1,
            sharded=True,
        )

    raise ValueError(
        f"Unknown sequence backbone {name!r}. "
        "Expected one of: auto, transformer, sharded_transformer."
    )


def build_image_backbone(
    *,
    name: str,
    feature_dim: int,
    n_layers: int,
    n_dit_layers: int,
    dit_num_heads: int,
    dit_hidden_size: int,
    ch_mult: Sequence[int],
    vocab_size: int,
    dropout_rate: float,
) -> nn.Module:
    """Create image backbone from a Hydra-selected name."""
    key = str(name).lower()
    if key == "auto":
        key = "unet5d"

    if key == "unet5d":
        return UNet5DBackbone(
            feature_dim=int(feature_dim),
            n_layers=int(n_layers),
            n_dit_layers=int(n_dit_layers),
            dit_num_heads=int(dit_num_heads),
            dit_hidden_size=int(dit_hidden_size),
            ch_mult=tuple(int(x) for x in ch_mult),
            output_channels=int(vocab_size),
            dropout_rate=float(dropout_rate),
        )

    raise ValueError(
        f"Unknown image backbone {name!r}. Expected one of: auto, unet5d."
    )
