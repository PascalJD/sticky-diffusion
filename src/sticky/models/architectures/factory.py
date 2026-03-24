from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn

from .image import ADMUNet5DBackbone, UNet5DBackbone
from .sequence import GPT2LikeBackbone, TransformerBackbone


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
    embed_input: bool = False,
    n_embed_classes: int = 1,
    hidden_dim: int | None = None,
    max_seq_len: int | None = None,
    causal: bool = False,
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
            embed_input=bool(embed_input),
            n_embed_classes=int(n_embed_classes),
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
            embed_input=bool(embed_input),
            n_embed_classes=int(n_embed_classes),
            sharded=True,
        )

    if key == "gpt2_like":
        return GPT2LikeBackbone(
            dim=int(feature_dim) * int(num_heads),
            n_layers=int(n_layers),
            n_heads=int(num_heads),
            output_channels=int(vocab_size),
            dropout_rate=float(dropout_rate),
            use_attn_dropout=bool(use_attn_dropout),
            embed_input=bool(embed_input),
            n_embed_classes=int(n_embed_classes),
            hidden_dim=None if hidden_dim is None else int(hidden_dim),
            max_seq_len=None if max_seq_len is None else int(max_seq_len),
            causal=bool(causal),
        )

    raise ValueError(
        f"Unknown sequence backbone {name!r}. "
        "Expected one of: auto, transformer, sharded_transformer, gpt2_like."
    )


def build_image_token_backbone(
    *,
    name: str,
    feature_dim: int,
    n_layers: int,
    n_dit_layers: int,
    dit_num_heads: int,
    dit_hidden_size: int,
    ch_mult: Sequence[int],
    output_channels: int,
    dropout_rate: float,
    adm_num_res_blocks: int,
    adm_attention_resolutions: Sequence[int],
    adm_num_heads: int,
    adm_num_head_channels: int,
    adm_num_heads_upsample: int,
    adm_conv_resample: bool,
    adm_use_scale_shift_norm: bool,
    adm_resblock_updown: bool,
    adm_use_conv_skip: bool,
    adm_use_new_attention_order: bool,
) -> nn.Module:
    """Create a shared 5D image-token backbone for embedded or continuous inputs."""
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
            output_channels=int(output_channels),
            dropout_rate=float(dropout_rate),
        )

    if key == "adm_unet5d":
        return ADMUNet5DBackbone(
            feature_dim=int(feature_dim),
            output_channels=int(output_channels),
            num_res_blocks=int(adm_num_res_blocks),
            attention_resolutions=tuple(int(v) for v in adm_attention_resolutions),
            channel_mult=tuple(int(x) for x in ch_mult),
            num_heads=int(adm_num_heads),
            num_head_channels=int(adm_num_head_channels),
            num_heads_upsample=int(adm_num_heads_upsample),
            dropout_rate=float(dropout_rate),
            conv_resample=bool(adm_conv_resample),
            use_scale_shift_norm=bool(adm_use_scale_shift_norm),
            resblock_updown=bool(adm_resblock_updown),
            use_conv_skip=bool(adm_use_conv_skip),
            use_new_attention_order=bool(adm_use_new_attention_order),
        )

    raise ValueError(
        f"Unknown image backbone {name!r}. "
        "Expected one of: auto, unet5d, adm_unet5d."
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
    adm_num_res_blocks: int,
    adm_attention_resolutions: Sequence[int],
    adm_num_heads: int,
    adm_num_head_channels: int,
    adm_num_heads_upsample: int,
    adm_conv_resample: bool,
    adm_use_scale_shift_norm: bool,
    adm_resblock_updown: bool,
    adm_use_conv_skip: bool,
    adm_use_new_attention_order: bool,
) -> nn.Module:
    """Backward-compatible wrapper for discrete image-token classifiers."""
    return build_image_token_backbone(
        name=name,
        feature_dim=feature_dim,
        n_layers=n_layers,
        n_dit_layers=n_dit_layers,
        dit_num_heads=dit_num_heads,
        dit_hidden_size=dit_hidden_size,
        ch_mult=ch_mult,
        output_channels=int(vocab_size),
        dropout_rate=dropout_rate,
        adm_num_res_blocks=adm_num_res_blocks,
        adm_attention_resolutions=adm_attention_resolutions,
        adm_num_heads=adm_num_heads,
        adm_num_head_channels=adm_num_head_channels,
        adm_num_heads_upsample=adm_num_heads_upsample,
        adm_conv_resample=adm_conv_resample,
        adm_use_scale_shift_norm=adm_use_scale_shift_norm,
        adm_resblock_updown=adm_resblock_updown,
        adm_use_conv_skip=adm_use_conv_skip,
        adm_use_new_attention_order=adm_use_new_attention_order,
    )
