# src/sticky/models/factories/_helpers.py
"""Shared helper utilities for per-family model builder functions."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def sampler_method(
    cfg: DictConfig,
    *,
    default: str,
    allow_sampler_alias: bool = False,
) -> str:
    sampler_cfg = cfg.get("sampler", {})
    if allow_sampler_alias:
        return str(
            sampler_cfg.get(
                "method",
                sampler_cfg.get("sampler", cfg.model.get("sampler", default)),
            )
        )
    return str(sampler_cfg.get("method", cfg.model.get("sampler", default)))


def sampling_grid(cfg: DictConfig, *, default: str) -> str:
    sampler_cfg = cfg.get("sampler", {})
    return str(sampler_cfg.get("sampling_grid", cfg.model.get("sampling_grid", default)))


def categorical_sampling_policy(
    cfg: DictConfig,
    *,
    default: str = "legacy_low",
) -> str:
    sampler_cfg = cfg.get("sampler", {})
    return str(
        sampler_cfg.get(
            "categorical_sampling_policy",
            cfg.model.get("categorical_sampling_policy", default),
        )
    )


def adm_backbone_kwargs(
    model_cfg: DictConfig,
    *,
    image_backbone_default: str,
    adm_num_res_blocks_default: int,
    adm_attention_resolutions_default: tuple[int, ...],
    adm_num_heads_default: int,
    adm_num_head_channels_default: int,
) -> dict[str, Any]:
    return {
        "image_backbone": str(model_cfg.get("image_backbone", image_backbone_default)),
        "adm_num_res_blocks": int(
            model_cfg.get("adm_num_res_blocks", adm_num_res_blocks_default)
        ),
        "adm_attention_resolutions": tuple(
            model_cfg.get(
                "adm_attention_resolutions",
                adm_attention_resolutions_default,
            )
        ),
        "adm_num_heads": int(model_cfg.get("adm_num_heads", adm_num_heads_default)),
        "adm_num_head_channels": int(
            model_cfg.get("adm_num_head_channels", adm_num_head_channels_default)
        ),
        "adm_num_heads_upsample": int(model_cfg.get("adm_num_heads_upsample", -1)),
        "adm_conv_resample": bool(model_cfg.get("adm_conv_resample", True)),
        "adm_use_scale_shift_norm": bool(
            model_cfg.get("adm_use_scale_shift_norm", True)
        ),
        "adm_resblock_updown": bool(model_cfg.get("adm_resblock_updown", False)),
        "adm_use_conv_skip": bool(model_cfg.get("adm_use_conv_skip", False)),
        "adm_use_new_attention_order": bool(
            model_cfg.get("adm_use_new_attention_order", False)
        ),
    }
