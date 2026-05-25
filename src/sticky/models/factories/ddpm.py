# src/sticky/models/factories/ddpm.py
"""Builder for the DDPM model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import adm_backbone_kwargs


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    del vocab_size
    from sticky.models.baselines.ddpm.ddpm_model import DDPM

    return DDPM(
        data_shape=data_shape,
        timesteps=int(cfg.model.get("timesteps", 1000)),
        beta_schedule=str(cfg.model.get("beta_schedule", "linear")),
        beta_start=float(cfg.model.get("beta_start", 1e-4)),
        beta_end=float(cfg.model.get("beta_end", 2e-2)),
        prediction_type=str(cfg.model.get("prediction_type", "eps")),
        variance_type=str(cfg.model.get("variance_type", "fixed_small")),
        clip_x0=bool(cfg.model.get("clip_x0", True)),
        feature_dim=int(cfg.model.get("feature_dim", 96)),
        ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="adm_unet2d",
            adm_num_res_blocks_default=4,
            adm_attention_resolutions_default=(2, 4),
            adm_num_heads_default=4,
            adm_num_head_channels_default=64,
        ),
        classes=int(cfg.model.get("classes", -1)),
    )
