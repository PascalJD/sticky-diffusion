# src/sticky/models/factories/md4.py
"""Builder for the MD4 model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    categorical_sampling_policy,
    sampler_method,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    from sticky.models.baselines.md4.md4_model import MD4

    return MD4(
        data_shape=data_shape,
        cont_time=bool(cfg.model.cont_time),
        timesteps=int(cfg.model.timesteps),
        feature_dim=int(cfg.model.feature_dim),
        num_heads=int(cfg.model.num_heads),
        antithetic_time_sampling=bool(cfg.model.antithetic_time_sampling),
        n_layers=int(cfg.model.n_layers),
        n_dit_layers=int(cfg.model.n_dit_layers),
        dit_num_heads=int(cfg.model.dit_num_heads),
        dit_hidden_size=int(cfg.model.dit_hidden_size),
        ch_mult=tuple(cfg.model.ch_mult),
        vocab_size=vocab_size,
        noise_schedule_type=str(cfg.model.noise_schedule_type),
        dropout_rate=float(cfg.model.dropout_rate),
        use_attn_dropout=bool(cfg.model.use_attn_dropout),
        mlp_type=str(cfg.model.mlp_type),
        depth_scaled_init=bool(cfg.model.depth_scaled_init),
        cond_type=str(cfg.model.cond_type),
        outside_embed=bool(cfg.model.outside_embed),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="auto",
            adm_num_res_blocks_default=2,
            adm_attention_resolutions_default=(2, 4, 8),
            adm_num_heads_default=4,
            adm_num_head_channels_default=-1,
        ),
        time_features=str(cfg.model.time_features),
        classes=int(cfg.model.classes),
        sampler=sampler_method(cfg, default="ancestral", allow_sampler_alias=True),
        sampling_grid=sampling_grid(cfg, default="cosine"),
        topp=float(cfg.get("sampler", {}).get("topp", cfg.model.get("topp", 0.98))),
        categorical_sampling_policy=categorical_sampling_policy(cfg),
        model_sharding=bool(cfg.model.model_sharding),
    )
