# src/sticky/models/factories/candi.py
"""Builder for the CANDI model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    categorical_sampling_policy,
    sampler_method,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    from sticky.models.baselines.candi.candi_model import CANDI

    sampler_cfg = cfg.get("sampler", {})
    return CANDI(
        data_shape=data_shape,
        vocab_size=vocab_size,
        cont_time=bool(cfg.model.get("cont_time", True)),
        timesteps=int(cfg.model.get("timesteps", 256)),
        representation=str(cfg.model.get("representation", "embed")),
        experimental=bool(cfg.model.get("experimental", True)),
        alpha_schedule_type=str(cfg.model.get("alpha_schedule_type", "linear")),
        schedule_eps=float(cfg.model.get("schedule_eps", 0.0)),
        pure_continuous=bool(cfg.model.get("pure_continuous", False)),
        use_percentile_scheduling=bool(
            cfg.model.get("use_percentile_scheduling", True)
        ),
        min_percentile=float(cfg.model.get("min_percentile", 0.01)),
        max_percentile=float(cfg.model.get("max_percentile", 0.45)),
        sigma_min=float(cfg.model.get("sigma_min", 0.2)),
        sigma_max=float(cfg.model.get("sigma_max", 4.0)),
        ode_step_scale=float(cfg.model.get("ode_step_scale", 1.0)),
        feature_dim=int(cfg.model.get("feature_dim", 96)),
        num_heads=int(cfg.model.get("num_heads", 12)),
        n_layers=int(cfg.model.get("n_layers", 32)),
        n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
        dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
        ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
        cond_type=str(cfg.model.get("cond_type", "adaln")),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="adm_unet5d",
            adm_num_res_blocks_default=4,
            adm_attention_resolutions_default=(2, 4),
            adm_num_heads_default=4,
            adm_num_head_channels_default=64,
        ),
        time_features=str(cfg.model.get("time_features", "t")),
        classes=int(cfg.model.get("classes", -1)),
        sampler=sampler_method(cfg, default="hybrid_cache"),
        sampling_grid=sampling_grid(cfg, default="cosine"),
        categorical_sampling_policy=categorical_sampling_policy(cfg),
        guidance_scale=float(
            sampler_cfg.get("guidance_scale", cfg.model.get("guidance_scale", 0.0))
        ),
    )
