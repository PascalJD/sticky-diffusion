# src/sticky/models/factories/d3pm.py
"""Builder for the D3PM model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    categorical_sampling_policy,
    sampler_method,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    from sticky.models.baselines.d3pm.d3pm_model import D3PM

    return D3PM(
        data_shape=data_shape,
        timesteps=int(cfg.model.get("timesteps", 256)),
        transition_type=str(cfg.model.get("transition_type", "gaussian")),
        transition_beta_schedule=str(
            cfg.model.get("transition_beta_schedule", "linear")
        ),
        beta_start=float(cfg.model.get("beta_start", 1e-4)),
        beta_end=float(cfg.model.get("beta_end", 2e-2)),
        cosine_s=float(cfg.model.get("cosine_s", 0.008)),
        max_beta=float(cfg.model.get("max_beta", 0.999)),
        auxiliary_loss_weight=float(cfg.model.get("auxiliary_loss_weight", 1e-3)),
        absorbing_state=int(cfg.model.get("absorbing_state", 128)),
        feature_dim=int(cfg.model.get("feature_dim", 96)),
        num_heads=int(cfg.model.get("num_heads", 12)),
        antithetic_time_sampling=bool(cfg.model.get("antithetic_time_sampling", True)),
        n_layers=int(cfg.model.get("n_layers", 32)),
        n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
        dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
        ch_mult=tuple(cfg.model.get("ch_mult", (3, 4, 4))),
        vocab_size=vocab_size,
        dropout_rate=float(cfg.model.get("dropout_rate", 0.1)),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", True)),
        cond_type=str(cfg.model.get("cond_type", "adaln_zero")),
        outside_embed=bool(cfg.model.get("outside_embed", False)),
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
        sampler=sampler_method(cfg, default="ancestral", allow_sampler_alias=True),
        sampling_grid=sampling_grid(cfg, default="uniform"),
        categorical_sampling_policy=categorical_sampling_policy(cfg),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
    )
