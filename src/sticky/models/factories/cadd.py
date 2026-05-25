# src/sticky/models/factories/cadd.py
"""Builder for the CADD model family."""
from __future__ import annotations

from omegaconf import DictConfig

from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    categorical_sampling_policy,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    from sticky.models.baselines.cadd.cadd_model import CADD

    sampler_cfg = cfg.get("sampler", {})
    latent_cfg = cfg.model.get("cadd_latent", {})
    return CADD(
        data_shape=data_shape,
        vocab_size=vocab_size,
        cont_time=bool(cfg.model.get("cont_time", True)),
        timesteps=int(cfg.model.get("timesteps", 512)),
        antithetic_time_sampling=bool(cfg.model.get("antithetic_time_sampling", False)),
        discrete_schedule_type=str(cfg.model.get("discrete_schedule_type", "linear")),
        continuous_schedule_type=str(
            latent_cfg.get(
                "continuous_schedule_type",
                cfg.model.get("continuous_schedule_type", "linear"),
            )
        ),
        continuous_latent_type=str(
            latent_cfg.get(
                "type",
                cfg.model.get("continuous_latent_type", "gaussian"),
            )
        ),
        schedule_eps=float(cfg.model.get("schedule_eps", 1e-4)),
        feature_dim=int(cfg.model.get("feature_dim", 128)),
        num_heads=int(cfg.model.get("num_heads", 12)),
        n_layers=int(cfg.model.get("n_layers", 32)),
        n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
        dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
        ch_mult=tuple(cfg.model.get("ch_mult", (1, 2, 2, 2))),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.0)),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
        cond_type=str(cfg.model.get("cond_type", "adaln")),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="auto",
            adm_num_res_blocks_default=2,
            adm_attention_resolutions_default=(2, 4, 8),
            adm_num_heads_default=4,
            adm_num_head_channels_default=-1,
        ),
        time_features=str(cfg.model.get("time_features", "t")),
        classes=int(cfg.model.get("classes", -1)),
        sampling_grid=sampling_grid(cfg, default="cosine"),
        temperature_schedule=str(
            sampler_cfg.get(
                "temperature_schedule",
                cfg.model.get("temperature_schedule", "cosine_decay"),
            )
        ),
        tau_max=float(sampler_cfg.get("tau_max", cfg.model.get("tau_max", 2.5))),
        logit_temperature=float(
            sampler_cfg.get(
                "logit_temperature",
                cfg.model.get("logit_temperature", 1.0),
            )
        ),
        z0_estimator=str(
            sampler_cfg.get("z0_estimator", cfg.model.get("z0_estimator", "hard"))
        ),
        K=int(sampler_cfg.get("K", cfg.model.get("K", 1))),
        force_decode_at_end=bool(
            sampler_cfg.get(
                "force_decode_at_end",
                cfg.model.get("force_decode_at_end", True),
            )
        ),
        categorical_sampling_policy=categorical_sampling_policy(cfg),
        remask_refine_enabled=bool(
            sampler_cfg.get(
                "remask_refine_enabled",
                cfg.model.get("remask_refine_enabled", False),
            )
        ),
        remask_refine_steps=int(
            sampler_cfg.get("remask_refine_steps", cfg.model.get("remask_refine_steps", 1))
        ),
        remask_refine_frac=float(
            sampler_cfg.get(
                "remask_refine_frac",
                cfg.model.get("remask_refine_frac", 0.0),
            )
        ),
        remask_refine_metric=str(
            sampler_cfg.get(
                "remask_refine_metric",
                cfg.model.get("remask_refine_metric", "entropy"),
            )
        ),
        remask_refine_sample_mode=str(
            sampler_cfg.get(
                "remask_refine_sample_mode",
                cfg.model.get("remask_refine_sample_mode", "sample"),
            )
        ),
    )
