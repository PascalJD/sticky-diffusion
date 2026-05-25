# src/sticky/models/factories/mdm.py
"""Builder and init function for the MDM (inpaint) model family."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from sticky.models._registry import register_init
from sticky.models.factories._helpers import (
    adm_backbone_kwargs,
    categorical_sampling_policy,
    sampler_method,
    sampling_grid,
)


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    from sticky.models.baselines.mdm.mdm_inpaint_model import MDMInpaint

    return MDMInpaint(
        data_shape=data_shape,
        cont_time=bool(cfg.model.get("cont_time", False)),
        timesteps=int(cfg.model.get("timesteps", 50)),
        feature_dim=int(cfg.model.get("feature_dim", 128)),
        num_heads=int(cfg.model.get("num_heads", 12)),
        antithetic_time_sampling=bool(cfg.model.get("antithetic_time_sampling", False)),
        n_layers=int(cfg.model.get("n_layers", 3)),
        n_dit_layers=int(cfg.model.get("n_dit_layers", 0)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 12)),
        dit_hidden_size=int(cfg.model.get("dit_hidden_size", 768)),
        ch_mult=tuple(cfg.model.get("ch_mult", (1,))),
        vocab_size=vocab_size,
        noise_schedule_type=str(cfg.model.get("noise_schedule_type", "loglinear")),
        dropout_rate=float(cfg.model.get("dropout_rate", 0.0)),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "gelu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
        cond_type=str(cfg.model.get("cond_type", "adaln")),
        outside_embed=bool(cfg.model.get("outside_embed", False)),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "gpt2_like")),
        sequence_mlp_hidden_dim=cfg.model.get("sequence_mlp_hidden_dim", None),
        sequence_max_length=cfg.model.get("sequence_max_length", None),
        sequence_causal=bool(cfg.model.get("sequence_causal", False)),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="auto",
            adm_num_res_blocks_default=2,
            adm_attention_resolutions_default=(2, 4),
            adm_num_heads_default=4,
            adm_num_head_channels_default=-1,
        ),
        time_features=str(cfg.model.get("time_features", "none")),
        classes=int(cfg.model.get("classes", -1)),
        sampler=sampler_method(cfg, default="top_prob_margin", allow_sampler_alias=True),
        sampling_grid=sampling_grid(cfg, default="loglinear"),
        categorical_sampling_policy=categorical_sampling_policy(cfg, default="exact"),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
    )


@register_init("mdm")
def init_mdm(model, cfg: DictConfig, rng) -> dict:
    rng_params, rng_sample = jax.random.split(rng, 2)
    batch_size = max(int(cfg.dataset.per_device_batch_size), 1)
    data_shape = tuple(getattr(model, "data_shape", tuple(cfg.dataset.data_shape)))
    dummy_x = jnp.zeros((batch_size,) + data_shape, dtype=jnp.int32)
    dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)
    dummy_cond = (
        jnp.zeros((batch_size,), dtype=jnp.int32)
        if int(cfg.model.classes) > 0
        else None
    )
    return model.init(
        {"params": rng_params, "sample": rng_sample},
        dummy_x, dummy_t,
        cond=dummy_cond,
        train=False,
    )
