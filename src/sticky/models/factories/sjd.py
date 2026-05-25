# src/sticky/models/factories/sjd.py
"""Builder and init function for the SJD model family."""
from __future__ import annotations

import jax.numpy as jnp
from omegaconf import DictConfig

from sticky.models._registry import register_init
from sticky.models.factories._helpers import adm_backbone_kwargs


def build_model(cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int):
    del data_shape

    from sticky.models.sjd.anchors import (
        anchor_learnable_from_mapping,
        anchor_table_config_from_mapping,
    )
    from sticky.models.sjd.freq_weighting import (
        hazard_weighting_mode,
        load_anchor_log_w,
    )
    from sticky.models.sjd.sjd_model import SJD

    hw_cfg = cfg.get("forward", {}).get("hazard_weighting", None)
    hw_mode = hazard_weighting_mode(hw_cfg)
    learnable_log_w = hw_mode == "learned"
    log_w_init = None
    if learnable_log_w:
        init = str(getattr(hw_cfg, "init", "zeros"))
        if init == "frequency":
            loaded = load_anchor_log_w(hw_cfg, vocab_size=int(vocab_size))
            if loaded is None:
                raise ValueError(
                    "hazard_weighting=learned init=frequency requires "
                    "enabled=true and a valid log_w_path."
                )
            # Match the runtime mean-zero re-centering so the at-init forward
            # pass reproduces the frozen `frequency` path bit-exactly.
            log_w_init = loaded - jnp.mean(loaded)

    return SJD(
        vocab_size=vocab_size,
        anchor_config=anchor_table_config_from_mapping(
            cfg.model,
            vocab_size=vocab_size,
        ),
        learnable_anchors=anchor_learnable_from_mapping(cfg.model, default=True),
        learnable_log_w=learnable_log_w,
        log_w_init=log_w_init,
        use_noisy_input_bias=bool(cfg.model.get("use_noisy_input_bias", False)),
        feature_dim=int(cfg.model.feature_dim),
        n_layers=int(cfg.model.n_layers),
        n_dit_layers=int(cfg.model.n_dit_layers),
        dit_num_heads=int(cfg.model.dit_num_heads),
        dit_hidden_size=int(cfg.model.dit_hidden_size),
        ch_mult=tuple(cfg.model.ch_mult),
        num_heads=int(cfg.model.num_heads),
        dropout_rate=float(cfg.model.dropout_rate),
        use_attn_dropout=bool(cfg.model.get("use_attn_dropout", True)),
        mlp_type=str(cfg.model.get("mlp_type", "swiglu")),
        depth_scaled_init=bool(cfg.model.get("depth_scaled_init", False)),
        cond_type=str(cfg.model.get("cond_type", "adaln")),
        model_sharding=bool(cfg.model.get("model_sharding", False)),
        sequence_backbone=str(cfg.model.get("sequence_backbone", "auto")),
        sequence_mlp_hidden_dim=cfg.model.get("sequence_mlp_hidden_dim", None),
        sequence_max_length=cfg.model.get("sequence_max_length", None),
        sequence_causal=bool(cfg.model.get("sequence_causal", False)),
        **adm_backbone_kwargs(
            cfg.model,
            image_backbone_default="auto",
            adm_num_res_blocks_default=2,
            adm_attention_resolutions_default=(2, 4, 8),
            adm_num_heads_default=4,
            adm_num_head_channels_default=-1,
        ),
    )


@register_init("sjd")
def init_sjd(model, cfg: DictConfig, rng) -> dict:
    # rng is used directly as the params key (no further split needed for this
    # family; the caller in init_state has already split off init_rng).
    rng_params = rng
    batch_size = max(int(cfg.dataset.per_device_batch_size), 1)
    anchor_dim = int(model.anchor_config.anchor_dim)
    dummy_z = jnp.zeros(
        (batch_size,) + tuple(cfg.dataset.data_shape) + (anchor_dim,),
        dtype=jnp.float32,
    )
    dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)
    dummy_ids = jnp.zeros(tuple(dummy_z.shape[:-1]), dtype=jnp.int32)
    return model.init(
        {"params": rng_params}, dummy_z, dummy_t,
        anchor_token_ids=dummy_ids, train=False,
    )
