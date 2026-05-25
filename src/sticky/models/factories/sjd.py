from __future__ import annotations

import jax.numpy as jnp
from omegaconf import DictConfig

from sticky.models._registry import register_init


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
