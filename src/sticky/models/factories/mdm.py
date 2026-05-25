from __future__ import annotations

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from sticky.models._registry import register_init


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
