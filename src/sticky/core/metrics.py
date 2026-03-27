from __future__ import annotations

import jax.numpy as jnp


def scale_loss_metrics_to_bits(loss_dict, data_shape):
    seq_len = jnp.prod(jnp.asarray(data_shape))
    scale = 1.0 / (seq_len * jnp.log(2.0))
    return {
        key: value * scale if "loss" in key else value
        for key, value in loss_dict.items()
    }
