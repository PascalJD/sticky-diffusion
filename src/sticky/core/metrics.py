from __future__ import annotations

import jax.numpy as jnp


def scale_bpd_components(loss_dict, data_shape):
    """Scale any keys named ``*loss*`` or ``*bpd*`` from nats/sample to bits/dim.

    Models historically emit per-sample cross-entropy under ``loss*`` names;
    the renamed MD4/CADD paths emit ``bpd*``. Both naming conventions are
    accepted so the scaler can be shared across families without a flag.
    """
    seq_len = jnp.prod(jnp.asarray(data_shape))
    scale = 1.0 / (seq_len * jnp.log(2.0))
    return {
        key: value * scale if ("loss" in key or "bpd" in key) else value
        for key, value in loss_dict.items()
    }
