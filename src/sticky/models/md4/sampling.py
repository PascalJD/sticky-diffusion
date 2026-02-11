from __future__ import annotations

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp


def get_attr(train_state: Any, key: str):
    # Mirror MD4 repo helper; supports dataclass or dict-like.
    if hasattr(train_state, key):
        return getattr(train_state, key)
    return train_state[key]


def _get_params(train_state: Any, *, use_ema: bool = True):
    if use_ema:
        ema = get_attr(train_state, "ema_params")
        if ema is not None:
            return ema
    return get_attr(train_state, "params")


def simple_generate(
    rng: jax.Array,
    train_state: Any,
    *,
    model: Any,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
):
    """Single-device generate. JIT-friendly."""
    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    T = int(model.timesteps if timesteps is None else timesteps)

    # Prior sample (all mask tokens for MD4)
    zt = model.apply(variables, batch_size, method=model.prior_sample)

    rng, sub_rng = jax.random.split(rng)

    def body_fn(i, z):
        return model.apply(
            variables,
            sub_rng,
            i,
            T,
            z,
            conditioning=conditioning,
            method=model.sample_step,
        )

    z0 = jax.lax.fori_loop(0, T, body_fn, zt)

    # Replace any remaining mask tokens
    return model.apply(
        variables,
        z0,
        conditioning=conditioning,
        method=model.decode,
    )


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
def generate(
    model: Any,
    train_state: Any,
    rng: jax.Array,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
):
    """Multi-device generate (per-device batch_size)."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    T = int(model.timesteps if timesteps is None else timesteps)

    zt = model.apply(variables, batch_size, method=model.prior_sample)

    rng, sub_rng = jax.random.split(rng)

    def body_fn(i, z):
        return model.apply(
            variables,
            sub_rng,
            i,
            T,
            z,
            conditioning=conditioning,
            method=model.sample_step,
        )

    z0 = jax.lax.fori_loop(0, T, body_fn, zt)

    return model.apply(
        variables,
        z0,
        conditioning=conditioning,
        method=model.decode,
    )
