from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


PRNGKey = jax.Array


def make_rng(seed: int) -> PRNGKey:
    return jax.random.key(int(seed))


def is_legacy_prng_key(key: Any) -> bool:
    shape = getattr(key, "shape", None)
    dtype = getattr(key, "dtype", None)
    try:
        return (shape == (2,)) and (jnp.dtype(dtype) == jnp.uint32)
    except Exception:
        return False


def ensure_prng_key(key: Any) -> PRNGKey:
    if is_legacy_prng_key(key):
        return jax.random.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))
    return key


def legacy_prng_key_data(key: Any) -> jax.Array:
    return jax.random.key_data(ensure_prng_key(key))
