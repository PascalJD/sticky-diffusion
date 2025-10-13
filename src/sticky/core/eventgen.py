from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Tuple

from .hazard import sample_unstick_time

Array = jnp.ndarray

def _sample_mark(key: jax.random.PRNGKey, log_m0: Array) -> Array:
    d, L = log_m0.shape
    k_i, k_l = jax.random.split(key)
    i = jax.random.randint(k_i, shape=(), minval=0, maxval=d)
    ell = jax.random.categorical(k_l, log_m0[i])
    return jnp.stack([i.astype(jnp.int32), ell.astype(jnp.int32)], axis=0)

def _sample_R_given(
    key: jax.random.PRNGKey, 
    mu_rest: Array, 
    bins: Array, 
    sigma_R: float,
    i: Array, 
    ell: Array
) -> Array:
    m = mu_rest.at[i].set(bins[ell])
    z = jax.random.normal(key, shape=mu_rest.shape)
    return m + sigma_R * z

def sample_events_coord(
    key: jax.random.PRNGKey,
    M: int,
    log_m0: Array, 
    bins: Array,
    mu_rest: Array, 
    sigma_R: float,
    beta_fn,
    c: float,
    T: float,
) -> Tuple[Array, Array, Array]:
    keys = jax.random.split(key, M * 4).reshape(M, 4, -1)

    def one(krow):
        k1, k2, k3, _ = krow[0], krow[1], krow[2], krow[3]
        mark = _sample_mark(k1, log_m0)  # (2,)
        i, ell = mark[0], mark[1]
        t, _ = sample_unstick_time(k2, beta_fn, c, T=T)
        y = _sample_R_given(k3, mu_rest, bins, sigma_R, i, ell)
        tau = T - t
        return tau, y, mark

    tau, y_evt, marks = jax.vmap(one)(keys)
    return tau, y_evt, marks

sample_events_coord_jit = jax.jit(
    sample_events_coord, static_argnames=("M", "beta_fn")
)