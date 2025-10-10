# src/sticky/core/eventgen.py
from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Tuple

from .hazard import sample_unstick_time_truncated
from .anchors import CoordAnchors

Array = jnp.ndarray

def _sample_mark(key: jax.random.PRNGKey, log_m0: Array) -> Tuple[int, int]:
    d, L = log_m0.shape
    key_i, key_l = jax.random.split(key)
    i = int(jax.random.randint(key_i, (), 0, d))
    ell = int(jax.random.categorical(key_l, log_m0[i]))
    return i, ell

def build_events_coord(
    key: jax.random.PRNGKey,
    d: int,
    L: int,
    log_m0: Array,
    anchors: CoordAnchors,
    beta_fn,
    c: float,
    T: float,
    M: int,
) -> Tuple[Array, Array, Array]:
    taus = []
    ys = []
    ms = []
    k = 0
    kgen = key
    while k < M:
        kgen, k1, k2, k3 = jax.random.split(kgen, 4)
        i, ell = _sample_mark(k1, log_m0)
        t, _ = sample_unstick_time_truncated(k2, beta_fn, c, T=T)
        y = anchors.sample_R(k3, i, ell)
        taus.append(float(T) - float(t))
        ys.append(y)
        ms.append(jnp.array([i, ell], dtype=jnp.int32))
        k += 1
    tau = jnp.array(taus)
    y_evt = jnp.stack(ys, axis=0)
    marks = jnp.stack(ms, axis=0)
    return tau, y_evt, marks