# src/sticky/core/anchors.py
from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Tuple, Dict, Any

Array = jnp.ndarray


class CoordAnchors:
    """
    Factorized per-coordinate anchors:
    A mark is (i, ell). Reverse jump sets y[i] <- b[ell].
    R_{i,ell} is Gaussian around that partially snapped vector.
    """
    def __init__(
        self, 
        d: int, 
        bins: Array, 
        log_m0: Array, 
        sigma_R: float, 
        mu_rest: Array|None=None
    ):
        self.d = d
        self.bins = bins
        self.L = bins.shape[0]
        self.log_m0 = log_m0  # (d,L)
        self.sigma_R = sigma_R
        self.mu_rest = jnp.zeros((d,)) if mu_rest is None else mu_rest

    def make_mean(self, i: int, ell: int) -> Array:
        m = self.mu_rest.copy()
        m = m.at[i].set(self.bins[ell])
        return m

    def log_r_mark(self, y: Array, i: int, ell: int) -> Array:
        m = self.make_mean(i, ell)
        diff = y - m
        q = jnp.sum(diff**2)
        d = y.shape[0]
        const = -0.5 * d * jnp.log(2*jnp.pi*self.sigma_R**2 + 1e-12)
        return const - 0.5 * q / (self.sigma_R**2 + 1e-12)

    def posterior_logits(self, y: Array, log_m_t: Array) -> Array:
        """
        Returns per-pixel logits over bins: 
        logits[i, ell] \propto log m_t[i,ell] + log r_{i,ell}(y).
        """
        def logits_i(i):
            m = jnp.stack(
                [self.log_r_mark(y, i, ell) for ell in range(self.L)], 
                axis=0
            )
            return log_m_t[i] + m  # (L,)
        return jax.vmap(logits_i)(jnp.arange(self.d))  # (d,L)

    def sample_R(self, key: jax.random.PRNGKey, i: int, ell: int) -> Array:
        m = self.make_mean(i, ell)
        z = jax.random.normal(key, m.shape)
        return m + self.sigma_R * z