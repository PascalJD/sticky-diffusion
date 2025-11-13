# src/sticky/core/jump.py
from __future__ import annotations
from dataclasses import dataclass
import jax, jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True)
class GaussianJump:
    """Forward 'unstick' kernel R_a: N(a, std^2 I)."""
    std: float = 0.05
    clip: float | None = None
    
    @property
    def var(self) -> float:
        return float(self.std ** 2)

    def sample(self, key: jax.random.PRNGKey, anchor: Array) -> Array:
        eps = jax.random.normal(key, shape=anchor.shape) * self.std
        y = anchor + eps
        if self.clip is not None:
            y = jnp.clip(y, -self.clip, self.clip)
        return y

    def logpdf(self, y: Array, anchor: Array) -> Array:
        diff = y - anchor
        dim = diff.shape[-1]
        m = jnp.sum(jnp.square(diff), axis=-1) / (self.var + 1e-12)
        logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi * (self.var + 1e-12))
        return -(logZ + 0.5 * m)

    def pdf(self, y: Array, anchor: Array) -> Array:
        return jnp.exp(self.logpdf(y, anchor))