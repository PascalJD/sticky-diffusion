# src/sticky/core/jump.py
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import jax, jax.numpy as jnp
from .sde_vp import alpha_sigma, _expand_like

Array = jnp.ndarray

@dataclass(frozen=True)
class GaussianJump:
    """Forward 'unstick' kernel R_a: N(a, std^2 I)."""
    std: float = 0.05
    clip: float | None = None
    beta: Any | None = None  # Not used
    
    @property
    def var(self) -> float:
        return float(self.std ** 2)

    def sample(self, key: jax.random.PRNGKey, anchor: Array, t: Array) -> Array:
        eps = jax.random.normal(key, shape=anchor.shape) * self.std
        y = anchor + eps
        if self.clip is not None:
            y = jnp.clip(y, -self.clip, self.clip)
        return y

    def logpdf(self, y: Array, anchor: Array, t: Array = None) -> Array:
        diff = y - anchor
        dim = diff.shape[-1]
        m = jnp.sum(jnp.square(diff), axis=-1) / (self.var + 1e-12)
        logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi * (self.var + 1e-12))
        return -(logZ + 0.5 * m)

    def pdf(self, y: Array, anchor: Array, t: Array = None) -> Array:
        return jnp.exp(self.logpdf(y, anchor, t))


@dataclass(frozen=True)
class VPMatchedGaussianJump:
    """r_t(y|a) = N(mu_r(t,a), tau(t)^2 I) with mu_r(t,a)=alpha(t)a, tau(t)=eta*sigma(t)."""
    beta: object
    eta: float = 0.7  # <1 => state dependence, =1 => r=q
    std_floor: float = 1e-3
    clip: float | None = None

    def _mean_std(self, anchor: Array, t: Array) -> tuple[Array, Array]:
        t = jnp.asarray(t, dtype=jnp.float32)
        alpha, sigma = alpha_sigma(self.beta, t)
        alpha = _expand_like(alpha, anchor)
        sigma = _expand_like(sigma, anchor)
        mean = alpha * anchor
        std = jnp.maximum(self.eta * sigma, self.std_floor)
        return mean, std

    def sample(self, key: jax.random.PRNGKey, anchor: Array, t: Array) -> Array:
        mean, std = self._mean_std(anchor, t)
        eps = jax.random.normal(key, shape=anchor.shape) * std
        y = mean + eps
        if self.clip is not None:
            y = jnp.clip(y, -self.clip, self.clip)
        return y

    def logpdf(self, y: Array, anchor: Array, t: Array) -> Array:
        mean, std = self._mean_std(anchor, t)
        var = std * std
        diff = y - mean
        dim = diff.shape[-1]
        m = jnp.sum((diff * diff) / (var + 1e-12), axis=-1)
        logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi) \
            + 0.5 * dim * jnp.log(var[..., 0] + 1e-12)
        return -(logZ + 0.5 * m)
