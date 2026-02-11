from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .sdes import alpha_sigma, _expand_like

Array = jnp.ndarray


@dataclass(frozen=True)
class VPMatchedGaussianJump:
    """VP-matched Gaussian jump proposal.

    r_t(y|a) = N(mu_r(t,a), tau(t)^2 I)
    mu_r(t,a) = alpha(t) * a
    tau(t) = eta * sigma(t)

    Parameters
    ----------
    beta:
        VP beta schedule (callable with attributes beta_min, beta_max, T, ...).
    eta:
        Proposal width factor.
    std_floor:
        Lower bound for tau(t) for numerical stability.
    clip:
        Optional clipping of sampled y.
    """

    beta: object
    eta: float = 0.7
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
        logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi)\
         + 0.5 * dim * jnp.log(var[..., 0] + 1e-12)
        return -(logZ + 0.5 * m)
