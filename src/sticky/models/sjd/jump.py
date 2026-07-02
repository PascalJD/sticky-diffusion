from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import alpha_sigma, _expand_like

Array = jnp.ndarray


@dataclass(frozen=True, eq=False)
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
    blur_kernel: Array | None = None   # (N, N) site-blending matrix; None == identity.
    # Where blur_kernel acts (inert when blur_kernel is None):
    #   "embedding" — mu = W @ E(X0), the legacy anchor-embedding blend;
    #   "value"     — mu = E(B v), blend raw pixel values then embed the
    #                 continuous blur (frozen sin8 CIFAR contract; validated
    #                 at task-build time). Consumers: corruption.sample_pair
    #                 and corruption.classifier_induced_score.
    blur_space: str = "embedding"

    def _mean_std(self, anchor: Array, t: Array) -> tuple[Array, Array]:
        t = jnp.asarray(t, dtype=jnp.float32)
        alpha, sigma = alpha_sigma(self.beta, t)
        alpha = _expand_like(alpha, anchor)
        sigma = _expand_like(sigma, anchor)
        mean = alpha * anchor
        std = jnp.maximum(self.eta * sigma, self.std_floor)
        return mean, std

    def sample(self, key: PRNGKey, anchor: Array, t: Array) -> Array:
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

    def apply_blur(self, anchor_field: Array) -> Array:
        """Apply the configured blur kernel to a per-site anchor EMBEDDING field.

        When blur_kernel is None this returns the input object unchanged
        (Python identity), guaranteeing bit-exact backward compatibility
        for the no-blur path. Otherwise dispatches to blur_means.

        NOTE: this is the blur_space="embedding" application only. Consumers
        that honor blur_space="value" (sample_pair, classifier_induced_score)
        branch BEFORE calling this; callers that cannot support value space
        must reject jumps with blur_space="value" + a kernel instead of
        silently blending embeddings (see sjd_elbo_loss.q_t_sample).
        """
        if self.blur_kernel is None:
            return anchor_field
        from .blur import blur_means
        return blur_means(anchor_field, self.blur_kernel)
