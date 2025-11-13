# src/sticky/core/sde_vp.py
from __future__ import annotations
from typing import Callable, Tuple
import jax, jax.numpy as jnp

Array = jnp.ndarray

def make_beta(
    beta_min: float, beta_max: float, T: float = 1.0
) -> Callable[[Array], Array]:
    """Linear beta(t) = beta_min + (beta_max - beta_min) * t / T, t in [0, T]."""
    slope = (beta_max - beta_min) / T
    def beta(t: Array) -> Array:
        t = jnp.asarray(t, dtype=jnp.float32)
        return beta_min + slope * t
    # annotate for downstream closed forms
    beta.kind = "linear"
    beta.beta_min = float(beta_min)
    beta.beta_max = float(beta_max)
    beta.beta_diff = float(beta_max - beta_min)
    beta.T = float(T)
    beta.slope = float(slope)
    return beta

def B_of_t(beta: Callable[[Array], Array], t: Array) -> Array:
    """Closed-form B(t) = \int_0^t beta(s) ds."""
    t = jnp.asarray(t, dtype=jnp.float32)
    bmin = beta.beta_min
    bdiff = beta.beta_diff
    TT = beta.T
    return bmin * t + 0.5 * (bdiff / TT) * (t**2)

def alpha_sigma(beta: Callable[[Array], Array], t: Array) -> Tuple[Array, Array]:
    # Perturbation kernel, (33) in Song et al.
    # N(\alpha(t)x_0,\sigma^2(t)I),
    B = B_of_t(beta, t)
    alpha = jnp.exp(-0.5 * B)
    sigma2 = jnp.clip(1.0 - jnp.exp(-B), a_min=1e-12)
    return alpha, jnp.sqrt(sigma2)

def _expand_like(v: Array, like: Array) -> Array:
    """Add trailing singleton dims so v broadcasts with like."""
    while v.ndim < like.ndim:
        v = v[..., None]
    return v

def vp_perturb(
    key: jax.random.PRNGKey,
    x0: Array,
    t: Array,
    beta: Callable[[Array], Array],
) -> Tuple[Array, Array]:
    """Sample X_t | X_0 under VP and return the DSM score target"""
    t = jnp.asarray(t, dtype=jnp.float32)
    alpha, sigma = alpha_sigma(beta, t)
    alpha = _expand_like(alpha, x0)
    sigma = _expand_like(sigma, x0)
    z = jax.random.normal(key, shape=x0.shape)
    xt = alpha * x0 + sigma * z
    target_score = -(xt - alpha * x0) / (jnp.square(sigma) + 1e-12)
    return xt, target_score

def vp_logpdf(xt: Array, x0: Array, t: Array, beta) -> Array:
    t = jnp.asarray(t, dtype=jnp.float32)
    alpha, sigma = alpha_sigma(beta, t)
    alpha = _expand_like(alpha, xt) 
    sigma = _expand_like(sigma, xt)
    diff = xt - alpha * x0
    dim = diff.shape[-1]
    var = jnp.square(sigma)
    mahal = jnp.sum(jnp.square(diff) / (var + 1e-12), axis=-1)
    log_det = dim * jnp.log(2 * jnp.pi) + dim * jnp.log(var[..., 0] + 1e-12)
    return -0.5 * (log_det + mahal)