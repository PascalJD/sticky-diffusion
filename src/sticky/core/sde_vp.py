# src/sticky/core/sde_vp.py
from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Callable, Tuple

Array = jnp.ndarray

def make_beta_linear(
    beta_min: float, beta_max: float, T: float = 1.0
) -> Callable[[Array], Array]:
    slope = (beta_max - beta_min) / T

    def beta(t: Array) -> Array:
        return beta_min + slope * t
        
    beta.kind = "linear"      
    beta.beta_min = float(beta_min)
    beta.beta_max = float(beta_max)
    beta.beta_diff = float(beta_max - beta_min)
    beta.T = float(T)
    beta.slope = float(slope)
    return beta

def B_of_t(beta: Callable[[Array], Array], t: Array) -> Array:
    """Compute B(t) = \int_0^t beta(s) ds.
    """
    # closed-form for linear schedule
    if hasattr(beta, "kind") and getattr(beta, "kind") == "linear": 
        bmin = getattr(beta, "beta_min")
        bdiff = getattr(beta, "beta_diff")
        TT = getattr(beta, "T")
        t = jnp.asarray(t)
        return bmin * t + 0.5 * (bdiff / TT) * (t ** 2)
    s = jnp.linspace(0.0, jnp.maximum(1e-12, jnp.asarray(t)), 2048)
    vals = beta(s)
    return jnp.trapezoid(vals, s)

def alpha_sigma(
    beta: Callable[[Array], Array], t: Array
) -> Tuple[Array, Array]:
    """alpha(t) = exp(-0.5 * B(t)), sigma(t) = sqrt(1 - alpha^2)."""
    B = B_of_t(beta, t)
    alpha = jnp.exp(-0.5 * B)
    sigma2 = jnp.maximum(1e-12, 1.0 - alpha * alpha)
    return alpha, jnp.sqrt(sigma2)

def vp_perturb(
    key: jax.random.PRNGKey,
    x0: Array,
    t: Array,
    beta: Callable[[Array], Array]
) -> Tuple[Array, Array]:
    """Sample X_t | X_0 under VP, and return the DSM target.
    """
    alpha, sigma = alpha_sigma(beta, t)
    z = jax.random.normal(key, shape=x0.shape)
    xt = alpha * x0 + sigma * z
    target_score = -(xt - alpha * x0) / (sigma**2 + 1e-12)
    return xt, target_score