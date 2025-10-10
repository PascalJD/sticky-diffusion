# src/sticky/core/hazard.py
from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Callable, Tuple

Array = jnp.ndarray

def make_lambda(
    beta_fn: Callable[[Array], Array], c: float=2.0
) -> Callable[[Array], Array]:
    return lambda t: c * beta_fn(t)

def integrated_lambda(
    beta_fn: Callable[[Array], Array], c: float, t: Array
) -> Array:
    from .sde_vp import B_of_t
    return c * B_of_t(beta_fn, t)

def survival_prob(
    beta_fn: Callable[[Array], Array], c: float, T: float
) -> Array:
    """P(no unstick by T) exp(-c B(T))."""
    return jnp.exp(-integrated_lambda(beta_fn, c, jnp.asarray(T)))

def _invert_cumhazard_linear(
    y: Array, beta_min: float, beta_diff: float, T: float, c: float
) -> Array:
    """Solve y = c * \int_0^t (beta_min + (beta_diff/T) s) ds for t >= 0.
       Integral: c*(beta_min t + 0.5 * beta_diff * t^2 / T) = y.
    """
    eps = 1e-12
    if abs(beta_diff) < 1e-12:
        return y / (c * (beta_min + eps))
    a = 0.5 * c * (beta_diff / T)
    b = c * beta_min
    disc = jnp.maximum(0.0, b * b + 4.0 * a * y)
    return (-b + jnp.sqrt(disc)) / (2.0 * a + 1e-18)

def sample_unstick_time_truncated(
    key: jax.random.PRNGKey,
    beta_fn: Callable[[Array], Array],
    c: float,
    T: float = 1.0
) -> Tuple[Array, Array]:
    """Draw t with pdf f(t) = \lambda_t exp(-\int_0^t \lambda_t).
    Returns: (t, happened) where `happened` is a boolean mask (here always True by truncation).
    Implemented by inverse-CDF with rejection: resample until t<T.
    In practice the expected number of retries is 1 / (1 - exp(-c B(T))).
    """
    from .sde_vp import B_of_t
    log_tail = -integrated_lambda(beta_fn, c, T)
    umin = jnp.exp(log_tail) 
    u = jax.random.uniform(key, minval=float(umin) + 1e-12, maxval=1.0 - 1e-12)
    y = -jnp.log(1.0 - u)

    if hasattr(beta_fn, "kind") and getattr(beta_fn, "kind") == "linear":
        t = _invert_cumhazard_linear(
            y, 
            getattr(beta_fn, "beta_min"), 
            getattr(beta_fn, "beta_diff"), 
            getattr(beta_fn, "T"), 
            c  #
        )
        return jnp.minimum(t, T - 1e-12), jnp.array(True)

    # Fallback: Newton but now t<T by construction of `u`
    def body(t):
        from .sde_vp import B_of_t
        B = B_of_t(beta_fn, t)
        f = c * B - y
        beta_t = beta_fn(t)
        step = f / (c * beta_t + 1e-8)
        return jnp.clip(t - step, 0.0, jnp.nextafter(T, 0.0))

    t = jnp.asarray(0.5 * T)
    for _ in range(32):
        t = body(t)
    return t, jnp.array(True)