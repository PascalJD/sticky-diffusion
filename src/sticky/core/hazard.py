from __future__ import annotations
import jax, jax.numpy as jnp
from typing import Callable, Tuple

Array = jnp.ndarray

def make_lambda(
    beta_fn: Callable[[Array], Array], c: float = 2.0
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
    H_T = integrated_lambda(beta_fn, c, jnp.asarray(T))
    return jnp.exp(-H_T)

def _invert_cumhazard_linear(
    y: Array, beta_min: float, beta_diff: float, T: float, c: float
) -> Array:
    """Solve y = c * (beta_min t + 0.5 * (beta_diff/T) t^2) for t in [0, T]."""
    eps = 1e-12
    if abs(beta_diff) < 1e-12:
        return y / (c * (beta_min + eps))
    a = 0.5 * c * (beta_diff / T)
    b = c * beta_min
    disc = jnp.maximum(0.0, b * b + 4.0 * a * y)
    t = (-b + jnp.sqrt(disc)) / (2.0 * a + 1e-18)
    return t

def sample_unstick_time(
    key: jax.random.PRNGKey,
    beta_fn: Callable[[Array], Array],
    c: float,
    T: float = 1.0
) -> Tuple[Array, Array]:
    from .sde_vp import B_of_t
    T_arr = jnp.asarray(T)
    H_T = c * B_of_t(beta_fn, jnp.asarray(T_arr))
    S_T = jnp.exp(-H_T)
    u = jax.random.uniform(key, ())
    y = -jnp.log1p(-u * (1.0 - jnp.exp(-H_T)))
    t = _invert_cumhazard_linear(
        y, beta_fn.beta_min, beta_fn.beta_diff, beta_fn.T, c
    )
    t = jnp.minimum(t, jnp.nextafter(jnp.asarray(T_arr), 0.0))
    return t, jnp.array(True)