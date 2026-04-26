from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp

from .sdes import _expand_like, alpha_sigma

Array = jnp.ndarray


def mixture_variance(
    beta: Callable[[Array], Array],
    t: Array,
    tau: Array,
    eta: float,
    std_floor: float = 1e-3,
) -> Array:
    """Variance v_t(tau) of the VP-matched mixture component
    (r_tau * K_{tau -> t})( . | a) = N( . ; alpha(t) a, v_t(tau) I_d).

    v_t(tau) = sigma^2(t) - (1 - eta^2) * (alpha^2(t) / alpha^2(tau)) * sigma^2(tau)

    For eta < 1, v_t(tau) is **strictly decreasing** in tau on (0, t), with
    v_t(0+) = sigma^2(t) (an early unstick has had the full VP semigroup to
    wash out the variance deficit) and v_t(t-) = eta^2 * sigma^2(t) (a
    particle that just unstuck has accumulated only the un-sticking variance).
    For eta = 1, v_t(tau) = sigma^2(t) is constant in tau.

    The result has the broadcast shape of `t` and `tau`. Floored to
    `std_floor**2` so downstream `sqrt`/`log` are safe when the schedule
    drives the closed-form variance below float32 resolution.
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    eta_arr = jnp.asarray(eta, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha_tau, sigma_tau = alpha_sigma(beta, tau)
    deficit = (1.0 - jnp.square(eta_arr)) * jnp.square(
        alpha_t / alpha_tau
    ) * jnp.square(sigma_tau)
    var = jnp.square(sigma_t) - deficit
    floor = jnp.square(jnp.asarray(std_floor, dtype=jnp.float32))
    return jnp.maximum(var, floor)


def mixture_component_mean_std(
    anchor: Array,
    t: Array,
    tau: Array,
    beta: Callable[[Array], Array],
    eta: float,
    std_floor: float = 1e-3,
) -> Tuple[Array, Array]:
    """Mean and isotropic std of the VP-matched mixture component at (t, tau).

    Mirrors `VPMatchedGaussianJump._mean_std`: `mean` broadcasts to
    `anchor.shape`, `std` keeps a trailing singleton axis so it broadcasts
    against feature-dim residuals.
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    alpha_t, _ = alpha_sigma(beta, t)
    alpha_t = _expand_like(alpha_t, anchor)
    mean = alpha_t * anchor
    var = mixture_variance(beta, t, tau, eta, std_floor=std_floor)
    std = jnp.sqrt(var)
    std = _expand_like(std, anchor)
    return mean, std


def mixture_component_logpdf(
    y: Array,
    anchor: Array,
    t: Array,
    tau: Array,
    beta: Callable[[Array], Array],
    eta: float,
    std_floor: float = 1e-3,
) -> Array:
    """log N(y; alpha(t) anchor, v_t(tau) I_d) for the VP-matched mixture
    component. Matches the broadcasting of `vp_logpdf` in `sdes.py`.
    """
    mean, std = mixture_component_mean_std(
        anchor, t, tau, beta, eta, std_floor=std_floor
    )
    diff = y - mean
    dim = diff.shape[-1]
    var = jnp.square(std)
    mahal = jnp.sum(jnp.square(diff) / (var + 1e-12), axis=-1)
    log_det = dim * jnp.log(2.0 * jnp.pi) + dim * jnp.log(var[..., 0] + 1e-12)
    return -0.5 * (log_det + mahal)
