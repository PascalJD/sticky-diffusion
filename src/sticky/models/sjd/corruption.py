from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from sticky.rng import PRNGKey

from .convolution import mixture_component_logpdf, mixture_component_mean_std
from .hazard import HazardSchedule
from .jump import VPMatchedGaussianJump
from .sdes import _expand_like, alpha_sigma

Array = jnp.ndarray


def sample_pair(
    key: PRNGKey,
    x0_anchor: Array,
    t: Array,
    beta,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
) -> Tuple[Array, Array]:
    """Draw a SJD-corrupted pair (X_t, never_unstuck_mask) per Algorithm 1.

    For each site of each example, with probability S_a(t) (`hazard.surv(t)`)
    the particle has not unstuck by t and `xt` is set to `x0_anchor` (caller
    masks the loss). Otherwise, an unstick time `tau` is drawn from the
    truncated density lam(tau) S(tau) / (1 - S(t)) on (0, t) by inverse-CDF
    on `(0, hazard.cdf(t))`, and `X_t ~ N(alpha(t) a, v_t(tau) I)` is drawn
    from the closed-form mixture component.

    Parameters
    ----------
    x0_anchor : shape (B, *site_shape, d).
    t         : shape (B,).

    Returns
    -------
    xt : shape (B, *site_shape, d).
    never_unstuck_mask : shape (B, *site_shape) (broadcast-compatible with x0_idx).
    """
    x0_anchor = jnp.asarray(x0_anchor, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    B = x0_anchor.shape[0]
    site_shape = x0_anchor.shape[1:-1]
    feat_ndim = 1  # last axis is the feature dim d.

    eta = float(jump.eta)
    std_floor = float(jump.std_floor)

    k_unstuck, k_tau, k_normal = jax.random.split(key, 3)

    surv_t = jnp.asarray(hazard.surv(t), dtype=jnp.float32)  # (B,)
    cdf_t = jnp.asarray(hazard.cdf(t), dtype=jnp.float32)  # (B,)
    site_pad = (1,) * len(site_shape)
    surv_t_b = jnp.broadcast_to(
        surv_t.reshape((B,) + site_pad), (B,) + site_shape
    )
    cdf_t_b = jnp.broadcast_to(
        cdf_t.reshape((B,) + site_pad), (B,) + site_shape
    )
    t_b = jnp.broadcast_to(t.reshape((B,) + site_pad), (B,) + site_shape)

    u_unstuck = jax.random.uniform(
        k_unstuck, shape=(B,) + site_shape, dtype=jnp.float32
    )
    never_unstuck_mask = u_unstuck < surv_t_b

    u_tau = jax.random.uniform(
        k_tau, shape=(B,) + site_shape, dtype=jnp.float32,
        minval=0.0, maxval=1.0,
    )
    tau = hazard.inv_cdf(u_tau * cdf_t_b)

    mean, std = mixture_component_mean_std(
        anchor=x0_anchor,
        t=t_b,
        tau=tau,
        beta=beta,
        eta=eta,
        std_floor=std_floor,
    )
    eps_n = jax.random.normal(k_normal, shape=x0_anchor.shape, dtype=jnp.float32)
    xt_unstuck = mean + std * eps_n

    mask_b = never_unstuck_mask.reshape((B,) + site_shape + (1,) * feat_ndim)
    xt = jnp.where(mask_b, x0_anchor, xt_unstuck)

    return xt, never_unstuck_mask


def _build_quadrature(
    t: Array, tau_grid_size: int, tau_grid: str
) -> Tuple[Array, Array]:
    """Build (nodes, log_w) of shape (tau_grid_size, B) for a per-example
    quadrature on (0, t)."""
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim != 1:
        raise ValueError(f"t must be shape (B,), got {t.shape}")
    if tau_grid == "uniform":
        i_arr = jnp.arange(tau_grid_size, dtype=jnp.float32) + 0.5
        nodes = (i_arr[:, None] / float(tau_grid_size)) * t[None, :]  # (tau_grid_size, B)
        log_h = jnp.log(t / float(tau_grid_size))  # (B,)
        log_w = jnp.broadcast_to(log_h[None, :], (tau_grid_size, t.shape[0]))
        return nodes, log_w
    if tau_grid == "simpson":
        if tau_grid_size % 2 == 0:
            raise ValueError(
                f"Simpson's rule requires odd tau_grid_size, got tau_grid_size={tau_grid_size}"
            )
        if tau_grid_size < 3:
            raise ValueError(
                f"Simpson's rule requires tau_grid_size >= 3, got tau_grid_size={tau_grid_size}"
            )
        i_arr = jnp.arange(tau_grid_size, dtype=jnp.float32)
        nodes = (i_arr[:, None] / float(tau_grid_size - 1)) * t[None, :]  # (tau_grid_size, B)
        w = jnp.ones((tau_grid_size,), dtype=jnp.float32)
        w = w.at[1:-1:2].set(4.0)
        w = w.at[2:-1:2].set(2.0)
        log_w_pat = jnp.log(w)  # (tau_grid_size,)
        log_h = jnp.log(t / float(tau_grid_size - 1))  # (B,)
        log_three = jnp.log(jnp.asarray(3.0, dtype=jnp.float32))
        log_w = log_w_pat[:, None] + log_h[None, :] - log_three
        return nodes, log_w
    raise ValueError(f"Unknown tau_grid={tau_grid!r}")


def _online_lse_step(
    m_carry: Array, l_carry: Array, log_x: Array
) -> Tuple[Array, Array]:
    """Single online log-sum-exp update.

    Carry semantics: at any point, ``logsumexp(stack_so_far) = m_carry + log(l_carry)``.
    Initialize ``m_carry = -inf`` and ``l_carry = 0``; after seeing all log_x_i,
    return ``m_final + log(l_final)``.
    """
    m_new = jnp.maximum(m_carry, log_x)
    # exp(-inf - finite) = 0 and 0 * 0 = 0, so the (-inf, 0) initial state
    # collapses to (log_x_0, 1) on the first step without producing NaNs.
    l_new = l_carry * jnp.exp(m_carry - m_new) + jnp.exp(log_x - m_new)
    return m_new, l_new


def mixture_logpdf(
    y: Array,
    anchor: Array,
    t: Array,
    beta,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    tau_grid_size: int = 32,
    tau_grid: str = "uniform",
) -> Array:
    """log p_t^ac(y | a) by 1-D log-sum-exp quadrature.

    Discretizes
        p_t^ac(y | a) = int_0^t lam(tau) S(tau) (r_tau * K_{tau->t})(y | a) dtau
    on a per-example tau-grid via streaming online log-sum-exp through
    ``jax.lax.scan``: peak memory is O(B * site_size) instead of the
    O(tau_grid_size * B * site_size) of a stacked ``logsumexp``. With
    ``tau_grid='uniform'`` the midpoint rule is used (tau_grid_size nodes at
    (i + 0.5) * t / tau_grid_size). With ``tau_grid='simpson'`` composite Simpson's
    rule is used and ``tau_grid_size`` must be odd.

    Output shape matches ``mixture_component_logpdf``: (B, *site_shape).
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    anchor = jnp.asarray(anchor, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    eta = float(jump.eta)
    std_floor = float(jump.std_floor)

    nodes, log_w = _build_quadrature(
        t=t, tau_grid_size=int(tau_grid_size), tau_grid=str(tau_grid)
    )
    # nodes, log_w: (tau_grid_size, B)

    lam_nodes = jnp.asarray(hazard.lam(nodes), dtype=jnp.float32)
    log_lam_nodes = jnp.log(
        jnp.maximum(lam_nodes, jnp.asarray(1e-30, dtype=jnp.float32))
    )
    log_S_nodes = -jnp.asarray(hazard.cum(nodes), dtype=jnp.float32)

    # mixture_component_logpdf returns shape y.shape[:-1] = (B, *site_shape).
    out_shape = y.shape[:-1]
    site_pad = (1,) * (len(out_shape) - 1)

    def step(carry, x):
        log_w_i, log_lam_i, log_S_i, tau_i = x
        log_p_i = mixture_component_logpdf(
            y=y, anchor=anchor, t=t, tau=tau_i,
            beta=beta, eta=eta, std_floor=std_floor,
        )
        weight_i = (log_w_i + log_lam_i + log_S_i).reshape((-1,) + site_pad)
        log_x = weight_i + log_p_i
        m_new, l_new = _online_lse_step(carry[0], carry[1], log_x)
        return (m_new, l_new), None

    m_init = jnp.full(out_shape, -jnp.inf, dtype=jnp.float32)
    l_init = jnp.zeros(out_shape, dtype=jnp.float32)
    xs = (log_w, log_lam_nodes, log_S_nodes, nodes)
    (m_final, l_final), _ = jax.lax.scan(step, (m_init, l_init), xs)
    return m_final + jnp.log(l_final)


def classifier_induced_score(
    y: Array,
    t: Array,
    *,
    anchor_logits: Array,
    anchors,
    beta,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    tau_grid_size: int = 32,
) -> Array:
    """Classifier-induced score nabla log p_t(y) on X_A (Appendix C.3).

    Computes the score of the SJD marginal p_t on X_A as the classifier-weighted
    average of per-anchor mixture scores. Per-anchor scores are themselves the
    time-posterior expectation of the per-tau component score:
        s_a(y, t) = -sum_i w_t(tau_i | y, a) (y - alpha(t) a) / v_t(tau_i),
    where the time-posterior `w_t(tau_i | y, a)` is the softmax over tau of
    `log_w_i + log lambda(tau_i) + log S(tau_i) + log N(y; alpha(t)a, v_t(tau_i) I)`.

    Output shape equals `y.shape`.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    anchor_logits = jnp.asarray(anchor_logits, dtype=jnp.float32)
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)

    B = int(y.shape[0])
    site_shape = y.shape[1:-1]
    d = int(y.shape[-1])
    L = int(a_table.shape[0])
    y_flat = y.reshape((B, -1, d))  # (B, S, d)
    S = int(y_flat.shape[1])

    probs = jax.nn.softmax(anchor_logits, axis=-1)
    probs_flat = probs.reshape((B, S, L))

    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha_t = alpha_t.astype(jnp.float32)
    sigma_t = sigma_t.astype(jnp.float32)

    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    tau_grid_size_i = int(tau_grid_size)

    # Uniform-midpoint quadrature on (0, t), per-example.
    i_arr = jnp.arange(tau_grid_size_i, dtype=jnp.float32) + 0.5
    nodes = (i_arr[:, None] / float(tau_grid_size_i)) * t[None, :]  # (tau_grid_size, B)
    log_h_per_i = jnp.log(t / float(tau_grid_size_i))  # (B,) — same for every i
    lam_nodes = jnp.asarray(hazard.lam(nodes), dtype=jnp.float32)
    log_lam = jnp.log(
        jnp.maximum(lam_nodes, jnp.asarray(1e-30, dtype=jnp.float32))
    )
    log_S = -jnp.asarray(hazard.cum(nodes), dtype=jnp.float32)

    # v_t(tau_i): (tau_grid_size, B), variance of the mixture component at tau_i.
    alpha_tau, sigma_tau = alpha_sigma(beta, nodes)
    alpha_tau = alpha_tau.astype(jnp.float32)
    sigma_tau = sigma_tau.astype(jnp.float32)
    deficit = (
        (1.0 - eta * eta)
        * jnp.square(alpha_t[None, :] / alpha_tau)
        * jnp.square(sigma_tau)
    )
    v = jnp.square(sigma_t)[None, :] - deficit  # (tau_grid_size, B)
    v = jnp.maximum(v, jnp.square(jnp.asarray(std_floor, dtype=jnp.float32)))

    # ||y - alpha(t) a||^2 over (b, s, l).
    dot = jnp.einsum("bsd,ld->bsl", y_flat, a_table)
    y_norm2 = jnp.sum(y_flat * y_flat, axis=-1, keepdims=True)
    a_norm2 = jnp.sum(a_table * a_table, axis=-1)[None, None, :]
    alpha_t_3 = alpha_t[:, None, None]  # (B, 1, 1)
    dist2 = (
        y_norm2
        - 2.0 * alpha_t_3 * dot
        + (alpha_t_3 * alpha_t_3) * a_norm2
    )  # (B, S, L)

    log_2pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))

    # Streaming online softmax + weighted-by-1/v sum, both shape (B, S, L).
    # Carry: (m, l, n) where l = sum exp(log_x_i - m), n = sum (1/v_i) exp(log_x_i - m).
    # After scan, e = n / l (the exp(m) factors cancel) and the score uses e.
    out_shape = (B, S, L)

    def step(carry, x):
        log_w_i, log_lam_i, log_S_i, v_i = x  # each (B,)
        v_i_b = v_i[:, None, None]  # (B, 1, 1)
        log_N_i = -0.5 * (
            d * (log_2pi + jnp.log(v_i_b)) + dist2 / v_i_b
        )  # (B, S, L)
        weight_i = (log_w_i + log_lam_i + log_S_i)[:, None, None]  # (B, 1, 1)
        log_x = weight_i + log_N_i  # (B, S, L)
        m_carry, l_carry, n_carry = carry
        m_new = jnp.maximum(m_carry, log_x)
        ratio = jnp.exp(m_carry - m_new)
        term = jnp.exp(log_x - m_new)
        l_new = l_carry * ratio + term
        n_new = n_carry * ratio + term * (1.0 / v_i_b)
        return (m_new, l_new, n_new), None

    m_init = jnp.full(out_shape, -jnp.inf, dtype=jnp.float32)
    l_init = jnp.zeros(out_shape, dtype=jnp.float32)
    n_init = jnp.zeros(out_shape, dtype=jnp.float32)
    log_h_xs = jnp.broadcast_to(log_h_per_i[None, :], (tau_grid_size_i, B))
    xs = (log_h_xs, log_lam, log_S, v)
    (m_final, l_final, n_final), _ = jax.lax.scan(
        step, (m_init, l_init, n_init), xs
    )
    e = n_final / l_final  # (B, S, L)

    # Score = -sum_l P_theta(a_l | y, t) e[b, s, l] (y - alpha(t) a_l).
    coeff = probs_flat * e  # (B, S, L)
    coeff_total = jnp.sum(coeff, axis=-1, keepdims=True)  # (B, S, 1)
    weighted_anchor_sum = jnp.einsum("bsl,ld->bsd", coeff, a_table)
    score_flat = -(coeff_total * y_flat - alpha_t_3 * weighted_anchor_sum)
    return score_flat.reshape(y.shape)
