from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from sticky.rng import PRNGKey

from .blur import (
    SIN8_VALUE_PERIOD,
    blur_values,
    blurred_posterior_mean,
    blurred_value_center,
    sinusoidal_value_embedding,
)
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
    log_w_per_site: Array | None = None,
    x0_idx: Array | None = None,
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
    log_w_per_site : optional shape (B, *site_shape). When provided, the forward
        hazard becomes lambda_t(a) = beta(t) * exp(log_w_per_site), so per the
        flux identity S_a(t) = S(t)^{w(a)} and tau is drawn from the truncated
        weighted density. When None, behavior is bit-identical to the
        anchor-agnostic baseline.
    x0_idx : optional shape (B, *site_shape) integer values of X_0. REQUIRED
        when the jump carries a blur kernel with blur_space="value" (the
        unstick mean is E((B v)) with v = x0_idx, evaluated at the CONTINUOUS
        blurred values); ignored otherwise (passing it on the embedding /
        no-blur path is a no-op and keeps those paths bit-exact).

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

    if log_w_per_site is not None:
        log_w_b = jnp.asarray(log_w_per_site, dtype=jnp.float32)
        if log_w_b.shape != (B,) + site_shape:
            raise ValueError(
                f"log_w_per_site must have shape {(B,) + site_shape}, "
                f"got {log_w_b.shape}"
            )
        # S_a(t) = S(t)^{w(a)} ; w(a) = exp(log_w_per_site).
        log_S_t_b = jnp.log(jnp.maximum(surv_t_b, jnp.asarray(1e-30, dtype=jnp.float32)))
        w_b = jnp.exp(log_w_b)
        log_S_a_t_b = w_b * log_S_t_b
        surv_t_b = jnp.exp(log_S_a_t_b)
        cdf_t_b = -jnp.expm1(log_S_a_t_b)

    u_unstuck = jax.random.uniform(
        k_unstuck, shape=(B,) + site_shape, dtype=jnp.float32
    )
    never_unstuck_mask = u_unstuck < surv_t_b

    u_tau = jax.random.uniform(
        k_tau, shape=(B,) + site_shape, dtype=jnp.float32,
        minval=0.0, maxval=1.0,
    )
    if log_w_per_site is None:
        tau = hazard.inv_cdf(u_tau * cdf_t_b)
    else:
        # B(tau) = -log(1 - U * (1 - S_a(t))) / w(a).  Pre-divide so the
        # existing inv_cdf (which inverts the *base* schedule's 1 - S) works
        # unchanged: feed it u_eff_baseline = 1 - exp(-B(tau)).
        u_eff_a = u_tau * cdf_t_b
        B_target = -jnp.log1p(-u_eff_a) / jnp.maximum(
            w_b, jnp.asarray(1e-12, dtype=jnp.float32)
        )
        u_eff_baseline = -jnp.expm1(-B_target)
        tau = hazard.inv_cdf(u_eff_baseline)

    # Un-stick center mu(X_0). Trace-time Python branch (blur_space is a
    # plain str field): the embedding/no-blur path below is the pre-existing
    # expression, textually untouched — bit-exact legacy behavior.
    blur_space = str(getattr(jump, "blur_space", "embedding"))
    if getattr(jump, "blur_kernel", None) is not None and blur_space == "value":
        if x0_idx is None:
            raise ValueError(
                "sample_pair with blur_space='value' requires x0_idx (the raw "
                "integer values of X_0); got None."
            )
        d = int(x0_anchor.shape[-1])
        if d % 2 != 0:
            raise ValueError(
                f"blur_space='value' requires an even anchor dim (sin/cos "
                f"pairs at freqs 1..d/2); got d={d}."
            )
        v0 = jnp.asarray(x0_idx, dtype=jnp.float32)
        if v0.shape != (B,) + site_shape:
            raise ValueError(
                f"x0_idx must have shape {(B,) + site_shape}, got {v0.shape}"
            )
        # mu = E((B v)): blur raw values (row-stochastic B keeps Bv inside
        # [0, 255] by convexity — validated at task-build time), then evaluate
        # the continuous sinusoidal embedding. No rounding / quantization.
        blurred_center = sinusoidal_value_embedding(
            blur_values(v0, jump.blur_kernel),
            num_freqs=d // 2,
            period=SIN8_VALUE_PERIOD,
        )
    else:
        blurred_center = jump.apply_blur(x0_anchor)  # identity when blur_kernel is None
    mean, std = mixture_component_mean_std(
        anchor=blurred_center,
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


def _squared_distance_all_anchors(
    y: Array, a_table: Array, alpha_t: Array
) -> Array:
    """``||y - alpha(t) a_l||^2`` for every anchor a_l, shape ``(B, *site_shape, L)``.

    Computed via ``||y||^2 - 2 alpha (y . a) + alpha^2 ||a||^2`` so no
    ``(B, *site_shape, L, d)`` intermediate is materialized — the ``d``-axis is
    contracted by a single matmul.
    """
    B = y.shape[0]
    d = y.shape[-1]
    site_shape = y.shape[1:-1]
    L = a_table.shape[0]
    y_flat = y.reshape((B, -1, d))
    dot = jnp.einsum("bsd,ld->bsl", y_flat, a_table)
    y_norm2 = jnp.sum(y_flat * y_flat, axis=-1, keepdims=True)
    a_norm2 = jnp.sum(a_table * a_table, axis=-1)[None, None, :]
    alpha_3 = alpha_t[:, None, None]
    dist2_flat = (
        y_norm2 - 2.0 * alpha_3 * dot + (alpha_3 * alpha_3) * a_norm2
    )
    return dist2_flat.reshape((B,) + site_shape + (L,))


# NOTE(blur): this log-pdf assumes W = I (per-site anchor means). This is
# harmless for the plug-in hazard/allocation at eta=1: the jump kernel r and
# every mixture component share mean and variance there, so the density
# factors cancel in the r/p ratio and the (mis)specified mean drops out —
# exactly as it does for the true W-blurred densities (cf. vp_matched.yaml
# "eta=1 cancels state dependency"). eta < 1 with W != I is rejected upstream.
def vp_jump_logpdf_all_anchors(
    y: Array, t: Array, a_table: Array, jump: VPMatchedGaussianJump
) -> Array:
    """``log r_a(y) = log N(y; alpha(t) a, eta^2 sigma^2(t) I_d)`` for every anchor.

    Returns shape ``(B, *site_shape, L)`` without expanding the ``d``-axis.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    a_table = jnp.asarray(a_table, dtype=jnp.float32)
    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    d = int(y.shape[-1])

    alpha_t, sigma_t = alpha_sigma(jump.beta, t)
    alpha_t = alpha_t.astype(jnp.float32)
    sigma_t = sigma_t.astype(jnp.float32)

    var = jnp.maximum(
        jnp.square(eta * sigma_t),
        jnp.square(jnp.asarray(std_floor, dtype=jnp.float32)),
    )
    log_2pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))

    dist2 = _squared_distance_all_anchors(y, a_table, alpha_t)
    site_pad = (1,) * (y.ndim - 2)
    var_b = var.reshape((-1,) + site_pad + (1,))
    return -0.5 * (
        d * (log_2pi + jnp.log(var_b)) + dist2 / (var_b + 1e-12)
    )


# NOTE(blur): this log-pdf assumes W = I (per-site anchor means). This is
# harmless for the plug-in hazard/allocation at eta=1: the jump kernel r and
# every mixture component share mean and variance there, so the density
# factors cancel in the r/p ratio and the (mis)specified mean drops out —
# exactly as it does for the true W-blurred densities (cf. vp_matched.yaml
# "eta=1 cancels state dependency"). eta < 1 with W != I is rejected upstream.
def mixture_logpdf_all_anchors(
    y: Array,
    t: Array,
    anchors,
    beta,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    tau_grid_size: int = 32,
    tau_grid: str = "uniform",
) -> Array:
    """``log p_t^ac(y | a_l)`` for every anchor a_l, shape ``(B, *site_shape, L)``.

    Same tau-quadrature integrand as ``mixture_logpdf`` but vectorized over L:
    the L-axis is added to the streaming online-LSE carry so peak memory stays
    at one ``(B, *site_shape, L)`` tensor (no stacked tau-components).

    Frequency-weighted hazards: ``anchors.effective_log_w`` (shape ``(L,)``) is
    forwarded as ``log w(a)`` per anchor; the integrand carries the same
    ``log w(a) + w(a) * log S(tau)`` factors as the per-anchor `mixture_logpdf`.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    _eff = getattr(anchors, "effective_log_w", None)
    if _eff is None:
        _eff = jnp.zeros((int(a_table.shape[0]),), dtype=jnp.float32)
    log_w_table = jnp.asarray(_eff, dtype=jnp.float32)

    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    d = int(y.shape[-1])
    L = int(a_table.shape[0])

    nodes, log_w_q = _build_quadrature(
        t=t, tau_grid_size=int(tau_grid_size), tau_grid=str(tau_grid)
    )
    lam_nodes = jnp.asarray(hazard.lam(nodes), dtype=jnp.float32)
    log_lam = jnp.log(
        jnp.maximum(lam_nodes, jnp.asarray(1e-30, dtype=jnp.float32))
    )
    log_S = -jnp.asarray(hazard.cum(nodes), dtype=jnp.float32)

    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha_t = alpha_t.astype(jnp.float32)
    sigma_t = sigma_t.astype(jnp.float32)
    alpha_tau, sigma_tau = alpha_sigma(beta, nodes)
    alpha_tau = alpha_tau.astype(jnp.float32)
    sigma_tau = sigma_tau.astype(jnp.float32)

    deficit = (
        (1.0 - eta * eta)
        * jnp.square(alpha_t[None, :] / alpha_tau)
        * jnp.square(sigma_tau)
    )
    v = jnp.maximum(
        jnp.square(sigma_t)[None, :] - deficit,
        jnp.square(jnp.asarray(std_floor, dtype=jnp.float32)),
    )

    dist2 = _squared_distance_all_anchors(y, a_table, alpha_t)

    site_pad = (1,) * (y.ndim - 2)
    log_2pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))
    log_w_a = log_w_table.reshape((1,) + site_pad + (L,))
    w_a = jnp.exp(log_w_a)

    out_shape = dist2.shape

    def step(carry, x):
        log_w_q_i, log_lam_i, log_S_i, v_i = x
        v_b = v_i.reshape((-1,) + site_pad + (1,))
        log_N = -0.5 * (
            d * (log_2pi + jnp.log(v_b)) + dist2 / v_b
        )
        prefix = (log_w_q_i + log_lam_i).reshape((-1,) + site_pad + (1,))
        log_S_b = log_S_i.reshape((-1,) + site_pad + (1,))
        weight = prefix + log_w_a + log_S_b * w_a
        log_x = weight + log_N
        m_new, l_new = _online_lse_step(carry[0], carry[1], log_x)
        return (m_new, l_new), None

    m_init = jnp.full(out_shape, -jnp.inf, dtype=jnp.float32)
    l_init = jnp.zeros(out_shape, dtype=jnp.float32)
    xs = (log_w_q, log_lam, log_S, v)
    (m_final, l_final), _ = jax.lax.scan(step, (m_init, l_init), xs)
    return m_final + jnp.log(l_final)


# NOTE(blur): this log-pdf assumes W = I (per-site anchor means). This is
# harmless for the plug-in hazard/allocation at eta=1: the jump kernel r and
# every mixture component share mean and variance there, so the density
# factors cancel in the r/p ratio and the (mis)specified mean drops out —
# exactly as it does for the true W-blurred densities (cf. vp_matched.yaml
# "eta=1 cancels state dependency"). eta < 1 with W != I is rejected upstream.
def mixture_logpdf(
    y: Array,
    anchor: Array,
    t: Array,
    beta,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    tau_grid_size: int = 32,
    tau_grid: str = "uniform",
    log_w_anchor: float | Array = 0.0,
) -> Array:
    """log p_t^ac(y | a) by 1-D log-sum-exp quadrature.

    Discretizes
        p_t^ac(y | a) = int_0^t lam_tau(a) S_a(tau) (r_tau * K_{tau->t})(y | a) dtau
    on a per-example tau-grid via streaming online log-sum-exp through
    ``jax.lax.scan``: peak memory is O(B * site_size) instead of the
    O(tau_grid_size * B * site_size) of a stacked ``logsumexp``. With
    ``tau_grid='uniform'`` the midpoint rule is used (tau_grid_size nodes at
    (i + 0.5) * t / tau_grid_size). With ``tau_grid='simpson'`` composite Simpson's
    rule is used and ``tau_grid_size`` must be odd.

    Frequency-weighted hazards: ``log_w_anchor`` is log w(a) where the forward
    hazard is lambda_tau(a) = beta(tau) * w(a) and S_a(tau) = S(tau)^{w(a)}.
    Default 0.0 reproduces the anchor-agnostic integrand bit-exactly.

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

    # log w(a) and w(a) shaped to broadcast against the per-example weight
    # tensor of shape (B, *site_pad). For scalar 0.0 these are also scalars.
    log_w_a_arr = jnp.asarray(log_w_anchor, dtype=jnp.float32)
    w_a_arr = jnp.exp(log_w_a_arr)

    # mixture_component_logpdf returns shape y.shape[:-1] = (B, *site_shape).
    out_shape = y.shape[:-1]
    site_pad = (1,) * (len(out_shape) - 1)

    def step(carry, x):
        log_w_i, log_lam_i, log_S_i, tau_i = x
        log_p_i = mixture_component_logpdf(
            y=y, anchor=anchor, t=t, tau=tau_i,
            beta=beta, eta=eta, std_floor=std_floor,
        )
        # Reshape per-tau scalars to (B, *site_pad), then add the per-anchor
        # log w(a) and the w(a)-scaled log S(tau).
        prefix_i = (log_w_i + log_lam_i).reshape((-1,) + site_pad)
        log_S_i_b = log_S_i.reshape((-1,) + site_pad)
        weight_i = prefix_i + log_w_a_arr + log_S_i_b * w_a_arr
        log_x = weight_i + log_p_i
        m_new, l_new = _online_lse_step(carry[0], carry[1], log_x)
        return (m_new, l_new), None

    m_init = jnp.full(out_shape, -jnp.inf, dtype=jnp.float32)
    l_init = jnp.zeros(out_shape, dtype=jnp.float32)
    xs = (log_w, log_lam_nodes, log_S_nodes, nodes)
    (m_final, l_final), _ = jax.lax.scan(step, (m_init, l_init), xs)
    return m_final + jnp.log(l_final)


# blur_kernel != None: resolved for eta=1 via the W-aware branch inside the
# function below (score uses mu = W @ Ehat with committed-site delta override
# in embedding space, or the plug-in mu = E(B vhat) in value space).
# eta < 1 with W != I remains out of scope and raises ValueError (the
# tau-mixture weights couple to the blurred mean; needs the quadrature
# generalization).
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
    posterior_temperature: float = 1.0,
    committed: Array | None = None,
    committed_idx: Array | None = None,
) -> Array:
    """Classifier-induced score nabla log p_t(y) on X_A (Appendix C.3).

    Computes the score of the SJD marginal p_t on X_A as the classifier-weighted
    average of per-anchor mixture scores. Per-anchor scores are themselves the
    time-posterior expectation of the per-tau component score:
        s_a(y, t) = -sum_i w_t(tau_i | y, a) (y - alpha(t) a) / v_t(tau_i),
    where the time-posterior `w_t(tau_i | y, a)` is the softmax over tau of
    `log_w_i + log lambda(tau_i) + log S(tau_i) + log N(y; alpha(t)a, v_t(tau_i) I)`.

    ``posterior_temperature`` (default 1.0) tempers the classifier posterior
    P_theta(a | y) = softmax(anchor_logits / temperature) that weights the
    per-anchor scores, i.e. it softens/sharpens the predicted-embedding mixture
    abar = sum_a P_theta(a|y) E(a) that drives the continuous drift. At 1.0 the
    branch below is bit-identical to the untempered softmax; a sampler may pass
    its logit_temperature here so temperature acts symmetrically on the drift
    and the discrete-jump allocation (sampling-strategy only; the score is no
    longer the exact marginal score when temperature != 1.0).

    W-aware branch: when ``jump.blur_kernel`` is not None the forward
    corruption blended anchors through W (``sample_pair`` line ~111), so the
    exact marginal score at eta=1 is

        s_i = -(y_i - alpha_t (W Ehat)_i) / max(sigma_t^2, std_floor^2),

    with Ehat_j the (tempered) classifier posterior mean at uncommitted sites
    and the committed anchor's embedding (a delta; the posterior collapses) at
    committed sites. ``committed``/``committed_idx`` (bool / int32 with -1 at
    uncommitted sites, shape ``(B,) + site_shape``) supply that override; both
    None means all-uncommitted. They are ignored when ``jump.blur_kernel`` is
    None, and the kernel-None path is bit-exact with the pre-W code (the
    branch is a trace-time Python check adding zero ops). eta != 1 with a
    kernel raises ValueError. Note the eta=1 quadrature limit of the legacy
    path is v = max(sigma_t^2, std_floor^2) with e = n/l equal to 1/v only
    algebraically (ULP-level scan rounding); the W branch uses the closed form.

    Value-space branch (``jump.blur_space == "value"``): the forward blended
    RAW values, mu_i = E((B v)_i). The exact induced mean E[E((B v)_i) | y]
    is intractable (E is nonlinear), so the drift uses the PLUG-IN

        vhat_j = sum_a P_theta^(j)(a | y) * a      (delta at committed sites),
        muhat_i = E((B vhat)_i),
        s_i = -(y_i - alpha_t muhat_i) / max(sigma_t^2, std_floor^2).

    Approximation quality is gated offline (G4: plug-in vs mean-field
    Monte-Carlo). Same eta=1 requirement and committed/committed_idx
    semantics as the embedding branch (committed_idx doubles as the
    committed VALUE — anchor index == pixel value under the frozen sin8
    contract, enforced at task-build time).

    Output shape equals `y.shape`.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    anchor_logits = jnp.asarray(anchor_logits, dtype=jnp.float32)
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    # `effective_log_w` is the AnchorTable property; duck-typed namespaces
    # (used in some tests) may not provide it — fall back to zeros.
    _eff_log_w = getattr(anchors, "effective_log_w", None)
    if _eff_log_w is None:
        _eff_log_w = jnp.zeros((int(a_table.shape[0]),), dtype=jnp.float32)
    log_w_table = jnp.asarray(_eff_log_w, dtype=jnp.float32)  # (L,)

    B = int(y.shape[0])
    site_shape = y.shape[1:-1]
    d = int(y.shape[-1])
    L = int(a_table.shape[0])
    y_flat = y.reshape((B, -1, d))  # (B, S, d)
    S = int(y_flat.shape[1])

    if float(posterior_temperature) == 1.0:
        probs = jax.nn.softmax(anchor_logits, axis=-1)
    else:
        probs = jax.nn.softmax(
            anchor_logits / jnp.asarray(posterior_temperature, dtype=jnp.float32),
            axis=-1,
        )
    probs_flat = probs.reshape((B, S, L))

    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha_t = alpha_t.astype(jnp.float32)
    sigma_t = sigma_t.astype(jnp.float32)

    # --- W-aware plug-in score (exact at eta=1; forward blur, Section 4.4). ---
    # Trace-time Python check: with blur_kernel=None this adds zero ops and the
    # legacy quadrature path below is textually untouched (bit-exact bypass).
    blur_kernel = getattr(jump, "blur_kernel", None)
    if blur_kernel is not None:
        eta_blur = float(jump.eta)
        if abs(eta_blur - 1.0) > 1e-9:
            raise ValueError(
                "classifier_induced_score with jump.blur_kernel is "
                f"exact only at eta=1; got eta={eta_blur}. Either sample with "
                "eta=1 or set sampler blur_score=false to force the legacy "
                "W=I score."
            )
        if (committed is None) != (committed_idx is None):
            raise ValueError(
                "committed and committed_idx must both be provided or both be None."
            )
        if committed is None:
            committed_site = jnp.zeros((B,) + tuple(site_shape), dtype=bool)
            committed_idx_site = -jnp.ones(
                (B,) + tuple(site_shape), dtype=jnp.int32
            )
        else:
            committed_site = jnp.asarray(committed, dtype=bool).reshape(
                (B,) + tuple(site_shape)
            )
            committed_idx_site = jnp.asarray(
                committed_idx, dtype=jnp.int32
            ).reshape((B,) + tuple(site_shape))
        blur_space = str(getattr(jump, "blur_space", "embedding"))
        if blur_space == "value":
            # Plug-in value-space center: vhat = posterior-mean VALUE
            # (anchor index == pixel value; L must be the 256-value 8-bit
            # grid so the arange below IS the value grid), delta at
            # committed sites, then muhat = E((B vhat)).
            if L != 256:
                raise ValueError(
                    f"blur_space='value' requires the 256-value pixel grid "
                    f"(anchor index == value); got L={L}."
                )
            if d % 2 != 0:
                raise ValueError(
                    f"blur_space='value' requires an even anchor dim "
                    f"(sin/cos pairs at freqs 1..d/2); got d={d}."
                )
            value_grid = jnp.arange(L, dtype=jnp.float32)  # (L,)
            v_mean = jnp.einsum("bsl,l->bs", probs_flat, value_grid).reshape(
                (B,) + tuple(site_shape)
            )
            mu = blurred_value_center(
                values_mean=v_mean,
                committed_mask=committed_site,
                committed_idx=committed_idx_site,
                kernel=blur_kernel,
                num_freqs=d // 2,
                period=SIN8_VALUE_PERIOD,
            )  # same shape as y
        else:
            m_flat = jnp.einsum("bsl,ld->bsd", probs_flat, a_table)  # (B, S, d)
            mu = blurred_posterior_mean(
                probs_mean=m_flat.reshape(y.shape),
                committed_mask=committed_site,
                committed_idx=committed_idx_site,
                a_table=a_table,
                kernel=blur_kernel,
            )  # same shape as y
        # v = max(sigma_t^2, std_floor^2): the exact eta=1 limit of the legacy
        # tau-quadrature (deficit == 0.0 at eta=1.0 in float) — deliberately
        # NOT board_sampling's 1e-12 floor.
        v_b = jnp.maximum(
            jnp.square(sigma_t),
            jnp.square(jnp.asarray(float(jump.std_floor), dtype=jnp.float32)),
        )  # (B,)
        score_flat = -(1.0 / v_b)[:, None, None] * (
            y_flat - alpha_t[:, None, None] * mu.reshape((B, S, d))
        )
        return score_flat.reshape(y.shape)

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

    # Per-anchor weighting: lambda_tau(a) = beta(tau) * w(a) and
    # S_a(tau) = S(tau)^{w(a)}. log_w_a_3 and w_a_3 broadcast as (1, 1, L).
    # When anchors.log_w is None (or zeros), these collapse to 0 and 1, so the
    # integrand is unchanged from the anchor-agnostic baseline.
    log_w_a_3 = log_w_table[None, None, :]
    w_a_3 = jnp.exp(log_w_a_3)

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
        # Per-anchor weighted integrand: log[w_q,i * beta(tau_i) * w(a) *
        # S(tau_i)^{w(a)}] = (log_h_i + log_lam_i) + log_w_a + w(a)*log_S_i.
        prefix_i = (log_w_i + log_lam_i)[:, None, None]  # (B, 1, 1)
        log_S_i_b = log_S_i[:, None, None]  # (B, 1, 1)
        weight_i = prefix_i + log_w_a_3 + log_S_i_b * w_a_3  # (B, 1, L)
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
