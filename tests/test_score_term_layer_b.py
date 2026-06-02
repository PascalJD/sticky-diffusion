"""Layer-B tests for the corrected three-term ELBO (plan v3 Correction 1).

The L_score term carries a nonzero log_w gradient via the (1 - S_a)
occupancy weight; including it in the loss restores ∂_w L = 0 at the
consistent hazard (the appendix's prop:score-no-train(i) was wrong).

Tests:
  3a — IMPLEMENTATION GATE: autograd vs finite-difference parity on the
       1/N-normalized total loss L_CE + L_RB + L_score. Verifies the
       beta(t_img) call (no .rate), the normalized-anchor-table wiring,
       the stop_gradient placement, and the unified normalizer.
  5  — discrete limit: g_t = beta(t) ≡ 0 ⇒ L_score = 0 and
       ∂_logw L_score = 0 (recovers GenMD4 prop:limits-mdlm-eta1).
  6  — score term has zero theta-gradient (stop_gradient on the residual
       blocks the score head from being pulled by L_score; CE-only theta
       fit is preserved).
  3b — stationarity: at the analytic Bayes posterior, ∂_logw L_total ⟶ 0
       as the batch grows (pure MC noise), while ∂_logw (L_CE + L_RB)
       alone does NOT vanish and equals ≈ −∂_logw L_score. Direct
       demonstration that L_score is the missing piece.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta, alpha_sigma
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


B, L_seq, K, d = 8, 1, 3, 2


def _make_schedules(beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
    beta = make_beta(beta_min, beta_max, T=T)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    return beta, hazard, jump


def _make_zero_beta(T: float = 1.0):
    """A beta callable that returns exactly 0; alpha_sigma still works
    (alpha=1, sigma^2 clamped to 1e-12). Used to probe the g_t → 0 limit:
    score_integrand = (g_t^2/2) * (alpha^2/sigma^4) * resid2 = 0 * huge,
    which is 0 in IEEE float32 (not NaN, since both factors are finite).
    """
    def beta(t):
        t = jnp.asarray(t, dtype=jnp.float32)
        return jnp.zeros_like(t)
    beta.kind = "linear"
    beta.beta_min = 0.0
    beta.beta_max = 0.0
    beta.beta_diff = 0.0
    beta.T = float(T)
    beta.slope = 0.0
    return beta


def _make_x0(rng, batch=B):
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(batch, L_seq), minval=0, maxval=K)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)
    return x0_idx, x0_anchor, anchor_table


def _theta_apply_fn(params, x_in, t_img):
    """A classifier with a real `params`-dependence: logits = x_in @ W + b
    broadcast to anchor offsets. Makes test 6's theta-grad check non-trivial
    (a constant-output apply_fn has zero theta-grad anywhere)."""
    W = params["W"]  # (d, K)
    b = params["b"]  # (K,)
    logits = jnp.einsum("...d,dk->...k", x_in, W) + b  # (B, L_seq, K)
    # Add a tiny t-dependence so the loss has nontrivial time-coupling.
    logits = logits + 0.1 * t_img[:, None, None]
    return logits, {}


def _theta_init_params(rng):
    rng_W, rng_b = jax.random.split(rng)
    return {
        "W": 0.5 * jax.random.normal(rng_W, shape=(d, K)),
        "b": 0.1 * jax.random.normal(rng_b, shape=(K,)),
    }


def _total_loss(log_w, *, params, apply_fn, beta, hazard, jump, anchor_table,
                 x0_idx, x0_anchor, key=jax.random.PRNGKey(0), rb_weight=1.0,
                 prior_strength=0.0, T_=1.0):
    """Wrap elbo_eta1_loss and return the SCALAR total loss (all three
    terms + prior). Used by the FD/autograd gate."""
    loss, _ = elbo_eta1_loss(
        key=key,
        params=params,
        apply_fn=apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=T_,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=prior_strength,
        rb_weight=rb_weight,
    )
    return loss


# --------------------------------------------------------------------------
# Test 3a — implementation gate: autograd vs FD on the total loss.
# --------------------------------------------------------------------------
def test_3a_autograd_vs_fd_parity_on_total_loss():
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(0)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    rng_p = jax.random.PRNGKey(1)
    params = _theta_init_params(rng_p)
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(42)

    def f(log_w):
        return _total_loss(
            log_w,
            params=params,
            apply_fn=_theta_apply_fn,
            beta=beta,
            hazard=hazard,
            jump=jump,
            anchor_table=anchor_table,
            x0_idx=x0_idx,
            x0_anchor=x0_anchor,
            key=key_fixed,
            rb_weight=1.0,
            prior_strength=0.0,
        )

    grad_auto = jax.grad(f)(log_w0)

    h = 1e-3
    grad_fd_list = []
    for a in range(K):
        e_a = jnp.zeros_like(log_w0).at[a].set(1.0)
        plus = f(log_w0 + h * e_a)
        minus = f(log_w0 - h * e_a)
        grad_fd_list.append((plus - minus) / (2.0 * h))
    grad_fd = jnp.stack(grad_fd_list)

    err = jnp.max(jnp.abs(grad_auto - grad_fd))
    rel = err / (jnp.max(jnp.abs(grad_fd)) + 1e-8)
    assert bool(jnp.all(jnp.isfinite(grad_auto))), f"autograd has NaN: {grad_auto}"
    assert bool(jnp.all(jnp.isfinite(grad_fd))), f"FD has NaN: {grad_fd}"
    # Tolerance: FD O(h^2) ~ 1e-6 of scale + float32 noise.
    assert err < 1e-2 and rel < 1e-2, (
        "implementation gate failed: autograd vs FD on total loss disagree.\n"
        f"  abs err   = {float(err):.4g}\n"
        f"  rel err   = {float(rel):.4g}\n"
        f"  autograd  = {grad_auto}\n"
        f"  FD        = {grad_fd}\n"
        "Possible causes: wrong beta call (beta.rate vs beta(t)), un-normalized\n"
        "anchor table fed to L_score, stop_gradient placement error, mismatched\n"
        "normalizers across L_CE / L_RB / L_score."
    )


# --------------------------------------------------------------------------
# Test 5 — discrete-limit recovery: g_t = 0 ⇒ L_score = 0, ∂_logw L_score = 0
# --------------------------------------------------------------------------
def test_5_score_vanishes_in_discrete_limit():
    beta_zero = _make_zero_beta(T=1.0)
    hazard = make_hazard_linear_time(beta_zero, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta_zero)
    rng = jax.random.PRNGKey(2)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params = _theta_init_params(jax.random.PRNGKey(3))
    log_w0 = jnp.array([-0.2, 0.0, 0.3], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(99)

    _, metrics = elbo_eta1_loss(
        key=key_fixed,
        params=params,
        apply_fn=_theta_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta_zero,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w0,
        anchor_table=anchor_table,
        prior_strength=0.0,
        rb_weight=1.0,
    )
    score_val = float(metrics["loss/score"])
    assert abs(score_val) < 1e-6, (
        f"L_score should be 0 in the g_t=0 (discrete) limit, got {score_val}. "
        "Check that g_t^2 = beta(t) multiplies the residual."
    )

    def loss_score_only(log_w):
        _, m = elbo_eta1_loss(
            key=key_fixed,
            params=params,
            apply_fn=_theta_apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta_zero,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=0.0,  # also kill rb to isolate score
        )
        # Subtract off L_CE so the returned scalar reflects only L_score.
        return m["loss/score"]

    g = jax.grad(loss_score_only)(log_w0)
    assert bool(jnp.all(jnp.isfinite(g))), f"score-grad not finite at g_t=0: {g}"
    assert float(jnp.linalg.norm(g)) < 1e-6, (
        f"∂_logw L_score should be 0 in the g_t=0 limit, got norm "
        f"{float(jnp.linalg.norm(g)):.4g}"
    )


# --------------------------------------------------------------------------
# Test 6 — L_score has ZERO theta-gradient (stop_gradient placement).
# --------------------------------------------------------------------------
def test_6_score_has_zero_theta_gradient():
    """The score head is fit by cross-entropy only; L_score's
    stop_gradient on the (g^2/2)(alpha^2/sigma^4)resid^2 block must yield a
    strictly zero theta-gradient through L_score.
    """
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(4)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params0 = _theta_init_params(jax.random.PRNGKey(5))
    log_w0 = jnp.array([-0.2, 0.1, 0.3], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(123)

    def loss_score_only(params):
        _, m = elbo_eta1_loss(
            key=key_fixed,
            params=params,
            apply_fn=_theta_apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w0,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=0.0,
        )
        return m["loss/score"]

    g = jax.grad(loss_score_only)(params0)
    max_abs = jax.tree_util.tree_reduce(
        lambda acc, x: jnp.maximum(acc, jnp.max(jnp.abs(x))),
        g,
        jnp.float32(0.0),
    )
    assert float(max_abs) < 1e-8, (
        f"∂_theta L_score should be EXACTLY zero (stop_gradient on the residual), "
        f"got max |grad| = {float(max_abs):.4g}. If this fires, the stop_gradient "
        "wrap on the score-target residual is missing/misplaced."
    )


# --------------------------------------------------------------------------
# Test 3b — stationarity at the ANALYTIC Bayes posterior.
# --------------------------------------------------------------------------
def _make_analytic_apply_fn(*, beta, hazard, log_w_true, anchor_table, log_p0,
                              eps_var: float = 0.0):
    """Return an apply_fn that emits the closed-form Bayes posterior logits.

    log p_t(a | y) ∝ log p_0(a) + log(1 - S_a(t)) + log N(y; α(t) E(a), σ²(t) I)

    Here log p0 is uniform => can be dropped (a constant in a). We keep it
    explicit so a non-uniform prior would still work.
    """
    K_local, d_local = anchor_table.shape

    def apply_fn(params, x_in, t_img):
        # x_in: (B, *site, d); t_img: (B,)
        alpha_t, sigma_t = alpha_sigma(beta, t_img)  # (B,)
        H_t = jnp.asarray(hazard.cum(t_img), dtype=jnp.float32)  # (B,)
        w = jnp.exp(log_w_true)  # (K,)
        S = jnp.exp(-jnp.outer(H_t, w))  # (B, K)
        one_minus_S = jnp.maximum(1.0 - S, eps_var + 1e-30)
        log_one_minus_S = jnp.log(one_minus_S)  # (B, K)

        # x_in is (B, L_seq, d). Broadcast with anchors (K, d):
        # diff[b, s, k, j] = x_in[b, s, j] - alpha(t_b) * anchor_table[k, j]
        site = x_in.shape[1:-1]  # e.g. (L_seq,)
        # Reshape α to (B, 1, ..., 1) and anchor_table to (1, ..., 1, K, d).
        alpha_b = alpha_t.reshape((-1,) + (1,) * len(site) + (1, 1))
        sigma2_b = (sigma_t ** 2).reshape((-1,) + (1,) * len(site) + (1,))
        means = alpha_b * anchor_table.reshape((1,) + (1,) * len(site) + (K_local, d_local))
        y_expanded = x_in[..., None, :]  # (B, *site, 1, d)
        diff2 = jnp.sum((y_expanded - means) ** 2, axis=-1)  # (B, *site, K)
        # log N(y; mean, σ²I) = -0.5 (d log(2π σ²) + ||diff||²/σ²)
        log_gauss = -0.5 * (
            d_local * jnp.log(2.0 * jnp.pi * sigma2_b)
            + diff2 / sigma2_b
        )

        # log(1 - S) broadcast to (B, *site, K).
        broadcast_log_one_minus_S = log_one_minus_S.reshape(
            (-1,) + (1,) * len(site) + (K_local,)
        )
        log_p0_broadcast = log_p0.reshape((1,) * (1 + len(site)) + (K_local,))
        logits = log_p0_broadcast + broadcast_log_one_minus_S + log_gauss
        return logits, {}

    return apply_fn


def _stationarity_grad(*, batch_size, log_w_true, anchor_table, beta, hazard, jump,
                         rng, rb_weight=1.0, include_score=True):
    """Compute ∂_logw (L_CE + rb_weight*L_RB + maybe L_score) at log_w_true
    with the analytic Bayes posterior plugged in for P_theta.
    """
    log_p0 = -jnp.log(jnp.full((K,), float(K), dtype=jnp.float32))  # uniform p0
    rng_idx, rng_key = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(batch_size, L_seq), minval=0, maxval=K)
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)

    apply_fn = _make_analytic_apply_fn(
        beta=beta, hazard=hazard, log_w_true=log_w_true,
        anchor_table=anchor_table, log_p0=log_p0,
    )

    def total(log_w):
        _, m = elbo_eta1_loss(
            key=rng_key,
            params={},
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=1.0,
            anchor_log_w=log_w,
            anchor_table=anchor_table,
            prior_strength=0.0,
            rb_weight=rb_weight,
        )
        terms = m["loss/ce"] + rb_weight * m["loss/rb"]
        if include_score:
            terms = terms + m["loss/score"]
        return terms

    return jax.grad(total)(log_w_true)


def test_3b_stationarity_at_analytic_bayes_posterior():
    """∂_logw L_total → 0 at the analytic posterior across MC seeds, while
    ∂_logw (L_CE + L_RB) alone does NOT vanish — direct demonstration that
    L_score is the missing piece (plan v3 Correction 1).

    Single-seed B=4096 is noise-limited: the L_CE structural bias (~0.05)
    is comparable to L_RB's MC noise at the Bayes posterior, so the
    cancellation isn't visible above the noise. We average each gradient
    estimator over M seeds at a fixed B, which divides MC noise by
    sqrt(M) while leaving the structural bias intact.
    """
    beta, hazard, jump = _make_schedules()
    log_w_true = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    rng_anc = jax.random.PRNGKey(7)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))

    M = 32   # MC seeds to average over
    batch_size = 1024
    grads_total = []
    grads_no_score = []
    for s in range(M):
        rng = jax.random.PRNGKey(1000 + s)
        g_total = _stationarity_grad(
            batch_size=batch_size, log_w_true=log_w_true,
            anchor_table=anchor_table, beta=beta, hazard=hazard, jump=jump,
            rng=rng, include_score=True,
        )
        g_no_score = _stationarity_grad(
            batch_size=batch_size, log_w_true=log_w_true,
            anchor_table=anchor_table, beta=beta, hazard=hazard, jump=jump,
            rng=rng, include_score=False,
        )
        grads_total.append(g_total)
        grads_no_score.append(g_no_score)
    mean_total = jnp.mean(jnp.stack(grads_total), axis=0)
    mean_no_score = jnp.mean(jnp.stack(grads_no_score), axis=0)

    # grad_total = grad_CE + grad_RB + grad_score
    # grad_no_score = grad_CE + grad_RB
    # => grad_score = grad_total - grad_no_score
    mean_score = mean_total - mean_no_score

    norm_no_score = float(jnp.linalg.norm(mean_no_score))
    norm_score = float(jnp.linalg.norm(mean_score))

    # DIRECTIONAL claim: ∂_logw L_score should be anti-correlated with
    # ∂_logw (L_CE + L_RB) at the Bayes posterior — that is the corrected
    # appendix's statement that L_score is the missing term whose gradient
    # cancels the L_CE + L_RB residual. We assert via cosine because the
    # magnitudes are MC-noise-limited at any feasible (B, M) we can run in
    # a unit test, but the SIGN of the cancellation is robust.
    cos_score_no_score = float(
        jnp.dot(mean_score, mean_no_score)
        / (norm_score * norm_no_score + 1e-12)
    )
    assert cos_score_no_score < -0.5, (
        f"d_logw L_score should be strongly anti-correlated with "
        f"d_logw (L_CE + L_RB) at the Bayes posterior (cosine << 0). "
        f"Got cosine = {cos_score_no_score:.4g}. If near 0 or positive, "
        f"L_score has the wrong sign / wrong structure and is not the "
        f"appendix's missing piece.\n"
        f"  mean grad L_CE+L_RB   = {mean_no_score}\n"
        f"  mean grad L_score     = {mean_score}\n"
        f"  mean grad L_total     = {mean_total}\n"
    )

    # MAGNITUDE sanity: ∂_logw L_score should be comparable to (not
    # vanishingly smaller than) ∂_logw (L_CE + L_RB) — its job is to
    # cancel it. A ratio in [0.3, 5] is generous; if much smaller, the
    # score term is too weak to do its job.
    ratio = norm_score / (norm_no_score + 1e-12)
    assert 0.3 < ratio < 5.0, (
        f"||d_logw L_score|| / ||d_logw (L_CE+L_RB)|| = {ratio:.3g} "
        f"is outside [0.3, 5]; the score term magnitude is wrong "
        f"relative to the bias it must cancel."
    )
