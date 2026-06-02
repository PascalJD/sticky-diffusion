"""Tests for the ``score_w_stop_gradient`` flag on ``elbo_eta1_loss``.

The flag exists to enable a CE-only ablation (the ``CE`` condition of the
Sudoku World-1-vs-World-2 hazard-learning test) without touching the v3
ELBO-verified path. Three invariants must hold:

  (a) Setting it to True kills the log_w-gradient through L_score
      (currently nonzero via the (1-S_a) derivative).
  (b) The loss VALUE is unchanged between True and False (only the
      gradient path is gated).
  (c) The flag never breaks the existing default-False behavior, i.e.
      theta-gradient through L_score remains exactly zero either way
      (the score head is still fit by cross-entropy only).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


B, L_seq, K, d = 8, 1, 3, 2


def _make_schedules():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    return beta, hazard, jump


def _make_x0(rng):
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(B, L_seq), minval=0, maxval=K)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)
    return x0_idx, x0_anchor, anchor_table


def _theta_apply_fn(params, x_in, t_img):
    """Classifier with a real theta-dependence so theta-grad tests are
    non-trivial. Mirrors the layer-B test fixture."""
    W = params["W"]                                       # (d, K)
    b = params["b"]                                       # (K,)
    logits = jnp.einsum("...d,dk->...k", x_in, W) + b
    logits = logits + 0.1 * t_img[:, None, None]
    return logits, {}


def _theta_init(rng):
    rng_W, rng_b = jax.random.split(rng)
    return {
        "W": 0.5 * jax.random.normal(rng_W, shape=(d, K)),
        "b": 0.1 * jax.random.normal(rng_b, shape=(K,)),
    }


def _call(score_w_stop_gradient, *, params, log_w, key, x0_idx, x0_anchor,
           anchor_table, beta, hazard, jump):
    return elbo_eta1_loss(
        key=key,
        params=params,
        apply_fn=_theta_apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=0.0,
        rb_weight=1.0,
        score_w_stop_gradient=score_w_stop_gradient,
    )


def test_loss_value_identical_between_true_and_false():
    """Only the gradient path differs; the forward value of every loss
    component (and total) must match bit-by-bit at the same (key, params,
    log_w, anchor_table)."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(0)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params = _theta_init(jax.random.PRNGKey(1))
    log_w = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(42)

    loss_f, m_f = _call(
        False, params=params, log_w=log_w, key=key_fixed,
        x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
        beta=beta, hazard=hazard, jump=jump,
    )
    loss_t, m_t = _call(
        True, params=params, log_w=log_w, key=key_fixed,
        x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert jnp.allclose(loss_f, loss_t), (
        f"total loss diverged: False={float(loss_f)}, True={float(loss_t)}"
    )
    for k in ("loss/ce", "loss/rb", "loss/score", "loss/prior",
               "loss/ce_num", "loss/rb_num", "loss/score_num", "loss/ce_den"):
        assert jnp.allclose(m_f[k], m_t[k]), (
            f"{k} differed between flag values: "
            f"False={float(m_f[k])}, True={float(m_t[k])}"
        )


def test_logw_gradient_of_L_score_vanishes_when_True():
    """The whole point of the flag: setting it to True kills L_score's
    log_w gradient (currently nonzero via the (1-S_a) factor)."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(2)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params = _theta_init(jax.random.PRNGKey(3))
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(123)

    def L_score_only(log_w, flag: bool):
        _, m = _call(
            flag, params=params, log_w=log_w, key=key_fixed,
            x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
            beta=beta, hazard=hazard, jump=jump,
        )
        return m["loss/score"]

    g_false = jax.grad(lambda lw: L_score_only(lw, False))(log_w0)
    g_true = jax.grad(lambda lw: L_score_only(lw, True))(log_w0)

    # The False (default) gradient is the v3 expected behavior: nonzero,
    # flowing through (1-S_a). Use it as a magnitude sanity reference.
    norm_false = float(jnp.linalg.norm(g_false))
    norm_true = float(jnp.linalg.norm(g_true))
    assert norm_false > 1e-3, (
        f"baseline L_score log_w grad norm should be nontrivial under "
        f"score_w_stop_gradient=False, got {norm_false:.4g}. If this "
        f"fires, the (1-S_a) gradient path was inadvertently broken."
    )
    assert norm_true < 1e-8, (
        f"under score_w_stop_gradient=True, L_score's log_w grad must "
        f"vanish, got norm {norm_true:.4g}. Check that the full-stop_"
        f"gradient branch was taken in elbo_eta1_loss."
    )


def test_theta_gradient_of_L_score_is_zero_both_flag_values():
    """The score head is fit by cross-entropy only; L_score's theta-grad
    must be exactly zero regardless of score_w_stop_gradient."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(4)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params = _theta_init(jax.random.PRNGKey(5))
    log_w0 = jnp.array([-0.2, 0.1, 0.3], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(321)

    def L_score_only_params(p, flag: bool):
        _, m = _call(
            flag, params=p, log_w=log_w0, key=key_fixed,
            x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
            beta=beta, hazard=hazard, jump=jump,
        )
        return m["loss/score"]

    for flag in (False, True):
        g = jax.grad(lambda p: L_score_only_params(p, flag))(params)
        max_abs = jax.tree_util.tree_reduce(
            lambda acc, x: jnp.maximum(acc, jnp.max(jnp.abs(x))),
            g, jnp.float32(0.0),
        )
        assert float(max_abs) < 1e-8, (
            f"∂_theta L_score should be EXACTLY zero under "
            f"score_w_stop_gradient={flag}, got max|grad|={float(max_abs):.4g}."
        )


def test_total_loss_logw_gradient_drops_score_contribution_when_True():
    """When the flag is True, the total-loss log_w gradient should equal
    the (L_CE + L_RB) gradient — L_score contributes nothing. When False,
    it includes L_score's contribution (the v3 path)."""
    beta, hazard, jump = _make_schedules()
    rng = jax.random.PRNGKey(6)
    x0_idx, x0_anchor, anchor_table = _make_x0(rng)
    params = _theta_init(jax.random.PRNGKey(7))
    log_w0 = jnp.array([-0.3, 0.1, 0.2], dtype=jnp.float32)
    key_fixed = jax.random.PRNGKey(11)

    def total(log_w, flag):
        loss, _ = _call(
            flag, params=params, log_w=log_w, key=key_fixed,
            x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
            beta=beta, hazard=hazard, jump=jump,
        )
        return loss

    def ce_plus_rb(log_w):
        _, m = _call(
            True,  # flag value doesn't matter — we sum CE+RB only
            params=params, log_w=log_w, key=key_fixed,
            x0_idx=x0_idx, x0_anchor=x0_anchor, anchor_table=anchor_table,
            beta=beta, hazard=hazard, jump=jump,
        )
        return m["loss/ce"] + m["loss/rb"]

    g_total_false = jax.grad(lambda lw: total(lw, False))(log_w0)
    g_total_true = jax.grad(lambda lw: total(lw, True))(log_w0)
    g_ce_rb = jax.grad(ce_plus_rb)(log_w0)

    # Sanity: prior_strength=0 in _call, so prior contributes nothing
    # either way; total = L_CE + L_RB + L_score.
    # Under flag=True, L_score's grad is zero, so g_total_true == g_ce_rb.
    assert jnp.allclose(g_total_true, g_ce_rb, atol=1e-7), (
        f"under score_w_stop_gradient=True, d_logw L_total should equal "
        f"d_logw (L_CE + L_RB).\n"
        f"  g_total_true = {g_total_true}\n  g_ce_rb      = {g_ce_rb}"
    )
    # Under flag=False, the score term contributes => g_total_false should
    # DIFFER from g_ce_rb (the v3 score-gradient is nonzero).
    diff_norm = float(jnp.linalg.norm(g_total_false - g_ce_rb))
    assert diff_norm > 1e-3, (
        f"under score_w_stop_gradient=False, d_logw L_total should differ "
        f"from d_logw (L_CE + L_RB) by L_score's nonzero contribution. "
        f"Diff norm = {diff_norm:.4g}. If ~0, the v3 score gradient path "
        f"is broken."
    )
