"""Tests for the ``rb_theta_stop_gradient`` flag on ``elbo_eta1_loss``.

It is gradient-routing ONLY: when ON it zeroes L_RB's theta-gradient and leaves
L_CE's theta-gradient and L_RB's log_w-gradient untouched; the loss VALUE is
identical to OFF (stop_gradient is forward-identity).
"""
import jax
import jax.numpy as jnp

from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss
from sticky.models.sjd.sdes import make_beta, B_of_t


class _MockHazard:  # anchor-agnostic lam_t = beta(t); cum = B_of_t
    def __init__(self, beta): self.beta = beta
    def lam(self, t): return self.beta(t)
    def cum(self, t): return B_of_t(self.beta, t)


class _MockJump:  # W = I
    eta = 1.0
    blur_kernel = None
    def apply_blur(self, x): return x


def _tnorm(g):
    return float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(g))))


def _setup():
    key = jax.random.PRNGKey(0)
    kp = jax.random.split(key, 5)
    B, S, K, d = 8, 16, 12, 4
    beta = make_beta(0.1, 20.0, 1.0)
    params = {"W": 0.1 * jax.random.normal(kp[0], (d, K)), "b": jnp.zeros((K,))}

    def apply_fn(p, Y, t, **kw):
        return jnp.einsum("...d,dk->...k", Y, p["W"]) + p["b"], {}

    table = jax.random.normal(kp[1], (K, d))
    x0_idx = jax.random.randint(kp[2], (B, S), 0, K)
    x0_anchor = jnp.take(table, x0_idx, axis=0)
    log_w = 0.2 * jax.random.normal(kp[3], (K,))
    common = dict(
        key=kp[4], apply_fn=apply_fn, x0_anchor=x0_anchor, x0_idx=x0_idx,
        beta=beta, hazard=_MockHazard(beta), jump=_MockJump(), T=1.0,
        anchor_table=table, rb_share_sample=True, rb_weight=1.0,
        prior_strength=0.01, score_w_stop_gradient=False,
    )
    return params, log_w, common


def _metrics_fn(common, flag):
    def f(p, w):
        return elbo_eta1_loss(
            params=p, anchor_log_w=w, rb_theta_stop_gradient=flag, **common
        )[1]
    return f


def _value_fn(common, flag):
    def f(p, w):
        return elbo_eta1_loss(
            params=p, anchor_log_w=w, rb_theta_stop_gradient=flag, **common
        )[0]
    return f


def test_flag_is_value_routing_only():
    """ON vs OFF: the loss VALUE is identical (stop_gradient is forward-identity)."""
    params, log_w, common = _setup()
    v_off = float(_value_fn(common, False)(params, log_w))
    v_on = float(_value_fn(common, True)(params, log_w))
    assert abs(v_off - v_on) < 1e-12, f"loss value changed by flag: {v_off} vs {v_on}"


def test_rb_theta_grad_zeroed_when_on():
    """L_RB's theta-grad is nonzero OFF and exactly zero ON (routing works)."""
    params, log_w, common = _setup()
    rb_off = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, False)(p, w)["loss/rb"])(params, log_w))
    rb_on = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, True)(p, w)["loss/rb"])(params, log_w))
    assert rb_off > 1e-6, f"expected nonzero L_RB theta-grad when OFF, got {rb_off}"
    assert rb_on < 1e-10, f"expected ~zero L_RB theta-grad when ON, got {rb_on}"


def test_ce_theta_grad_untouched():
    """L_CE's theta-grad is identical OFF vs ON (L_CE must not be detached)."""
    params, log_w, common = _setup()
    ce_off = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, False)(p, w)["loss/ce"])(params, log_w))
    ce_on = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, True)(p, w)["loss/ce"])(params, log_w))
    assert ce_off > 1e-6, "sanity: L_CE should train theta"
    assert abs(ce_off - ce_on) < 1e-9, f"L_CE theta-grad changed by flag: {ce_off} vs {ce_on}"


def test_score_theta_grad_zero_both():
    """L_score's theta-grad is zero regardless of the flag (residual stop_gradient)."""
    params, log_w, common = _setup()
    for flag in (False, True):
        sc = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, flag)(p, w)["loss/score"])(params, log_w))
        assert sc < 1e-10, f"L_score theta-grad should be 0 (flag={flag}), got {sc}"


def test_rb_logw_grad_preserved_when_on():
    """OVER-DETACH GUARD: L_RB's log_w-grad is identical OFF vs ON.

    The hazard must still learn from L_RB under the flag. If this fires, the
    stop_gradient leaked into the log_w path (e.g. wrapped one_minus_S/lam_hat,
    or the whole RB term)."""
    params, log_w, common = _setup()
    rb_w_off = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, False)(p, w)["loss/rb"], argnums=1)(params, log_w))
    rb_w_on = _tnorm(jax.grad(lambda p, w: _metrics_fn(common, True)(p, w)["loss/rb"], argnums=1)(params, log_w))
    assert rb_w_off > 1e-6, "sanity: L_RB should feed log_w"
    assert abs(rb_w_off - rb_w_on) < 1e-9, f"L_RB log_w-grad changed by flag: {rb_w_off} vs {rb_w_on}"
