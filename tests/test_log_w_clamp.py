"""Layer-A tests for the e2e ELBO NaN fix (plan v3 H1+H2).

Validates that the single chokepoint clamp in ``TokenAnchors.log_w_float()``
prevents float32 overflow in downstream code paths (the prior term in
particular, where unguarded ``0.0 * inf = NaN`` under IEEE 754).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.sjd.anchors import AnchorTableConfig, TokenAnchors
from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


# Tiny dims for fast tests.
B, L_seq, K, d = 4, 1, 3, 2


def _make_anchors(*, log_w_init, log_w_clip):
    cfg = AnchorTableConfig(family="normal", vocab_size=K, anchor_dim=d)
    return TokenAnchors(
        config=cfg,
        learnable=True,
        learnable_log_w=True,
        log_w_init=log_w_init,
        log_w_clip=log_w_clip,
    )


def _init_and_get_log_w(anchors):
    """Init a TokenAnchors module and return its (clamped, recentered) log_w."""
    dummy_ids = jnp.zeros((1,), dtype=jnp.int32)
    variables = anchors.init(jax.random.PRNGKey(0), dummy_ids)
    return anchors.apply(variables, method=anchors.log_w_float)


def test_log_w_clamp_applied_after_recentering():
    """With raw log_w = [100, -100, 0]: recentering makes mean=0 (no shift, since
    mean is already 0), then clamp [-3, 3] bounds the entries.
    """
    raw = jnp.array([100.0, -100.0, 0.0], dtype=jnp.float32)
    anchors = _make_anchors(log_w_init=raw, log_w_clip=(-3.0, 3.0))
    log_w = _init_and_get_log_w(anchors)
    assert log_w.shape == (K,)
    assert bool(jnp.all(jnp.isfinite(log_w)))
    # Recentered values: raw - mean(raw) = raw (mean is 0). Then clamped to [-3, 3].
    expected = jnp.array([3.0, -3.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(log_w, expected), f"got {log_w}, expected {expected}"


def test_log_w_clamp_none_passes_through():
    """log_w_clip=None disables the clamp; values are returned raw (modulo
    recentering). Preserves back-compat for configs that don't set the clamp.
    """
    raw = jnp.array([5.0, -2.0, 0.0], dtype=jnp.float32)
    anchors = _make_anchors(log_w_init=raw, log_w_clip=None)
    log_w = _init_and_get_log_w(anchors)
    # Recentered: raw - mean = [5 - 1, -2 - 1, 0 - 1] = [4, -3, -1]
    expected = raw - jnp.mean(raw)
    assert jnp.allclose(log_w, expected), f"got {log_w}, expected {expected}"


def test_log_w_clamp_recentering_then_clamp_order():
    """The clamp is applied AFTER recentering. Raw values that violate the
    clamp before recentering may be valid after, and vice versa.
    Raw = [10, 10, 10]; mean=10; recentered=[0,0,0]; clamp([-3,3]) = [0,0,0].
    """
    raw = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    anchors = _make_anchors(log_w_init=raw, log_w_clip=(-3.0, 3.0))
    log_w = _init_and_get_log_w(anchors)
    expected = jnp.zeros((K,), dtype=jnp.float32)
    assert jnp.allclose(log_w, expected), f"got {log_w}, expected {expected}"


def test_prior_safe_under_clamp_when_raw_log_w_would_overflow():
    """Plan v3 H2: with prior_strength=0.0 and raw log_w containing 100.0,
    w = exp(log_w) would overflow float32 to inf, and 0.0 * inf = NaN. The
    clamp in log_w_float() bounds the values fed to the loss, so the prior
    term stays finite end-to-end.
    """
    # Raw log_w with one entry guaranteed to overflow if not clamped:
    # exp(100) ~ 2.7e43 >> float32 max (3.4e38).
    raw = jnp.array([100.0, -50.0, -50.0], dtype=jnp.float32)
    anchors = _make_anchors(log_w_init=raw, log_w_clip=(-3.0, 3.0))
    log_w = _init_and_get_log_w(anchors)
    assert bool(jnp.all(jnp.isfinite(log_w)))

    # Plug the clamped log_w into the loss with prior_strength=0.0. Without
    # the clamp, w**2 would overflow inside the prior and 0.0 * inf = NaN.
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    rng = jax.random.PRNGKey(0)
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(B, L_seq), minval=0, maxval=K)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)

    def _fake_apply(params, x_in, t_img):
        return jnp.zeros(x_in.shape[:-1] + (K,), dtype=jnp.float32), {}

    loss, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(1),
        params={},
        apply_fn=_fake_apply,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=0.0,
    )
    assert bool(jnp.isfinite(loss)), f"loss not finite: {loss}"
    assert bool(jnp.isfinite(metrics["loss/prior"])), (
        f"prior NaN'd despite clamp: {metrics['loss/prior']}"
    )


def test_prior_safe_at_clamp_boundary_with_nonzero_strength():
    """With the clamp at [-3, 3], w_max = exp(3) ~ 20. (w-1)^2 ~ 361, finite.
    Multiplied by any prior_strength, the prior term stays finite.
    """
    raw = jnp.array([100.0, -100.0, 50.0], dtype=jnp.float32)  # all out of range
    anchors = _make_anchors(log_w_init=raw, log_w_clip=(-3.0, 3.0))
    log_w = _init_and_get_log_w(anchors)
    w = jnp.exp(log_w)
    assert bool(jnp.all(jnp.isfinite(w))), f"w not finite: {w}"

    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    rng = jax.random.PRNGKey(2)
    rng_idx, rng_anc = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, shape=(B, L_seq), minval=0, maxval=K)
    anchor_table = jax.random.normal(rng_anc, shape=(K, d))
    x0_anchor = jnp.take(anchor_table, x0_idx, axis=0)

    def _fake_apply(params, x_in, t_img):
        return jnp.zeros(x_in.shape[:-1] + (K,), dtype=jnp.float32), {}

    loss, metrics = elbo_eta1_loss(
        key=jax.random.PRNGKey(3),
        params={},
        apply_fn=_fake_apply,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
        anchor_log_w=log_w,
        anchor_table=anchor_table,
        prior_strength=0.01,
    )
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["loss/prior"]))
    assert metrics["loss/prior"] > 0.0
