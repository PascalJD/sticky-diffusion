from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses_slack import ce_allocation_loss_with_slack
from sticky.models.sjd.sdes import make_beta


VOCAB = 9
SEQ_LEN = 81
SLACK_LEN = 27
JOINT_LEN = SEQ_LEN + SLACK_LEN


def _make_apply_fn(captured: dict):
    """Returns zero logits of shape (B, 108, 9) and records the slack input shape."""
    def apply_fn(params, cell_xt, slack_xt, t_img):
        del params, t_img
        captured["cell_xt_shape"] = tuple(cell_xt.shape)
        captured["slack_xt_shape"] = tuple(slack_xt.shape)
        B = cell_xt.shape[0]
        return jnp.zeros((B, JOINT_LEN, VOCAB), dtype=jnp.float32), {}
    return apply_fn


def _make_inputs(B: int = 4, seed: int = 0):
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.6)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.97, std_floor=1e-3)
    anchor_table = jnp.eye(VOCAB, dtype=jnp.float32)

    rng = jax.random.PRNGKey(seed)
    rng_idx, rng_slack = jax.random.split(rng)
    x0_idx = jax.random.randint(rng_idx, (B, SEQ_LEN), 0, VOCAB).astype(jnp.int32)
    x0_anchor = anchor_table[x0_idx]  # (B, 81, 9)
    slack_x0 = jnp.ones((B, SLACK_LEN, VOCAB), dtype=jnp.float32)
    return dict(
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        slack_x0=slack_x0,
        beta=beta,
        hazard=hazard,
        jump=jump,
        anchor_table=anchor_table,
        T=1.0,
    )


def test_loss_returns_finite_metrics_with_full_diagnostics():
    captured: dict = {}
    apply_fn = _make_apply_fn(captured)
    inputs = _make_inputs(B=8)

    loss, metrics = ce_allocation_loss_with_slack(
        key=jax.random.PRNGKey(42),
        params={},
        apply_fn=apply_fn,
        **inputs,
    )

    assert captured["cell_xt_shape"] == (8, SEQ_LEN, VOCAB)
    assert captured["slack_xt_shape"] == (8, SLACK_LEN, VOCAB)
    assert np.isfinite(float(loss))

    expected_keys = {
        "loss",
        "loss/ce_nll_bits",
        "loss/acc_top1",
        "loss/frac_active",
        "loss/frac_never_unstuck",
        "loss/slack_residual_l2",
        "loss/slack_sigma_t_mean",
        "t/mean",
        "t/std",
        "state_dep/log_ratio_mean",
        "state_dep/log_ratio_std",
    }
    missing = expected_keys - set(metrics.keys())
    assert not missing, f"missing metrics: {missing}"
    for key in expected_keys:
        assert np.isfinite(float(metrics[key])), f"{key} not finite"


def test_loss_value_matches_uniform_baseline_for_zero_logits():
    """With zero logits, per-site NLL is log(V). The aggregate loss equals
    log(V) on the active sites and 0 on masked sites; the weighted mean is
    therefore log(V) whenever effective_loss_weight > 0."""
    inputs = _make_inputs(B=64)
    captured: dict = {}
    loss, metrics = ce_allocation_loss_with_slack(
        key=jax.random.PRNGKey(0),
        params={},
        apply_fn=_make_apply_fn(captured),
        **inputs,
    )
    expected = float(jnp.log(VOCAB))
    np.testing.assert_allclose(float(loss), expected, atol=1e-5)
    np.testing.assert_allclose(
        float(metrics["loss/acc_top1"]), 1.0 / VOCAB, atol=1e-2
    )


def test_slack_residual_l2_metric_matches_sigma_t():
    """The residual = slack_x_t - alpha(t) * slack_x0 has per-coordinate std
    sigma(t). Per-row L2 / sqrt(d) is therefore an unbiased estimate of
    sigma(t), and averaging over a large enough batch + over the U(0, T)
    time distribution should converge to E[sigma(t)] which the loss reports
    as `loss/slack_sigma_t_mean`."""
    inputs = _make_inputs(B=2048)
    captured: dict = {}
    _, metrics = ce_allocation_loss_with_slack(
        key=jax.random.PRNGKey(0),
        params={},
        apply_fn=_make_apply_fn(captured),
        **inputs,
    )
    measured = float(metrics["loss/slack_residual_l2"])
    expected = float(metrics["loss/slack_sigma_t_mean"])
    assert measured >= 0.0
    np.testing.assert_allclose(measured, expected, atol=0.02)


def test_slack_residual_l2_matches_known_sigma_at_fixed_time(monkeypatch):
    """Force t = 0.5 deterministically and check the per-coordinate residual
    std equals sigma(0.5) within Monte Carlo noise."""
    from sticky.models.sjd import losses_slack as ls
    from sticky.models.sjd.sdes import alpha_sigma

    inputs = _make_inputs(B=4096)
    fixed_t = jnp.full((4096,), 0.5, dtype=jnp.float32)

    real_uniform = jax.random.uniform

    def patched_uniform(key, shape, *, minval=0.0, maxval=1.0, dtype=jnp.float32):
        if shape == (4096,) and float(maxval) == 1.0 and float(minval) == 0.0:
            return fixed_t
        return real_uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)

    monkeypatch.setattr(jax.random, "uniform", patched_uniform)

    captured: dict = {}
    _, metrics = ls.ce_allocation_loss_with_slack(
        key=jax.random.PRNGKey(0),
        params={},
        apply_fn=_make_apply_fn(captured),
        **inputs,
    )
    _, sigma_at_t = alpha_sigma(inputs["beta"], jnp.asarray([0.5], dtype=jnp.float32))
    expected_sigma = float(sigma_at_t[0])
    measured = float(metrics["loss/slack_residual_l2"])
    np.testing.assert_allclose(measured, expected_sigma, atol=0.02)


def test_given_mask_excludes_clue_sites_from_loss():
    """When given_mask is all True, no site contributes to the loss; the
    weighted mean falls back to the safety denom and the loss is 0."""
    inputs = _make_inputs(B=4)
    given_mask = jnp.ones((4, SEQ_LEN), dtype=jnp.bool_)
    captured: dict = {}
    loss, metrics = ce_allocation_loss_with_slack(
        key=jax.random.PRNGKey(0),
        params={},
        apply_fn=_make_apply_fn(captured),
        given_mask=given_mask,
        **inputs,
    )
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(metrics["loss/frac_active"]), 0.0)


def test_rejects_wrong_slack_shape():
    inputs = _make_inputs(B=4)
    bad_inputs = dict(inputs)
    bad_inputs["slack_x0"] = jnp.ones((4, 26, 9), dtype=jnp.float32)
    captured: dict = {}
    try:
        ce_allocation_loss_with_slack(
            key=jax.random.PRNGKey(0),
            params={},
            apply_fn=_make_apply_fn(captured),
            **bad_inputs,
        )
    except ValueError as exc:
        assert "(B, 27, 9)" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong slack shape")


def test_jit_traces_under_jax_jit():
    inputs = _make_inputs(B=4)
    captured: dict = {}
    apply_fn = _make_apply_fn(captured)

    @jax.jit
    def step(key, x0_anchor, x0_idx, slack_x0):
        return ce_allocation_loss_with_slack(
            key=key,
            params={},
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            slack_x0=slack_x0,
            beta=inputs["beta"],
            hazard=inputs["hazard"],
            jump=inputs["jump"],
            anchor_table=inputs["anchor_table"],
            T=inputs["T"],
        )

    loss, metrics = step(
        jax.random.PRNGKey(0),
        inputs["x0_anchor"],
        inputs["x0_idx"],
        inputs["slack_x0"],
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["state_dep/log_ratio_mean"]))
