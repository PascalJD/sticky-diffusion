"""Unit tests for log_lambda_hat_per_anchor — the eta=1 closed-form
log lambda_hat_t(a), extracted to be shared between the existing plug-in
sampler and the new elbo_eta1 training loss.

These tests pin down the formula:
    log lambda_hat_t(a) = log lam(t) + log w(a) + w(a) * log S(t)
                          - log(1 - S(t)^{w(a)})
per appendix eq:c2-collapse, where lam(t) = hazard.lam(t) is the SJD
forward hazard schedule (the appendix's "beta(t)" — a notation collision
with the VP noise schedule, which is a separate function), S(t) = exp(-H(t)).
"""
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.plugin_intensity import log_lambda_hat_per_anchor


def _schedules():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    return beta, hazard


def test_anchor_agnostic_w_equals_one():
    """w(a) = 1 for all a => log lam_hat(a) is independent of a and matches
    log[lam(t) * S(t) / (1 - S(t))]."""
    beta, hazard = _schedules()
    K = 5
    log_w = jnp.zeros((K,), dtype=jnp.float32)
    t = jnp.array([0.3, 0.7], dtype=jnp.float32)  # B=2

    log_lh = log_lambda_hat_per_anchor(
        t_img=t, hazard=hazard, log_w_table=log_w,
    )
    assert log_lh.shape == (2, K), f"expected (2,{K}), got {log_lh.shape}"
    # All anchors equal at w == 1
    assert jnp.allclose(log_lh[0, 0], log_lh[0, 1:], atol=1e-6)
    assert jnp.allclose(log_lh[1, 0], log_lh[1, 1:], atol=1e-6)
    # Match anchor-agnostic closed form using hazard.lam, not VP beta
    H_t = hazard.cum(t)  # (B,)
    expected = jnp.log(hazard.lam(t)) - H_t - jnp.log(-jnp.expm1(-H_t))
    assert jnp.allclose(log_lh[:, 0], expected, atol=1e-5)


def test_non_trivial_w_matches_closed_form():
    """For non-trivial w(a), each anchor matches
        log lam(t) + log w(a) + w(a) log S(t) - log(1 - S(t)^{w(a)})."""
    beta, hazard = _schedules()
    log_w = jnp.array([-0.5, 0.0, 0.7], dtype=jnp.float32)
    w = jnp.exp(log_w)  # [0.6065, 1.0, 2.0138]
    t = jnp.array([0.5], dtype=jnp.float32)

    log_lh = log_lambda_hat_per_anchor(
        t_img=t, hazard=hazard, log_w_table=log_w,
    )
    assert log_lh.shape == (1, 3), f"expected (1,3), got {log_lh.shape}"

    H_t = float(hazard.cum(t)[0])
    log_lam_t = float(jnp.log(hazard.lam(t))[0])
    for a in range(3):
        wa = float(w[a])
        log_w_a = float(log_w[a])
        log_S_a = -wa * H_t  # log S(t)^{w(a)}
        log_one_minus = float(jnp.log(-jnp.expm1(log_S_a)))
        expected_a = log_lam_t + log_w_a + log_S_a - log_one_minus
        got = float(log_lh[0, a])
        assert abs(got - expected_a) < 1e-4, (
            f"anchor {a}: got {got}, want {expected_a}"
        )
