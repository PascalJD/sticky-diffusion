from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from sticky.models.sjd.hazard import lam_off_star, make_hazard_linear_time
from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.sitewise_hazard import (
    confidence_to_w,
    sitewise_confidence,
    sitewise_lam_off,
)


def _hazard():
    beta = make_beta(beta_min=0.1, beta_max=0.3, T=1.0)
    return beta, make_hazard_linear_time(beta, kappa=1.7)


def test_confidence_metrics_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(4, 6, 9)).astype(np.float32))
    for metric in ("margin", "top_prob", "entropy"):
        conf = sitewise_confidence(logits, metric)
        assert conf.shape == (4, 6)
        assert float(jnp.min(conf)) >= 0.0
        assert float(jnp.max(conf)) <= 1.0


def test_margin_high_for_peaked_logits_and_low_for_uniform():
    peaked = jnp.asarray([[[10.0, 0.0, 0.0]]], dtype=jnp.float32)
    uniform = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    assert float(sitewise_confidence(peaked, "margin")[0, 0]) > 0.9
    assert abs(float(sitewise_confidence(uniform, "margin")[0, 0])) < 1e-6


def test_confidence_to_w_interpolates_endpoints():
    r = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    w = confidence_to_w(r, w_min=0.5, w_max=2.0)
    np.testing.assert_allclose(np.asarray(w), np.asarray([2.0, 1.25, 0.5]), atol=1e-6)


def test_sitewise_lam_off_reduces_to_scalar_when_w_is_one():
    beta, hazard = _hazard()
    t_img = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    w_i = jnp.ones((2, 5), dtype=jnp.float32)
    lam_sw = sitewise_lam_off(hazard, t_img, w_i)
    lam_scalar = lam_off_star(hazard, t_img)
    # Broadcast scalar across sites.
    np.testing.assert_allclose(
        np.asarray(lam_sw),
        np.broadcast_to(np.asarray(lam_scalar)[:, None], lam_sw.shape),
        rtol=1e-5, atol=1e-8,
    )


def test_sitewise_lam_off_matches_closed_form():
    beta, hazard = _hazard()
    t_img = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    w_i = jnp.asarray([[0.5, 1.0, 2.0], [1.5, 0.8, 1.2]], dtype=jnp.float32)
    got = np.asarray(sitewise_lam_off(hazard, t_img, w_i, eps=1e-12))

    S = np.asarray(hazard.surv(t_img))
    lam = np.asarray(hazard.lam(t_img))
    w_np = np.asarray(w_i)
    S_i = np.power(np.maximum(S[:, None], 1e-30), w_np)
    expected = w_np * lam[:, None] * S_i / np.maximum(1.0 - S_i, 1e-12)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
