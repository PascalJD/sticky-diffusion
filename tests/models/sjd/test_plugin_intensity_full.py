from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import jax.numpy as jnp

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import plugin_hazard_and_allocation
from sticky.models.sjd.sdes import make_beta


def _plugin_inputs():
    beta = make_beta(beta_min=0.1, beta_max=0.3, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.7)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.75, std_floor=1e-3)
    anchors = SimpleNamespace(
        table_float=jnp.asarray(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.5],
            ],
            dtype=jnp.float32,
        )
    )
    logits = jnp.asarray(
        [
            [[2.0, 0.0, -1.0], [0.1, 0.5, 0.2]],
            [[-0.3, 1.2, 0.0], [1.5, -0.2, -0.7]],
        ],
        dtype=jnp.float32,
    )
    y = jnp.asarray(
        [
            [[-0.4, 0.2], [0.7, -0.1]],
            [[0.5, 0.9], [-0.8, 0.3]],
        ],
        dtype=jnp.float32,
    )
    t_img = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    return beta, hazard, jump, anchors, logits, y, t_img


def test_plugin_intensity_returns_normalized_probs():
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()

    lam_total, choice_probs = plugin_hazard_and_allocation(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
    )

    assert lam_total.shape == logits.shape[:-1]
    assert choice_probs.shape == logits.shape
    assert lam_total.dtype == jnp.float32
    assert choice_probs.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(jnp.sum(choice_probs, axis=-1)),
        np.ones(logits.shape[:-1], dtype=np.float32),
        atol=1e-6,
    )


def test_sitewise_lam_off_none_is_bit_identical_to_scalar():
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()
    lam_legacy, probs_legacy = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t_img, anchors=anchors, beta=beta,
        hazard=hazard, jump=jump,
    )
    lam_new, probs_new = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t_img, anchors=anchors, beta=beta,
        hazard=hazard, jump=jump, sitewise_lam_off=None,
    )
    np.testing.assert_array_equal(np.asarray(lam_legacy), np.asarray(lam_new))
    np.testing.assert_array_equal(np.asarray(probs_legacy), np.asarray(probs_new))


def test_sitewise_lam_off_preserves_choice_probs_when_uniform_per_site():
    """A per-site constant log_lam_base cancels out of softmax-over-anchors,
    so normalized choice_probs must match the scalar-broadcast baseline."""
    from sticky.models.sjd.hazard import lambda_off_star

    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()
    # Build a (B, S) sitewise tensor equal to the scalar baseline broadcast,
    # then multiply per-site by a varying factor (still uniform over anchors).
    scalar = lambda_off_star(hazard, t_img).astype(jnp.float32)  # (B,)
    B, S = logits.shape[0], logits.shape[1]
    varying = jnp.asarray(np.linspace(0.3, 3.0, S, dtype=np.float32))  # (S,)
    sitewise = scalar[:, None] * varying[None, :]  # (B, S)

    lam_baseline, probs_baseline = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t_img, anchors=anchors, beta=beta,
        hazard=hazard, jump=jump,
    )
    lam_sw, probs_sw = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t_img, anchors=anchors, beta=beta,
        hazard=hazard, jump=jump, sitewise_lam_off=sitewise,
    )
    # Choice probs must be invariant (axis-wise softmax of shifted logits).
    np.testing.assert_allclose(
        np.asarray(probs_sw), np.asarray(probs_baseline), atol=1e-6
    )
    # lam_total scales proportionally to the per-site factor.
    scale = (sitewise / scalar[:, None]).reshape(B, S)
    np.testing.assert_allclose(
        np.asarray(lam_sw),
        np.asarray(lam_baseline) * np.asarray(scale),
        rtol=1e-5, atol=1e-6,
    )


def test_plugin_hazard_and_allocation_normalizes_choice_probs():
    """The plug-in's choice probabilities must normalize to 1 along the
    anchor axis. Smoke check on the (now-singular) corrected SJD path."""
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()
    lam_total, choice_probs = plugin_hazard_and_allocation(
        logits=logits, y=y, t_img=t_img, anchors=anchors, beta=beta,
        hazard=hazard, jump=jump, tau_grid_size=32,
    )
    assert lam_total.shape == logits.shape[:-1]
    assert choice_probs.shape == logits.shape
    np.testing.assert_allclose(
        np.asarray(jnp.sum(choice_probs, axis=-1)),
        np.ones(choice_probs.shape[:-1], dtype=np.float32),
        atol=1e-6,
    )
