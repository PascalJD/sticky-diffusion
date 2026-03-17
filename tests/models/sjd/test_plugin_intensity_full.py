from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import plugin_intensity_and_probs
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


def test_plugin_intensity_full_returns_normalized_probs():
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()

    lam_total, choice_probs = plugin_intensity_and_probs(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        intensity_mode="full",
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


def test_plugin_intensity_chunked_aliases_to_full():
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()

    lam_full, probs_full = plugin_intensity_and_probs(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        intensity_mode="full",
    )

    with pytest.warns(FutureWarning, match="aliases to the full materialized backend"):
        lam_alias, probs_alias = plugin_intensity_and_probs(
            logits=logits,
            y=y,
            t_img=t_img,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            intensity_mode="chunked",
            chunk_size=1,
        )

    np.testing.assert_allclose(np.asarray(lam_alias), np.asarray(lam_full), atol=1e-6)
    np.testing.assert_allclose(np.asarray(probs_alias), np.asarray(probs_full), atol=1e-6)


def test_plugin_intensity_rejects_unknown_mode():
    beta, hazard, jump, anchors, logits, y, t_img = _plugin_inputs()

    with pytest.raises(ValueError, match="Unknown intensity_mode"):
        plugin_intensity_and_probs(
            logits=logits,
            y=y,
            t_img=t_img,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            intensity_mode="streaming",
        )
