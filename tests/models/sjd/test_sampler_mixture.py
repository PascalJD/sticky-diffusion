from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp

import sticky.models.sjd.sampler as sampler_mod
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sampler import SamplerConfig, reverse_sample
from sticky.models.sjd.sdes import make_beta


def _stub_hazard():
    """A real hazard built on a flat beta — satisfies hazard.surv/cdf/inv_cdf/
    lam/cum for the score path, while the test monkeypatches the plug-in."""
    return make_hazard_linear_time(make_beta(0.1, 0.1, T=1.0), kappa=1.0)


def _stub_jump(beta):
    """A real VPMatchedGaussianJump with eta=0.5 — provides eta/std_floor to
    classifier_induced_score on the score path that runs alongside the
    monkeypatched plugin."""
    return VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)


def test_reverse_sample_forces_final_jump_with_direct_mixture(monkeypatch):
    choice_probs = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=jnp.float32)
    lam_total = jnp.asarray([[1e-6, 1e-6]], dtype=jnp.float32)

    def fake_plugin_hazard_and_allocation(**kwargs):
        del kwargs
        return lam_total, choice_probs

    monkeypatch.setattr(
        sampler_mod,
        "plugin_hazard_and_allocation",
        fake_plugin_hazard_and_allocation,
    )

    captured = {}

    def fake_sample_mixture_categorical(
        key,
        *,
        destination_probs,
        stay_prob,
        change_prob,
        policy,
    ):
        del key
        captured["policy"] = policy

        def _capture(dest, stay, change):
            captured["destination_probs"] = np.asarray(dest)
            captured["stay_prob"] = np.asarray(stay)
            captured["change_prob"] = np.asarray(change)

        jax.debug.callback(_capture, destination_probs, stay_prob, change_prob)
        a_idx = jnp.argmax(destination_probs, axis=-1).astype(jnp.int32)
        stay_mask = jnp.zeros(destination_probs.shape[:-1], dtype=bool)
        return a_idx, stay_mask

    monkeypatch.setattr(
        sampler_mod,
        "sample_mixture_categorical",
        fake_sample_mixture_categorical,
    )

    def apply_model(params, y, t_img):
        del params, t_img
        logits = jnp.zeros(y.shape[:-1] + (2,), dtype=jnp.float32)
        return logits, None

    anchors = SimpleNamespace(
        table_float=jnp.asarray(
            [
                [-1.0],
                [2.0],
            ],
            dtype=jnp.float32,
        )
    )
    beta = make_beta(beta_min=0.1, beta_max=0.1, T=1.0)
    cfg = SamplerConfig(
        T=1.0,
        n_steps=1,
        alloc_mode="sample",
        force_classify_at_end=True,
        categorical_sampling_policy="exact",
    )

    result = reverse_sample(
        jax.random.PRNGKey(0),
        params=None,
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=_stub_hazard(),
        jump=_stub_jump(beta),
        shape=(2,),
        batch_size=1,
        cfg=cfg,
    )

    np.testing.assert_array_equal(
        np.asarray(result.k),
        np.asarray([[1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.committed),
        np.asarray([[True, True]]),
    )
    np.testing.assert_array_equal(
        captured["destination_probs"],
        np.asarray(choice_probs),
    )
    np.testing.assert_array_equal(
        captured["stay_prob"],
        np.asarray([[0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        captured["change_prob"],
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    assert captured["policy"] == "exact"


def test_reverse_sample_uses_policy_for_final_fill(monkeypatch):
    lam_total = jnp.zeros((1, 2), dtype=jnp.float32)
    choice_probs = jnp.asarray([[[0.5, 0.5], [0.5, 0.5]]], dtype=jnp.float32)
    captured = {}

    def fake_plugin_hazard_and_allocation(**kwargs):
        del kwargs
        return lam_total, choice_probs

    monkeypatch.setattr(
        sampler_mod,
        "plugin_hazard_and_allocation",
        fake_plugin_hazard_and_allocation,
    )

    def fake_sample_mixture_categorical(
        key,
        *,
        destination_probs,
        stay_prob,
        change_prob,
        policy,
    ):
        del key, destination_probs, stay_prob, change_prob
        captured["mixture_policy"] = policy
        stay_mask = jnp.ones((1, 2), dtype=bool)
        return jnp.zeros((1, 2), dtype=jnp.int32), stay_mask

    monkeypatch.setattr(
        sampler_mod,
        "sample_mixture_categorical",
        fake_sample_mixture_categorical,
    )

    def fake_categorical_sample_from_logits(key, logits, *, policy):
        del key
        captured["fill_policy"] = policy

        def _capture(logits_val):
            captured["fill_logits"] = np.asarray(logits_val)

        jax.debug.callback(_capture, logits)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    monkeypatch.setattr(
        sampler_mod,
        "categorical_sample_from_logits",
        fake_categorical_sample_from_logits,
    )

    def apply_model(params, y, t_img):
        del params, y, t_img
        logits = jnp.asarray([[[3.0, -1.0], [-2.0, 4.0]]], dtype=jnp.float32)
        return logits, None

    anchors = SimpleNamespace(
        table_float=jnp.asarray(
            [
                [-1.0],
                [2.0],
            ],
            dtype=jnp.float32,
        )
    )
    beta = make_beta(beta_min=0.1, beta_max=0.1, T=1.0)
    cfg = SamplerConfig(
        T=1.0,
        n_steps=1,
        alloc_mode="sample",
        force_classify_at_end=False,
        categorical_sampling_policy="exact",
    )

    result = reverse_sample(
        jax.random.PRNGKey(0),
        params=None,
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=_stub_hazard(),
        jump=_stub_jump(beta),
        shape=(2,),
        batch_size=1,
        cfg=cfg,
    )

    assert captured["mixture_policy"] == "exact"
    assert captured["fill_policy"] == "exact"
    np.testing.assert_array_equal(
        np.asarray(result.k_filled),
        np.asarray([[0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        captured["fill_logits"],
        np.asarray([[[3.0, -1.0], [-2.0, 4.0]]], dtype=np.float32),
    )


def _real_sampler_inputs():
    """Real (non-monkeypatched) inputs for end-to-end sampler smoke tests."""
    import pytest  # local to avoid affecting earlier tests

    from sticky.models.sjd.anchors import AnchorTable
    from sticky.models.sjd.hazard import make_hazard_linear_time
    from sticky.models.sjd.jump import VPMatchedGaussianJump

    anchors = AnchorTable(table_float=jnp.eye(4, dtype=jnp.float32))
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=2.0)

    def apply_model(params, y, t_img):
        del params, t_img
        a = anchors.table_float
        # Distance-based logits so the score is non-trivial.
        diff = y[..., None, :] - a[None, None, :, :]
        return -jnp.sum(jnp.square(diff), axis=-1), {}

    return anchors, beta, hazard, apply_model


def test_reverse_sample_runs_end_to_end():
    """Smoke test: reverse_sample runs through the real plugin / classifier
    code paths and produces a finite sample of the correct shape."""
    from sticky.models.sjd.jump import VPMatchedGaussianJump
    anchors, beta, hazard, apply_model = _real_sampler_inputs()
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)

    cfg = SamplerConfig(
        T=1.0,
        n_steps=2,
        alloc_mode="argmax",
        categorical_sampling_policy="exact",
        force_classify_at_end=True,
        tau_grid_size=16,
    )
    result = reverse_sample(
        jax.random.PRNGKey(0),
        params={},
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        shape=(5,),
        batch_size=2,
        cfg=cfg,
    )
    assert result.k_filled.shape == (2, 5)
    assert bool(jnp.all(jnp.isfinite(result.k_filled.astype(jnp.float32))))
    assert bool(jnp.all(result.k_filled >= 0))
    assert bool(jnp.all(result.k_filled < 4))

