from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp

import sticky.models.sjd.sampler as sampler_mod
from sticky.models.sjd.sampler import SamplerConfig, reverse_sample
from sticky.models.sjd.sdes import make_beta


def test_reverse_sample_forces_final_jump_with_direct_mixture(monkeypatch):
    choice_probs = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=jnp.float32)
    lam_total = jnp.asarray([[1e-6, 1e-6]], dtype=jnp.float32)

    def fake_plugin_intensity_and_probs(**kwargs):
        del kwargs
        return lam_total, choice_probs

    monkeypatch.setattr(
        sampler_mod,
        "plugin_intensity_and_probs",
        fake_plugin_intensity_and_probs,
    )

    captured = {}

    def fake_sample_mixture_categorical(
        key,
        *,
        destination_probs,
        stay_prob,
        change_prob,
    ):
        del key

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
        intensity_mode="full",
        force_classify_at_end=True,
    )

    result = reverse_sample(
        jax.random.PRNGKey(0),
        params=None,
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=SimpleNamespace(),
        jump=SimpleNamespace(),
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
