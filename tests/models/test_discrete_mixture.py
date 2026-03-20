from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from sticky.models.discrete_mixture import (
    categorical_sample_from_logits,
    categorical_sample_from_probs,
    normalize_probs,
    sample_mixture_categorical,
)


def test_normalize_probs_uses_last_entry_as_zero_mass_fallback():
    probs = jnp.zeros((2, 3), dtype=jnp.float32)
    normalized = normalize_probs(probs)
    expected = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(normalized), expected)


def test_sample_mixture_categorical_preserves_shape_and_dtype():
    destination_probs = jnp.asarray(
        [
            [[0.2, 0.8], [0.7, 0.3]],
            [[0.6, 0.4], [0.1, 0.9]],
        ],
        dtype=jnp.float32,
    )
    dest_idx, is_stay = sample_mixture_categorical(
        jax.random.PRNGKey(0),
        destination_probs=destination_probs,
        stay_prob=jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        change_prob=jnp.asarray([0.75, 0.5], dtype=jnp.float32),
    )

    assert dest_idx.shape == (2, 2)
    assert is_stay.shape == (2, 2)
    assert dest_idx.dtype == jnp.int32
    assert is_stay.dtype == jnp.bool_


def test_sample_mixture_categorical_has_deterministic_edge_cases():
    destination_probs = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)

    dest_idx, is_stay = sample_mixture_categorical(
        jax.random.PRNGKey(1),
        destination_probs=destination_probs,
        stay_prob=0.0,
        change_prob=1.0,
    )
    np.testing.assert_array_equal(np.asarray(dest_idx), np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(is_stay), np.asarray([False]))

    dest_idx, is_stay = sample_mixture_categorical(
        jax.random.PRNGKey(2),
        destination_probs=destination_probs,
        stay_prob=1.0,
        change_prob=0.0,
    )
    np.testing.assert_array_equal(np.asarray(is_stay), np.asarray([True]))
    np.testing.assert_array_equal(np.asarray(dest_idx), np.asarray([0], dtype=np.int32))


def test_sample_mixture_categorical_matches_expected_mixture_law():
    destination_probs = jnp.asarray([0.7, 0.3], dtype=jnp.float32)
    stay_prob = 0.4
    change_prob = 0.6
    expected = np.asarray([0.42, 0.18, 0.4], dtype=np.float32)

    n_samples = 20000
    keys = jax.random.split(jax.random.PRNGKey(7), n_samples)

    def draw(key):
        dest_idx, is_stay = sample_mixture_categorical(
            key,
            destination_probs=destination_probs,
            stay_prob=stay_prob,
            change_prob=change_prob,
        )
        return jnp.where(is_stay, 2, dest_idx)

    draws = np.asarray(jax.vmap(draw)(keys))
    counts = np.bincount(draws, minlength=3).astype(np.float32)
    freqs = counts / counts.sum()
    np.testing.assert_allclose(freqs, expected, atol=0.02, rtol=0.0)


def test_sample_mixture_categorical_legacy_low_logits_matches_probs_path():
    destination_logits = jnp.log(jnp.asarray([0.7, 0.3], dtype=jnp.float32))
    destination_probs = jax.nn.softmax(destination_logits, axis=-1)
    stay_prob = 0.4
    change_prob = 0.6
    keys = jax.random.split(jax.random.PRNGKey(11), 256)

    def draw_from_logits(key):
        dest_idx, is_stay = sample_mixture_categorical(
            key,
            destination_logits=destination_logits,
            stay_prob=stay_prob,
            change_prob=change_prob,
            policy="legacy_low",
        )
        return jnp.where(is_stay, 2, dest_idx)

    def draw_from_probs(key):
        dest_idx, is_stay = sample_mixture_categorical(
            key,
            destination_probs=destination_probs,
            stay_prob=stay_prob,
            change_prob=change_prob,
            policy="legacy_low",
        )
        return jnp.where(is_stay, 2, dest_idx)

    np.testing.assert_array_equal(
        np.asarray(jax.vmap(draw_from_logits)(keys)),
        np.asarray(jax.vmap(draw_from_probs)(keys)),
    )


def test_categorical_sample_from_logits_legacy_low_passes_explicit_low_mode(monkeypatch):
    captured = {}

    def fake_categorical(key, logits, *, axis=-1, shape=None, mode=None):
        del key, shape
        captured["axis"] = axis
        captured["mode"] = mode
        return jnp.argmax(logits, axis=axis).astype(jnp.int32)

    monkeypatch.setattr(jax.random, "categorical", fake_categorical)

    logits = jnp.asarray([[1.0, 3.0, -2.0]], dtype=jnp.float32)
    sample = categorical_sample_from_logits(
        jax.random.key(0),
        logits,
        policy="legacy_low",
    )

    assert captured["axis"] == -1
    assert captured["mode"] == "low"
    np.testing.assert_array_equal(np.asarray(sample), np.asarray([1], dtype=np.int32))


def test_sample_mixture_categorical_accepts_logits_for_exact_policy():
    destination_logits = jnp.log(jnp.asarray([0.7, 0.3], dtype=jnp.float32))
    stay_prob = 0.4
    change_prob = 0.6
    expected = np.asarray([0.42, 0.18, 0.4], dtype=np.float32)
    keys = jax.random.split(jax.random.PRNGKey(13), 20000)

    def draw(key):
        dest_idx, is_stay = sample_mixture_categorical(
            key,
            destination_logits=destination_logits,
            stay_prob=stay_prob,
            change_prob=change_prob,
            policy="exact",
        )
        return jnp.where(is_stay, 2, dest_idx)

    draws = np.asarray(jax.vmap(draw)(keys))
    counts = np.bincount(draws, minlength=3).astype(np.float32)
    freqs = counts / counts.sum()
    np.testing.assert_allclose(freqs, expected, atol=0.02, rtol=0.0)


def test_categorical_sample_from_probs_exact_handles_tiny_probabilities_under_jit():
    probs = jnp.asarray([0.0, 1e-30, 1.0], dtype=jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(17), 4096)

    draw_many = jax.jit(
        jax.vmap(
            lambda key: categorical_sample_from_probs(
                key,
                probs,
                policy="exact",
            )
        )
    )

    draws = np.asarray(draw_many(keys))
    assert draws.shape == (4096,)
    assert draws.dtype == np.int32
    assert int(draws.min()) >= 0
    assert int(draws.max()) < 3
    assert not np.any(draws == 0)
