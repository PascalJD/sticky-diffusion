from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from sticky.models.discrete_mixture import normalize_probs, sample_mixture_categorical


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
