from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.sdes import alpha_sigma, make_beta
from sticky.models.sjd.slack_corruption import sample_slack_pair


_beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)


def test_sample_slack_pair_returns_correct_shape_and_dtype():
    key = jax.random.PRNGKey(0)
    slack_x0 = jnp.ones((4, 27, 9), dtype=jnp.float32)
    t = jnp.full((4,), 0.5, dtype=jnp.float32)
    slack_x_t = sample_slack_pair(key, slack_x0, t, _beta)
    assert slack_x_t.shape == (4, 27, 9)
    assert slack_x_t.dtype == jnp.float32


def test_sample_slack_pair_at_tiny_t_is_close_to_clean():
    """At t -> 0, alpha -> 1 and sigma -> 0, so slack_x_t ≈ slack_x0."""
    key = jax.random.PRNGKey(1)
    slack_x0 = jnp.ones((8, 27, 9), dtype=jnp.float32)
    t = jnp.full((8,), 1e-4, dtype=jnp.float32)
    slack_x_t = sample_slack_pair(key, slack_x0, t, _beta)
    # alpha(1e-4) ≈ 1, sigma(1e-4) ≈ small, so distance to ones is small.
    l2 = jnp.linalg.norm(slack_x_t - slack_x0, axis=-1)
    assert float(jnp.mean(l2)) < 0.1


def test_sample_slack_pair_at_t_one_matches_vp_terminal_statistics():
    """At t = 1, slack is heavily noised; mean ≈ alpha * x0, std ≈ sigma."""
    key = jax.random.PRNGKey(2)
    B = 1024
    slack_x0 = jnp.ones((B, 27, 9), dtype=jnp.float32)
    t = jnp.full((B,), 1.0, dtype=jnp.float32)
    alpha, sigma = alpha_sigma(_beta, t[:1])
    slack_x_t = sample_slack_pair(key, slack_x0, t, _beta)
    # Empirical mean over batch should be near alpha * 1 = alpha.
    mean = float(jnp.mean(slack_x_t))
    std = float(jnp.std(slack_x_t))
    np.testing.assert_allclose(mean, float(alpha[0]), atol=0.02)
    np.testing.assert_allclose(std, float(sigma[0]), atol=0.05)


def test_sample_slack_pair_is_deterministic_given_key():
    slack_x0 = jnp.ones((2, 27, 9), dtype=jnp.float32)
    t = jnp.full((2,), 0.3, dtype=jnp.float32)
    out_a = sample_slack_pair(jax.random.PRNGKey(7), slack_x0, t, _beta)
    out_b = sample_slack_pair(jax.random.PRNGKey(7), slack_x0, t, _beta)
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


def test_sample_slack_pair_changes_with_key():
    slack_x0 = jnp.ones((2, 27, 9), dtype=jnp.float32)
    t = jnp.full((2,), 0.3, dtype=jnp.float32)
    out_a = sample_slack_pair(jax.random.PRNGKey(7), slack_x0, t, _beta)
    out_b = sample_slack_pair(jax.random.PRNGKey(8), slack_x0, t, _beta)
    assert not jnp.allclose(out_a, out_b)
