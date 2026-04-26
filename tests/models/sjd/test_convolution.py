from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from sticky.models.sjd.convolution import (
    mixture_component_logpdf,
    mixture_component_mean_std,
    mixture_variance,
)
from sticky.models.sjd.sdes import alpha_sigma, make_beta


def test_eta_one_recovers_q_t():
    """At eta=1, v_t(tau) = sigma^2(t) for any tau in (0, t)."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    t = jnp.asarray(0.7, dtype=jnp.float32)
    _, sigma_t = alpha_sigma(beta, t)
    expected = float(jnp.square(sigma_t))
    taus = jnp.linspace(0.05, 0.65, 8, dtype=jnp.float32)
    for tau in taus:
        v = float(mixture_variance(beta, t, tau, eta=1.0))
        np.testing.assert_allclose(v, expected, atol=1e-6, rtol=0.0)


def test_monotone_in_tau():
    """v_t(tau) is strictly monotone in tau in (0, t).

    Note: the paper text on page 11 calls this "increasing", but the same
    paragraph's limits (v -> sigma^2(t) as tau -> 0+, v -> eta^2 sigma^2(t)
    as tau -> t-) and the closed-form expression imply strictly *decreasing*
    in tau for eta < 1. We test the actual monotonicity given by the formula.
    """
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    t = jnp.asarray(0.8, dtype=jnp.float32)
    taus = jnp.linspace(0.02, 0.78, 50, dtype=jnp.float32)
    vs = jnp.stack([mixture_variance(beta, t, tau, eta=0.5) for tau in taus])
    diffs = jnp.diff(vs)
    assert bool(jnp.all(diffs < 0.0)), (
        f"variance not strictly monotone in tau: diffs={np.asarray(diffs)}"
    )


def test_terminal_limit_is_one():
    """At t=T with alpha(T) ~ 0, v_T(tau) ~ 1 regardless of tau and eta,
    for tau bounded away from T."""
    beta = make_beta(beta_min=0.1, beta_max=200.0, T=1.0)
    t = jnp.asarray(1.0, dtype=jnp.float32)
    alpha_T, _ = alpha_sigma(beta, t)
    assert float(alpha_T) < 1e-3, f"alpha(T)={float(alpha_T)} not small enough"

    for tau in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9):
        for eta in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            v = float(
                mixture_variance(
                    beta, t, jnp.asarray(tau, dtype=jnp.float32), eta=eta
                )
            )
            assert abs(v - 1.0) < 1e-3, (
                f"tau={tau}, eta={eta}, v={v}, |v-1|={abs(v - 1.0)}"
            )


def test_logpdf_matches_jax_normal():
    """mixture_component_logpdf agrees with elementwise jax.scipy.stats.norm.logpdf
    summed over the feature axis."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    key = jax.random.PRNGKey(0)
    k_y, k_a, k_t, k_scale = jax.random.split(key, 4)
    B, d = 4, 5
    y = jax.random.normal(k_y, (B, d))
    anchor = jax.random.normal(k_a, (B, d))
    t = jax.random.uniform(
        k_t, (B,), minval=0.1, maxval=0.9
    ).astype(jnp.float32)
    tau = (
        t
        * jax.random.uniform(k_scale, (B,), minval=0.1, maxval=0.9)
    ).astype(jnp.float32)
    eta = 0.7

    mean, std = mixture_component_mean_std(anchor, t, tau, beta, eta)
    std_full = jnp.broadcast_to(std, mean.shape)
    expected = jnp.sum(norm.logpdf(y, loc=mean, scale=std_full), axis=-1)

    actual = mixture_component_logpdf(y, anchor, t, tau, beta, eta)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5
    )
