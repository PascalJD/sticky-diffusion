from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.hazard import (
    make_hazard_linear_time,
    make_hazard_poly_alpha,
)
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta, vp_logpdf
from sticky.models.sjd.corruption import (
    mixture_logpdf,
    sample_pair,
)


def test_mixture_logpdf_simpson_more_accurate():
    """Simpson with tau_grid_size=33 and uniform with tau_grid_size=32 should agree to 1e-2
    on a smooth integrand."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5)

    B = 4
    d = 3
    key = jax.random.PRNGKey(1)
    k_y, k_a = jax.random.split(key)
    y = jax.random.normal(k_y, (B, d))
    anchor = jax.random.normal(k_a, (B, d))
    t = jnp.full((B,), 0.7, dtype=jnp.float32)

    uniform = mixture_logpdf(
        y, anchor, t, beta, hazard, jump, tau_grid_size=32, tau_grid="uniform"
    )
    simpson = mixture_logpdf(
        y, anchor, t, beta, hazard, jump, tau_grid_size=33, tau_grid="simpson"
    )

    np.testing.assert_allclose(
        np.asarray(uniform), np.asarray(simpson), atol=1e-2, rtol=0.0
    )


def test_never_unstuck_mass():
    """Empirical never_unstuck rate matches hazard.surv(t) at MC tolerance."""
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)

    N = 100000
    d = 3
    a = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.tile(a[None, :], (N, 1))
    t_val = 0.6
    t = jnp.full((N,), t_val, dtype=jnp.float32)

    _, mask = sample_pair(
        jax.random.PRNGKey(7), x0, t, beta, hazard, jump
    )

    expected = float(hazard.surv(jnp.asarray(t_val, dtype=jnp.float32)))
    actual = float(jnp.mean(mask.astype(jnp.float32)))

    np.testing.assert_allclose(actual, expected, atol=1e-2)
