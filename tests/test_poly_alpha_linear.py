"""poly_alpha(p=1) gives linear survival S(t) = 1 - t (CANDI-equivalent hazard).

CANDI's "log-linear noise" is the sigma parameterization; the survival fraction
is linear, which is exactly poly_alpha at p=1. This pins that mapping so the
FIXED forward stack is not silently using a different schedule.
"""
import jax.numpy as jnp

from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.sdes import make_beta


def test_poly_alpha_p1_is_linear_survival():
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    haz = make_hazard_poly_alpha(beta, p=1.0, eps=1e-5)
    # Stay below the eps tail (t <= 1 - eps) where S(t) = (1 - t)^1 exactly.
    ts = jnp.linspace(0.0, 0.99, 100)
    surv = haz.surv(ts)
    expected = 1.0 - ts
    assert jnp.max(jnp.abs(surv - expected)) < 1e-4
    # CDF is the complement.
    assert jnp.max(jnp.abs(haz.cdf(ts) - ts)) < 1e-4
