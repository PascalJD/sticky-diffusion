# src/sticky/core/hazard.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import jax, jax.numpy as jnp
from .sde_vp import B_of_t, alpha_sigma

Array = jnp.ndarray

@dataclass(frozen=True)
class HazardSchedule:
    """Container for a 1D time-dependent hazard lambda(t) on [0, T]."""
    T: float
    lam: Callable[[Array], Array] 
    cum: Callable[[Array], Array]  # cum(t) = \int_0^t lambda(t)
    surv: Callable[[Array], Array]  # S(t) = exp(-cum(t))
    cdf: Callable[[Array], Array]  # F(t) = 1 - S(t)
    inv_cdf: Callable[[Array], Array]  # F^{-1}(u), u in [0, F(T)]

    def first_event_time(self, key: jax.random.PRNGKey, shape=()) -> Array:
        """Inverse-transform sample of the first event time on [0, T]"""
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        FT = self.cdf(jnp.asarray(self.T, dtype=jnp.float32))
        # If u > F(T), no event occurs by T -> return +inf 
        # (caller can clamp to T)
        return jnp.where(u <= FT, self.inv_cdf(u), jnp.inf)

def _invert_B_linear(beta, B_star: Array) -> Array:
    bmin, bdiff, TT = beta.beta_min, beta.beta_diff, beta.T
    B_star = jnp.asarray(B_star, dtype=jnp.float32)
    if abs(bdiff) < 1e-12:
        return B_star / max(bmin, 1e-12)
    A = (bdiff / TT) * 0.5
    disc = jnp.sqrt(jnp.maximum(bmin * bmin + 4.0 * A * B_star, 0.0))
    return (-bmin + disc) / (2.0 * A)

def make_hazard_early(beta, kappa: float = 5.0) -> HazardSchedule:
    """lambda(t)= k beta(t) alpha(t)^2 with alpha^2 = exp(-B)."""
    T = float(beta.T)

    def lam(t):
        B = B_of_t(beta, t)
        return kappa * beta(t) * jnp.exp(-B)

    def cum(t):
        B = B_of_t(beta, t)
        return kappa * (1.0 - jnp.exp(-B))

    def surv(t): return jnp.exp(-cum(t))
    def cdf(t):  return 1.0 - surv(t)

    def inv_cdf(u):
        u = jnp.asarray(u, dtype=jnp.float32)
        F_T = cdf(jnp.asarray(T, jnp.float32))
        u_eff = jnp.clip(u, 0.0, F_T)
        alpha2 = 1.0 + (1.0 / kappa) * jnp.log1p(-u_eff)
        B_star = -jnp.log(jnp.clip(alpha2, a_min=1e-12))
        return _invert_B_linear(beta, B_star)

    return HazardSchedule(
        T=T, lam=lam, cum=cum, surv=surv, cdf=cdf, inv_cdf=inv_cdf
    )