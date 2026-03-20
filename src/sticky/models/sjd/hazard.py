from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import B_of_t

Array = jnp.ndarray


@dataclass(frozen=True)
class HazardSchedule:

    T: float
    lam: Callable[[Array], Array]
    cum: Callable[[Array], Array]  # cum(t) = \int_0^t lambda(s) ds
    surv: Callable[[Array], Array]  # S(t) = exp(-cum(t))
    cdf: Callable[[Array], Array]  # F(t) = 1 - S(t)
    inv_cdf: Callable[[Array], Array]  # F^{-1}(u), u in [0, F(T)]

    def first_event_time(self, key: PRNGKey, shape=()) -> Array:
        """Inverse-transform sample of the first event time on [0, T]."""
        u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
        FT = self.cdf(jnp.asarray(self.T, dtype=jnp.float32))
        # If u > F(T), no event occurs by T -> return +inf (caller can clamp to T)
        return jnp.where(u <= FT, self.inv_cdf(u), jnp.inf)


def _invert_B_linear(beta, B_star: Array) -> Array:
    """Invert B(t) for the linear-beta schedule."""
    bmin, bdiff, TT = beta.beta_min, beta.beta_diff, beta.T
    B_star = jnp.asarray(B_star, dtype=jnp.float32)
    if abs(bdiff) < 1e-12:
        return B_star / max(bmin, 1e-12)
    A = (bdiff / TT) * 0.5
    disc = jnp.sqrt(jnp.maximum(bmin * bmin + 4.0 * A * B_star, 0.0))
    return (-bmin + disc) / (2.0 * A)


def _validate_kappa(kappa: float) -> float:
    k = float(kappa)
    if k <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    return k


def _make_common_terms(
    *,
    T: float,
    lam_fn: Callable[[Array], Array],
    cum_fn: Callable[[Array], Array],
    inv_cdf_fn: Callable[[Array], Array],
) -> HazardSchedule:
    def lam(t):
        t = jnp.asarray(t, dtype=jnp.float32)
        return lam_fn(t)

    def cum(t):
        t = jnp.asarray(t, dtype=jnp.float32)
        return cum_fn(t)

    def surv(t):
        return jnp.exp(-cum(t))

    def cdf(t):
        return 1.0 - surv(t)

    def inv_cdf(u):
        u = jnp.asarray(u, dtype=jnp.float32)
        F_T = cdf(jnp.asarray(T, dtype=jnp.float32))
        u_eff = jnp.clip(u, 0.0, F_T)
        return inv_cdf_fn(u_eff)

    return HazardSchedule(
        T=float(T),
        lam=lam,
        cum=cum,
        surv=surv,
        cdf=cdf,
        inv_cdf=inv_cdf,
    )


def _invert_cum_bisect(
    *,
    cum_fn: Callable[[Array], Array],
    T: float,
    h_target: Array,
    steps: int = 64,
) -> Array:
    """Monotone inverse of cumulative hazard via bisection on [0, T]."""
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    h_target = jnp.asarray(h_target, dtype=jnp.float32)
    lo = jnp.zeros_like(h_target)
    hi = jnp.full_like(h_target, float(T))

    def body_fn(_, state):
        lo_i, hi_i = state
        mid = 0.5 * (lo_i + hi_i)
        h_mid = cum_fn(mid)
        go_right = h_mid < h_target
        lo_i = jnp.where(go_right, mid, lo_i)
        hi_i = jnp.where(go_right, hi_i, mid)
        return (lo_i, hi_i)

    lo, hi = jax.lax.fori_loop(0, int(steps), body_fn, (lo, hi))
    return 0.5 * (lo + hi)


def make_hazard_early(beta, kappa: float = 5.0) -> HazardSchedule:
    k = _validate_kappa(kappa)
    T = float(beta.T)

    def lam(t):
        B = B_of_t(beta, t)
        return k * beta(t) * jnp.exp(-B)

    def cum(t):
        B = B_of_t(beta, t)
        return k * (1.0 - jnp.exp(-B))

    def surv(t):
        return jnp.exp(-cum(t))

    def cdf(t):
        return 1.0 - surv(t)

    def inv_cdf(u):
        u = jnp.asarray(u, dtype=jnp.float32)
        F_T = cdf(jnp.asarray(T, jnp.float32))
        u_eff = jnp.clip(u, 0.0, F_T)
        alpha2 = 1.0 + (1.0 / k) * jnp.log1p(-u_eff)
        B_star = -jnp.log(jnp.clip(alpha2, a_min=1e-12))
        return _invert_B_linear(beta, B_star)

    return HazardSchedule(T=T, lam=lam, cum=cum, surv=surv, cdf=cdf, inv_cdf=inv_cdf)


def make_hazard_linear_time(beta, kappa: float = 3.0) -> HazardSchedule:
    """Cumulative hazard linear in time: H(t)=kappa*(t/T)."""
    k = _validate_kappa(kappa)
    T = float(beta.T)
    rate = k / max(T, 1e-12)

    def lam_fn(t):
        return jnp.full_like(t, rate, dtype=jnp.float32)

    def cum_fn(t):
        return jnp.asarray(rate, dtype=jnp.float32) * t

    def inv_cdf_fn(u_eff):
        h = -jnp.log1p(-u_eff)
        return h / jnp.maximum(jnp.asarray(rate, dtype=jnp.float32), 1e-12)

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def make_hazard_poly_alpha(
    beta,
    p: float = 1.0,
    eps: float = 1e-5,
) -> HazardSchedule:
    """Polynomial survival schedule with a matched finite tail.

    For t <= T(1 - eps), this matches S(t) = (1 - t/T)^p exactly. For the last
    eps-fraction of the interval, it switches to a constant hazard chosen to
    match both the value and slope of the cumulative hazard at the splice point.
    This keeps lam, cum, surv, and inv_cdf mutually consistent while avoiding
    the singularity at t = T.
    """
    T = float(beta.T)
    p_f = float(p)
    eps_f = float(eps)
    if p_f <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")
    if (eps_f <= 0.0) or (eps_f >= 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    inv_T = 1.0 / max(T, 1e-12)
    T_arr = jnp.asarray(T, dtype=jnp.float32)
    eps_arr = jnp.asarray(eps_f, dtype=jnp.float32)
    tail_start = T_arr * (1.0 - eps_arr)
    tail_rate = jnp.asarray(p_f * inv_T / eps_f, dtype=jnp.float32)
    tail_cum_start = jnp.asarray(-p_f * jnp.log(eps_arr), dtype=jnp.float32)

    def _t_clipped(t):
        return jnp.clip(jnp.asarray(t, dtype=jnp.float32), 0.0, T_arr)

    def _one_minus_exact(t):
        tau = _t_clipped(t) * inv_T
        return jnp.clip(1.0 - tau, a_min=0.0, a_max=1.0)

    def lam_fn(t):
        one_minus = jnp.maximum(_one_minus_exact(t), eps_arr)
        return (p_f * inv_T) / one_minus

    def cum_fn(t):
        t = _t_clipped(t)
        one_minus = _one_minus_exact(t)
        exact_cum = -p_f * jnp.log(jnp.maximum(one_minus, eps_arr))
        tail_cum = tail_cum_start + tail_rate * (t - tail_start)
        return jnp.where(t <= tail_start, exact_cum, tail_cum)

    def inv_cdf_fn(u_eff):
        u_eff = jnp.asarray(u_eff, dtype=jnp.float32)
        h = -jnp.log1p(-u_eff)
        one_minus = jnp.exp(-h / p_f)
        exact_t = T_arr * (1.0 - one_minus)
        tail_t = tail_start + (h - tail_cum_start) / jnp.maximum(tail_rate, 1e-12)
        t = jnp.where(h <= tail_cum_start, exact_t, tail_t)
        return jnp.clip(t, 0.0, T_arr)

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def make_hazard_cosine_alpha(
    beta,
    eps: float = 1e-4,
) -> HazardSchedule:
    """Hazard whose survival matches the MD4 cosine alpha schedule.

    With tau=t/T, this uses the clipped MD4 keep-probability

      S(t) = alpha(t) = (1 - 2*eps) * (1 - sin(pi * tau / 2)) + eps.

    This matches the forward masking survival used by MD4's cosine schedule,
    including its endpoint regularization.
    """
    T = float(beta.T)
    eps_f = float(eps)
    if not (0.0 <= eps_f < 0.5):
        raise ValueError(f"eps must be in [0, 0.5), got {eps}")

    T_arr = jnp.asarray(T, dtype=jnp.float32)
    eps_arr = jnp.asarray(eps_f, dtype=jnp.float32)
    scale = jnp.asarray(1.0 - 2.0 * eps_f, dtype=jnp.float32)
    half_pi_over_T = 0.5 * jnp.pi / max(T, 1e-12)

    def _theta(t):
        t = jnp.clip(jnp.asarray(t, dtype=jnp.float32), 0.0, T_arr)
        return half_pi_over_T * t

    def _surv_exact(t):
        theta = _theta(t)
        raw = 1.0 - jnp.sin(theta)
        return scale * raw + eps_arr

    def lam_fn(t):
        theta = _theta(t)
        numer = scale * half_pi_over_T * jnp.cos(theta)
        denom = jnp.maximum(_surv_exact(t), 1e-12)
        return numer / denom

    def cum_fn(t):
        return -jnp.log(jnp.maximum(_surv_exact(t), 1e-12))

    def inv_cdf_fn(u_eff):
        # Since the clipped MD4 alpha has S(0)=1-eps, the induced event CDF has
        # F(0)=eps. Values below that map to t=0.
        sin_arg = (jnp.asarray(u_eff, dtype=jnp.float32) - eps_arr) / jnp.maximum(scale, 1e-12)
        sin_arg = jnp.clip(sin_arg, 0.0, 1.0)
        return (2.0 * T_arr / jnp.pi) * jnp.arcsin(sin_arg)

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def make_hazard_linear_B(beta, kappa: float = 3.0) -> HazardSchedule:
    """Cumulative hazard linear in VP accumulated noise: H(t)=kappa*B(t)/B(T)."""
    k = _validate_kappa(kappa)
    T = float(beta.T)
    B_T = jnp.maximum(B_of_t(beta, jnp.asarray(T, dtype=jnp.float32)), 1e-12)

    def lam_fn(t):
        return (k / B_T) * beta(t)

    def cum_fn(t):
        return k * B_of_t(beta, t) / B_T

    def inv_cdf_fn(u_eff):
        h = -jnp.log1p(-u_eff)
        B_star = (B_T / k) * h
        return _invert_B_linear(beta, B_star)

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def make_hazard_cosine(beta, kappa: float = 3.0) -> HazardSchedule:
    """Cosine-late cumulative hazard: H(t)=kappa*(1-cos(pi*t/(2T)))."""
    k = _validate_kappa(kappa)
    T = float(beta.T)
    half_pi_over_T = 0.5 * jnp.pi / max(T, 1e-12)

    def _s(t):
        return half_pi_over_T * t

    def lam_fn(t):
        return k * half_pi_over_T * jnp.sin(_s(t))

    def cum_fn(t):
        return k * (1.0 - jnp.cos(_s(t)))

    def inv_cdf_fn(u_eff):
        h = -jnp.log1p(-u_eff)
        arg = 1.0 - (h / k)
        arg = jnp.clip(arg, -1.0, 1.0)
        return (2.0 * T / jnp.pi) * jnp.arccos(arg)

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def make_hazard_edm(
    beta,
    kappa: float = 3.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    inv_steps: int = 64,
    eps: float = 1e-8,
) -> HazardSchedule:
    """EDM-style cumulative hazard using a normalized Normal-CDF in log-sigma.

    Define u(t)=log(sigma(t)) for VP sigma(t)=sqrt(1-exp(-B(t))). Then:
      H(t)=kappa * normalize( Phi((u(t)-p_mean)/p_std) ),
    where normalization enforces H(0)=0 and H(T)=kappa.
    """
    T = float(beta.T)
    k = _validate_kappa(kappa)
    p_std_f = float(p_std)
    if p_std_f <= 0.0:
        raise ValueError(f"p_std must be > 0, got {p_std}")

    eps_f = float(eps)
    inv_steps_i = int(inv_steps)

    def _log_sigma(t):
        B = B_of_t(beta, t)
        sigma2 = jnp.clip(1.0 - jnp.exp(-B), a_min=eps_f)
        return 0.5 * jnp.log(sigma2)

    def _normal_cdf(z):
        return 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))

    def _normal_pdf(z):
        return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)

    u0 = _log_sigma(jnp.asarray(0.0, dtype=jnp.float32))
    uT = _log_sigma(jnp.asarray(T, dtype=jnp.float32))
    z0 = (u0 - float(p_mean)) / p_std_f
    zT = (uT - float(p_mean)) / p_std_f
    c0 = _normal_cdf(z0)
    cT = _normal_cdf(zT)
    c_span = jnp.maximum(cT - c0, 1e-12)

    def _g(t):
        u = _log_sigma(t)
        z = (u - float(p_mean)) / p_std_f
        c = _normal_cdf(z)
        return jnp.clip((c - c0) / c_span, 0.0, 1.0)

    def cum_fn(t):
        return k * _g(t)

    def lam_fn(t):
        B = B_of_t(beta, t)
        exp_neg_B = jnp.exp(-B)
        sigma2 = jnp.clip(1.0 - exp_neg_B, a_min=eps_f)
        dlog_sigma_dt = 0.5 * beta(t) * exp_neg_B / sigma2
        u = _log_sigma(t)
        z = (u - float(p_mean)) / p_std_f
        dcdf_dt = _normal_pdf(z) * dlog_sigma_dt / p_std_f
        dg_dt = dcdf_dt / c_span
        return jnp.maximum(k * dg_dt, 0.0)

    def inv_cdf_fn(u_eff):
        h = -jnp.log1p(-u_eff)
        return _invert_cum_bisect(
            cum_fn=cum_fn,
            T=T,
            h_target=h,
            steps=inv_steps_i,
        )

    return _make_common_terms(T=T, lam_fn=lam_fn, cum_fn=cum_fn, inv_cdf_fn=inv_cdf_fn)


def lam_time_star(h: HazardSchedule, t: Array) -> Array:
    r"""\lambda^*_time(t) from the SJD paper (Proposition 5.3)."""
    t = jnp.asarray(t, dtype=jnp.float32)
    M = h.surv(t)
    return h.lam(t) * M


def lam_off_star(h: HazardSchedule, t: Array, eps: float = 1e-12) -> Array:
    r"""\lambda^*_off(t) from the SJD paper (Proposition 5.3)."""
    t = jnp.asarray(t, dtype=jnp.float32)
    M = h.surv(t)
    return (h.lam(t) * M) / jnp.maximum(1.0 - M, eps)
