# src/sticky/core/sampler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .sde_vp import alpha_sigma, B_of_t
Array = jnp.ndarray


@dataclass(frozen=True)
class SamplerConfig:
    T: float = 1.0
    n_steps: int = 250
    alloc_mode: str = "argmax"  # "argmax" | "sample"
    score_from_classifier: bool = True
    score_scale: float = 1.0
    force_classify_at_end: bool = True
    init_std: float = 1.0 
    eps_denom: float = 1e-12
    rng_fold: int = 4 


@dataclass(frozen=True)
class SampleResult:
    y: Array
    k: Array 
    stuck_mask: Array 
    forced_mask: Array
    t_stick: Array
    metrics: Dict[str, Array]


def classifier_induced_score(
    logits: Array, 
    y: Array, 
    t: Array,  
    beta, 
    anchor_table: Array,
    eps: float = 1e-12,
) -> Array:
    # probs over anchors
    probs = jax.nn.softmax(logits, axis=-1)
    # expected anchor vector \hat{z0}
    z0 = jnp.tensordot(probs, anchor_table, axes=[(-1,), (0,)])

    # time scalars -> broadcast
    alpha, sigma = alpha_sigma(beta, t)  # (B,)
    while alpha.ndim < y.ndim:
        alpha = alpha[..., None]
        sigma = sigma[..., None]

    inv_sigma2 = 1.0 / jnp.maximum(sigma * sigma, eps)
    s = (alpha * inv_sigma2) * z0 - inv_sigma2 * y
    return s


def _apply_classifier(alloc_apply: Callable, params_cls, y: Array, t: Array) -> Array:
    """Helper: (B,H,W,d),(B,) -> (B,H,W,L)"""
    return alloc_apply(params_cls, y, t)


def _apply_intensity(haz_apply: Callable, params_haz, y: Array, t: Array) -> Array:
    """Helper: (B,H,W,d),(B,) -> (B,H,W)"""
    return haz_apply(params_haz, y, t)


def _argmax_or_sample(
    key: jax.random.PRNGKey, logits: Array, mode: str = "argmax"
) -> Array:
    """Return indices (B,H,W) from (B,H,W,L) logits."""
    if mode == "sample":
        u = jax.random.uniform(key, shape=logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
        g = -jnp.log(-jnp.log(u))
        k = jnp.argmax(logits + g, axis=-1)
    else:
        k = jnp.argmax(logits, axis=-1)
    return k


def reverse_sample(
    key: jax.random.PRNGKey,
    params_cls,
    params_haz,
    apply_classifier: Callable[[object, Array, Array], Array], 
    apply_intensity: Callable[[object, Array, Array], Array],
    anchors,  # AnalogBitsAnchors (has .table_float (L,d), .d, .L)
    beta,
    shape_hw: Tuple[int, int],
    B: int = 16,
    cfg: SamplerConfig = SamplerConfig(),
) -> SampleResult:
    H, W = shape_hw
    d, L = int(anchors.d), int(anchors.L)
    table = anchors.table_float  # (L,d), jnp

    T = float(cfg.T)
    n = int(cfg.n_steps)
    h = T / n
    t_grid = jnp.linspace(T, 0.0, n + 1, dtype=jnp.float32)  # [t0=T, ..., tn=0]

    # Init y_T ~ N(0, init_std^2 I)
    key, k0 = jax.random.split(key)
    y = cfg.init_std * jax.random.normal(
        k0, shape=(B, H, W, d), dtype=jnp.float32
    )

    # Book-keeping
    stuck = jnp.zeros((B, H, W), dtype=bool)
    forced = jnp.zeros_like(stuck)
    t_stick = jnp.full((B, H, W), jnp.inf, dtype=jnp.float32)
    k_idx = jnp.full((B, H, W), -1, dtype=jnp.int32)

    # metrics accumulators
    total_sites = B * H * W
    acc_events = jnp.array(0, dtype=jnp.int32)

    def one_step(carry, step_idx):
        y, stuck, t_stick, k_idx, acc_events, key = carry
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        t_curr = t_grid[step_idx]
        t_next = t_grid[step_idx + 1]
        t_b = jnp.full((y.shape[0],), t_curr, dtype=jnp.float32)

        # Evaluate allocator logits once (we reuse both for score and allocation)
        logits = _apply_classifier(apply_classifier, params_cls, y, t_b)

        # Score on XA
        if cfg.score_from_classifier:
            s = classifier_induced_score(
                logits=logits,
                y=y,
                t=t_b,
                beta=beta,
                anchor_table=table,
                eps=cfg.eps_denom,
            ) * cfg.score_scale
        else:
            s = jnp.zeros_like(y)

        beta_t = beta(t_curr).astype(jnp.float32)  # scalar
        f = -0.5 * beta_t * y
        drift = f - beta_t * s

        # Propose event via piecewise-constant hazard for not-stuck sites
        lam = _apply_intensity(apply_intensity, params_haz, y, t_b) 
        lam = jnp.where(stuck, 0.0, lam)  # no hazard once stuck
        p_evt = 1.0 - jnp.exp(-jnp.clip(lam * h, a_min=0.0))
        u_evt = jax.random.uniform(k1, shape=p_evt.shape)
        will_stick = (u_evt < p_evt) & (~stuck)

        # Reverse Euler-M (advance to t_next)
        noise = jax.random.normal(k2, shape=y.shape)
        y_next = y + drift * (t_next - t_curr) + jnp.sqrt(jnp.maximum(beta_t * h, 0.0)) * noise

        # Allocate anchors for the ones that stick this step
        k_new = _argmax_or_sample(k3, logits, mode=cfg.alloc_mode)
        y_anchor = jnp.take(table, k_new, axis=0)

        # Replace only where will_stick:
        # - gather anchor vectors for chosen k_new
        k_new_flat = k_new.reshape(-1)
        table_g = table[k_new_flat]
        y_anchor = table_g.reshape((B, H, W, d))

        y_final = jnp.where(will_stick[..., None], y_anchor, y_next)
        stuck_next = stuck | will_stick
        t_stick_next = jnp.where(will_stick, t_next, t_stick)
        k_idx_next = jnp.where(will_stick, k_new, k_idx)
        acc_events_next = acc_events + jnp.sum(will_stick, dtype=jnp.int32)

        return (y_final, stuck_next, t_stick_next, k_idx_next, acc_events_next, k4), (y_final, stuck_next)

    # Scan over all steps (n steps from T->0)
    carry0 = (y, stuck, t_stick, k_idx, acc_events, key)
    carryT, _ = jax.lax.scan(one_step, carry0, jnp.arange(0, n))
    y, stuck, t_stick, k_idx, acc_events, key = carryT

    # Optional: force classification at the end for sites that never stuck
    if cfg.force_classify_at_end:
        key, kf = jax.random.split(key)
        t0 = jnp.zeros((B,), dtype=jnp.float32)
        logits_0 = _apply_classifier(apply_classifier, params_cls, y, t0)
        k_force = _argmax_or_sample(kf, logits_0, mode="argmax")
        # gather anchor vectors and apply only where not stuck
        y_force = jnp.take(table, k_force, axis=0)
        y = jnp.where(stuck[..., None], y, y_force)
        k_idx = jnp.where(stuck, k_idx, k_force)
        forced = ~stuck

    # Final metrics
    num_stuck = jnp.sum(stuck, dtype=jnp.int32)
    num_forced = jnp.sum(forced, dtype=jnp.int32)
    metrics = dict(
        ratio_stuck_hazard=num_stuck / total_sites,
        ratio_forced_end=num_forced / total_sites,
        mean_t_stick=jnp.where(num_stuck > 0, jnp.mean(t_stick[stuck]), jnp.array(jnp.inf)),
        num_events=acc_events,
        total_sites=jnp.array(total_sites, dtype=jnp.int32),
    )

    return SampleResult(
        y=y, 
        k=k_idx, 
        stuck_mask=stuck, 
        forced_mask=forced, 
        t_stick=t_stick, 
        metrics=metrics
    )