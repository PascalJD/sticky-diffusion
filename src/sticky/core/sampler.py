# src/sticky/core/sampler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from sticky.core.anchors import AnalogBitsAnchors
from sticky.core.hazard import HazardSchedule, lam_off_star
from sticky.core.jump import GaussianJump, VPMatchedGaussianJump
from sticky.core.plugin_intensity import plugin_per_anchor_intensity
from sticky.core.sde_vp import alpha_sigma

Array = jnp.ndarray


@dataclass(frozen=True)
class SamplerConfig:
    T: float = 1.0
    n_steps: int = 250

    # score side
    score_from_classifier: bool = True
    score_scale: float = 1.0

    # jump side
    hazard_mode: str = "plugin"  # "plugin" | "learned"
    alloc_mode: str = "argmax"  # "argmax" | "sample"
    log_ratio_clip: float = 10.0

    # init / numerics
    init_std: float = 1.0
    eps_denom: float = 1e-12
    rng_fold: int = 4

    # optional final projection/commit
    force_classify_at_end: bool = True


@dataclass
class ReverseSampleResult:
    # k: committed anchor indices, -1 for uncommitted if force_classify_at_end=False
    k: Array
    # k_filled: always filled with a final classifier decision (useful for visualization)
    k_filled: Array
    committed: Array
    metrics: Dict[str, Array]


def _choose_anchor(
    *,
    key: jax.random.PRNGKey,
    scores: Array,  # (B,H,W,L), nonnegative
    mode: str,
    eps: float = 1e-20,
) -> Array:
    """Choose anchor indices from per-anchor nonnegative scores."""
    if mode == "argmax":
        return jnp.argmax(scores, axis=-1).astype(jnp.int32)
    if mode == "sample":
        return jax.random.categorical(key, jnp.log(scores + eps), axis=-1).astype(jnp.int32)
    raise ValueError(f"Unknown alloc_mode={mode!r}")


def reverse_sample(
    key: jax.random.PRNGKey,
    *,
    params,
    apply_model: Callable[[object, Array, Array], Tuple[Array, Array]],
    anchors: AnalogBitsAnchors,
    beta,
    hazard: HazardSchedule,
    jump: GaussianJump | VPMatchedGaussianJump,
    shape_hw: Tuple[int, int],
    B: int,
    cfg: SamplerConfig,
) -> ReverseSampleResult:
    """
    Reverse-time sampler with:
      - classifier-induced VP score on off-anchor states
      - plug-in hazard / intensities OR learned hazard head
      - optional force sticking at the end

    This is a splitting scheme: EM diffusion step, then Bernoulli jump step.
    """
    H, W = shape_hw
    L = anchors.L
    d = anchors.d

    dt = float(cfg.T) / float(cfg.n_steps)

    k0, k_loop = jax.random.split(key, 2)
    y = cfg.init_std * jax.random.normal(k0, shape=(B, H, W, d), dtype=jnp.float32)

    committed = jnp.zeros((B, H, W), dtype=bool)
    k_idx = -jnp.ones((B, H, W), dtype=jnp.int32)

    # accumulators for metrics
    jump_count = jnp.asarray(0.0, jnp.float32)

    def step_fn(i: int, carry):
        key, y, committed, k_idx, jump_count = carry

        # forward-time parameter for this reverse-time slice
        t_scalar = jnp.asarray(cfg.T - dt * i, dtype=jnp.float32)
        t_img = jnp.full((B,), t_scalar, dtype=jnp.float32)

        # Diffusion step (EM)
        if not cfg.score_from_classifier:
            raise NotImplementedError("Only classifier-induced score is implemented.")

        key, k_eps = jax.random.split(key, 2)
        logits, _ = apply_model(params, y, t_img)
        probs = jax.nn.softmax(logits, axis=-1)

        # mu = sum_a p(a|y,t) a
        mu = jnp.tensordot(probs, anchors.table_float, axes=[-1, 0])

        alpha, sigma = alpha_sigma(beta, t_img)
        alpha = alpha[:, None, None, None]
        sigma2 = (sigma * sigma)[:, None, None, None]
        denom = jnp.maximum(sigma2, cfg.eps_denom)

        score = -(y - alpha * mu) / denom
        score = score * cfg.score_scale

        bt = beta(t_img)[:, None, None, None]
        drift = (+0.5 * bt) * y + bt * score

        # update only uncommitted sites
        m = (~committed)[..., None].astype(jnp.float32)
        noise = jax.random.normal(k_eps, shape=y.shape, dtype=jnp.float32)
        y = y + m * (drift * dt + jnp.sqrt(bt * dt) * noise)

        # Jump step
        # Recompute on the post-diffusion y
        logits2, rho2 = apply_model(params, y, t_img)
        probs2 = jax.nn.softmax(logits2, axis=-1)

        if cfg.hazard_mode == "plugin":
            lam_total, Lam = plugin_per_anchor_intensity(
                logits=logits2,
                y=y,
                t_img=t_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                log_ratio_clip=float(cfg.log_ratio_clip),
            )

        elif cfg.hazard_mode == "learned":
            # hazard head predicts rho(y,t), and we scale by the time-only off-anchor baseline
            lam_off = lam_off_star(hazard, t_img)[:, None, None]
            lam_total = lam_off * rho2
            Lam = lam_total[..., None] * probs2
            Lam = jnp.maximum(Lam, 1e-20)
            lam_total = jnp.maximum(lam_total, 1e-20)
        else:
            raise ValueError(f"Unknown hazard_mode={cfg.hazard_mode!r}")

        # no jumps once committed
        lam_total = jnp.where(committed, 0.0, lam_total)
        Lam = jnp.where(committed[..., None], 0.0, Lam)

        p_jump = 1.0 - jnp.exp(-lam_total * dt)

        key, k_u, k_a = jax.random.split(key, 3)
        u = jax.random.uniform(k_u, shape=(B, H, W), minval=0.0, maxval=1.0)

        jump_mask = (~committed) & (u < p_jump)

        a_idx = _choose_anchor(key=k_a, scores=Lam, mode=cfg.alloc_mode)

        # commit: set discrete index + snap y to the chosen anchor vector
        k_idx = jnp.where(jump_mask, a_idx, k_idx)
        a_vec = anchors.table_float[a_idx]
        y = jnp.where(jump_mask[..., None], a_vec, y)
        committed = committed | jump_mask

        jump_count = jump_count + jnp.sum(jump_mask.astype(jnp.float32))

        return (key, y, committed, k_idx, jump_count)

    carry = (k_loop, y, committed, k_idx, jump_count)
    carry = jax.lax.fori_loop(0, cfg.n_steps, step_fn, carry)
    key, y, committed, k_idx, jump_count = carry

    jump_frac = jump_count / jnp.asarray(B * H * W, jnp.float32)

    # Optional force sticking at end
    # Always compute a filled k for convenience/visualization.
    t0 = jnp.zeros((B,), dtype=jnp.float32)
    logits_end, _ = apply_model(params, y, t0)
    probs_end = jax.nn.softmax(logits_end, axis=-1)
    key, k_end = jax.random.split(key)
    if cfg.alloc_mode == "sample":
        k_fill = jax.random.categorical(k_end, jnp.log(probs_end + 1e-20), axis=-1).astype(jnp.int32)
    else:
        k_fill = jnp.argmax(probs_end, axis=-1).astype(jnp.int32)

    k_filled = jnp.where(committed, k_idx, k_fill)

    if cfg.force_classify_at_end:
        a_vec_end = anchors.table_float[k_fill]
        y = jnp.where(committed[..., None], y, a_vec_end)
        committed = jnp.ones_like(committed, dtype=bool)
        k_idx = k_filled
    
    metrics = {
        "sampling/frac_committed_pre_force": jnp.mean((k_idx != -1).astype(jnp.float32)),
        "sampling/frac_committed_final": jnp.mean(committed.astype(jnp.float32)),
        "sampling/jump_count": jump_count,
        "sampling/jump_frac_total": jump_frac,
    }


    return ReverseSampleResult(k=k_idx, k_filled=k_filled, committed=committed, metrics=metrics)
