"""Reverse-time sampler for Sticky Jump Diffusion.

This is the SJD analogue of MD4 sampling, but in *continuous* anchor-embedding
space. The sampler evolves a VP reverse SDE between jumps and uses a
plug-in reverse hazard to "stick" to anchors.

The implementation is adapted to the I/O contract of
`sticky.models.sjd.backward.ContinuousClassifier`:

    apply_model(params, y, t_img) -> (logits, aux)

where:
    y      : (B, ..., d) continuous state
    t_img  : (B,)        per-example continuous time in [0, T]
    logits : (B, ..., L) logits over L anchors

We only implement the plug-in hazard branch (no learned hazard head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from sticky.models.discrete_mixture import (
    categorical_sample_from_logits,
    sample_mixture_categorical,
)
from sticky.rng import PRNGKey

from .hazard import HazardSchedule
from .jump import VPMatchedGaussianJump
from .plugin_intensity import plugin_intensity_and_choice, plugin_intensity_and_probs
from .sdes import alpha_sigma


Array = jnp.ndarray


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the reverse-time SJD sampler."""
    T: float = 1.0
    n_steps: int = 250
    sampling_grid: str = "uniform"  # "uniform" | "cosine"
    score_from_classifier: bool = True
    score_scale: float = 1.0
    logit_temperature: float = 1.0
    alloc_mode: str = "sample"  # "argmax" | "sample"
    categorical_sampling_policy: str = "legacy_low"  # legacy_low | jax_high | exact
    hazard_mode: str = "plugin"
    intensity_mode: str = "full"  # Only "full" is implemented; "chunked" is a deprecated alias.
    log_ratio_clip: float = 10.0
    intensity_chunk_size: int = 256  # Deprecated compatibility knob; ignored by the full backend.
    init_std: float = 1.0
    eps_denom: float = 1e-12
    force_classify_at_end: bool = True  # legacy name: force final plug-in jump at t=dt
    refresh_logits_after_em_step: bool = False


@struct.dataclass
class ReverseSampleResult:
    # k: committed anchor indices, -1 for uncommitted if force_classify_at_end=False
    # k_filled: convenience fill if any sites remain uncommitted at the end
    k: Array
    k_filled: Array
    committed: Array
    metrics: Dict[str, Array]


def _broadcast_time_to_batch(t_scalar: Array, batch_size: int) -> Array:
    """Convert a scalar time to a (B,) vector."""
    return jnp.full((batch_size,), jnp.asarray(t_scalar, dtype=jnp.float32), dtype=jnp.float32)


def _expand_like(v: Array, like: Array) -> Array:
    """Broadcast (B,) vector to match `like` by adding trailing singleton dims."""
    while v.ndim < like.ndim:
        v = v[..., None]
    return v


def make_sampling_time_grid(*, T: float, n_steps: int, sampling_grid: str) -> Array:
    """Return reverse-time step boundaries from T down to 0."""
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")

    key = str(sampling_grid).strip().lower()
    idx = jnp.arange(int(n_steps) + 1, dtype=jnp.float32)
    u = 1.0 - idx / jnp.asarray(float(n_steps), dtype=jnp.float32)

    if key == "uniform":
        scaled = u
    elif key == "cosine":
        scaled = jnp.cos(0.5 * jnp.pi * (1.0 - u))
    else:
        raise ValueError(
            f"Unknown sampling_grid={sampling_grid!r}. Expected 'uniform' or 'cosine'."
        )

    scaled = scaled.at[0].set(1.0)
    scaled = scaled.at[-1].set(0.0)
    return jnp.asarray(float(T), dtype=jnp.float32) * jnp.clip(scaled, 0.0, 1.0)


def reverse_sample(
    key: PRNGKey,
    *,
    params: Any,
    apply_model: Callable[[Any, Array, Array], Tuple[Array, Any]],
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    shape: Tuple[int, ...],
    batch_size: int,
    cfg: SamplerConfig,
    known_idx: Array | None = None,
    known_mask: Array | None = None,
) -> ReverseSampleResult:
    """
    This is a splitting scheme:
        1) Evaluate classifier once at current state y_t.
        2) Euler-Maruyama reverse VP diffusion step (only on uncommitted sites).
        3) Sample a one-step stay/commit transition for uncommitted sites.

    By default the reverse score uses the raw classifier distribution, while
    `logit_temperature` only affects jump-time anchor allocation. If
    `refresh_logits_after_em_step=True`, the jump step makes a second classifier
    call on the post-EM state at the next time slice. If
    `force_classify_at_end=True`, the final strictly positive slice forces a
    plug-in jump for any remaining uncommitted sites.

    Returns:
        ReverseSampleResult with indices and sampling metrics.
    """
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    L = int(a_table.shape[0])
    d = int(a_table.shape[1])
    if float(cfg.logit_temperature) <= 0.0:
        raise ValueError(
            f"logit_temperature must be > 0, got {cfg.logit_temperature}"
        )
    logit_temperature = jnp.asarray(float(cfg.logit_temperature), dtype=jnp.float32)
    time_grid = make_sampling_time_grid(
        T=float(cfg.T),
        n_steps=int(cfg.n_steps),
        sampling_grid=cfg.sampling_grid,
    )

    k0, k_loop = jax.random.split(key, 2)
    y = cfg.init_std * jax.random.normal(k0, shape=(batch_size,) + tuple(shape) + (d,), dtype=jnp.float32)

    committed = jnp.zeros((batch_size,) + tuple(shape), dtype=bool)
    k_idx = -jnp.ones((batch_size,) + tuple(shape), dtype=jnp.int32)
    known_frac = jnp.asarray(0.0, dtype=jnp.float32)

    if (known_idx is None) != (known_mask is None):
        raise ValueError("known_idx and known_mask must both be provided or both be None.")
    if known_idx is not None and known_mask is not None:
        known_idx = jnp.asarray(known_idx, dtype=jnp.int32)
        known_mask = jnp.asarray(known_mask, dtype=bool)
        if known_idx.shape != committed.shape or known_mask.shape != committed.shape:
            raise ValueError(
                "known_idx/known_mask must match sampled token shape "
                f"{committed.shape}; got {known_idx.shape} and {known_mask.shape}."
            )
        known_idx_clipped = jnp.clip(known_idx, 0, L - 1)
        known_vec = a_table[known_idx_clipped]
        y = jnp.where(known_mask[..., None], known_vec, y)
        committed = known_mask
        k_idx = jnp.where(known_mask, known_idx_clipped, k_idx)
        known_frac = jnp.mean(known_mask.astype(jnp.float32))

    jump_count = jnp.asarray(0.0, jnp.float32)
    lam_sum_active = jnp.asarray(0.0, jnp.float32)
    p_jump_sum_active = jnp.asarray(0.0, jnp.float32)
    active_count_total = jnp.asarray(0.0, jnp.float32)
    frac_committed_pre_force = jnp.asarray(0.0, dtype=jnp.float32)

    def step_fn(i: int, carry):
        (
            key,
            y,
            committed,
            k_idx,
            jump_count,
            lam_sum_active,
            p_jump_sum_active,
            active_count_total,
            frac_committed_pre_force,
        ) = carry

        t_scalar = time_grid[i]
        next_t_scalar = time_grid[i + 1]
        dt = t_scalar - next_t_scalar
        t_img = _broadcast_time_to_batch(t_scalar, batch_size)
        is_last_step = i == int(cfg.n_steps) - 1

        # Diffusion step (Euler-Maruyama)
        if not cfg.score_from_classifier:
            raise NotImplementedError("Only classifier-induced score is implemented.")

        y_for_model = y
        if cfg.alloc_mode == "sample":
            key, k_eps, k_mix = jax.random.split(key, 3)
        else:
            key, k_eps, k_u, k_a = jax.random.split(key, 4)
        logits_score, _ = apply_model(params, y_for_model, t_img)
        probs = jax.nn.softmax(logits_score, axis=-1)

        probs2 = probs.reshape((-1, L)).astype(jnp.float32)
        mu2 = probs2 @ a_table
        mu = mu2.reshape(y.shape)

        alpha, sigma = alpha_sigma(beta, t_img)
        alpha = _expand_like(alpha, y)
        sigma = _expand_like(sigma, y)
        sigma2 = sigma * sigma
        denom = jnp.maximum(sigma2, cfg.eps_denom)

        score = -(y - alpha * mu) / denom
        score = score * float(cfg.score_scale)

        bt = beta(t_img)
        bt = _expand_like(bt, y)

        drift = (+0.5 * bt) * y + bt * score

        # Update only uncommitted sites.
        m = (~committed)[..., None].astype(jnp.float32)
        noise = jax.random.normal(k_eps, shape=y.shape, dtype=jnp.float32)
        y = y + m * (drift * dt + jnp.sqrt(bt * dt) * noise)

        jump_y = y_for_model
        jump_t_img = t_img
        jump_logits = logits_score
        if cfg.refresh_logits_after_em_step:
            refreshed_t_img = _broadcast_time_to_batch(next_t_scalar, batch_size)
            refreshed_y = y
            refreshed_logits, _ = apply_model(params, refreshed_y, refreshed_t_img)
            jump_t_img = refreshed_t_img
            jump_y = refreshed_y
            jump_logits = refreshed_logits
            if cfg.force_classify_at_end:
                jump_t_img = jnp.where(is_last_step, t_img, jump_t_img)
                jump_y = jnp.where(is_last_step, y_for_model, jump_y)
                jump_logits = jnp.where(is_last_step, logits_score, jump_logits)

        # Jump step (plugin hazard). Temperature only changes allocation.
        if cfg.alloc_mode == "sample":
            lam_total, choice_probs = plugin_intensity_and_probs(
                logits=jump_logits,
                y=jump_y,
                t_img=jump_t_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                logit_temperature=float(cfg.logit_temperature),
                intensity_mode=cfg.intensity_mode,
                log_ratio_clip=float(cfg.log_ratio_clip),
                chunk_size=int(cfg.intensity_chunk_size),
            )
        else:
            lam_total, a_idx = plugin_intensity_and_choice(
                key=k_a,
                logits=jump_logits,
                y=jump_y,
                t_img=jump_t_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                alloc_mode=cfg.alloc_mode,
                categorical_sampling_policy=cfg.categorical_sampling_policy,
                logit_temperature=float(cfg.logit_temperature),
                intensity_mode=cfg.intensity_mode,
                log_ratio_clip=float(cfg.log_ratio_clip),
                chunk_size=int(cfg.intensity_chunk_size),
            )

        # No jumps once committed.
        active = (~committed).astype(jnp.float32)
        lam_total = jnp.where(committed, 0.0, lam_total)

        p_jump = 1.0 - jnp.exp(-lam_total * dt)
        lam_sum_active = lam_sum_active + jnp.sum(lam_total * active)
        p_jump_sum_active = p_jump_sum_active + jnp.sum(p_jump * active)
        active_count_total = active_count_total + jnp.sum(active)

        if cfg.force_classify_at_end:
            frac_committed_pre_force = jnp.where(
                is_last_step,
                jnp.mean(committed.astype(jnp.float32)),
                frac_committed_pre_force,
            )

        if cfg.alloc_mode == "sample":
            p_jump_sample = jnp.where(
                is_last_step & bool(cfg.force_classify_at_end),
                active,
                p_jump,
            )
            # Intentionally draw once from the joint {stay} U {anchors} mixture.
            # In JAX we do not factor this into jump/no-jump plus anchor choice,
            # so we preserve the categorical behavior that motivated the SJD fix.
            a_idx, stay_mask = sample_mixture_categorical(
                k_mix,
                destination_probs=choice_probs,
                stay_prob=1.0 - p_jump_sample,
                change_prob=p_jump_sample,
                policy=cfg.categorical_sampling_policy,
            )
            jump_mask = (~committed) & (~stay_mask)
        else:
            u = jax.random.uniform(k_u, shape=committed.shape, minval=0.0, maxval=1.0)
            jump_mask = (~committed) & (u < p_jump)
            if cfg.force_classify_at_end:
                jump_mask = jnp.where(is_last_step, ~committed, jump_mask)

        # Commit: set discrete index + snap y to the chosen anchor vector.
        k_idx = jnp.where(jump_mask, a_idx, k_idx)
        a_vec = a_table[a_idx]  # (..., d)
        y = jnp.where(jump_mask[..., None], a_vec, y)
        committed = committed | jump_mask

        jump_count = jump_count + jnp.sum(jump_mask.astype(jnp.float32))
        return (
            key,
            y,
            committed,
            k_idx,
            jump_count,
            lam_sum_active,
            p_jump_sum_active,
            active_count_total,
            frac_committed_pre_force,
        )

    carry = (
        k_loop,
        y,
        committed,
        k_idx,
        jump_count,
        lam_sum_active,
        p_jump_sum_active,
        active_count_total,
        frac_committed_pre_force,
    )
    carry = jax.lax.fori_loop(0, int(cfg.n_steps), step_fn, carry)
    (
        key,
        y,
        committed,
        k_idx,
        jump_count,
        lam_sum_active,
        p_jump_sum_active,
        active_count_total,
        frac_committed_pre_force,
    ) = carry

    # Fraction of sites that jumped at least once.
    n_sites = jnp.asarray(committed.size, dtype=jnp.float32)
    jump_frac = jump_count / jnp.maximum(n_sites, 1.0)
    denom_active = jnp.maximum(active_count_total, 1.0)
    lam_mean_active = lam_sum_active / denom_active
    p_jump_mean_active = p_jump_sum_active / denom_active

    final_committed_frac = jnp.mean(committed.astype(jnp.float32))

    # Fill any remaining sites only as a convenience when the terminal forced
    # jump is disabled.
    if cfg.force_classify_at_end:
        k_filled = k_idx
    else:
        t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        logits_end, _ = apply_model(params, y, t0)
        key, k_end = jax.random.split(key)
        if cfg.alloc_mode == "sample":
            k_fill = categorical_sample_from_logits(
                k_end,
                logits_end,
                policy=cfg.categorical_sampling_policy,
            )
        else:
            k_fill = jnp.argmax(logits_end, axis=-1).astype(jnp.int32)
        k_filled = jnp.where(committed, k_idx, k_fill)
        frac_committed_pre_force = final_committed_frac

    metrics = {
        "sampling/frac_committed_pre_force": frac_committed_pre_force,
        "sampling/frac_committed_final": final_committed_frac,
        "sampling/fill_frac_by_final_jump": jnp.clip(
            final_committed_frac - frac_committed_pre_force,
            a_min=0.0,
            a_max=1.0,
        ),
        "sampling/known_frac": known_frac,
        "sampling/jump_count": jump_count,
        "sampling/jump_frac_total": jump_frac,
        "sampling/lam_mean_active": lam_mean_active,
        "sampling/p_jump_mean_active": p_jump_mean_active,
        "sampling/score_scale": jnp.asarray(float(cfg.score_scale), dtype=jnp.float32),
        "sampling/logit_temperature": logit_temperature,
    }

    return ReverseSampleResult(k=k_idx, k_filled=k_filled, committed=committed, metrics=metrics)
