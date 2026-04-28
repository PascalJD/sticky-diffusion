"""Reverse-time predictor sampler for Sticky Jump Diffusion.

The sampler evolves the continuous VP reverse SDE between jumps and uses the
plug-in reverse hazard to stick uncommitted sites to anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from sticky.models.common.discrete_mixture import (
    categorical_sample_from_logits,
    sample_mixture_categorical,
)
from sticky.rng import PRNGKey

from .anchors import clamp_known_state, gather_anchor_at_label
from .corruption import classifier_induced_score
from .hazard import HazardSchedule
from .jump import VPMatchedGaussianJump
from .plugin_intensity import plugin_hazard_and_allocation, plugin_hazard_and_anchor
from .sdes import _expand_like


Array = jnp.ndarray


# Back-compat alias for tests importing the previous private name.
_gather_committed_anchor = gather_anchor_at_label


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the reverse-time SJD predictor sampler."""

    T: float = 1.0
    n_steps: int = 250
    sampling_grid: str = "uniform"  # "uniform" | "cosine"
    score_from_classifier: bool = True
    score_scale: float = 1.0
    logit_temperature: float = 1.0
    alloc_mode: str = "sample"  # "argmax" | "sample"
    categorical_sampling_policy: str = "legacy_low"
    hazard_mode: str = "plugin"
    log_ratio_clip: float = 10.0
    init_std: float = 1.0
    eps_denom: float = 1e-12
    force_classify_at_end: bool = True
    refresh_logits_after_em_step: bool = False
    metrics_count_nfe: bool = True
    tau_grid_size: int = 32


@struct.dataclass
class ReverseSampleResult:
    # k: committed anchor indices, -1 for uncommitted if force_classify_at_end=False.
    # k_filled: convenience fill if any sites remain uncommitted at the end.
    k: Array
    k_filled: Array
    committed: Array
    metrics: Dict[str, Array]


def _broadcast_time_to_batch(t_scalar: Array, batch_size: int) -> Array:
    return jnp.full(
        (batch_size,),
        jnp.asarray(t_scalar, dtype=jnp.float32),
        dtype=jnp.float32,
    )


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
    Splitting scheme:
      1. Evaluate the classifier at the current continuous state.
      2. Take one Euler-Maruyama reverse VP step on uncommitted sites.
      3. Sample one stay/commit transition for uncommitted sites.
    """
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    if a_table.ndim == 3:
        # Per-position table (P, L, d).
        L = int(a_table.shape[1])
        d = int(a_table.shape[2])
    else:
        L = int(a_table.shape[0])
        d = int(a_table.shape[1])
    if float(cfg.logit_temperature) <= 0.0:
        raise ValueError(f"logit_temperature must be > 0, got {cfg.logit_temperature}")
    time_grid = make_sampling_time_grid(
        T=float(cfg.T),
        n_steps=int(cfg.n_steps),
        sampling_grid=cfg.sampling_grid,
    )

    k0, k_loop = jax.random.split(key, 2)
    y = cfg.init_std * jax.random.normal(
        k0,
        shape=(batch_size,) + tuple(shape) + (d,),
        dtype=jnp.float32,
    )
    committed = jnp.zeros((batch_size,) + tuple(shape), dtype=bool)
    k_idx = -jnp.ones((batch_size,) + tuple(shape), dtype=jnp.int32)

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
        known_idx = jnp.clip(known_idx, 0, L - 1)
        y, committed, k_idx = clamp_known_state(
            y=y,
            committed=committed,
            k_idx=k_idx,
            known_mask=known_mask,
            known_idx=known_idx,
            a_table=a_table,
        )

    def _predictive_stats(
        logits: Array,
        y_state: Array,
        t_img_state: Array,
    ) -> tuple[Array, Array]:
        score = classifier_induced_score(
            y=y_state,
            t=t_img_state,
            anchor_logits=logits,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            tau_grid_size=int(cfg.tau_grid_size),
        ) * float(cfg.score_scale)
        bt = _expand_like(beta(t_img_state), y_state)
        return score, bt

    frac_committed_pre_force = jnp.asarray(0.0, dtype=jnp.float32)
    nfe_total = jnp.asarray(0.0, dtype=jnp.float32)

    def step_fn(i: int, carry):
        (
            key,
            y,
            committed,
            k_idx,
            frac_committed_pre_force,
            nfe_total,
        ) = carry

        t_scalar = time_grid[i]
        next_t_scalar = time_grid[i + 1]
        dt = t_scalar - next_t_scalar
        t_img = _broadcast_time_to_batch(t_scalar, batch_size)
        is_last_step = i == int(cfg.n_steps) - 1

        if not cfg.score_from_classifier:
            raise NotImplementedError("Only classifier-induced score is implemented.")

        if cfg.alloc_mode == "sample":
            key, k_eps, k_mix = jax.random.split(key, 3)
        else:
            key, k_eps, k_u, k_a = jax.random.split(key, 4)

        y_for_model = y
        logits_score, _ = apply_model(params, y_for_model, t_img)
        if cfg.metrics_count_nfe:
            nfe_total = nfe_total + 1.0
        score, bt = _predictive_stats(logits_score, y_for_model, t_img)
        drift = (+0.5 * bt) * y + bt * score

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
            if cfg.metrics_count_nfe:
                nfe_total = nfe_total + 1.0
            jump_t_img = refreshed_t_img
            jump_y = refreshed_y
            jump_logits = refreshed_logits
            if cfg.force_classify_at_end:
                jump_t_img = jnp.where(is_last_step, t_img, jump_t_img)
                jump_y = jnp.where(is_last_step, y_for_model, jump_y)
                jump_logits = jnp.where(is_last_step, logits_score, jump_logits)

        if cfg.alloc_mode == "sample":
            lam_total, choice_probs = plugin_hazard_and_allocation(
                logits=jump_logits,
                y=jump_y,
                t_img=jump_t_img,
                anchors=anchors,
                beta=beta,
                hazard=hazard,
                jump=jump,
                logit_temperature=float(cfg.logit_temperature),
                log_ratio_clip=float(cfg.log_ratio_clip),
                tau_grid_size=int(cfg.tau_grid_size),
            )
        else:
            lam_total, a_idx = plugin_hazard_and_anchor(
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
                log_ratio_clip=float(cfg.log_ratio_clip),
                tau_grid_size=int(cfg.tau_grid_size),
            )

        active = (~committed).astype(jnp.float32)
        lam_total = jnp.where(committed, 0.0, lam_total)
        p_jump = 1.0 - jnp.exp(-lam_total * dt)

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

        k_idx = jnp.where(jump_mask, a_idx, k_idx)
        y = jnp.where(
            jump_mask[..., None], gather_anchor_at_label(a_table, a_idx), y
        )
        committed = committed | jump_mask

        return (
            key,
            y,
            committed,
            k_idx,
            frac_committed_pre_force,
            nfe_total,
        )

    carry = (
        k_loop,
        y,
        committed,
        k_idx,
        frac_committed_pre_force,
        nfe_total,
    )
    carry = jax.lax.fori_loop(0, int(cfg.n_steps), step_fn, carry)
    (
        key,
        y,
        committed,
        k_idx,
        frac_committed_pre_force,
        nfe_total,
    ) = carry

    final_committed_frac = jnp.mean(committed.astype(jnp.float32))

    if cfg.force_classify_at_end:
        k_filled = k_idx
    else:
        t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        logits_end, _ = apply_model(params, y, t0)
        if cfg.metrics_count_nfe:
            nfe_total = nfe_total + 1.0
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
        "sampling/nfe_total": nfe_total,
    }

    return ReverseSampleResult(
        k=k_idx,
        k_filled=k_filled,
        committed=committed,
        metrics=metrics,
    )
