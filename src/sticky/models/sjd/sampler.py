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

from sticky.models.common.discrete_mixture import (
    categorical_sample_from_probs,
    categorical_sample_from_logits,
    sample_mixture_categorical,
)
from sticky.rng import PRNGKey

from .anchors import clamp_known_state
from .corrector import corrector_step_size, masked_sum, normalize_pc_gate_name, pc_gate_from_probs
from .hazard import HazardSchedule
from .jump import VPMatchedGaussianJump
from .plugin_intensity import plugin_intensity_and_choice, plugin_intensity_and_probs
from .sdes import _expand_like, alpha_sigma
from .sitewise_hazard import (
    confidence_to_w,
    sitewise_confidence,
    sitewise_lam_off as sitewise_lam_off_fn,
)


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
    pc_enabled: bool = False
    corrector_substeps: int = 0
    corrector_step_scale: float = 0.0
    pc_gate: str = "constant_one"
    pc_clamp_known: bool = True
    pc_refresh_logits_after_langevin: bool = False
    pc_allow_unstick_unknown_only: bool = True
    metrics_count_nfe: bool = True
    # Sitewise reverse off-hazard matched to teacher-conditioned training.
    # The current teacher-conditioned training hazard is sitewise-only (class-
    # reduced confidence), so these knobs modulate lam_total / p_commit only
    # and leave anchor choice_probs mathematically unchanged.
    sitewise_hazard_mode: str = "none"  # "none" | "margin" | "top_prob" | "entropy"
    sitewise_hazard_w_min: float = 0.5
    sitewise_hazard_w_max: float = 2.0
    sitewise_hazard_eps: float = 1e-12
    # Perturb-check corrector gate knobs.
    pc_eta_reopen: float = 0.2
    pc_remask_window_lo: float = 0.15
    pc_remask_window_hi: float = 0.85
    pc_max_reopens_per_site: int = 1


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


def _site_mask_count(mask: Array) -> Array:
    return jnp.sum(jnp.asarray(mask, dtype=jnp.float32))


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
    jump_reopen: VPMatchedGaussianJump | None = None,
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
        raise ValueError(f"logit_temperature must be > 0, got {cfg.logit_temperature}")
    logit_temperature = jnp.asarray(float(cfg.logit_temperature), dtype=jnp.float32)
    time_grid = make_sampling_time_grid(
        T=float(cfg.T),
        n_steps=int(cfg.n_steps),
        sampling_grid=cfg.sampling_grid,
    )
    pc_gate = normalize_pc_gate_name(cfg.pc_gate)
    pc_active = bool(cfg.pc_enabled) and int(cfg.corrector_substeps) > 0 and float(cfg.corrector_step_scale) > 0.0

    if pc_gate == "perturb_check" and jump_reopen is None:
        jump_reopen = VPMatchedGaussianJump(
            beta=jump.beta,
            eta=float(cfg.pc_eta_reopen),
            std_floor=jump.std_floor,
            clip=jump.clip,
        )

    k0, k_loop = jax.random.split(key, 2)
    y = cfg.init_std * jax.random.normal(
        k0,
        shape=(batch_size,) + tuple(shape) + (d,),
        dtype=jnp.float32,
    )

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
        known_idx = jnp.clip(known_idx, 0, L - 1)
        y, committed, k_idx = clamp_known_state(
            y=y,
            committed=committed,
            k_idx=k_idx,
            known_mask=known_mask,
            known_idx=known_idx,
            a_table=a_table,
        )
        known_frac = jnp.mean(known_mask.astype(jnp.float32))

    def _predictive_stats(logits: Array, y_state: Array, t_img_state: Array) -> tuple[Array, Array, Array]:
        probs = jax.nn.softmax(logits, axis=-1)
        mu = (probs.reshape((-1, L)).astype(jnp.float32) @ a_table).reshape(y_state.shape)
        alpha, sigma = alpha_sigma(beta, t_img_state)
        alpha = _expand_like(alpha, y_state)
        sigma = _expand_like(sigma, y_state)
        sigma2 = sigma * sigma
        denom = jnp.maximum(sigma2, cfg.eps_denom)
        score = (-(y_state - alpha * mu) / denom) * float(cfg.score_scale)
        bt = _expand_like(beta(t_img_state), y_state)
        return probs, score, bt

    def _sample_commit_indices(key_choice: Array, *, choice_probs: Array) -> Array:
        if cfg.alloc_mode == "sample":
            return categorical_sample_from_probs(
                key_choice,
                choice_probs,
                policy=cfg.categorical_sampling_policy,
            )
        return jnp.argmax(choice_probs, axis=-1).astype(jnp.int32)

    jump_count = jnp.asarray(0.0, jnp.float32)
    lam_sum_active = jnp.asarray(0.0, jnp.float32)
    p_jump_sum_active = jnp.asarray(0.0, jnp.float32)
    active_count_total = jnp.asarray(0.0, jnp.float32)
    frac_committed_pre_force = jnp.asarray(0.0, dtype=jnp.float32)
    nfe_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_commit_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_unstick_attempts_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_unstick_accepts_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_langevin_updates_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_gate_cont_sum = jnp.asarray(0.0, dtype=jnp.float32)
    pc_gate_cont_count = jnp.asarray(0.0, dtype=jnp.float32)
    pc_gate_committed_sum = jnp.asarray(0.0, dtype=jnp.float32)
    pc_gate_committed_count = jnp.asarray(0.0, dtype=jnp.float32)
    reopen_count = jnp.zeros((batch_size,) + tuple(shape), dtype=jnp.int32)
    pc_reopen_total = jnp.asarray(0.0, dtype=jnp.float32)
    pc_reopen_gate_sum = jnp.asarray(0.0, dtype=jnp.float32)
    pc_reopen_gate_count = jnp.asarray(0.0, dtype=jnp.float32)

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
            nfe_total,
            pc_commit_total,
            pc_unstick_attempts_total,
            pc_unstick_accepts_total,
            pc_langevin_updates_total,
            pc_gate_cont_sum,
            pc_gate_cont_count,
            pc_gate_committed_sum,
            pc_gate_committed_count,
            reopen_count,
            pc_reopen_total,
            pc_reopen_gate_sum,
            pc_reopen_gate_count,
        ) = carry

        t_scalar = time_grid[i]
        next_t_scalar = time_grid[i + 1]
        dt = t_scalar - next_t_scalar
        t_img = _broadcast_time_to_batch(t_scalar, batch_size)
        is_last_step = i == int(cfg.n_steps) - 1

        if not cfg.score_from_classifier:
            raise NotImplementedError("Only classifier-induced score is implemented.")

        y_for_model = y
        if cfg.alloc_mode == "sample":
            key, k_eps, k_mix = jax.random.split(key, 3)
        else:
            key, k_eps, k_u, k_a = jax.random.split(key, 4)
        logits_score, _ = apply_model(params, y_for_model, t_img)
        if cfg.metrics_count_nfe:
            nfe_total = nfe_total + 1.0
        probs, score, bt = _predictive_stats(logits_score, y_for_model, t_img)
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

        sitewise_lam_off_jump = None
        if str(cfg.sitewise_hazard_mode) != "none":
            # Reuse jump_logits already computed for this reverse step — no
            # extra NFE. Confidence is derived from the same tensor that
            # drives plugin_intensity's allocation math.
            r_i_jump = sitewise_confidence(jump_logits, str(cfg.sitewise_hazard_mode))
            w_i_jump = confidence_to_w(
                r_i_jump,
                w_min=float(cfg.sitewise_hazard_w_min),
                w_max=float(cfg.sitewise_hazard_w_max),
            )
            sitewise_lam_off_jump = sitewise_lam_off_fn(
                hazard, jump_t_img, w_i_jump, eps=float(cfg.sitewise_hazard_eps)
            )

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
                sitewise_lam_off=sitewise_lam_off_jump,
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
                sitewise_lam_off=sitewise_lam_off_jump,
            )

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
        a_vec = a_table[a_idx]
        y = jnp.where(jump_mask[..., None], a_vec, y)
        committed = committed | jump_mask
        jump_count = jump_count + jnp.sum(jump_mask.astype(jnp.float32))

        if pc_active:
            corr_t_img = _broadcast_time_to_batch(next_t_scalar, batch_size)
            eps_corr = corrector_step_size(dt=dt, step_scale=float(cfg.corrector_step_scale))
            alpha_u = _expand_like(hazard.lam(corr_t_img), committed)
            p_unstick_base = 1.0 - jnp.exp(-alpha_u * eps_corr)

            for _ in range(int(cfg.corrector_substeps)):
                # Sudoku experiments should only ever allow corrector unsticking on
                # committed unknown/non-clue sites. When the flag is disabled we
                # treat the anchor->continuous move as fully disabled rather than
                # letting it reach clue cells or introducing ambiguous semantics.
                if bool(cfg.pc_allow_unstick_unknown_only):
                    committed_unknown = committed if known_mask is None else (
                        committed & (~known_mask)
                    )
                else:
                    committed_unknown = jnp.zeros_like(committed, dtype=jnp.bool_)

                k_anchor = jnp.clip(k_idx, 0, L - 1)
                anchor_vec = a_table[k_anchor]
                key, k_prop, k_accept = jax.random.split(key, 3)

                if pc_gate == "perturb_check":
                    # --- Perturb-and-check reopen gate ---
                    # 1. Remask window: only activate when normalised time is
                    #    within [lo, hi].
                    tau = next_t_scalar / jnp.asarray(float(cfg.T), dtype=jnp.float32)
                    tau_in_window = (
                        (tau >= jnp.asarray(float(cfg.pc_remask_window_lo), dtype=jnp.float32))
                        & (tau <= jnp.asarray(float(cfg.pc_remask_window_hi), dtype=jnp.float32))
                    )

                    # 2. Eligible sites: committed-unknown within reopen budget.
                    reopen_eligible = committed_unknown & (
                        reopen_count < int(cfg.pc_max_reopens_per_site)
                    )

                    # 3. Perturb committed sites with the smaller-eta kernel.
                    perturbed_y = jnp.where(
                        reopen_eligible[..., None],
                        jump_reopen.sample(k_prop, anchor_vec, corr_t_img),
                        y,
                    )
                    if bool(cfg.pc_clamp_known):
                        perturbed_y, _, _ = clamp_known_state(
                            y=perturbed_y,
                            committed=committed,
                            k_idx=k_idx,
                            known_mask=known_mask,
                            known_idx=known_idx,
                            a_table=a_table,
                        )

                    # 4. Evaluate the model on the perturbed state (+1 NFE).
                    perturbed_logits, _ = apply_model(params, perturbed_y, corr_t_img)
                    if cfg.metrics_count_nfe:
                        nfe_total = nfe_total + 1.0

                    # 5. SJD recommit distribution on the perturbed state.
                    _, perturbed_choice_probs = plugin_intensity_and_probs(
                        logits=perturbed_logits,
                        y=perturbed_y,
                        t_img=corr_t_img,
                        anchors=anchors,
                        beta=beta,
                        hazard=hazard,
                        jump=jump,
                        logit_temperature=float(cfg.logit_temperature),
                        intensity_mode=cfg.intensity_mode,
                        log_ratio_clip=float(cfg.log_ratio_clip),
                        chunk_size=int(cfg.intensity_chunk_size),
                    )

                    # 6. Compare current anchor prob vs best alternative.
                    current_prob = jnp.take_along_axis(
                        perturbed_choice_probs,
                        jnp.clip(k_anchor, 0, L - 1)[..., None],
                        axis=-1,
                    )[..., 0]
                    # Mask out the current anchor to find the best alternative.
                    anchor_range = jnp.arange(L, dtype=jnp.int32)
                    is_current = anchor_range == k_anchor[..., None]  # (..., L)
                    masked_probs = jnp.where(
                        is_current,
                        jnp.float32(-1.0),
                        perturbed_choice_probs,
                    )
                    alt_prob = jnp.max(masked_probs, axis=-1)
                    reopen_gate = jnp.clip(alt_prob - current_prob, 0.0, 1.0)

                    # 7. Zero the gate outside the remask window.
                    reopen_gate = jnp.where(tau_in_window, reopen_gate, 0.0)

                    # 8. Reopen probability = base unstick rate * gate.
                    p_reopen = jnp.where(
                        reopen_eligible,
                        jnp.clip(p_unstick_base * reopen_gate, 0.0, 1.0),
                        0.0,
                    )

                    # 9. Accept / reject.
                    accept_reopen = reopen_eligible & (
                        jax.random.uniform(
                            k_accept,
                            shape=committed.shape,
                            minval=0.0,
                            maxval=1.0,
                        )
                        < p_reopen
                    )

                    # 10. Update state & metrics.
                    y = jnp.where(accept_reopen[..., None], perturbed_y, y)
                    committed = committed & (~accept_reopen)
                    k_idx = jnp.where(accept_reopen, -1, k_idx)
                    reopen_count = reopen_count + accept_reopen.astype(jnp.int32)

                    pc_unstick_attempts_total = pc_unstick_attempts_total + _site_mask_count(reopen_eligible)
                    pc_unstick_accepts_total = pc_unstick_accepts_total + _site_mask_count(accept_reopen)
                    pc_reopen_total = pc_reopen_total + _site_mask_count(accept_reopen)
                    pc_reopen_gate_sum = pc_reopen_gate_sum + masked_sum(reopen_gate, reopen_eligible)
                    pc_reopen_gate_count = pc_reopen_gate_count + _site_mask_count(reopen_eligible)
                    pc_gate_committed_sum = pc_gate_committed_sum + masked_sum(reopen_gate, committed_unknown)
                    pc_gate_committed_count = pc_gate_committed_count + _site_mask_count(committed_unknown)

                    if bool(cfg.pc_clamp_known):
                        y, committed, k_idx = clamp_known_state(
                            y=y,
                            committed=committed,
                            k_idx=k_idx,
                            known_mask=known_mask,
                            known_idx=known_idx,
                            a_table=a_table,
                        )
                else:
                    # --- Existing unstick logic (constant_one / entropy / margin) ---
                    proposed_y = jump.sample(k_prop, anchor_vec, corr_t_img)
                    if bool(cfg.pc_clamp_known):
                        proposed_y, _, _ = clamp_known_state(
                            y=proposed_y,
                            committed=committed,
                            k_idx=k_idx,
                            known_mask=known_mask,
                            known_idx=known_idx,
                            a_table=a_table,
                        )

                    proposal_gate = jnp.ones(committed.shape, dtype=jnp.float32)
                    if pc_gate != "constant_one":
                        proposal_logits, _ = apply_model(params, proposed_y, corr_t_img)
                        if cfg.metrics_count_nfe:
                            nfe_total = nfe_total + 1.0
                        proposal_gate = pc_gate_from_probs(
                            jax.nn.softmax(proposal_logits, axis=-1),
                            gate=pc_gate,
                        )

                    p_unstick = jnp.where(
                        committed_unknown,
                        jnp.clip(proposal_gate * p_unstick_base, 0.0, 1.0),
                        0.0,
                    )
                    accept_unstick = committed_unknown & (
                        jax.random.uniform(
                            k_accept,
                            shape=committed.shape,
                            minval=0.0,
                            maxval=1.0,
                        )
                        < p_unstick
                    )
                    pc_unstick_attempts_total = pc_unstick_attempts_total + _site_mask_count(committed_unknown)
                    pc_unstick_accepts_total = pc_unstick_accepts_total + _site_mask_count(accept_unstick)
                    pc_gate_committed_sum = pc_gate_committed_sum + masked_sum(proposal_gate, committed_unknown)
                    pc_gate_committed_count = pc_gate_committed_count + _site_mask_count(committed_unknown)

                    y = jnp.where(accept_unstick[..., None], proposed_y, y)
                    committed = committed & (~accept_unstick)
                    k_idx = jnp.where(accept_unstick, -1, k_idx)
                    if bool(cfg.pc_clamp_known):
                        y, committed, k_idx = clamp_known_state(
                            y=y,
                            committed=committed,
                            k_idx=k_idx,
                            known_mask=known_mask,
                            known_idx=known_idx,
                            a_table=a_table,
                        )

                logits_corr, _ = apply_model(params, y, corr_t_img)
                if cfg.metrics_count_nfe:
                    nfe_total = nfe_total + 1.0
                probs_corr, score_corr, _ = _predictive_stats(logits_corr, y, corr_t_img)
                continuous_mask = ~committed

                if bool(cfg.pc_refresh_logits_after_langevin):
                    key, k_langevin = jax.random.split(key)
                    langevin_noise = jax.random.normal(k_langevin, shape=y.shape, dtype=jnp.float32)
                    langevin_mask = continuous_mask[..., None].astype(jnp.float32)
                    y = y + langevin_mask * (
                        eps_corr * score_corr + jnp.sqrt(2.0 * eps_corr) * langevin_noise
                    )
                    pc_langevin_updates_total = pc_langevin_updates_total + _site_mask_count(continuous_mask)
                    if bool(cfg.pc_clamp_known):
                        y, committed, k_idx = clamp_known_state(
                            y=y,
                            committed=committed,
                            k_idx=k_idx,
                            known_mask=known_mask,
                            known_idx=known_idx,
                            a_table=a_table,
                        )

                    logits_commit, _ = apply_model(params, y, corr_t_img)
                    if cfg.metrics_count_nfe:
                        nfe_total = nfe_total + 1.0
                    probs_commit = jax.nn.softmax(logits_commit, axis=-1)
                    commit_y = y
                    continuous_mask = ~committed
                else:
                    logits_commit = logits_corr
                    probs_commit = probs_corr
                    commit_y = y

                sitewise_lam_off_pc = None
                if str(cfg.sitewise_hazard_mode) != "none":
                    r_i_pc = sitewise_confidence(logits_commit, str(cfg.sitewise_hazard_mode))
                    w_i_pc = confidence_to_w(
                        r_i_pc,
                        w_min=float(cfg.sitewise_hazard_w_min),
                        w_max=float(cfg.sitewise_hazard_w_max),
                    )
                    sitewise_lam_off_pc = sitewise_lam_off_fn(
                        hazard, corr_t_img, w_i_pc, eps=float(cfg.sitewise_hazard_eps)
                    )
                lam_pc, choice_probs_pc = plugin_intensity_and_probs(
                    logits=logits_commit,
                    y=commit_y,
                    t_img=corr_t_img,
                    anchors=anchors,
                    beta=beta,
                    hazard=hazard,
                    jump=jump,
                    logit_temperature=float(cfg.logit_temperature),
                    intensity_mode=cfg.intensity_mode,
                    log_ratio_clip=float(cfg.log_ratio_clip),
                    chunk_size=int(cfg.intensity_chunk_size),
                    sitewise_lam_off=sitewise_lam_off_pc,
                )
                # perturb_check only gates the anchor->continuous reopen move.
                # Once a site is continuous again, recommit with the standard
                # SJD plug-in rule, i.e. no extra margin/entropy gate here.
                if pc_gate == "perturb_check":
                    gate_cont = jnp.ones(continuous_mask.shape, dtype=jnp.float32)
                else:
                    gate_cont = pc_gate_from_probs(probs_commit, gate=pc_gate)

                p_commit = jnp.where(
                    continuous_mask,
                    jnp.clip(gate_cont * (1.0 - jnp.exp(-lam_pc * eps_corr)), 0.0, 1.0),
                    0.0,
                )
                pc_gate_cont_sum = pc_gate_cont_sum + masked_sum(gate_cont, continuous_mask)
                pc_gate_cont_count = pc_gate_cont_count + _site_mask_count(continuous_mask)

                key, k_commit, k_commit_select, k_langevin = jax.random.split(key, 4)
                commit_idx = _sample_commit_indices(k_commit_select, choice_probs=choice_probs_pc)
                commit_mask = continuous_mask & (
                    jax.random.uniform(
                        k_commit,
                        shape=continuous_mask.shape,
                        minval=0.0,
                        maxval=1.0,
                    )
                    < p_commit
                )
                pc_commit_total = pc_commit_total + _site_mask_count(commit_mask)
                committed = committed | commit_mask
                k_idx = jnp.where(commit_mask, commit_idx, k_idx)
                y = jnp.where(commit_mask[..., None], a_table[commit_idx], commit_y)

                if not bool(cfg.pc_refresh_logits_after_langevin):
                    continuous_post = ~committed
                    langevin_noise = jax.random.normal(k_langevin, shape=y.shape, dtype=jnp.float32)
                    y = y + continuous_post[..., None].astype(jnp.float32) * (
                        eps_corr * score_corr + jnp.sqrt(2.0 * eps_corr) * langevin_noise
                    )
                    pc_langevin_updates_total = pc_langevin_updates_total + _site_mask_count(continuous_post)

                if bool(cfg.pc_clamp_known):
                    y, committed, k_idx = clamp_known_state(
                        y=y,
                        committed=committed,
                        k_idx=k_idx,
                        known_mask=known_mask,
                        known_idx=known_idx,
                        a_table=a_table,
                    )

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
            nfe_total,
            pc_commit_total,
            pc_unstick_attempts_total,
            pc_unstick_accepts_total,
            pc_langevin_updates_total,
            pc_gate_cont_sum,
            pc_gate_cont_count,
            pc_gate_committed_sum,
            pc_gate_committed_count,
            reopen_count,
            pc_reopen_total,
            pc_reopen_gate_sum,
            pc_reopen_gate_count,
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
        nfe_total,
        pc_commit_total,
        pc_unstick_attempts_total,
        pc_unstick_accepts_total,
        pc_langevin_updates_total,
        pc_gate_cont_sum,
        pc_gate_cont_count,
        pc_gate_committed_sum,
        pc_gate_committed_count,
        reopen_count,
        pc_reopen_total,
        pc_reopen_gate_sum,
        pc_reopen_gate_count,
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
        nfe_total,
        pc_commit_total,
        pc_unstick_attempts_total,
        pc_unstick_accepts_total,
        pc_langevin_updates_total,
        pc_gate_cont_sum,
        pc_gate_cont_count,
        pc_gate_committed_sum,
        pc_gate_committed_count,
        reopen_count,
        pc_reopen_total,
        pc_reopen_gate_sum,
        pc_reopen_gate_count,
    ) = carry

    n_sites = jnp.asarray(committed.size, dtype=jnp.float32)
    jump_frac = jump_count / jnp.maximum(n_sites, 1.0)
    denom_active = jnp.maximum(active_count_total, 1.0)
    lam_mean_active = lam_sum_active / denom_active
    p_jump_mean_active = p_jump_sum_active / denom_active
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

    batch_den = jnp.maximum(jnp.asarray(float(batch_size), dtype=jnp.float32), 1.0)
    n_steps_den = jnp.maximum(jnp.asarray(float(cfg.n_steps), dtype=jnp.float32), 1.0)
    unstick_attempts_per_board = pc_unstick_attempts_total / batch_den
    unstick_accepts_per_board = pc_unstick_accepts_total / batch_den
    pc_commit_per_board = pc_commit_total / batch_den
    langevin_updates_per_board = pc_langevin_updates_total / batch_den

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
        "sampling/nfe_total": nfe_total,
        "sampling/continuous_to_anchor_commits_total": pc_commit_per_board,
        "sampling/continuous_to_anchor_commits_per_step": pc_commit_per_board / n_steps_den,
        "sampling/anchor_to_continuous_unstick_attempts_total": unstick_attempts_per_board,
        "sampling/anchor_to_continuous_unstick_accepts_total": unstick_accepts_per_board,
        "sampling/anchor_to_continuous_accept_rate": unstick_accepts_per_board
        / jnp.maximum(unstick_attempts_per_board, 1.0),
        "sampling/langevin_updates_total": langevin_updates_per_board,
        "sampling/langevin_updates_per_step": langevin_updates_per_board / n_steps_den,
        "sampling/gate_mean_continuous": pc_gate_cont_sum
        / jnp.maximum(pc_gate_cont_count, 1.0),
        "sampling/gate_mean_committed": pc_gate_committed_sum
        / jnp.maximum(pc_gate_committed_count, 1.0),
        "sampling/pc_reopen_total": pc_reopen_total / batch_den,
        "sampling/pc_reopen_gate_mean": pc_reopen_gate_sum
        / jnp.maximum(pc_reopen_gate_count, 1.0),
    }

    return ReverseSampleResult(
        k=k_idx,
        k_filled=k_filled,
        committed=committed,
        metrics=metrics,
    )
