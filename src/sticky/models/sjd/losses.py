from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import _expand_like, alpha_sigma, vp_perturb
from .state_dependency import state_dependency_metrics

Array = jnp.ndarray
Metrics = Dict[str, Array]


def _teacher_confidence(logits: Array, metric: str) -> Array:
    """Per-site confidence r_i ∈ [0, 1] from detached teacher logits."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(logp)
    if metric == "margin":
        top2 = jax.lax.top_k(probs, k=2)[0]
        conf = top2[..., 0] - top2[..., 1]
    elif metric == "top_prob":
        conf = jnp.max(probs, axis=-1)
    elif metric == "entropy":
        K = int(probs.shape[-1])
        ent = -jnp.sum(probs * logp, axis=-1)
        conf = 1.0 - ent / jnp.log(jnp.float32(max(K, 2)))
    else:
        raise ValueError(f"Unknown teacher_confidence_metric={metric!r}")
    return jnp.clip(conf, 0.0, 1.0)


def ce_allocation_loss(
    key: PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Tuple[Array, dict]],
    x0_anchor: Array,
    x0_idx: Array,
    beta,
    hazard: Optional[object],
    T: float,
    jump: Optional[object] = None,
    anchor_table: Optional[Array] = None,
    state_dep_log_ratio_clip: float = 10.0,
    given_mask: Optional[Array] = None,
    time_sampling: str = "uniform",
    loss_weighting: str = "uniform",
    solver_rank: Optional[Array] = None,
    order_conditioning: str = "none",
    order_w_min: float = 0.5,
    order_w_max: float = 2.0,
    forward_site_hazard_mode: str = "none",
    x0_anchor_teacher: Optional[Array] = None,
    teacher_params: Any = None,
    teacher_apply_fn: Optional[Callable[[object, Array, Array], Tuple[Array, dict]]] = None,
    teacher_confidence_metric: str = "margin",
    teacher_w_min: float = 0.5,
    teacher_w_max: float = 2.0,
    teacher_neutral_weight: float = 1.0,
    teacher_log_metrics: bool = True,
) -> Tuple[Array, Metrics]:
    x0_idx = x0_idx.astype(jnp.int32)
    if given_mask is None:
        given_mask = jnp.zeros_like(x0_idx, dtype=jnp.bool_)
    else:
        given_mask = jnp.asarray(given_mask, dtype=jnp.bool_)
        if given_mask.shape != x0_idx.shape:
            raise ValueError(
                "given_mask must match x0_idx shape, got "
                f"{given_mask.shape} vs {x0_idx.shape}."
            )

    # Resolve effective mode. Legacy `order_conditioning="solver"` + solver_rank
    # is mapped to the unified "solver" mode so existing configs keep working.
    mode = str(forward_site_hazard_mode)
    if mode == "none" and str(order_conditioning) == "solver" and solver_rank is not None:
        mode = "solver"
    if mode not in ("none", "solver", "teacher_margin"):
        raise ValueError(f"Unknown forward_site_hazard_mode={mode!r}")
    teacher_mode = mode == "teacher_margin"
    solver_mode = mode == "solver"
    if teacher_mode and (
        teacher_params is None
        or teacher_apply_fn is None
        or x0_anchor_teacher is None
    ):
        raise ValueError(
            "forward_site_hazard_mode='teacher_margin' requires teacher_params, "
            "teacher_apply_fn, and x0_anchor_teacher."
        )
    if solver_mode and solver_rank is None:
        raise ValueError("forward_site_hazard_mode='solver' requires solver_rank.")

    B = int(x0_anchor.shape[0])

    # Bit-for-bit compat: keep the 3-way split when teacher mode is off so
    # existing runs produce identical randomness.
    if teacher_mode:
        key_t, key_vp, key_probe, key_mask = jax.random.split(key, 4)
    else:
        key_t, key_vp, key_mask = jax.random.split(key, 3)

    # One global time per example. For odd B, ceil(B/2) base samples and
    # their complements give >=B elements; truncate to exactly B.
    if time_sampling == "antithetic" and B >= 2:
        half_B = (B + 1) // 2
        t_half = jax.random.uniform(
            key_t, shape=(half_B,), minval=0.0, maxval=float(T)
        )
        t_complement = float(T) - t_half
        t_img = jnp.concatenate([t_half, t_complement], axis=0)[:B]
    else:
        t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))

    # Student VP corruption with broadcasting over spatial/token dimensions.
    x_t, _ = vp_perturb(key_vp, x0_anchor, t_img, beta)

    # Base scalar survival S(t) — shape (B,).
    if hazard is not None:
        p_base_scalar = jnp.clip(
            jnp.asarray(hazard.surv(t_img), dtype=jnp.float32), 0.0, 1.0
        )
    else:
        p_base_scalar = jnp.zeros_like(t_img, dtype=jnp.float32)

    # Build sitewise commit probabilities p_committed (shape x0_idx.shape).
    order_w_i = None
    teacher_w_i = None
    teacher_conf = None
    probe_committed = None
    if solver_mode:
        rank = jnp.asarray(solver_rank, dtype=jnp.float32)
        order_w_i = jnp.asarray(order_w_min, dtype=jnp.float32) + (
            jnp.asarray(order_w_max - order_w_min, dtype=jnp.float32) * rank
        )
        p_base = _expand_like(p_base_scalar, order_w_i)
        p_committed = jnp.clip(
            jnp.power(jnp.maximum(p_base, 1e-12), order_w_i), 0.0, 1.0
        )
    elif teacher_mode:
        # Teacher-space corruption reuses the student's key_vp → identical z,
        # only the anchor mean differs.
        x_t_teacher, _ = vp_perturb(key_vp, x0_anchor_teacher, t_img, beta)
        # Probe commit mask sampled from the base scalar survival.
        p_probe = _expand_like(p_base_scalar, x0_idx)
        probe_committed = jax.random.bernoulli(
            key_probe, p=p_probe, shape=x0_idx.shape
        )
        probe_effective = jnp.logical_or(probe_committed, given_mask)
        x_probe = jnp.where(
            probe_effective[..., None], x0_anchor_teacher, x_t_teacher
        )
        # Exactly one detached teacher forward pass (train=False, no dropout).
        teacher_logits, _ = teacher_apply_fn(teacher_params, x_probe, t_img)
        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        teacher_conf = jax.lax.stop_gradient(
            _teacher_confidence(teacher_logits, str(teacher_confidence_metric))
        )
        # Sitewise exponent. Neutral weight on probe-committed and given sites,
        # where the teacher saw the anchor directly and its confidence is trivial.
        teacher_w_raw = (
            jnp.asarray(teacher_w_min, dtype=jnp.float32)
            + jnp.asarray(teacher_w_max - teacher_w_min, dtype=jnp.float32)
            * (1.0 - teacher_conf)
        )
        teacher_w_i = jnp.where(
            probe_effective,
            jnp.asarray(teacher_neutral_weight, dtype=jnp.float32),
            teacher_w_raw,
        )
        p_base = _expand_like(p_base_scalar, teacher_w_i)
        p_committed = jnp.clip(
            jnp.power(jnp.maximum(p_base, 1e-12), teacher_w_i), 0.0, 1.0
        )
    else:
        p_committed = _expand_like(p_base_scalar, x0_idx)

    committed = jax.random.bernoulli(key_mask, p=p_committed, shape=x0_idx.shape)

    # Conditional sequence tasks (for example Sudoku) reserve some tokens as
    # known/given context. Those tokens must stay visible in the corrupted
    # model input and must never contribute to the allocation loss.
    committed = jnp.logical_or(committed, given_mask)
    x_in = jnp.where(committed[..., None], x0_anchor, x_t)

    logits, _ = apply_fn(params, x_in, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)

    # NLL against the true token/anchor index.
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    suffix_mask = ~given_mask
    effective_loss_mask = suffix_mask & (~committed)
    effective_loss_weight = effective_loss_mask.astype(jnp.float32)
    w_t = None
    if loss_weighting == "alpha_deriv":
        alpha_t, _ = alpha_sigma(beta, t_img)
        beta_t = beta(t_img)
        w_t = 0.5 * beta_t * alpha_t / jnp.maximum(1.0 - alpha_t, 1e-8)
        w_t = w_t / jnp.maximum(jnp.mean(w_t), 1e-8)
        w_t = _expand_like(w_t, effective_loss_weight)
        effective_loss_weight = effective_loss_weight * w_t
    effective_loss_count = jnp.sum(effective_loss_weight)
    denom = jnp.maximum(effective_loss_count, 1.0)
    loss = jnp.sum(nll * effective_loss_weight) / denom

    # Diagnostics.
    probs = jnp.exp(logp)
    pred_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    correct = (pred_idx == x0_idx).astype(jnp.float32)
    acc_top1 = jnp.sum(correct * effective_loss_weight) / denom

    ent = -jnp.sum(probs * logp, axis=-1)
    alloc_entropy = jnp.sum(ent * effective_loss_weight) / denom

    suffix_weight = suffix_mask.astype(jnp.float32)
    suffix_count = jnp.maximum(jnp.sum(suffix_weight), 1.0)
    committed_suffix = (suffix_mask & committed).astype(jnp.float32)
    frac_uncommitted = jnp.sum(effective_loss_weight) / suffix_count
    frac_committed = jnp.sum(committed_suffix) / suffix_count

    metrics: Metrics = {
        "CE/acc_top1_event": acc_top1,
        "CE/alloc_entropy": alloc_entropy,
        "CE/ce_perplexity": jnp.exp(loss),
        "CE/ce_nll_bits": loss / jnp.log(2.0),
        "CE/frac_event": frac_uncommitted,
        "CE/frac_uncommitted": frac_uncommitted,
        "CE/frac_committed": frac_committed,
        "CE/time_mean": jnp.mean(t_img),
        "CE/time_std": jnp.std(t_img),
    }
    if w_t is not None:
        metrics["CE/loss_weight_mean"] = jnp.mean(w_t)
        metrics["CE/loss_weight_std"] = jnp.std(w_t)

    if solver_mode:
        rank_f = jnp.asarray(solver_rank, dtype=jnp.float32)
        committed_f = committed.astype(jnp.float32)
        easy_mask = ((rank_f < 0.25) & suffix_mask).astype(jnp.float32)
        hard_mask = ((rank_f > 0.75) & suffix_mask).astype(jnp.float32)
        easy_denom = jnp.maximum(jnp.sum(easy_mask), 1.0)
        hard_denom = jnp.maximum(jnp.sum(hard_mask), 1.0)
        metrics["CE/order_w_mean"] = jnp.mean(order_w_i)
        metrics["CE/order_committed_easy"] = jnp.sum(committed_f * easy_mask) / easy_denom
        metrics["CE/order_committed_hard"] = jnp.sum(committed_f * hard_mask) / hard_denom

    if teacher_mode and bool(teacher_log_metrics):
        suffix_f = suffix_weight
        suffix_denom = suffix_count
        probe_f = probe_committed.astype(jnp.float32)
        meaningful = suffix_mask & (~probe_committed)
        meaningful_f = meaningful.astype(jnp.float32)
        meaningful_denom = jnp.maximum(jnp.sum(meaningful_f), 1.0)
        conf_mean = jnp.sum(teacher_conf * meaningful_f) / meaningful_denom
        conf_var = (
            jnp.sum(jnp.square(teacher_conf - conf_mean) * meaningful_f)
            / meaningful_denom
        )
        metrics["CE/teacher_conf_mean"] = conf_mean
        metrics["CE/teacher_conf_std"] = jnp.sqrt(jnp.maximum(conf_var, 0.0))
        conf_flat = teacher_conf.reshape(-1)
        metrics["CE/teacher_conf_q25"] = jnp.quantile(conf_flat, 0.25)
        metrics["CE/teacher_conf_q75"] = jnp.quantile(conf_flat, 0.75)
        w_mean = jnp.sum(teacher_w_i * suffix_f) / suffix_denom
        w_var = jnp.sum(jnp.square(teacher_w_i - w_mean) * suffix_f) / suffix_denom
        metrics["CE/teacher_w_mean"] = w_mean
        metrics["CE/teacher_w_std"] = jnp.sqrt(jnp.maximum(w_var, 0.0))
        metrics["CE/teacher_probe_committed_fraction"] = (
            jnp.sum(probe_f * suffix_f) / suffix_denom
        )
        metrics["CE/teacher_final_committed_fraction"] = (
            jnp.sum(committed_suffix) / suffix_denom
        )
        if solver_rank is not None:
            rank_f = jnp.asarray(solver_rank, dtype=jnp.float32)
            easy_m = ((rank_f < 0.25) & suffix_mask).astype(jnp.float32)
            hard_m = ((rank_f > 0.75) & suffix_mask).astype(jnp.float32)
            easy_d = jnp.maximum(jnp.sum(easy_m), 1.0)
            hard_d = jnp.maximum(jnp.sum(hard_m), 1.0)
            metrics["CE/teacher_conf_easy"] = jnp.sum(teacher_conf * easy_m) / easy_d
            metrics["CE/teacher_conf_hard"] = jnp.sum(teacher_conf * hard_m) / hard_d

    if (jump is not None) and (anchor_table is not None):
        metrics.update(
            state_dependency_metrics(
                y=x_t,
                t_img=t_img,
                logits=logits,
                uncommitted_mask=effective_loss_mask,
                anchor_table=anchor_table,
                beta=beta,
                jump=jump,
                hazard=hazard,
                log_ratio_clip=float(state_dep_log_ratio_clip),
            )
        )

    return loss, metrics
