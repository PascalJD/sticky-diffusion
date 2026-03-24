from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

from .sdes import vp_perturb
from .state_dependency import state_dependency_metrics

Array = jnp.ndarray
Metrics = Dict[str, Array]


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

    B = int(x0_anchor.shape[0])
    key_t, key_vp, key_mask = jax.random.split(key, 3)

    # One global time per example.
    t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))

    # VP corruption with broadcasting over spatial/token dimensions.
    # x0_anchor: (B, ..., d)
    x_t, _ = vp_perturb(key_vp, x0_anchor, t_img, beta)

    # Match sampler conditioning: some sites are already committed (anchors),
    # others remain continuous/noisy.
    if hazard is not None:
        p_committed = jnp.clip(
            jnp.asarray(hazard.surv(t_img), dtype=jnp.float32), 0.0, 1.0
        )
    else:
        p_committed = jnp.zeros_like(t_img, dtype=jnp.float32)
    while p_committed.ndim < x0_idx.ndim:
        p_committed = p_committed[..., None]
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
    }

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
