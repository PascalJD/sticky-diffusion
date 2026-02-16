from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from .sdes import vp_perturb

Array = jnp.ndarray
Metrics = Dict[str, Array]


def ce_allocation_loss(
    key: jax.random.PRNGKey,
    params,
    apply_fn: Callable[[object, Array, Array], Tuple[Array, dict]],
    x0_anchor: Array,
    x0_idx: Array,
    beta,
    hazard: Optional[object],
    T: float,
) -> Tuple[Array, Metrics]:
    x0_idx = x0_idx.astype(jnp.int32)

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
    x_in = jnp.where(committed[..., None], x0_anchor, x_t)

    logits, _ = apply_fn(params, x_in, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)

    # NLL against the true token/anchor index.
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    uncommitted = (~committed).astype(jnp.float32)
    uncommitted_count = jnp.sum(uncommitted)
    denom = jnp.maximum(uncommitted_count, 1.0)
    loss = jnp.sum(nll * uncommitted) / denom

    # Diagnostics.
    probs = jnp.exp(logp)
    pred_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    correct = (pred_idx == x0_idx).astype(jnp.float32)
    acc_top1 = jnp.sum(correct * uncommitted) / denom

    ent = -jnp.sum(probs * logp, axis=-1)
    alloc_entropy = jnp.sum(ent * uncommitted) / denom
    frac_uncommitted = jnp.mean(uncommitted)
    frac_committed = 1.0 - frac_uncommitted

    metrics: Metrics = {
        "CE/acc_top1_event": acc_top1,
        "CE/alloc_entropy": alloc_entropy,
        "CE/ce_perplexity": jnp.exp(loss),
        "CE/ce_nll_bits": loss / jnp.log(2.0),
        "CE/frac_event": frac_uncommitted,
        "CE/frac_uncommitted": frac_uncommitted,
        "CE/frac_committed": frac_committed,
    }

    return loss, metrics
