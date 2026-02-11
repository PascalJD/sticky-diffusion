from __future__ import annotations

from typing import Callable, Dict, Tuple

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
    T: float,
) -> Tuple[Array, Metrics]:
    x0_idx = x0_idx.astype(jnp.int32)

    B = int(x0_anchor.shape[0])
    key_t, key_vp = jax.random.split(key, 2)

    # One global time per example.
    t_img = jax.random.uniform(key_t, shape=(B,), minval=0.0, maxval=float(T))

    # VP corruption with broadcasting over spatial/token dimensions.
    # x0_anchor: (B, ..., d)
    x_t, _ = vp_perturb(key_vp, x0_anchor, t_img, beta)

    logits, _ = apply_fn(params, x_t, t_img)
    logp = jax.nn.log_softmax(logits, axis=-1)

    # NLL against the true token/anchor index.
    nll = -jnp.take_along_axis(logp, x0_idx[..., None], axis=-1).squeeze(-1)
    loss = jnp.mean(nll)

    # Diagnostics.
    probs = jnp.exp(logp)
    pred_idx = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    acc_top1 = jnp.mean(pred_idx == x0_idx)

    ent = -jnp.sum(probs * logp, axis=-1)
    alloc_entropy = jnp.mean(ent)

    metrics: Metrics = {
        "CE/acc_top1_event": acc_top1,
        "CE/alloc_entropy": alloc_entropy,
        "CE/ce_perplexity": jnp.exp(loss),
        "CE/ce_nll_bits": loss / jnp.log(2.0),
        "CE/frac_event": jnp.array(1.0, jnp.float32),
    }

    return loss, metrics
