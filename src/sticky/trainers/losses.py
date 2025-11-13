# src/sticky/trainers/losses.py
from __future__ import annotations
from typing import Callable
import jax, jax.numpy as jnp

from sticky.core.sde_vp import vp_perturb, vp_logpdf
from sticky.core.hazard import HazardSchedule
from sticky.core.jump import GaussianJump

Array = jnp.ndarray

def ce_allocation_loss(
    params,
    apply_classifier: Callable[[object, Array, Array], Array],
    y_event: Array,
    t_event: Array,
    anchor_idx: Array,  # ground-truth anchor id for each sample
    mask_has_event: Array,  # ignore items with no event by T
) -> Array:
    logits = apply_classifier(params, y_event, t_event)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, anchor_idx[:, None], axis=-1).squeeze(-1)
    nll = jnp.where(mask_has_event, nll, 0.0)
    denom = jnp.maximum(jnp.sum(mask_has_event), 1.0)
    return jnp.sum(nll) / denom

def dhm_loss(
    key: jax.random.PRNGKey,
    params,
    apply_intensity: Callable[[object, Array, Array], Array],
    x0_anchor: Array,
    t: Array,
    beta,
    hazard: HazardSchedule,
    jump: GaussianJump,
    weight_fn: Callable[[Array], Array] | None = None,
) -> Array:
    # sample X_t using closed-form VP
    xt, _ = vp_perturb(key, x0_anchor, t, beta)

    # compute target
    log_r = jump.logpdf(xt, x0_anchor)
    log_q = vp_logpdf(xt, x0_anchor, t, beta)
    lam_fwd = hazard.lam(t)
    surv = hazard.surv(t)
    lam_hat = lam_fwd * surv * jnp.exp(jnp.clip(log_r - log_q, -50.0, 50.0))

    lam_pred = apply_intensity(params, xt, t)

    w = weight_fn(t) if weight_fn is not None else jnp.ones_like(t)
    mse = jnp.square(lam_pred - lam_hat)
    return jnp.mean(w * mse)