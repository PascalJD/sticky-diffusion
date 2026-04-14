from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from .hazard import HazardSchedule, lam_off_star
from .jump import VPMatchedGaussianJump
from .sdes import _expand_like, vp_logpdf

Array = jnp.ndarray


def _masked_mean(x: Array, mask: Array) -> Array:
    w = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(w), 1.0)
    return jnp.sum(x.astype(jnp.float32) * w) / denom


def state_dependency_metrics(
    *,
    y: Array,
    t_img: Array,
    logits: Array,
    uncommitted_mask: Array,
    anchor_table: Array,
    beta,
    jump: VPMatchedGaussianJump,
    hazard: Optional[HazardSchedule] = None,
    log_ratio_clip: float = 10.0,
) -> Dict[str, Array]:
    """State-dependence diagnostics for the jump modulation ratio.

    The reverse jump modulation contains a state-dependent factor proportional
    to r_t(y|a) / q_t(y|a). Computing the full expectation over all anchors is
    O(B·S·L), so this logs a cheap proxy using the model's argmax anchor.
    """
    if logits.shape[:-1] != y.shape[:-1]:
        raise ValueError(
            "Expected logits and y to share site shape, got "
            f"logits={logits.shape}, y={y.shape}"
        )

    a_table = jnp.asarray(anchor_table, dtype=jnp.float32)
    pred_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    a_pred = jnp.take(a_table, pred_idx, axis=0)

    log_r = jump.logpdf(y, a_pred, t_img)
    log_q = vp_logpdf(y, a_pred, t_img, beta)
    log_ratio = (log_r - log_q).astype(jnp.float32)

    ratio = jnp.exp(
        jnp.clip(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))
    )
    mask = uncommitted_mask.astype(jnp.float32)

    log_ratio_mean = _masked_mean(log_ratio, mask)
    centered = log_ratio - log_ratio_mean
    log_ratio_std = jnp.sqrt(jnp.maximum(_masked_mean(centered * centered, mask), 0.0))

    metrics: Dict[str, Array] = {
        "state_dep/log_ratio_mean": log_ratio_mean,
        "state_dep/log_ratio_std": log_ratio_std,
        "state_dep/log_ratio_abs_mean": _masked_mean(jnp.abs(log_ratio), mask),
        "state_dep/ratio_mean": _masked_mean(ratio, mask),
    }

    if hazard is not None:
        lam_base = _expand_like(lam_off_star(hazard, t_img).astype(jnp.float32), ratio)
        lam_mod = lam_base * ratio
        metrics["state_dep/lam_base_mean"] = _masked_mean(lam_base, mask)
        metrics["state_dep/lam_mod_mean"] = _masked_mean(lam_mod, mask)

    return metrics
