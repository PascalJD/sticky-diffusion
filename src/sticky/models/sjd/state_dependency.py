from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from .corruption import mixture_logpdf
from .hazard import HazardSchedule
from .jump import VPMatchedGaussianJump

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
    tau_grid_size: int = 32,
    anchor_log_w: Optional[Array] = None,
) -> Dict[str, Array]:
    """Active-mask mean/std of the DHM log-ratio log r_a(y) - log p_t(y|a).

    Computes a cheap proxy on the model's argmax anchor `a_pred = argmax(logits)`
    rather than averaging over all anchors:
      log_ratio_i = log r_{a_pred,i}(y_i) - log p_t(y_i | a_pred,i)
    where the denominator is the SJD off-anchor conditional p_t(y|a) computed
    by `mixture_logpdf` via tau-quadrature (Appendix C.1/C.2). The numerator is
    `jump.logpdf` (the VP-matched un-sticking kernel r_a).

    Under a frequency-weighted hazard ``lambda_tau(a) = beta(tau) * w(a)``, the
    denominator's integrand picks up ``w(a)`` and ``S(tau)^{w(a)}``; pass
    ``anchor_log_w`` (shape (K,) = log w over the vocab) so the metric tracks
    the same log-ratio the model actually trains on.

    Returns exactly two keys, per the Prompt B whitelist:
      state_dep/log_ratio_mean, state_dep/log_ratio_std.
    """
    if logits.shape[:-1] != y.shape[:-1]:
        raise ValueError(
            "Expected logits and y to share site shape, got "
            f"logits={logits.shape}, y={y.shape}"
        )
    if hazard is None:
        raise ValueError(
            "state_dependency_metrics requires a HazardSchedule to evaluate "
            "the SJD off-anchor mixture in the denominator."
        )

    a_table = jnp.asarray(anchor_table, dtype=jnp.float32)
    pred_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if a_table.ndim == 3:
        # Per-position table (P, V, d). Site axis must equal P; gather
        # (s_idx, pred_idx[..., s]) → a_pred shape (..., P, d).
        P = int(a_table.shape[0])
        if int(pred_idx.shape[-1]) != P:
            raise ValueError(
                f"state_dependency_metrics: per-position a_table P={P} != "
                f"flat site size {int(pred_idx.shape[-1])}."
            )
        s_idx = jnp.broadcast_to(jnp.arange(P), pred_idx.shape)
        a_pred = a_table[s_idx, pred_idx]
    else:
        a_pred = jnp.take(a_table, pred_idx, axis=0)

    if anchor_log_w is not None:
        log_w_per_site = jnp.take(
            jnp.asarray(anchor_log_w, dtype=jnp.float32), pred_idx, axis=0
        )
    else:
        log_w_per_site = jnp.float32(0.0)

    log_r = jump.logpdf(y, a_pred, t_img)
    log_p = mixture_logpdf(
        y=y,
        anchor=a_pred,
        t=t_img,
        beta=beta,
        hazard=hazard,
        jump=jump,
        tau_grid_size=int(tau_grid_size),
        log_w_anchor=log_w_per_site,
    )
    log_ratio = (log_r - log_p).astype(jnp.float32)
    log_ratio = jnp.clip(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))

    mask = uncommitted_mask.astype(jnp.float32)
    log_ratio_mean = _masked_mean(log_ratio, mask)
    centered = log_ratio - log_ratio_mean
    log_ratio_std = jnp.sqrt(jnp.maximum(_masked_mean(centered * centered, mask), 0.0))

    return {
        "state_dep/log_ratio_mean": log_ratio_mean,
        "state_dep/log_ratio_std": log_ratio_std,
    }
