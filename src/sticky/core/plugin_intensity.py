# src/sticky/core/plugin_intensity.py
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from sticky.core.hazard import lam_off_star
from sticky.core.sde_vp import vp_logpdf

Array = jnp.ndarray

def plugin_per_anchor_intensity(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors,
    beta,
    hazard,
    jump,
    log_ratio_clip: float = 10.0,
    eps: float = 1e-20,
) -> Tuple[Array, Array]:
    """
    Returns:
      lam_total: (B,H,W)
      Lam:       (B,H,W,L)  per-anchor intensities that sum to lam_total
    """
    probs = jax.nn.softmax(logits, axis=-1)  # P_theta(a|y,t)

    # Baseline reverse intensity conditional on being off-anchor
    lam_base = lam_off_star(hazard, t_img)
    lam_base = lam_base[:, None, None]

    # Broadcast anchors and time
    a = anchors.table_float 
    a_all = a[None, None, None, :, :]
    y_all = y[..., None, :]
    t_all = t_img[:, None, None, None]

    # log r/q
    log_r = jump.logpdf(y_all, a_all, t_all)
    log_q = vp_logpdf(y_all, a_all, t_all, beta)

    log_ratio = jnp.clip(log_r - log_q, -log_ratio_clip, log_ratio_clip)
    ratio = jnp.exp(log_ratio)

    # Per-anchor intensities
    Lam = lam_base[..., None] * probs * ratio
    Lam = jnp.maximum(Lam, eps)

    lam_total = jnp.sum(Lam, axis=-1)
    lam_total = jnp.maximum(lam_total, eps)

    return lam_total, Lam
