from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .hazard import HazardSchedule, lam_off_star
from .sdes import vp_logpdf, alpha_sigma
from .jump import VPMatchedGaussianJump


Array = jnp.ndarray


def plugin_per_anchor_intensity(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,  # Closed form for plugin in Gaussian case
    log_ratio_clip: float = 10.0,
    eps: float = 1e-20,
) -> Tuple[Array, Array]:
    """
    Plug-in reverse per-anchor intensities, memory-safe.
    """
    probs = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)

    B = probs.shape[0]
    site_shape = probs.shape[1:-1]
    L = probs.shape[-1]

    y_flat = y.reshape((B, -1, y.shape[-1])).astype(jnp.float32)
    probs_flat = probs.reshape((B, -1, L)) 

    a = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    d = int(a.shape[-1])

    dot = jnp.einsum("bnd,ld->bnl", y_flat, a)
    y_norm2 = jnp.sum(y_flat * y_flat, axis=-1, keepdims=True)   
    a_norm2 = jnp.sum(a * a, axis=-1)[None, None, :]

    alpha, sigma = alpha_sigma(beta, t_img)
    alpha = alpha[:, None, None]
    sigma2 = (sigma * sigma)[:, None, None]

    var_q = jnp.maximum(sigma2, 1e-12)
    eta = float(jump.eta)
    std_floor = float(jump.std_floor)

    var_r = jnp.maximum(
        (eta * eta) * sigma2, (std_floor * std_floor) + 1e-12
    )
    dist2 = y_norm2 - 2.0 * alpha * dot + (alpha * alpha) * a_norm2

    inv_r = 1.0 / var_r
    inv_q = 1.0 / var_q

    log_ratio = -0.5 * (d * jnp.log(var_r / var_q) + dist2 * (inv_r - inv_q))
    log_ratio = jnp.clip(
        log_ratio, -float(log_ratio_clip), float(log_ratio_clip)
    )
    ratio = jnp.exp(log_ratio).astype(jnp.float32) 

    lam_base = lam_off_star(hazard, t_img).astype(jnp.float32)
    lam_base = lam_base[:, None, None]

    Lam_flat = lam_base * probs_flat * ratio
    Lam_flat = jnp.maximum(Lam_flat, eps)
    lam_total_flat = jnp.maximum(jnp.sum(Lam_flat, axis=-1), eps)

    Lam = Lam_flat.reshape((B,) + site_shape + (L,))
    lam_total = lam_total_flat.reshape((B,) + site_shape)
    return lam_total, Lam