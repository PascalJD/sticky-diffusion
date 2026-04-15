"""Sitewise reverse off-hazard helpers matched to teacher-conditioned training.

The teacher-conditioned forward hazard used in `ce_allocation_loss` is
**sitewise only**: the teacher's per-class logits are reduced into a single
confidence scalar r_i per site (margin / top_prob / entropy), so sites are
modulated as S_i(t) = S(t)^{w_i}. Inference-time matching therefore also
produces a *sitewise* — not per-anchor — reverse off-hazard:

    w_i(r_i)       = w_min + (w_max - w_min) * (1 - r_i)
    S_i(t)         = S(t) ** w_i
    lam_off_i(t)   = w_i * hazard.lam(t) * S_i(t) / (1 - S_i(t) + eps)

Because w_i does not depend on the anchor/class axis, this helper does NOT
produce per-anchor weights w_{i,a}. Applying the sitewise tensor inside
`plugin_intensity` leaves the normalized choice-probabilities mathematically
unchanged — only the total intensity / commit probability is affected.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jnp.ndarray


_VALID_METRICS = ("margin", "top_prob", "entropy")


def sitewise_confidence(logits: Array, metric: str) -> Array:
    """Per-site confidence r_i ∈ [0, 1] reduced over the class/anchor axis.

    Matches the reduction used by `ce_allocation_loss` so training and
    sampling see consistent `r_i` from the same logits.
    """
    metric = str(metric)
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown sitewise hazard metric={metric!r}. "
            f"Expected one of {_VALID_METRICS}."
        )
    logits = logits.astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(logp)
    if metric == "margin":
        top2 = jax.lax.top_k(probs, k=2)[0]
        conf = top2[..., 0] - top2[..., 1]
    elif metric == "top_prob":
        conf = jnp.max(probs, axis=-1)
    else:  # entropy
        K = int(probs.shape[-1])
        ent = -jnp.sum(probs * logp, axis=-1)
        conf = 1.0 - ent / jnp.log(jnp.float32(max(K, 2)))
    return jnp.clip(conf, 0.0, 1.0)


def confidence_to_w(
    r_i: Array, *, w_min: float, w_max: float
) -> Array:
    """Map per-site confidence to the sitewise forward-hazard exponent w_i."""
    r_i = jnp.asarray(r_i, dtype=jnp.float32)
    return (
        jnp.asarray(w_min, dtype=jnp.float32)
        + jnp.asarray(w_max - w_min, dtype=jnp.float32) * (1.0 - r_i)
    )


def sitewise_lam_off(
    hazard: Any,
    t_img: Array,
    w_i: Array,
    *,
    eps: float = 1e-12,
) -> Array:
    """Sitewise matched reverse off-hazard tensor with shape of w_i.

    Formula:
        S_i(t)       = S(t) ** w_i
        lam_off_i(t) = w_i * hazard.lam(t) * S_i(t) / (1 - S_i(t) + eps)

    Args:
        hazard: forward hazard schedule with `surv(t)` and `lam(t)` methods.
        t_img: (B,) per-example time.
        w_i: (B, ...) sitewise exponent. Broadcast rules match the shape of w_i.
        eps: numerical floor on the denominator.
    """
    w_i = jnp.asarray(w_i, dtype=jnp.float32)
    t_img = jnp.asarray(t_img, dtype=jnp.float32)
    S_scalar = jnp.clip(
        jnp.asarray(hazard.surv(t_img), dtype=jnp.float32), 0.0, 1.0
    )
    lam_scalar = jnp.asarray(hazard.lam(t_img), dtype=jnp.float32)
    # Broadcast scalar survival/lam against the sitewise exponent.
    S_b = S_scalar.reshape((w_i.shape[0],) + (1,) * (w_i.ndim - 1))
    lam_b = lam_scalar.reshape((w_i.shape[0],) + (1,) * (w_i.ndim - 1))
    S_i = jnp.power(jnp.maximum(S_b, 1e-30), w_i)
    denom = jnp.maximum(1.0 - S_i, jnp.asarray(eps, dtype=jnp.float32))
    return w_i * lam_b * S_i / denom
