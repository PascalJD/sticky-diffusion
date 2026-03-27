from __future__ import annotations

import warnings
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from sticky.models.common.discrete_mixture import categorical_sample_from_probs, normalize_probs
from sticky.rng import PRNGKey

from .hazard import HazardSchedule, lam_off_star
from .jump import VPMatchedGaussianJump
from .sdes import alpha_sigma


Array = jnp.ndarray


def _validate_intensity_mode(intensity_mode: str) -> None:
    mode = str(intensity_mode).strip().lower()
    if mode == "full":
        return
    if mode == "chunked":
        warnings.warn(
            "SJD intensity_mode='chunked' is deprecated and now aliases to the "
            "full materialized backend; intensity_chunk_size is ignored.",
            FutureWarning,
            stacklevel=2,
        )
        return
    raise ValueError(
        f"Unknown intensity_mode={intensity_mode!r}. "
        "Expected one of: full, chunked."
    )


def _sample_choice_from_probs(
    *,
    key: PRNGKey,
    choice_probs: Array,
    alloc_mode: str,
    categorical_sampling_policy: str = "legacy_low",
) -> Array:
    if alloc_mode == "argmax":
        return jnp.argmax(choice_probs, axis=-1).astype(jnp.int32)
    if alloc_mode == "sample":
        return categorical_sample_from_probs(
            key,
            choice_probs,
            policy=categorical_sampling_policy,
        )
    raise ValueError(f"Unknown alloc_mode={alloc_mode!r}")


def _full_intensity_and_probs(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    logit_temperature: float,
    log_ratio_clip: float,
    eps: float,
) -> Tuple[Array, Array]:
    """Dense plugin implementation that materializes [B, S, L]-scale tensors."""
    if logit_temperature <= 0.0:
        raise ValueError(f"logit_temperature must be > 0, got {logit_temperature}")

    logits = logits.astype(jnp.float32)
    y = y.astype(jnp.float32)

    B = logits.shape[0]
    site_shape = logits.shape[1:-1]
    L = int(logits.shape[-1])
    logp = jax.nn.log_softmax(logits, axis=-1).astype(jnp.float32)

    y_flat = y.reshape((B, -1, y.shape[-1]))
    logp_flat = logp.reshape((B, -1, L))

    a = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    if int(a.shape[0]) != L:
        raise ValueError(
            f"Anchor table has {a.shape[0]} rows, but logits last dim is {L}."
        )
    d = int(a.shape[-1])

    dot = jnp.einsum("bnd,ld->bnl", y_flat, a)
    y_norm2 = jnp.sum(y_flat * y_flat, axis=-1, keepdims=True)
    a_norm2 = jnp.sum(a * a, axis=-1)[None, None, :]

    alpha, sigma = alpha_sigma(beta, t_img)
    alpha = alpha.astype(jnp.float32)[:, None, None]
    sigma2 = jnp.square(sigma).astype(jnp.float32)[:, None, None]

    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    var_q = jnp.maximum(sigma2, 1e-12)
    var_r = jnp.maximum(
        (eta * eta) * sigma2,
        (std_floor * std_floor) + 1e-12,
    )

    dist2 = y_norm2 - 2.0 * alpha * dot + (alpha * alpha) * a_norm2
    inv_r = 1.0 / var_r
    inv_q = 1.0 / var_q
    log_ratio = -0.5 * (d * jnp.log(var_r / var_q) + dist2 * (inv_r - inv_q))
    log_ratio = jnp.clip(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))

    lam_base = lam_off_star(hazard, t_img).astype(jnp.float32)[:, None, None]
    log_lam_base = jnp.log(jnp.maximum(lam_base, jnp.asarray(eps, dtype=jnp.float32)))
    logw_raw = log_lam_base + logp_flat + log_ratio
    lam_total_flat = jnp.maximum(
        jnp.sum(jnp.exp(logw_raw), axis=-1),
        jnp.asarray(eps, dtype=jnp.float32),
    )

    if logit_temperature == 1.0:
        logp_choice_flat = logp_flat
    else:
        logp_choice = jax.nn.log_softmax(
            logits / jnp.asarray(logit_temperature, dtype=jnp.float32),
            axis=-1,
        ).astype(jnp.float32)
        logp_choice_flat = logp_choice.reshape((B, -1, L))

    logw_choice = log_lam_base + logp_choice_flat + log_ratio
    choice_probs_flat = jax.nn.softmax(logw_choice, axis=-1).astype(jnp.float32)

    lam_total = lam_total_flat.reshape((B,) + site_shape)
    choice_probs = choice_probs_flat.reshape((B,) + site_shape + (L,))
    return lam_total, normalize_probs(choice_probs)


def plugin_intensity_and_probs(
    *,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    logit_temperature: float = 1.0,
    intensity_mode: str = "full",
    log_ratio_clip: float = 10.0,
    chunk_size: int = 256,
    eps: float = 1e-20,
) -> Tuple[Array, Array]:
    """Compute total plugin intensity and per-anchor allocation probabilities.

    We intentionally keep only the full materialized backend for now. The old
    chunked path is removed until the stable sampler behavior is validated, and
    "chunked" is kept only as a deprecated alias for older eval configs.
    """
    del chunk_size  # Deprecated compatibility knob; ignored by the full backend.
    _validate_intensity_mode(intensity_mode)
    return _full_intensity_and_probs(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        logit_temperature=logit_temperature,
        log_ratio_clip=log_ratio_clip,
        eps=eps,
    )


def plugin_intensity_and_choice(
    *,
    key: PRNGKey,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    alloc_mode: str,
    categorical_sampling_policy: str = "legacy_low",
    logit_temperature: float = 1.0,
    intensity_mode: str = "full",
    log_ratio_clip: float = 10.0,
    chunk_size: int = 256,
    eps: float = 1e-20,
) -> Tuple[Array, Array]:
    """Backward-compatible wrapper returning the sampled or argmax anchor."""
    lam_total, choice_probs = plugin_intensity_and_probs(
        logits=logits,
        y=y,
        t_img=t_img,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        logit_temperature=logit_temperature,
        intensity_mode=intensity_mode,
        log_ratio_clip=log_ratio_clip,
        chunk_size=chunk_size,
        eps=eps,
    )
    a_idx = _sample_choice_from_probs(
        key=key,
        choice_probs=choice_probs,
        alloc_mode=alloc_mode,
        categorical_sampling_policy=categorical_sampling_policy,
    )
    return lam_total, a_idx
