from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .hazard import HazardSchedule, lam_off_star
from .jump import VPMatchedGaussianJump
from .sdes import alpha_sigma


Array = jnp.ndarray


def _flatten_sites(x: Array) -> Tuple[Array, tuple[int, ...], int]:
    """Flatten spatial/token axes into one axis while preserving batch."""
    site_shape = x.shape[1:-1]
    sites_per_example = 1
    for dim in site_shape:
        sites_per_example *= int(dim)
    flat = x.reshape((x.shape[0] * sites_per_example, x.shape[-1]))
    return flat, site_shape, sites_per_example


def _full_intensity_and_choice(
    *,
    key: jax.random.PRNGKey,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    alloc_mode: str,
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

    if alloc_mode == "argmax":
        a_idx_flat = jnp.argmax(logw_choice, axis=-1).astype(jnp.int32)
    elif alloc_mode == "sample":
        a_idx_flat = jax.random.categorical(
            key,
            logw_choice,
            axis=-1,
        ).astype(jnp.int32)
    else:
        raise ValueError(f"Unknown alloc_mode={alloc_mode!r}")

    lam_total = lam_total_flat.reshape((B,) + site_shape)
    a_idx = a_idx_flat.reshape((B,) + site_shape)
    return lam_total, a_idx


def _chunked_intensity_and_choice(
    *,
    key: jax.random.PRNGKey,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    alloc_mode: str,
    logit_temperature: float,
    log_ratio_clip: float,
    chunk_size: int,
    eps: float,
) -> Tuple[Array, Array]:
    """Chunked plugin implementation with O(B*S*chunk_size) memory."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if logit_temperature <= 0.0:
        raise ValueError(f"logit_temperature must be > 0, got {logit_temperature}")

    logits = logits.astype(jnp.float32)
    y = y.astype(jnp.float32)

    B = logits.shape[0]
    L = int(logits.shape[-1])

    y_flat, site_shape, sites_per_example = _flatten_sites(y)
    logits_flat = logits.reshape((B * sites_per_example, L))

    a = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    if int(a.shape[0]) != L:
        raise ValueError(
            f"Anchor table has {a.shape[0]} rows, but logits last dim is {L}."
        )

    d = int(a.shape[-1])
    a_norm2 = jnp.sum(a * a, axis=-1)
    y_norm2 = jnp.sum(y_flat * y_flat, axis=-1)

    alpha, sigma = alpha_sigma(beta, t_img)
    alpha_flat = jnp.repeat(alpha.astype(jnp.float32), sites_per_example)
    sigma2_flat = jnp.repeat(jnp.square(sigma).astype(jnp.float32), sites_per_example)

    eta = float(jump.eta)
    std_floor = float(jump.std_floor)
    var_q = jnp.maximum(sigma2_flat, 1e-12)
    var_r = jnp.maximum((eta * eta) * sigma2_flat, (std_floor * std_floor) + 1e-12)

    inv_diff = (1.0 / var_r) - (1.0 / var_q)
    ratio_const = -0.5 * d * jnp.log(var_r / var_q)

    lam_base = lam_off_star(hazard, t_img).astype(jnp.float32)
    lam_base_flat = jnp.repeat(lam_base, sites_per_example)
    log_lam_base = jnp.log(jnp.maximum(lam_base_flat, eps))

    log_norm = jax.nn.logsumexp(logits_flat, axis=-1, keepdims=True)
    if logit_temperature == 1.0:
        log_norm_choice = log_norm
    else:
        log_norm_choice = jax.nn.logsumexp(
            logits_flat / jnp.asarray(logit_temperature, dtype=jnp.float32),
            axis=-1,
            keepdims=True,
        )

    n_sites_total = logits_flat.shape[0]
    n_chunks = (L + chunk_size - 1) // chunk_size
    offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    neg_inf = jnp.array(-jnp.inf, dtype=jnp.float32)

    def _chunk_logw(chunk_idx: Array) -> tuple[Array, Array, Array]:
        start = chunk_idx * int(chunk_size)
        idx = start + offsets
        valid = idx < L
        idx_clipped = jnp.minimum(idx, L - 1)

        logits_chunk = logits_flat[:, idx_clipped]
        logits_chunk = jnp.where(valid[None, :], logits_chunk, neg_inf)
        logp_chunk = logits_chunk - log_norm
        if logit_temperature == 1.0:
            logp_choice_chunk = logp_chunk
        else:
            logits_choice_chunk = logits_chunk / jnp.asarray(
                logit_temperature,
                dtype=jnp.float32,
            )
            logp_choice_chunk = logits_choice_chunk - log_norm_choice

        a_chunk = a[idx_clipped]
        dot = y_flat @ a_chunk.T

        dist2 = (
            y_norm2[:, None]
            - 2.0 * alpha_flat[:, None] * dot
            + jnp.square(alpha_flat)[:, None] * a_norm2[idx_clipped][None, :]
        )
        log_ratio = ratio_const[:, None] - 0.5 * dist2 * inv_diff[:, None]
        log_ratio = jnp.clip(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))
        log_ratio = jnp.where(valid[None, :], log_ratio, neg_inf)

        logw_raw = log_lam_base[:, None] + logp_chunk + log_ratio
        logw_choice = log_lam_base[:, None] + logp_choice_chunk + log_ratio
        return logw_raw, logw_choice, idx

    if alloc_mode == "argmax":

        def body_fn(i: int, carry):
            lam_total, best_score, best_idx = carry
            logw_raw, logw_choice, idx = _chunk_logw(jnp.asarray(i, dtype=jnp.int32))

            weights = jnp.exp(logw_raw)
            lam_total = lam_total + jnp.sum(weights, axis=-1)

            local = jnp.argmax(logw_choice, axis=-1).astype(jnp.int32)
            score = jnp.max(logw_choice, axis=-1)
            idx_global = jnp.minimum(idx[local], L - 1).astype(jnp.int32)
            take = score > best_score
            best_score = jnp.where(take, score, best_score)
            best_idx = jnp.where(take, idx_global, best_idx)
            return lam_total, best_score, best_idx

        init = (
            jnp.zeros((n_sites_total,), dtype=jnp.float32),
            jnp.full((n_sites_total,), neg_inf, dtype=jnp.float32),
            jnp.zeros((n_sites_total,), dtype=jnp.int32),
        )
        lam_total_flat, _, a_idx_flat = jax.lax.fori_loop(0, int(n_chunks), body_fn, init)

    elif alloc_mode == "sample":

        def body_fn(i: int, carry):
            key_i, lam_total, best_score, best_idx = carry
            key_i, key_g = jax.random.split(key_i)

            logw_raw, logw_choice, idx = _chunk_logw(jnp.asarray(i, dtype=jnp.int32))
            weights = jnp.exp(logw_raw)
            lam_total = lam_total + jnp.sum(weights, axis=-1)

            gumbel = -jnp.log(
                -jnp.log(
                    jax.random.uniform(
                        key_g,
                        shape=logw_choice.shape,
                        minval=jnp.array(eps, dtype=jnp.float32),
                        maxval=jnp.array(1.0, dtype=jnp.float32),
                    )
                )
            )
            draw = logw_choice + gumbel
            local = jnp.argmax(draw, axis=-1).astype(jnp.int32)
            score = jnp.max(draw, axis=-1)
            idx_global = jnp.minimum(idx[local], L - 1).astype(jnp.int32)
            take = score > best_score
            best_score = jnp.where(take, score, best_score)
            best_idx = jnp.where(take, idx_global, best_idx)
            return key_i, lam_total, best_score, best_idx

        init = (
            key,
            jnp.zeros((n_sites_total,), dtype=jnp.float32),
            jnp.full((n_sites_total,), neg_inf, dtype=jnp.float32),
            jnp.zeros((n_sites_total,), dtype=jnp.int32),
        )
        _, lam_total_flat, _, a_idx_flat = jax.lax.fori_loop(0, int(n_chunks), body_fn, init)

    else:
        raise ValueError(f"Unknown alloc_mode={alloc_mode!r}")

    lam_total_flat = jnp.maximum(lam_total_flat, eps)
    lam_total = lam_total_flat.reshape((B,) + site_shape)
    a_idx = a_idx_flat.reshape((B,) + site_shape).astype(jnp.int32)
    return lam_total, a_idx


def plugin_intensity_and_choice(
    *,
    key: jax.random.PRNGKey,
    logits: Array,
    y: Array,
    t_img: Array,
    anchors: Any,
    beta: Any,
    hazard: HazardSchedule,
    jump: VPMatchedGaussianJump,
    alloc_mode: str,
    logit_temperature: float = 1.0,
    intensity_mode: str = "chunked",
    log_ratio_clip: float = 10.0,
    chunk_size: int = 256,
    eps: float = 1e-20,
) -> Tuple[Array, Array]:
    """Compute total plugin intensity and selected anchor index.

    `logit_temperature` only affects the per-jump anchor allocation; the total
    plugin intensity is always computed from the raw classifier distribution.

    `intensity_mode` controls the implementation:
      - `chunked` (default): low-memory streaming over anchor chunks.
      - `full`: dense pre-refactor implementation over all anchors at once.
    """
    mode = str(intensity_mode).lower()
    if mode == "full":
        return _full_intensity_and_choice(
            key=key,
            logits=logits,
            y=y,
            t_img=t_img,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            alloc_mode=alloc_mode,
            logit_temperature=logit_temperature,
            log_ratio_clip=log_ratio_clip,
            eps=eps,
        )
    if mode == "chunked":
        return _chunked_intensity_and_choice(
            key=key,
            logits=logits,
            y=y,
            t_img=t_img,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump,
            alloc_mode=alloc_mode,
            logit_temperature=logit_temperature,
            log_ratio_clip=log_ratio_clip,
            chunk_size=chunk_size,
            eps=eps,
        )
    raise ValueError(
        f"Unknown intensity_mode={intensity_mode!r}. "
        "Expected one of: chunked, full."
    )
