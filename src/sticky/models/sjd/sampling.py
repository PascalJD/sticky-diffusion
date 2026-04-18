from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .sampler import SamplerConfig, ReverseSampleResult, reverse_sample

Array = jnp.ndarray


def simple_generate(
    *,
    rng: Array,
    params,
    model,
    anchors,
    beta,
    hazard,
    jump,
    batch_size: int,
    shape: Sequence[int],
    cfg: SamplerConfig,
    known_idx: Array | None = None,
    known_mask: Array | None = None,
    jump_reopen: "VPMatchedGaussianJump | None" = None,
) -> ReverseSampleResult:

    shape_tup: Tuple[int, ...] = tuple(int(x) for x in shape)

    def apply_model(p, y, t_img):
        return model.apply({"params": p}, y, t=t_img, train=False)

    return reverse_sample(
        rng,
        params=params,
        apply_model=apply_model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        shape=shape_tup,
        batch_size=int(batch_size),
        cfg=cfg,
        known_idx=known_idx,
        known_mask=known_mask,
        jump_reopen=jump_reopen,
    )


def conditional_generate_board(
    *,
    rng: Array,
    params,
    model,
    anchors,
    beta,
    hazard,
    jump,
    known_tokens: Array,
    known_token_mask: Array,
    cfg: SamplerConfig,
    jump_reopen: "VPMatchedGaussianJump | None" = None,
) -> tuple[Array, dict[str, Array]]:
    known_tokens = jnp.asarray(known_tokens, dtype=jnp.int32)
    known_token_mask = jnp.asarray(known_token_mask, dtype=jnp.bool_)
    if known_tokens.shape != known_token_mask.shape:
        raise ValueError(
            "known_tokens and known_token_mask must have matching shapes, got "
            f"{known_tokens.shape} vs {known_token_mask.shape}."
        )

    known_idx = jnp.where(
        known_token_mask,
        jnp.clip(known_tokens - 1, 0, anchors.table_float.shape[0] - 1),
        0,
    ).astype(jnp.int32)
    out = simple_generate(
        rng=rng,
        params=params,
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        batch_size=int(known_tokens.shape[0]),
        shape=tuple(int(dim) for dim in known_tokens.shape[1:]),
        cfg=cfg,
        known_idx=known_idx,
        known_mask=known_token_mask,
        jump_reopen=jump_reopen,
    )
    board = (jnp.asarray(out.k_filled, dtype=jnp.int32) + 1).astype(jnp.int32)
    board = jnp.where(known_token_mask, known_tokens, board).astype(jnp.int32)
    return board, dict(out.metrics)
