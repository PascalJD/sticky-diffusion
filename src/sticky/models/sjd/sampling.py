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
    )
