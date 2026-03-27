from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from sticky.models import masked_discrete_core as masked_core


Array = jnp.ndarray


class MaskingSchedule(nn.Module):
    data_shape: tuple[int, ...]
    schedule_fn_type: str = "cosine"
    eps: float = 1e-4

    def __call__(self, t: Array) -> Array:
        return masked_core.masked_logit_schedule(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
        )

    def dalpha(self, t: Array) -> Array:
        return masked_core.clipped_schedule_dalpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
        )

    def alpha(self, t: Array) -> Array:
        return masked_core.clipped_schedule_alpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
        )

    def dgamma_times_alpha(self, t: Array) -> Array:
        return masked_core.masked_dgamma_times_alpha(
            t,
            schedule_fn_type=self.schedule_fn_type,
            eps=self.eps,
        )


def mask_token_id(vocab_size: int) -> int:
    return masked_core.mask_token_id(vocab_size)


def forward_sample(
    module: Any,
    x: Array,
    t: Array,
    *,
    noise_schedule: MaskingSchedule,
    vocab_size: int,
) -> Array:
    keep_prob = noise_schedule.alpha(masked_core.reverse_broadcast(t, x.ndim))
    return masked_core.sample_masked_discrete_forward(
        module.make_rng("sample"),
        x,
        keep_prob=keep_prob,
        mask_token_id=mask_token_id(vocab_size),
    )


def prior_sample(*, batch_size: int, data_shape: tuple[int, ...], vocab_size: int) -> Array:
    return masked_core.make_masked_token_prior(
        batch_size=batch_size,
        data_shape=data_shape,
        vocab_size=vocab_size,
    )


def get_sampling_grid(
    i: int,
    timesteps: int,
    *,
    sampling_grid: str,
) -> tuple[Array, Array]:
    return masked_core.make_sampling_time_pair(
        i,
        timesteps,
        sampling_grid=sampling_grid,
    )


def get_cond_embedding(
    module: Any,
    conditioning: Array | None,
    *,
    classes: int,
) -> Array | None:
    if (classes > 0) and (conditioning is not None):
        return module.cond_embeddings(conditioning)
    return None


def sample_training_times(
    module: Any,
    *,
    batch_size: int,
    antithetic_time_sampling: bool,
) -> Array:
    rng = module.make_rng("sample")
    if antithetic_time_sampling:
        t0 = jax.random.uniform(rng)
        return jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / batch_size), 1.0)
    return jax.random.uniform(rng, shape=[batch_size])
