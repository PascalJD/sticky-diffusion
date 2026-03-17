from __future__ import annotations

import functools
from typing import Any, Optional

import jax


def get_attr(train_state: Any, key: str):
    if hasattr(train_state, key):
        return getattr(train_state, key)
    return train_state[key]


def _get_params(train_state: Any, *, use_ema: bool = True):
    if use_ema:
        ema = get_attr(train_state, "ema_params")
        if ema is not None:
            return ema
    return get_attr(train_state, "params")


def validate_timesteps(*, model: Any, timesteps: int | None) -> int:
    total_steps = int(model.timesteps if timesteps is None else timesteps)
    if total_steps != int(model.timesteps):
        raise ValueError(
            "DDPM currently requires sample_timesteps == model.timesteps. "
            f"Got sample_timesteps={total_steps} and model.timesteps={int(model.timesteps)}."
        )
    return total_steps


def simple_generate(
    rng: jax.Array,
    train_state: Any,
    *,
    model: Any,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
):
    """Single-device generate for DDPM. JIT-friendly."""
    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    total_steps = validate_timesteps(model=model, timesteps=timesteps)

    rng, prior_rng, step_rng = jax.random.split(rng, 3)
    state = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={"sample": prior_rng},
    )

    def body_fn(i, x_t):
        return model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            x_t,
            conditioning=conditioning,
            method=model.sample_step,
        )

    state = jax.lax.fori_loop(0, total_steps, body_fn, state)
    return model.apply(
        variables,
        state,
        conditioning=conditioning,
        method=model.decode,
    )


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
def generate(
    model: Any,
    train_state: Any,
    rng: jax.Array,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
):
    """Multi-device generate (per-device batch_size)."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    total_steps = validate_timesteps(model=model, timesteps=timesteps)

    rng, prior_rng, step_rng = jax.random.split(rng, 3)
    state = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={"sample": prior_rng},
    )

    def body_fn(i, x_t):
        return model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            x_t,
            conditioning=conditioning,
            method=model.sample_step,
        )

    state = jax.lax.fori_loop(0, total_steps, body_fn, state)
    return model.apply(
        variables,
        state,
        conditioning=conditioning,
        method=model.decode,
    )
