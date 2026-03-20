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
            "D3PM currently requires sample_timesteps == model.timesteps. "
            f"Got sample_timesteps={total_steps} and model.timesteps={int(model.timesteps)}."
        )
    return total_steps


def validate_sampling_grid(*, model: Any) -> str:
    sampling_grid = str(getattr(model, "sampling_grid", "uniform")).lower()
    if sampling_grid != "uniform":
        raise ValueError(
            "D3PM currently only supports sampling_grid='uniform' because "
            "reverse sampling follows the exact trained discrete chain."
        )
    return sampling_grid


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
    """Single-device generate for D3PM. JIT-friendly."""
    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    total_steps = validate_timesteps(model=model, timesteps=timesteps)
    validate_sampling_grid(model=model)

    rng, prior_rng, step_rng = jax.random.split(rng, 3)
    state = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={"sample": prior_rng},
    )

    def body_fn(i, xt):
        return model.apply(
            variables,
            jax.random.fold_in(step_rng, i),
            i,
            total_steps,
            xt,
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
    validate_sampling_grid(model=model)

    rng, prior_rng, step_rng = jax.random.split(rng, 3)
    state = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={"sample": prior_rng},
    )

    def body_fn(i, xt):
        return model.apply(
            variables,
            jax.random.fold_in(step_rng, i),
            i,
            total_steps,
            xt,
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
