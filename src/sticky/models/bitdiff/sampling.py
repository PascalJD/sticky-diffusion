from __future__ import annotations

import functools
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp


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
    validate_sampler_contract(model=model)
    total_steps = int(model.timesteps if timesteps is None else timesteps)
    if total_steps <= 0:
        raise ValueError(f"BitDiffusion requires sample timesteps > 0, got {total_steps}.")
    return total_steps


def validate_sampler_contract(*, model: Any) -> None:
    sampler = str(getattr(model, "sampler", "ddim")).lower()
    stochasticity = float(getattr(model, "stochasticity", 0.0))
    if sampler == "ddpm" and not math.isclose(stochasticity, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "BitDiffusion sampler='ddpm' is a DDIM-style alias with fixed eta=1.0. "
            "Set stochasticity=1.0 or use sampler='ddim' for custom eta."
        )


def _initial_sampling_state(model: Any, variables: dict[str, Any], batch_size: int, rng: jax.Array):
    latent = model.apply(
        variables,
        batch_size,
        method=model.prior_sample,
        rngs={"sample": rng},
    )
    if bool(getattr(model, "self_conditioning", False)):
        return {
            "latent": latent,
            "self_cond": jnp.zeros_like(latent),
        }
    return latent


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
    """Single-device generate for Bit Diffusion. JIT-friendly."""
    params = _get_params(train_state, use_ema=use_ema)
    variables = {"params": params}

    total_steps = validate_timesteps(model=model, timesteps=timesteps)

    rng, prior_rng, step_rng = jax.random.split(rng, 3)
    state = _initial_sampling_state(model, variables, batch_size, prior_rng)

    def body_fn(i, current_state):
        return model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            current_state,
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
    state = _initial_sampling_state(model, variables, batch_size, prior_rng)

    def body_fn(i, current_state):
        return model.apply(
            variables,
            step_rng,
            i,
            total_steps,
            current_state,
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
