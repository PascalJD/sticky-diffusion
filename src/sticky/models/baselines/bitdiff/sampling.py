from __future__ import annotations

import functools
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp

from sticky.core.sampling_loop import generate_from_simple_generate, simple_generate_loop


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


def _initialize_state(model: Any, variables: dict[str, Any], batch_size: int, rng: jax.Array):
    _, prior_rng, step_rng = jax.random.split(rng, 3)
    state = _initial_sampling_state(model, variables, batch_size, prior_rng)
    return state, step_rng


def _sample_step(
    model: Any,
    variables: dict[str, Any],
    step_rng: jax.Array,
    i: int,
    total_steps: int,
    state: Any,
    conditioning: Optional[jax.Array],
):
    return model.apply(
        variables,
        step_rng,
        i,
        total_steps,
        state,
        conditioning=conditioning,
        method=model.sample_step,
    )


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
    return simple_generate_loop(
        rng,
        train_state,
        model=model,
        batch_size=batch_size,
        conditioning=conditioning,
        timesteps=timesteps,
        use_ema=use_ema,
        resolve_total_steps=lambda current_model, current_timesteps: validate_timesteps(
            model=current_model,
            timesteps=current_timesteps,
        ),
        init_state=_initialize_state,
        sample_step=_sample_step,
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
    return generate_from_simple_generate(
        simple_generate=simple_generate,
        model=model,
        train_state=train_state,
        rng=rng,
        batch_size=batch_size,
        conditioning=conditioning,
        timesteps=timesteps,
        use_ema=use_ema,
    )
