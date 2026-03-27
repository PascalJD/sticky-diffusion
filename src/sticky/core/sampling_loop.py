from __future__ import annotations

from typing import Any, Callable, Optional

import jax


def get_attr(train_state: Any, key: str):
    if hasattr(train_state, key):
        return getattr(train_state, key)
    return train_state[key]


def get_params(train_state: Any, *, use_ema: bool = True):
    if use_ema:
        ema = get_attr(train_state, "ema_params")
        if ema is not None:
            return ema
    return get_attr(train_state, "params")


def simple_generate_loop(
    rng: jax.Array,
    train_state: Any,
    *,
    model: Any,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
    resolve_total_steps: Callable[[Any, Optional[int]], int],
    init_state: Callable[[Any, dict[str, Any], int, jax.Array], tuple[Any, jax.Array]],
    sample_step: Callable[[Any, dict[str, Any], jax.Array, int, int, Any, Optional[jax.Array]], Any],
):
    params = get_params(train_state, use_ema=use_ema)
    variables = {"params": params}
    total_steps = resolve_total_steps(model, timesteps)
    state, step_rng = init_state(model, variables, batch_size, rng)

    def body_fn(i, current_state):
        return sample_step(model, variables, step_rng, i, total_steps, current_state, conditioning)

    state = jax.lax.fori_loop(0, total_steps, body_fn, state)
    return model.apply(
        variables,
        state,
        conditioning=conditioning,
        method=model.decode,
    )


def generate_from_simple_generate(
    *,
    simple_generate: Callable[..., Any],
    model: Any,
    train_state: Any,
    rng: jax.Array,
    batch_size: int,
    conditioning: Optional[jax.Array] = None,
    timesteps: Optional[int] = None,
    use_ema: bool = True,
):
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    return simple_generate(
        rng,
        train_state,
        model=model,
        batch_size=batch_size,
        conditioning=conditioning,
        timesteps=timesteps,
        use_ema=use_ema,
    )
