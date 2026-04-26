from __future__ import annotations

from collections.abc import Callable
from typing import Dict

import jax
import jax.numpy as jnp
import optax

from sticky.training.state import TrainState


Array = jnp.ndarray


def params_for_sampling(state: TrainState):
    return state.ema_params if state.ema_params is not None else state.params


def make_wrapped_train_step(
    train_step_fn: Callable[[TrainState, Dict[str, Array], str | None], tuple[TrainState, Dict[str, Array]]],
    *,
    use_pmap: bool,
    axis_name: str = "batch",
):
    if use_pmap:
        return jax.pmap(
            lambda st, b: train_step_fn(st, b, axis_name=axis_name),
            axis_name=axis_name,
            donate_argnums=(0,),
        )
    return jax.jit(
        lambda st, b: train_step_fn(st, b, axis_name=None),
        donate_argnums=(0,),
    )


def make_wrapped_eval_step(
    eval_step_fn: Callable[[Dict[str, Array], Array, Dict[str, Array], str | None], Dict[str, Array]],
    *,
    use_pmap: bool,
    axis_name: str = "batch",
):
    if use_pmap:
        return jax.pmap(
            lambda p, b, r: eval_step_fn(p, r, b, axis_name=axis_name),
            axis_name=axis_name,
        )
    return jax.jit(lambda p, b, r: eval_step_fn(p, r, b, axis_name=None))


def make_train_step_fn(*, task, model, tx: optax.GradientTransformation, ema_rate: float):
    def loss_and_metrics(params, rng, batch, train: bool):
        return task.loss_fn(
            rng=rng,
            model=model,
            params=params,
            batch=batch,
            train=train,
        )

    def train_step_fn(state: TrainState, batch: Dict[str, Array], axis_name: str | None):
        rng, step_rng = jax.random.split(state.rng)
        if axis_name is not None:
            step_rng = jax.random.fold_in(step_rng, jax.lax.axis_index(axis_name))
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            state.params, step_rng, batch, True
        )

        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=axis_name)
            metrics = jax.tree.map(
                lambda x: jax.lax.pmean(x, axis_name=axis_name), metrics
            )
            loss = jax.lax.pmean(loss, axis_name=axis_name)

        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        if ema_rate > 0.0:
            new_ema_params = jax.tree_util.tree_map(
                lambda e, p: e + (1.0 - ema_rate) * (p - e),
                state.ema_params,
                new_params,
            )
        else:
            new_ema_params = None

        new_state = state.replace(
            step=state.step + 1,
            rng=rng,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
        )
        metrics = dict(metrics)
        metrics["train/loss"] = loss
        return new_state, metrics

    return train_step_fn
