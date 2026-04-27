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


def _microbatch_slice(
    batch: Dict[str, Array],
    *,
    idx: Array,
    microbatch_size: int,
) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    offset = idx * int(microbatch_size)
    for key, value in batch.items():
        starts = (offset,) + (0,) * (value.ndim - 1)
        sizes = (int(microbatch_size),) + tuple(value.shape[1:])
        out[key] = jax.lax.dynamic_slice(value, starts, sizes)
    return out


def make_train_step_fn(
    *,
    task,
    model,
    tx: optax.GradientTransformation,
    ema_rate: float,
    num_microbatches: int = 1,
):
    num_microbatches = int(num_microbatches)
    if num_microbatches <= 0:
        raise ValueError(f"num_microbatches must be positive, got {num_microbatches}.")

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

        grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
        if num_microbatches <= 1:
            (loss, metrics), grads = grad_fn(state.params, step_rng, batch, True)
            microbatch_size = int(next(iter(batch.values())).shape[0])
        else:
            batch_size = int(next(iter(batch.values())).shape[0])
            if batch_size % num_microbatches != 0:
                raise ValueError(
                    "Batch size must be divisible by training.num_microbatches, got "
                    f"batch_size={batch_size} num_microbatches={num_microbatches}."
                )
            microbatch_size = batch_size // num_microbatches

            def compute_microbatch(microbatch_idx):
                mb = _microbatch_slice(
                    batch,
                    idx=microbatch_idx,
                    microbatch_size=microbatch_size,
                )
                mb_rng = jax.random.fold_in(step_rng, microbatch_idx)
                return grad_fn(state.params, mb_rng, mb, True)

            (loss, metrics), grads = compute_microbatch(jnp.asarray(0, dtype=jnp.int32))

            def accumulate_microbatch(microbatch_idx, carry):
                grad_accum, loss_accum, metrics_accum = carry
                (mb_loss, mb_metrics), mb_grads = compute_microbatch(microbatch_idx)
                grad_accum = jax.tree.map(jnp.add, grad_accum, mb_grads)
                loss_accum = loss_accum + mb_loss
                metrics_accum = jax.tree.map(jnp.add, metrics_accum, mb_metrics)
                return grad_accum, loss_accum, metrics_accum

            grads, loss, metrics = jax.lax.fori_loop(
                1,
                num_microbatches,
                accumulate_microbatch,
                (grads, loss, metrics),
            )
            inv_microbatches = 1.0 / float(num_microbatches)
            grads = jax.tree.map(lambda x: x * inv_microbatches, grads)
            loss = loss * inv_microbatches
            metrics = jax.tree.map(lambda x: x * inv_microbatches, metrics)

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
        metrics["train/num_microbatches"] = jnp.asarray(
            num_microbatches,
            dtype=jnp.float32,
        )
        metrics["train/microbatch_size"] = jnp.asarray(
            microbatch_size,
            dtype=jnp.float32,
        )
        return new_state, metrics

    return train_step_fn
