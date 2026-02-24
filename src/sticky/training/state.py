from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct
from omegaconf import DictConfig


Array = jnp.ndarray


@struct.dataclass
class TrainState:
    step: Array
    rng: Array
    params: Any
    ema_params: Any
    opt_state: optax.OptState


def shard_batch(batch: Dict[str, Array]) -> Dict[str, Array]:
    """(B, ...) -> (n_local_devices, B//n_local_devices, ...)"""
    n = jax.local_device_count()
    out: Dict[str, Array] = {}
    for k, v in batch.items():
        assert v.shape[0] % n == 0, (
            f"Batch dim for {k} must be divisible by local_device_count."
        )
        out[k] = v.reshape((n, v.shape[0] // n) + v.shape[1:])
    return out


def make_lr_schedule(cfg: DictConfig):
    warmup_steps = int(cfg.optim.warmup_steps)
    total_steps = int(cfg.training.num_train_steps)
    base_lr = float(cfg.optim.learning_rate)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps),
        end_value=0.0,
    )


def make_optimizer(cfg: DictConfig):
    lr_schedule = make_lr_schedule(cfg)
    return optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=float(cfg.optim.b2),
        weight_decay=float(cfg.optim.weight_decay),
    )


def init_state(cfg: DictConfig, model, rng: jax.random.PRNGKey):
    name = str(cfg.model.name)
    batch_size = (
        int(cfg.dataset.per_device_batch_size)
        if int(cfg.dataset.per_device_batch_size) > 0
        else 1
    )

    if name in ("md4", "cadd"):
        rng, rng_params, rng_sample = jax.random.split(rng, 3)
        dummy_x = jnp.zeros(
            (batch_size,) + tuple(cfg.dataset.data_shape), dtype=jnp.int32
        )
        dummy_cond = (
            jnp.zeros((batch_size,), dtype=jnp.int32)
            if int(cfg.model.classes) > 0
            else None
        )
        variables = model.init(
            {"params": rng_params, "sample": rng_sample},
            dummy_x,
            cond=dummy_cond,
            train=False,
        )

    elif name == "sjd":
        rng, rng_params = jax.random.split(rng, 2)

        anchor_dim = int(cfg.model.anchor_dim)
        dummy_z = jnp.zeros(
            (batch_size,) + tuple(cfg.dataset.data_shape) + (anchor_dim,),
            dtype=jnp.float32,
        )
        dummy_t = jnp.zeros((batch_size,), dtype=jnp.float32)
        variables = model.init({"params": rng_params}, dummy_z, dummy_t, train=False)
        dummy_ids = jnp.zeros(tuple(dummy_z.shape[:-1]), dtype=jnp.int32)

        rng, rng_anchors = jax.random.split(rng, 2)
        _, mutated = model.apply(
            variables,
            dummy_ids,
            method=model.embed,
            mutable=["params"],
            rngs={"params": rng_anchors},
        )
        variables = {**variables, "params": mutated["params"]}

    else:
        raise ValueError(f"Unknown model.name={name!r} for init")

    params = variables["params"]
    tx = make_optimizer(cfg)
    opt_state = tx.init(params)

    ema_rate = float(cfg.training.ema_rate)
    if ema_rate > 0.0:
        ema_params = jax.tree_util.tree_map(lambda x: x, params)
    else:
        ema_params = None

    state = TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        rng=rng,
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
    )
    return state, tx
