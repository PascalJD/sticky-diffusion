from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
from flax.jax_utils import replicate, unreplicate

from sticky.training.state import TrainState, shard_batch
from sticky.training.step import make_train_step_fn, make_wrapped_train_step


class _DummyTask:
    def loss_fn(self, *, rng, model, params, batch, train):
        del rng, model, train
        pred = params["w"] * batch["x"]
        loss = jnp.mean(jnp.square(pred - batch["y"]))
        return loss, {"mse": loss}


def _make_state():
    params = {"w": jnp.asarray(1.0, dtype=jnp.float32)}
    tx = optax.sgd(learning_rate=0.1)
    return (
        TrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            rng=jax.random.PRNGKey(0),
            params=params,
            ema_params=jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params),
            opt_state=tx.init(params),
        ),
        tx,
    )


def test_train_step_compiles_on_single_device():
    state, tx = _make_state()
    train_step_fn = make_train_step_fn(
        task=_DummyTask(),
        model=SimpleNamespace(),
        tx=tx,
        ema_rate=0.9,
    )
    train_step = make_wrapped_train_step(train_step_fn, use_pmap=False)

    batch = {
        "x": jnp.asarray([[1.0], [2.0]], dtype=jnp.float32),
        "y": jnp.asarray([[0.0], [0.0]], dtype=jnp.float32),
    }
    new_state, metrics = train_step(state, batch)

    loss = float(jax.device_get(metrics["train/loss"]))
    assert new_state.step.shape == ()
    assert int(jax.device_get(new_state.step)) == 1
    assert loss >= 0.0


def test_train_step_compiles_with_pmap():
    state, tx = _make_state()
    train_step_fn = make_train_step_fn(
        task=_DummyTask(),
        model=SimpleNamespace(),
        tx=tx,
        ema_rate=0.9,
    )
    train_step = make_wrapped_train_step(train_step_fn, use_pmap=True, axis_name="batch")

    per_device_batch = 2
    batch = {
        "x": jnp.ones((jax.local_device_count() * per_device_batch, 1), dtype=jnp.float32),
        "y": jnp.zeros((jax.local_device_count() * per_device_batch, 1), dtype=jnp.float32),
    }

    new_state, metrics = train_step(replicate(state), shard_batch(batch))
    metrics_host = unreplicate(metrics)
    state_host = unreplicate(new_state)

    assert int(jax.device_get(state_host.step)) == 1
    assert float(jax.device_get(metrics_host["train/loss"])) >= 0.0
