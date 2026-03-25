from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from sticky.rng import ensure_prng_key, legacy_prng_key_data, make_rng
from sticky.training.persistence import CheckpointWriter
from sticky.training.state import TrainState


def test_checkpoint_writer_saves_typed_rng_keys_compatibly(tmp_path):
    state = TrainState(
        step=jnp.asarray(7, dtype=jnp.int32),
        rng=make_rng(11),
        params={"w": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
        ema_params={"w": jnp.asarray([1.5, 2.5], dtype=jnp.float32)},
        opt_state=(),
    )
    writer = CheckpointWriter(root_dir=tmp_path / "checkpoints", every_steps=1)

    saved = writer.maybe_save_best(
        target=state,
        step_i=7,
        metrics={"eval/fid": 3.14},
    )

    assert saved is True
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(writer.best_dir),
        target=state.replace(rng=legacy_prng_key_data(state.rng)),
        step=7,
    )
    restored_rng = ensure_prng_key(restored.rng)

    assert int(jax.device_get(restored.step)) == 7
    assert jnp.allclose(restored.params["w"], state.params["w"])
    assert jnp.allclose(restored.ema_params["w"], state.ema_params["w"])
    assert jnp.array_equal(
        jax.random.key_data(restored_rng),
        jax.random.key_data(state.rng),
    )


def test_checkpoint_writer_can_update_best_on_equal_ties(tmp_path):
    writer = CheckpointWriter(
        root_dir=tmp_path / "checkpoints",
        every_steps=1,
        best_metric_key="eval/solve_rate",
        best_mode="max",
        best_update_on_equal=True,
    )
    state = TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        rng=make_rng(0),
        params={"w": jnp.asarray([1.0], dtype=jnp.float32)},
        ema_params=None,
        opt_state=(),
    )

    assert writer.maybe_save_best(target=state, step_i=10, metrics={"eval/solve_rate": 0.0}) is True
    assert writer.maybe_save_best(target=state, step_i=20, metrics={"eval/solve_rate": 0.0}) is True

    meta = writer.best_meta_path.read_text(encoding="utf-8")
    assert '"step": 20' in meta
