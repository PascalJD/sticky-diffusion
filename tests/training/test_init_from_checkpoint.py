from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from omegaconf import OmegaConf

from sticky.rng import legacy_prng_key_data, make_rng
from sticky.training.state import TrainState, init_state


class _DummySJDModel:
    anchor_config = SimpleNamespace(anchor_dim=2)

    def init(self, rngs, z_t, t, *, anchor_token_ids=None, train=False):
        del rngs, z_t, t, train
        assert anchor_token_ids is not None
        return {
            "params": {
                "classifier": {"w": jnp.asarray([1.0], dtype=jnp.float32)},
                "anchors": {"table": jnp.ones((4, 2), dtype=jnp.float32)},
            }
        }


def _base_cfg(init_from_checkpoint=None, **overrides):
    training = {
        "num_train_steps": 1,
        "ema_rate": 0.9,
        "init_from_checkpoint": init_from_checkpoint,
        "init_load_ema": overrides.get("init_load_ema", True),
        "init_reset_optimizer": overrides.get("init_reset_optimizer", True),
        "init_reset_step": overrides.get("init_reset_step", True),
        "init_checkpoint_step": overrides.get("init_checkpoint_step", None),
    }
    return OmegaConf.create(
        {
            "dataset": {"per_device_batch_size": 2, "data_shape": [4]},
            "model": {
                "name": "sjd",
                "classes": -1,
                "anchor": {"family": "normal", "dim": 2, "learnable": True},
            },
            "optim": {
                "warmup_steps": 0,
                "learning_rate": 1e-3,
                "b2": 0.999,
                "weight_decay": 0.0,
            },
            "training": training,
        }
    )


def test_init_from_checkpoint_restores_params_and_resets_step(tmp_path):
    cfg_fresh = _base_cfg()
    fresh_state, _ = init_state(cfg_fresh, _DummySJDModel(), make_rng(0))

    modified_params = jax.tree.map(lambda x: x + 42.0, fresh_state.params)
    modified_ema = jax.tree.map(lambda x: x + 7.0, fresh_state.params)
    saved_state = fresh_state.replace(
        step=jnp.array(5000, dtype=jnp.int32),
        params=modified_params,
        ema_params=modified_ema,
    )
    serializable_state = saved_state.replace(rng=legacy_prng_key_data(saved_state.rng))
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    checkpoints.save_checkpoint(str(ckpt_dir), serializable_state, step=5000, keep=1)

    cfg_resume = _base_cfg(init_from_checkpoint=str(ckpt_dir))
    resumed_state, tx = init_state(cfg_resume, _DummySJDModel(), make_rng(0))

    assert tx is not None
    assert int(jax.device_get(resumed_state.step)) == 0

    for leaf_resumed, leaf_saved in zip(
        jax.tree.leaves(resumed_state.params),
        jax.tree.leaves(modified_params),
    ):
        np.testing.assert_array_equal(np.asarray(leaf_resumed), np.asarray(leaf_saved))

    for leaf_resumed, leaf_saved in zip(
        jax.tree.leaves(resumed_state.ema_params),
        jax.tree.leaves(modified_ema),
    ):
        np.testing.assert_array_equal(np.asarray(leaf_resumed), np.asarray(leaf_saved))


def test_init_from_checkpoint_null_is_noop():
    cfg = _base_cfg(init_from_checkpoint=None)
    state, tx = init_state(cfg, _DummySJDModel(), make_rng(0))
    assert tx is not None
    assert int(jax.device_get(state.step)) == 0
    np.testing.assert_array_equal(
        np.asarray(state.params["classifier"]["w"]),
        np.asarray([1.0]),
    )
