from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

import jax
import jax.numpy as jnp

from sticky.rng import ensure_prng_key, legacy_prng_key_data, make_rng
from sticky.training.state import init_state, make_optimizer


class _DummyDiscreteModel:
    def init(self, rngs, x, t=None, *, cond=None, train=False):
        del rngs, x, t, cond, train
        return {"params": {"w": jnp.asarray([1.0], dtype=jnp.float32)}}


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

    def apply(self, *args, **kwargs):
        raise AssertionError("init_state should not call model.apply for SJD anchor init")


def _base_cfg(model_name: str):
    model_cfg = {
        "name": model_name,
        "classes": -1,
    }
    if model_name == "sjd":
        model_cfg["anchor"] = {
            "family": "normal",
            "dim": 2,
            "learnable": True,
        }

    return OmegaConf.create(
        {
            "dataset": {
                "per_device_batch_size": 2,
                "data_shape": [4],
            },
            "model": model_cfg,
            "optim": {
                "warmup_steps": 0,
                "learning_rate": 1e-3,
                "b2": 0.999,
                "weight_decay": 0.0,
            },
            "training": {
                "num_train_steps": 1,
                "ema_rate": 0.9,
            },
        }
    )


def test_init_state_supports_all_model_branches():
    for model_name, model in (
        ("md4", _DummyDiscreteModel()),
        ("mdm", _DummyDiscreteModel()),
        ("cadd", _DummyDiscreteModel()),
        ("ddpm", _DummyDiscreteModel()),
        ("sjd", _DummySJDModel()),
    ):
        state, tx = init_state(_base_cfg(model_name), model, make_rng(0))
        assert tx is not None
        assert int(jax.device_get(state.step)) == 0
        assert state.ema_params is not None
        if model_name == "sjd":
            assert "anchors" in state.params
            assert state.params["anchors"]["table"].shape == (4, 2)


def test_sjd_init_state_does_not_need_second_mutable_apply():
    cfg = _base_cfg("sjd")
    state, _ = init_state(cfg, _DummySJDModel(), make_rng(0))

    assert state.params["anchors"]["table"].shape == (4, 2)


def test_typed_rng_smoke_round_trips_through_legacy_key_data():
    key = make_rng(11)
    split_key = jax.random.split(key, 2)[0]
    folded_key = jax.random.fold_in(split_key, 7)
    legacy_key = legacy_prng_key_data(folded_key)
    restored_key = ensure_prng_key(legacy_key)

    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(folded_key)),
        np.asarray(jax.random.key_data(restored_key)),
    )


def test_init_state_returns_typed_rng_key():
    state, _ = init_state(_base_cfg("md4"), _DummyDiscreteModel(), make_rng(0))

    assert state.rng.shape == ()
    assert jax.random.key_data(state.rng).shape == (2,)


def test_make_optimizer_routes_log_w_through_separate_chain():
    """make_optimizer routes params['anchors']['log_w'] through its own Adam
    with LR ≈ multiplier × base_lr. Verifies the multi_transform routing — at
    step 1 with g=1 everywhere and weight_decay=0, both Adam and AdamW produce
    update ≈ −lr, so the per-leaf update ratio equals the LR ratio.
    """
    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd", "learnable_anchors": True, "classes": -1},
            "optim": {
                "warmup_steps": 0,
                "learning_rate": 1e-4,
                "b2": 0.999,
                "weight_decay": 0.0,
                "lr_schedule": "warmup_constant",
                "grad_clip_norm": 0.0,
            },
            "training": {
                "num_train_steps": 100,
                "max_steps": 100,
                "log_w_lr_multiplier": 100.0,
            },
        }
    )
    params = {
        "anchors": {
            "table": jnp.zeros((4, 2), dtype=jnp.float32),
            "log_w": jnp.zeros((4,), dtype=jnp.float32),
        },
        "classifier": {"W": jnp.zeros((2, 4), dtype=jnp.float32)},
    }
    tx = make_optimizer(cfg, params)
    state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, state, params)

    log_w_step = float(jnp.mean(jnp.abs(updates["anchors"]["log_w"])))
    classifier_step = float(jnp.mean(jnp.abs(updates["classifier"]["W"])))
    assert classifier_step > 0.0, "classifier update is zero — multi_transform misrouted"
    ratio = log_w_step / classifier_step
    assert 90.0 < ratio < 110.0, (
        f"expected log_w step ~100x classifier step, got ratio={ratio:.3f}"
    )


def test_make_optimizer_log_w_lr_multiplier_overridable():
    cfg = OmegaConf.create(
        {
            "model": {"name": "sjd", "learnable_anchors": True, "classes": -1},
            "optim": {
                "warmup_steps": 0,
                "learning_rate": 1e-4,
                "b2": 0.999,
                "weight_decay": 0.0,
                "lr_schedule": "warmup_constant",
                "grad_clip_norm": 0.0,
            },
            "training": {
                "num_train_steps": 10,
                "max_steps": 10,
                "log_w_lr_multiplier": 25.0,
            },
        }
    )
    params = {
        "anchors": {
            "table": jnp.zeros((4, 2), dtype=jnp.float32),
            "log_w": jnp.zeros((4,), dtype=jnp.float32),
        },
        "classifier": {"W": jnp.zeros((2, 4), dtype=jnp.float32)},
    }
    tx = make_optimizer(cfg, params)
    state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, state, params)
    ratio = float(jnp.mean(jnp.abs(updates["anchors"]["log_w"]))) / float(
        jnp.mean(jnp.abs(updates["classifier"]["W"]))
    )
    assert 22.0 < ratio < 28.0, f"expected ~25x with multiplier=25, got {ratio:.3f}"
