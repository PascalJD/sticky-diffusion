from __future__ import annotations

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import sticky.training.loop as loop_mod


def test_make_pmap_batch_iterator_prefetch_preserves_shapes():
    per_device_batch = 2
    batch = {
        "image": jnp.zeros(
            (jax.local_device_count() * per_device_batch, 4, 4, 3),
            dtype=jnp.float32,
        ),
        "label": jnp.zeros(
            (jax.local_device_count() * per_device_batch,),
            dtype=jnp.int32,
        ),
    }

    iterator = loop_mod._make_pmap_batch_iterator(
        iter([batch]),
        prefetch_buffer_size=2,
    )
    prefetched = next(iterator)

    assert prefetched["image"].shape == (
        jax.local_device_count(),
        per_device_batch,
        4,
        4,
        3,
    )
    assert prefetched["label"].shape == (
        jax.local_device_count(),
        per_device_batch,
    )


def test_sync_flag_toggles_blocking_behavior(monkeypatch):
    calls = []

    def fake_block_until_ready(x):
        calls.append(x)
        return x

    monkeypatch.setattr(loop_mod.jax, "block_until_ready", fake_block_until_ready)

    metric = jnp.asarray(1.0, dtype=jnp.float32)
    loop_mod._maybe_sync_training_metric(metric, sync=False)
    assert calls == []

    loop_mod._maybe_sync_training_metric(metric, sync=True)
    assert len(calls) == 1


def test_resolve_num_train_steps_prefers_epoch_derived_budget():
    cfg = OmegaConf.create(
        {
            "dataset": {"batch_size": 128, "drop_remainder": True},
            "training": {"num_train_steps": 123, "num_train_epochs": 300},
        }
    )

    class _Task:
        def train_num_examples(self):
            return 1800

    num_steps = loop_mod._resolve_num_train_steps(cfg, _Task())

    assert num_steps == 4200


def test_resolve_num_train_steps_uses_explicit_steps_without_epochs():
    cfg = OmegaConf.create(
        {
            "dataset": {"batch_size": 128, "drop_remainder": True},
            "training": {"num_train_steps": 777},
        }
    )

    num_steps = loop_mod._resolve_num_train_steps(cfg, object())

    assert num_steps == 777
