from __future__ import annotations

import jax
import jax.numpy as jnp

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
