from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.tasks.sudoku_sjd import SudokuSJDTask


class _DummySudokuModel:
    def embed(self, token_ids):
        del token_ids
        raise AssertionError("embed should be dispatched through apply(method=...)")

    def anchor_table(self):
        raise AssertionError("anchor_table should be dispatched through apply(method=...)")

    def apply(self, variables, *args, method=None, **kwargs):
        del kwargs
        params = variables["params"]
        if method is not None:
            method_name = getattr(method, "__name__", "")
            if method_name == "embed":
                token_ids = args[0]
                return jax.nn.one_hot(token_ids, num_classes=10, dtype=jnp.float32)
            if method_name == "anchor_table":
                return params["anchors"]["table"]
            raise AssertionError(f"Unexpected method call: {method_name}")

        return params["logits"], {}


def _make_task() -> SudokuSJDTask:
    return SudokuSJDTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=1,
        eval_batch_size=1,
        data_shape=(243,),
        vocab_size=10,
        num_classes=-1,
        beta=make_beta(0.1, 20.0),
        log_state_dependency=False,
    )


def test_sudoku_loss_ignores_given_clue_prefix():
    task = _make_task()
    model = _DummySudokuModel()

    x0_idx = jnp.zeros((1, 243), dtype=jnp.int32)
    logits = -20.0 * jnp.ones((1, 243, 10), dtype=jnp.float32)
    logits = logits.at[:, :, 0].set(20.0)
    logits = logits.at[:, :3, 0].set(-20.0)
    logits = logits.at[:, :3, 1].set(20.0)
    params = {"logits": logits}

    loss_masked, metrics_masked = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=model,
        params=params,
        batch={
            "image": x0_idx,
            "start_index": jnp.asarray([[1]], dtype=jnp.int32),
        },
        train=False,
    )
    loss_unmasked, _ = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=model,
        params=params,
        batch={
            "image": x0_idx,
            "start_index": jnp.asarray([[0]], dtype=jnp.int32),
        },
        train=False,
    )

    assert float(loss_masked) < 1e-6
    assert float(loss_unmasked) > 0.1
    assert float(metrics_masked["CE/frac_uncommitted"]) == 1.0
