from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.tasks.sudoku_mdlm import SudokuMDLMTask


class _DummyMDLMModel:
    def __init__(self):
        self.known_token_mask = None
        self.loss_mask = None

    def apply(self, variables, x, *, train, known_token_mask, loss_mask, rngs):
        del variables, x, train, rngs
        self.known_token_mask = known_token_mask
        self.loss_mask = loss_mask
        loss = loss_mask.astype(jnp.float32).sum()
        return {
            "loss": loss,
            "loss_diff": loss,
            "loss_prior": jnp.asarray(0.0, dtype=jnp.float32),
            "loss_recon": jnp.asarray(0.0, dtype=jnp.float32),
        }


def _make_task() -> SudokuMDLMTask:
    return SudokuMDLMTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=1,
        eval_batch_size=1,
        data_shape=(243,),
        vocab_size=10,
        num_classes=-1,
    )


def test_sudoku_mdlm_loss_masks_out_the_clue_prefix():
    task = _make_task()
    model = _DummyMDLMModel()
    start_index = jnp.asarray([[0], [2], [81]], dtype=jnp.int32)
    token_pos = jnp.arange(243, dtype=jnp.int32)[None, :]
    expected_known_token_mask = token_pos < (3 * start_index)
    expected_loss_mask = ~expected_known_token_mask

    loss, stats = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=model,
        params={},
        batch={
            "image": jnp.zeros((3, 243), dtype=jnp.int32),
            "start_index": start_index,
        },
        train=False,
    )

    assert model.known_token_mask is not None
    assert model.loss_mask is not None
    assert bool(jnp.array_equal(model.known_token_mask, expected_known_token_mask))
    assert bool(jnp.array_equal(model.loss_mask, expected_loss_mask))
    assert bool(jnp.array_equal(model.known_token_mask.sum(axis=1), jnp.asarray([0, 6, 243])))
    assert bool(jnp.array_equal(model.loss_mask.sum(axis=1), jnp.asarray([243, 237, 0])))
    assert float(loss) == 480.0
    assert float(stats["loss_diff"]) == 480.0
