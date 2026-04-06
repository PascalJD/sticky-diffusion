from __future__ import annotations

import jax
import jax.numpy as jnp

import sticky.tasks.sudoku_mdlm as sudoku_mdlm_mod
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
        data_shape=(81,),
        vocab_size=10,
        num_classes=-1,
    )


def test_sudoku_mdlm_loss_masks_only_unknown_cells():
    task = _make_task()
    model = _DummyMDLMModel()
    clue_mask = jnp.asarray(
        [
            [False] * 81,
            ([True] * 6) + ([False] * 75),
            [True] * 81,
        ],
        dtype=jnp.bool_,
    )
    expected_known_token_mask = clue_mask
    expected_loss_mask = ~expected_known_token_mask

    loss, stats = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=model,
        params={},
        batch={
            "solution_board": jnp.ones((3, 81), dtype=jnp.int32),
            "clue_mask": clue_mask,
        },
        train=False,
    )

    assert model.known_token_mask is not None
    assert model.loss_mask is not None
    assert bool(jnp.array_equal(model.known_token_mask, expected_known_token_mask))
    assert bool(jnp.array_equal(model.loss_mask, expected_loss_mask))
    assert bool(jnp.array_equal(model.known_token_mask.sum(axis=1), jnp.asarray([0, 6, 81])))
    assert bool(jnp.array_equal(model.loss_mask.sum(axis=1), jnp.asarray([81, 75, 0])))
    assert float(loss) == 156.0
    assert float(stats["loss_diff"]) == 156.0


def test_sudoku_mdlm_loader_and_count_plumbing_is_stable(monkeypatch):
    task = SudokuMDLMTask(
        data_dir="/tmp/sudoku",
        train_file="train.npy",
        test_file="test.npy",
        batch_size=8,
        eval_batch_size=4,
        data_shape=(81,),
        vocab_size=10,
        num_classes=-1,
        drop_remainder=True,
        shuffle=True,
        mmap=False,
        max_train_examples=11,
        max_test_examples=7,
        auto_download=False,
        download_timeout_sec=33,
        download_retries=5,
    )
    iterator_calls = []
    count_calls = []

    def fake_make_sudoku_board_iterator(**kwargs):
        iterator_calls.append(dict(kwargs))
        return iter([{"solution_board": "ok"}])

    def fake_get_sudoku_board_num_examples(**kwargs):
        count_calls.append(dict(kwargs))
        return 123

    monkeypatch.setattr(
        sudoku_mdlm_mod,
        "make_sudoku_board_iterator",
        fake_make_sudoku_board_iterator,
    )
    monkeypatch.setattr(
        sudoku_mdlm_mod,
        "get_sudoku_board_num_examples",
        fake_get_sudoku_board_num_examples,
    )

    train_it, eval_it = task.make_dataloaders(seed=17)

    assert next(train_it) == {"solution_board": "ok"}
    assert next(eval_it) == {"solution_board": "ok"}
    assert iterator_calls == [
        {
            "split": "train",
            "batch_size": 8,
            "seed": 17,
            "data_dir": "/tmp/sudoku",
            "train_file": "train.npy",
            "test_file": "test.npy",
            "shuffle": True,
            "repeat": True,
            "drop_remainder": True,
            "mmap": False,
            "max_examples": 11,
            "auto_download": False,
            "download_timeout_sec": 33,
            "download_retries": 5,
        },
        {
            "split": "test",
            "batch_size": 4,
            "seed": 18,
            "data_dir": "/tmp/sudoku",
            "train_file": "train.npy",
            "test_file": "test.npy",
            "shuffle": False,
            "repeat": False,
            "drop_remainder": False,
            "mmap": False,
            "max_examples": 7,
            "auto_download": False,
            "download_timeout_sec": 33,
            "download_retries": 5,
        },
    ]

    assert task.train_num_examples() == 123
    assert count_calls == [
        {
            "split": "train",
            "data_dir": "/tmp/sudoku",
            "train_file": "train.npy",
            "test_file": "test.npy",
            "mmap": False,
            "max_examples": 11,
            "auto_download": False,
            "download_timeout_sec": 33,
            "download_retries": 5,
        }
    ]
