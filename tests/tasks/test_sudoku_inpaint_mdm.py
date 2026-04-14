from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from sticky.tasks.sudoku_inpaint_mdm import SudokuInpaintMDMTask


class _DummyInpaintModel:
    def __init__(
        self,
        *,
        vocab_size: int = 10,
        timesteps: int = 4,
        ce_values: jnp.ndarray | None = None,
    ):
        self.vocab_size = int(vocab_size)
        self.timesteps = int(timesteps)
        self.ce_values = ce_values
        self.last_x_t = None
        self.last_t = None

    @property
    def mask_token_id(self) -> int:
        return 0

    def predict_logits(self, zt, t=None, *, cond=None, train: bool = False):
        raise AssertionError("predict_logits should be called through apply")

    def apply(self, variables, zt, t=None, *, method=None, train: bool = False, rngs=None):
        del variables, train, rngs
        self.last_x_t = zt
        self.last_t = t
        method_name = getattr(method, "__name__", "")
        if method_name != "predict_logits":
            raise AssertionError(f"Unexpected method call: {method_name}")
        return jnp.zeros(zt.shape + (self.vocab_size,), dtype=jnp.float32)

    def token_cross_entropy(self, logits, targets):
        del logits
        if self.ce_values is not None:
            return jnp.asarray(self.ce_values, dtype=jnp.float32)
        return jnp.ones(targets.shape, dtype=jnp.float32)


def _make_task() -> SudokuInpaintMDMTask:
    return SudokuInpaintMDMTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=1,
        data_shape=(81,),
        vocab_size=10,
        num_classes=-1,
    )


def _make_batch() -> dict[str, np.ndarray]:
    solution = np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, [0, 10, 20]] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
        "image": solution,
    }


def test_clue_positions_are_never_masked_during_training(monkeypatch):
    task = _make_task()
    batch = _make_batch()
    model = _DummyInpaintModel(timesteps=4)

    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.full(shape, 3, dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype=dtype),
    )

    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    clue_mask = jnp.asarray(batch["clue_mask"], dtype=jnp.bool_)
    clue_board = jnp.asarray(batch["clue_board"], dtype=jnp.int32)

    assert model.last_x_t is not None
    assert bool(jnp.array_equal(model.last_x_t[clue_mask], clue_board[clue_mask]))
    assert float(metrics["masked_clue_token_count"]) == 0.0
    assert float(loss) > 0.0


def test_loss_only_uses_masked_non_clue_cells(monkeypatch):
    task = _make_task()
    batch = _make_batch()
    ce_values = jnp.arange(81, dtype=jnp.float32)[None, :]
    model = _DummyInpaintModel(timesteps=4, ce_values=ce_values)

    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.full(shape, 1, dtype=dtype),
    )

    draws = np.ones((1, 81), dtype=np.float32)
    draws[0, 5] = 0.0
    draws[0, 6] = 0.0
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: jnp.asarray(draws, dtype=dtype),
    )

    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(1),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    expected_loss = float((ce_values[0, 5] + ce_values[0, 6]) / 2.0)
    assert np.isclose(float(loss), expected_loss)
    assert np.isclose(float(metrics["loss_ce"]), expected_loss)
    assert float(metrics["masked_token_count"]) == 2.0


def test_zero_masked_non_clue_examples_are_finite(monkeypatch):
    task = _make_task()
    batch = _make_batch()
    model = _DummyInpaintModel(timesteps=4)

    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: jnp.ones(shape, dtype=dtype),
    )

    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(2),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    assert np.isfinite(float(loss))
    assert float(loss) == 0.0
    assert float(metrics["masked_token_count"]) == 0.0
    assert float(metrics["examples_with_zero_masked_unknown_tokens"]) == 1.0
