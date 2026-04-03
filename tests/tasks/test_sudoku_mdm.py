from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import sticky.tasks.sudoku_base as sudoku_base_mod
from sticky.data.sudoku import (
    SUDOKU_PACKED_SEQ_LEN,
    SUDOKU_SEP_TOKEN_ID,
    SUDOKU_VOCAB_SIZE,
)
from sticky.tasks.sudoku_mdm import SudokuMDMTask


class _DummyMDMModel:
    def __init__(
        self,
        *,
        vocab_size: int = 12,
        timesteps: int = 4,
        token_reweighting: bool = False,
        alpha: float = 0.25,
        gamma: float = 1.0,
        time_reweighting: str = "none",
        logits: jnp.ndarray | None = None,
    ):
        self.vocab_size = int(vocab_size)
        self.timesteps = int(timesteps)
        self.token_reweighting = bool(token_reweighting)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.time_reweighting = str(time_reweighting)
        self._logits = logits
        self.last_x_t = None
        self.last_t = None

    @property
    def mask_token_id(self) -> int:
        return int(self.vocab_size)

    def predict_logits(self, zt, t=None, *, cond=None, train: bool = False):
        raise AssertionError("predict_logits should be called through apply")

    def apply(self, variables, zt, t=None, *, method=None, train: bool = False, rngs=None):
        del variables, train, rngs
        self.last_x_t = zt
        self.last_t = t
        method_name = getattr(method, "__name__", "")
        if method_name != "predict_logits":
            raise AssertionError(f"Unexpected method call: {method_name}")
        if self._logits is not None:
            return self._logits
        shape = zt.shape + (self.vocab_size,)
        return jnp.zeros(shape, dtype=jnp.float32)

    def token_cross_entropy(self, logits, targets):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]

    def apply_token_reweighting(self, token_loss):
        token_loss = jnp.asarray(token_loss, dtype=jnp.float32)
        if not self.token_reweighting:
            return token_loss
        return self.alpha * (1.0 - jnp.exp(-token_loss)) ** self.gamma * token_loss

    def time_weights(self, t):
        t = jnp.asarray(t, dtype=jnp.int32)
        if self.time_reweighting == "original":
            return 1.0 / (t.astype(jnp.float32) + 1.0)
        if self.time_reweighting == "linear":
            return float(self.timesteps) - t.astype(jnp.float32)
        if self.time_reweighting == "none":
            return jnp.ones_like(t, dtype=jnp.float32)
        raise AssertionError(f"Unexpected time_reweighting={self.time_reweighting!r}")


def _make_task() -> SudokuMDMTask:
    return SudokuMDMTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=1,
        data_shape=(243,),
        vocab_size=10,
        num_classes=-1,
    )


def _legacy_batch(*, start_index: int = 2) -> dict[str, np.ndarray]:
    return {
        "image": np.asarray([np.arange(243, dtype=np.int32) % 10], dtype=np.int32),
        "start_index": np.asarray([[start_index]], dtype=np.int32),
        "puzzle": np.zeros((1, 81), dtype=np.int32),
    }


def test_sudoku_mdm_task_spec_uses_packed_sequence_geometry():
    task = _make_task()

    assert task.spec.name == "mdm_sudoku"
    assert task.spec.task_type == "text"
    assert task.spec.data_shape == (SUDOKU_PACKED_SEQ_LEN,)
    assert task.spec.vocab_size == SUDOKU_VOCAB_SIZE


def test_sudoku_mdm_dataloader_augments_legacy_batches_with_packed_fields(monkeypatch):
    task = _make_task()
    legacy_batch = {
        "image": np.asarray(
            [
                np.arange(243, dtype=np.int32) % 10,
                np.roll(np.arange(243, dtype=np.int32) % 10, 7),
            ],
            dtype=np.int32,
        ),
        "start_index": np.asarray([[0], [2]], dtype=np.int32),
        "puzzle": np.zeros((2, 81), dtype=np.int32),
    }
    iterator_calls = []

    def fake_make_sudoku_iterator(**kwargs):
        iterator_calls.append(dict(kwargs))
        return iter([legacy_batch])

    monkeypatch.setattr(sudoku_base_mod, "make_sudoku_iterator", fake_make_sudoku_iterator)

    train_it, eval_it = task.make_dataloaders(seed=11)
    train_batch = next(train_it)
    eval_batch = next(eval_it)

    assert iterator_calls[0]["split"] == "train"
    assert iterator_calls[1]["split"] == "test"

    np.testing.assert_array_equal(train_batch["image"], legacy_batch["image"])
    np.testing.assert_array_equal(train_batch["start_index"], legacy_batch["start_index"])
    np.testing.assert_array_equal(train_batch["puzzle"], legacy_batch["puzzle"])
    assert train_batch["packed_seq"].shape == (2, SUDOKU_PACKED_SEQ_LEN)
    assert train_batch["prompt_mask"].shape == (2, SUDOKU_PACKED_SEQ_LEN)
    assert train_batch["response_mask"].shape == (2, SUDOKU_PACKED_SEQ_LEN)
    np.testing.assert_array_equal(train_batch["sep_index"], np.asarray([[0], [6]], dtype=np.int32))
    np.testing.assert_array_equal(
        train_batch["response_start_index"],
        np.asarray([[1], [7]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        train_batch["eos_index"],
        np.asarray([[244], [244]], dtype=np.int32),
    )
    assert int(train_batch["packed_seq"][0, 0]) == SUDOKU_SEP_TOKEN_ID
    np.testing.assert_array_equal(eval_batch["packed_seq"], train_batch["packed_seq"])


def test_sudoku_mdm_prompt_tokens_are_never_masked(monkeypatch):
    task = _make_task()
    batch = task._augment_batch(_legacy_batch(start_index=2))
    model = _DummyMDMModel(timesteps=4)

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

    x = jnp.asarray(batch["packed_seq"], dtype=jnp.int32)
    prompt_mask = jnp.asarray(batch["prompt_mask"], dtype=jnp.bool_)
    response_mask = jnp.asarray(batch["response_mask"], dtype=jnp.bool_)

    assert model.last_x_t is not None
    assert bool(jnp.array_equal(model.last_x_t[prompt_mask], x[prompt_mask]))
    assert bool(jnp.all(model.last_x_t[response_mask] == model.mask_token_id))
    assert float(metrics["masked_prompt_token_count"]) == 0.0
    assert float(metrics["masked_response_fraction"]) == 1.0
    assert float(loss) > 0.0


def test_sudoku_mdm_loss_is_zero_when_no_response_tokens_are_masked(monkeypatch):
    task = _make_task()
    batch = task._augment_batch(_legacy_batch(start_index=2))
    model = _DummyMDMModel(timesteps=4)

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
        rng=jax.random.PRNGKey(1),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    np.testing.assert_array_equal(
        np.asarray(model.last_x_t),
        np.asarray(batch["packed_seq"]),
    )
    assert float(loss) == 0.0
    assert float(metrics["loss_ce"]) == 0.0
    assert float(metrics["masked_token_count"]) == 0.0
    assert float(metrics["examples_with_zero_masked_response_tokens"]) == 1.0


def test_sudoku_mdm_masking_probability_depends_on_timestep(monkeypatch):
    task = _make_task()
    legacy = {
        "image": np.asarray(
            [
                np.arange(243, dtype=np.int32) % 10,
                np.roll(np.arange(243, dtype=np.int32) % 10, 5),
            ],
            dtype=np.int32,
        ),
        "start_index": np.asarray([[2], [2]], dtype=np.int32),
        "puzzle": np.zeros((2, 81), dtype=np.int32),
    }
    batch = task._augment_batch(legacy)
    model = _DummyMDMModel(timesteps=4)

    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.asarray([0, 3], dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: jnp.full(shape, 0.5, dtype=dtype),
    )

    _, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(2),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    response_mask = jnp.asarray(batch["response_mask"], dtype=jnp.bool_)
    masked = model.last_x_t == model.mask_token_id

    assert int(masked[0].sum()) == 0
    assert int(masked[1].sum()) == int(response_mask[1].sum())
    assert float(metrics["examples_with_zero_masked_response_tokens"]) == 1.0
    assert np.isclose(float(metrics["mask_prob_mean"]), 0.625)


def test_sudoku_mdm_neutral_reweighting_matches_plain_masked_token_ce(monkeypatch):
    task = _make_task()
    batch = task._augment_batch(_legacy_batch(start_index=2))
    x = jnp.asarray(batch["packed_seq"], dtype=jnp.int32)
    seq_len = int(x.shape[1])
    vocab_size = 12
    target_pos = 10

    logits = -1.0e9 * jnp.ones((1, seq_len, vocab_size), dtype=jnp.float32)
    logits = logits.at[0, jnp.arange(seq_len), x[0]].set(0.0)
    logits = logits.at[0, target_pos, :].set(0.0)
    model = _DummyMDMModel(
        timesteps=4,
        token_reweighting=False,
        time_reweighting="none",
        logits=logits,
    )

    uniform = jnp.ones((1, seq_len), dtype=jnp.float32)
    uniform = uniform.at[0, target_pos].set(0.0)
    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: uniform.astype(dtype),
    )

    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(3),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    expected = np.log(vocab_size)
    assert np.isclose(float(loss), expected)
    assert np.isclose(float(metrics["loss_ce"]), expected)
    assert float(metrics["masked_token_count"]) == 1.0


def test_sudoku_mdm_reweighting_matches_ye_formula(monkeypatch):
    task = _make_task()
    batch = task._augment_batch(_legacy_batch(start_index=2))
    x = jnp.asarray(batch["packed_seq"], dtype=jnp.int32)
    seq_len = int(x.shape[1])
    vocab_size = 12
    target_pos = 10

    logits = -1.0e9 * jnp.ones((1, seq_len, vocab_size), dtype=jnp.float32)
    logits = logits.at[0, jnp.arange(seq_len), x[0]].set(0.0)
    logits = logits.at[0, target_pos, :].set(0.0)
    model = _DummyMDMModel(
        timesteps=4,
        token_reweighting=True,
        alpha=0.25,
        gamma=1.0,
        time_reweighting="linear",
        logits=logits,
    )

    uniform = jnp.ones((1, seq_len), dtype=jnp.float32)
    uniform = uniform.at[0, target_pos].set(0.0)
    monkeypatch.setattr(
        jax.random,
        "randint",
        lambda key, shape, minval, maxval, dtype: jnp.asarray([1], dtype=dtype),
    )
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key, shape, dtype=jnp.float32: uniform.astype(dtype),
    )

    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(4),
        model=model,
        params={},
        batch=batch,
        train=False,
    )

    plain = np.log(vocab_size)
    expected = (model.timesteps - 1) * (
        model.alpha * (1.0 - np.exp(-plain)) ** model.gamma * plain
    )
    assert np.isclose(float(metrics["loss_ce"]), plain)
    assert np.isclose(float(loss), expected)
