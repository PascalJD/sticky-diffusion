from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import sticky.tasks.sudoku_inpaint_sjd_slack as task_mod
from sticky.tasks.sudoku_inpaint_sjd_slack import SudokuInpaintSJDSlackTask


def _beta(t):
    t = jnp.asarray(t, dtype=jnp.float32)
    return jnp.ones_like(t)


def _valid_solution_board() -> np.ndarray:
    sol = np.array(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )
    return sol


def _batch():
    solution = _valid_solution_board()
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :20] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    # Build slack via the production helper to exercise the data path end-to-end.
    from sticky.data.sudoku import compute_slack_vectors

    slack_x0 = compute_slack_vectors(solution)
    return {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
        "slack_x0": slack_x0,
    }


class _DummySJDModel:
    def embed(self, token_ids):
        # Simplex-vertex embedding: one-hot in R^9.
        return jax.nn.one_hot(jnp.asarray(token_ids, dtype=jnp.int32), 9)

    def anchor_table(self):
        return jnp.eye(9, dtype=jnp.float32)

    def apply(self, variables, *args, method=None, **kwargs):
        del variables, kwargs
        if method is not None and getattr(method, "__name__", "") == "embed":
            return self.embed(args[0])
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self.anchor_table()
        # Forward call: args[0] is cell_xt (B, 81, 9). Return joint logits.
        cell_xt = jnp.asarray(args[0], dtype=jnp.float32)
        B = int(cell_xt.shape[0])
        return jnp.zeros((B, 108, 9), dtype=jnp.float32), {}


def _make_task() -> SudokuInpaintSJDSlackTask:
    return SudokuInpaintSJDSlackTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=1,
        eval_batch_size=1,
        data_shape=(81,),
        vocab_size=9,
        num_classes=-1,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
    )


def test_loss_passes_slack_through_to_loss(monkeypatch):
    captured = {}

    def _fake_loss_with_slack(**kwargs):
        captured.update(kwargs)
        return (
            jnp.asarray(0.5, dtype=jnp.float32),
            {"loss/slack_residual_l2": jnp.asarray(0.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss_with_slack", _fake_loss_with_slack)

    task = _make_task()
    loss, metrics = task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=_DummySJDModel(),
        params={},
        batch=_batch(),
        train=False,
    )

    assert tuple(captured["slack_x0"].shape) == (1, 27, 9)
    np.testing.assert_array_equal(
        np.asarray(captured["slack_x0"]),
        np.ones((1, 27, 9), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(captured["x0_idx"]),
        np.asarray(_valid_solution_board() - 1, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(captured["given_mask"]), np.asarray(_batch()["clue_mask"])
    )
    assert float(loss) == 0.5
    assert float(metrics["clean_index_min"]) == 0.0
    assert float(metrics["clean_index_max"]) == 8.0


def test_loss_apply_fn_forwards_slack_kwarg(monkeypatch):
    """The closure built inside loss_fn must call model.apply with slack_y_t kw."""
    captured = {}

    def _fake_loss_with_slack(**kwargs):
        # Invoke the inner apply_fn once so we can observe its kwargs.
        apply_fn = kwargs["apply_fn"]
        cell = jnp.zeros((1, 81, 9), dtype=jnp.float32)
        slack = jnp.ones((1, 27, 9), dtype=jnp.float32)
        t = jnp.full((1,), 0.3, dtype=jnp.float32)
        apply_fn({}, cell, slack, t)
        return jnp.asarray(0.0, dtype=jnp.float32), {}

    class _RecordingModel(_DummySJDModel):
        def apply(self, variables, *args, method=None, **kwargs):
            if method is None:
                captured["apply_kwargs"] = dict(kwargs)
                captured["apply_args"] = tuple(arg.shape for arg in args)
                B = args[0].shape[0]
                return jnp.zeros((B, 108, 9), dtype=jnp.float32), {}
            return super().apply(variables, *args, method=method, **kwargs)

    monkeypatch.setattr(task_mod, "ce_allocation_loss_with_slack", _fake_loss_with_slack)

    task = _make_task()
    task.loss_fn(
        rng=jax.random.PRNGKey(0),
        model=_RecordingModel(),
        params={},
        batch=_batch(),
        train=False,
    )

    assert "slack_y_t" in captured["apply_kwargs"]
    assert captured["apply_kwargs"]["slack_y_t"].shape == (1, 27, 9)
    assert captured["apply_args"][0] == (1, 81, 9)


def test_loss_fn_traces_under_jit(monkeypatch):
    def _fake_loss_with_slack(**kwargs):
        del kwargs
        return (
            jnp.asarray(0.0, dtype=jnp.float32),
            {"loss/slack_residual_l2": jnp.asarray(0.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss_with_slack", _fake_loss_with_slack)
    task = _make_task()

    @jax.jit
    def step(rng, batch_solution, batch_clue_mask, batch_slack):
        return task.loss_fn(
            rng=rng,
            model=_DummySJDModel(),
            params={},
            batch={
                "solution_board": batch_solution,
                "clue_mask": batch_clue_mask,
                "slack_x0": batch_slack,
            },
            train=False,
        )

    batch = _batch()
    loss, _ = step(
        jax.random.PRNGKey(0),
        jnp.asarray(batch["solution_board"]),
        jnp.asarray(batch["clue_mask"]),
        jnp.asarray(batch["slack_x0"]),
    )
    assert float(loss) == 0.0
