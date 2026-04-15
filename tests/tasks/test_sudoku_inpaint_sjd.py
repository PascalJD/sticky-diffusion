from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import sticky.tasks.sudoku_inpaint_sjd as task_mod
from sticky.tasks.sudoku_inpaint_sjd import SudokuInpaintSJDTask


def _beta(t):
    t = jnp.asarray(t, dtype=jnp.float32)
    return jnp.ones_like(t)


def _batch():
    solution = np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :20] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return {
        "solution_board": solution,
        "clue_board": clue_board,
        "clue_mask": clue_mask,
    }


class _DummySJDModel:
    def embed(self, token_ids):
        return jnp.asarray(token_ids, dtype=jnp.float32)[..., None]

    def anchor_table(self):
        return jnp.arange(18, dtype=jnp.float32).reshape(9, 2)

    def apply(self, variables, *args, method=None, **kwargs):
        del variables, kwargs
        if method is not None and getattr(method, "__name__", "") == "embed":
            return self.embed(args[0])
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self.anchor_table()
        xt = jnp.asarray(args[0], dtype=jnp.float32)
        return jnp.zeros(xt.shape[:-1] + (9,), dtype=jnp.float32), {}


def test_loss_uses_clean_digit_indices_and_excludes_clues(monkeypatch):
    captured = {}

    def _fake_ce_allocation_loss(**kwargs):
        captured.update(kwargs)
        return (
            jnp.asarray(0.25, dtype=jnp.float32),
            {"CE/acc_top1_event": jnp.asarray(1.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss", _fake_ce_allocation_loss)

    task = SudokuInpaintSJDTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=2,
        data_shape=(81,),
        vocab_size=9,
        num_classes=-1,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
    )

    loss, metrics = task.loss_fn(
        rng=jnp.asarray([0, 1], dtype=jnp.uint32),
        model=_DummySJDModel(),
        params={},
        batch=_batch(),
        train=False,
    )

    expected_idx = jnp.asarray(_batch()["solution_board"], dtype=jnp.int32) - 1
    np.testing.assert_array_equal(np.asarray(captured["x0_idx"]), np.asarray(expected_idx))
    np.testing.assert_array_equal(
        np.asarray(captured["given_mask"]),
        np.asarray(_batch()["clue_mask"]),
    )
    assert int(jnp.min(captured["x0_idx"])) >= 0
    assert int(jnp.max(captured["x0_idx"])) <= 8
    assert float(loss) == 0.25
    assert float(metrics["clean_index_min"]) == 0.0
    assert float(metrics["clean_index_max"]) == 8.0
    assert float(metrics["given_fraction"]) > 0.0


def test_teacher_margin_threads_teacher_params(monkeypatch):
    captured = {}

    def _fake_ce_allocation_loss(**kwargs):
        captured.update(kwargs)
        return (
            jnp.asarray(0.5, dtype=jnp.float32),
            {"CE/acc_top1_event": jnp.asarray(1.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss", _fake_ce_allocation_loss)

    task = SudokuInpaintSJDTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=2,
        data_shape=(81,),
        vocab_size=9,
        num_classes=-1,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
        forward_site_hazard_mode="teacher_margin",
        teacher_confidence_metric="top_prob",
        teacher_w_min=0.3,
        teacher_w_max=1.7,
    )
    assert task.requires_teacher_params is True

    task.loss_fn(
        rng=jnp.asarray([0, 1], dtype=jnp.uint32),
        model=_DummySJDModel(),
        params={"foo": jnp.zeros(())},
        batch=_batch(),
        train=True,
        teacher_params={"bar": jnp.zeros(())},
    )

    assert captured["forward_site_hazard_mode"] == "teacher_margin"
    assert captured["teacher_confidence_metric"] == "top_prob"
    assert float(captured["teacher_w_min"]) == 0.3
    assert float(captured["teacher_w_max"]) == 1.7
    assert captured["teacher_params"] is not None
    assert captured["teacher_apply_fn"] is not None
    assert captured["x0_anchor_teacher"] is not None


def test_loss_fn_traces_under_jit(monkeypatch):
    def _fake_ce_allocation_loss(**kwargs):
        del kwargs
        return (
            jnp.asarray(0.25, dtype=jnp.float32),
            {"CE/acc_top1_event": jnp.asarray(1.0, dtype=jnp.float32)},
        )

    monkeypatch.setattr(task_mod, "ce_allocation_loss", _fake_ce_allocation_loss)

    task = SudokuInpaintSJDTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=2,
        data_shape=(81,),
        vocab_size=9,
        num_classes=-1,
        beta=_beta,
        hazard=object(),
        jump=object(),
        T=1.0,
        log_state_dependency=False,
    )

    batch = {
        key: jnp.asarray(value)
        for key, value in _batch().items()
    }

    @jax.jit
    def _run(rng, solution_board, clue_board, clue_mask):
        loss, metrics = task.loss_fn(
            rng=rng,
            model=_DummySJDModel(),
            params={},
            batch={
                "solution_board": solution_board,
                "clue_board": clue_board,
                "clue_mask": clue_mask,
            },
            train=True,
        )
        return loss, metrics["clean_index_min"], metrics["clean_index_max"]

    loss, clean_min, clean_max = _run(
        jax.random.PRNGKey(0),
        batch["solution_board"],
        batch["clue_board"],
        batch["clue_mask"],
    )

    assert float(loss) == 0.25
    assert float(clean_min) == 0.0
    assert float(clean_max) == 8.0
