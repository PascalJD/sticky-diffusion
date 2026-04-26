from __future__ import annotations

import numpy as np

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.eval.sudoku import build_sudoku_eval_logger
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.training.state import init_state
from sticky.training.step import make_train_step_fn, params_for_sampling

import sticky.eval.sudoku as sudoku_eval_mod


jax = __import__("jax")
jnp = __import__("jax.numpy", fromlist=["jnp"])


def _board_batch():
    solution = np.asarray(
        [[((row * 3 + row // 3 + col) % 9) + 1 for row in range(9) for col in range(9)]],
        dtype=np.int32,
    )
    clue_mask = np.zeros((1, 81), dtype=np.bool_)
    clue_mask[0, :24] = True
    clue_board = np.where(clue_mask, solution, 0).astype(np.int32)
    return {
        "solution_board": jnp.asarray(solution, dtype=jnp.int32),
        "clue_board": jnp.asarray(clue_board, dtype=jnp.int32),
        "clue_mask": jnp.asarray(clue_mask, dtype=jnp.bool_),
    }


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(config_root())):
        return compose(config_name=config_name, overrides=overrides)


def test_sjd_sudoku_single_train_step_and_eval_smoke(monkeypatch):
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku",
            "eval=sudoku_sjd",
            "experiment.dataset.batch_size=1",
            "experiment.dataset.eval_batch_size=1",
            "experiment.training.num_train_epochs=1",
            "experiment.training.num_train_steps=1",
            "experiment.training.eval_every_steps=1",
            "experiment.training.checkpoint_every_steps=0",
            "experiment.training.log_every_steps=1",
            "experiment.sampler.n_steps=2",
            "eval.sudoku_num_batches=1",
            "eval.sudoku_num_batches_per_sampler=1",
            "eval.sudoku_prop52_enabled=false",
            "eval.sudoku_write_progress_csv=false",
            "eval.sudoku_write_latest_csv=false",
        ],
    )
    cfg.eval.sudoku_eval_sjd_runs = {
        "linear_survival": {
            "kind": "policy",
            "policy": "linear_survival",
            "n_steps": 2,
            "sampling_grid": "uniform",
            "eta": 1.0,
        },
        "plugin_hazard_eta_0p97": {
            "kind": "policy",
            "policy": "plugin_hazard",
            "n_steps": 2,
            "sampling_grid": "uniform",
            "eta": 0.97,
        },
    }

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    rng = jax.random.PRNGKey(int(cfg.experiment.training.seed))
    state, tx = init_state(cfg.experiment, model, rng)
    train_step_fn = make_train_step_fn(
        task=task,
        model=model,
        tx=tx,
        ema_rate=float(cfg.experiment.training.ema_rate),
    )

    batch = _board_batch()
    state, metrics = train_step_fn(state, batch, axis_name=None)
    assert float(jax.device_get(metrics["train/loss"])) >= 0.0

    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_board_iterator",
        lambda **kwargs: iter(
            [
                {
                    "solution_board": np.asarray(batch["solution_board"]),
                    "clue_board": np.asarray(batch["clue_board"]),
                    "clue_mask": np.asarray(batch["clue_mask"]),
                }
            ]
        ),
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg.experiment,
        eval_cfg=cfg.eval,
        task=task,
        model=model,
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    eval_metrics = maybe_eval(
        int(jax.device_get(state.step)),
        params_for_sampling(state),
        force_fid=True,
        force_is=False,
    )

    assert "eval/plugin_hazard_eta_0p97/solve_rate" in eval_metrics
    assert "eval/linear_survival/full_cell_acc" in eval_metrics


def test_sjd_sudoku_plugin_policy_and_sampler_smoke(monkeypatch):
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku",
            "eval=sudoku_sjd",
            "experiment.dataset.batch_size=1",
            "experiment.dataset.eval_batch_size=1",
            "experiment.training.num_train_epochs=1",
            "experiment.training.num_train_steps=1",
            "experiment.training.eval_every_steps=1",
            "experiment.training.checkpoint_every_steps=0",
            "experiment.training.log_every_steps=1",
            "experiment.sampler.n_steps=2",
            "eval.sudoku_num_batches=1",
            "eval.sudoku_num_batches_per_sampler=1",
            "eval.sudoku_prop52_enabled=false",
            "eval.sudoku_write_progress_csv=false",
            "eval.sudoku_write_latest_csv=false",
        ],
    )
    cfg.eval.sudoku_eval_sjd_runs = {
        "plugin_hazard": {
            "kind": "policy",
            "policy": "plugin_hazard",
            "n_steps": 2,
            "sampling_grid": "uniform",
            "eta": 0.97,
        },
        "predictor_only": {
            "kind": "sampler",
            "sampler": "sjd_sudoku",
        },
    }

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    rng = jax.random.PRNGKey(int(cfg.experiment.training.seed))
    state, _ = init_state(cfg.experiment, model, rng)
    batch = _board_batch()
    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_board_iterator",
        lambda **kwargs: iter(
            [
                {
                    "solution_board": np.asarray(batch["solution_board"]),
                    "clue_board": np.asarray(batch["clue_board"]),
                    "clue_mask": np.asarray(batch["clue_mask"]),
                }
            ]
        ),
    )

    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg.experiment,
        eval_cfg=cfg.eval,
        task=task,
        model=model,
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    eval_metrics = maybe_eval(
        int(jax.device_get(state.step)),
        params_for_sampling(state),
        force_fid=True,
        force_is=False,
    )
    assert "eval/plugin_hazard/full_cell_acc" in eval_metrics
    assert "eval/predictor_only/sampling/nfe_total" in eval_metrics
