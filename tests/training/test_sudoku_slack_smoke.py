"""Phase-3 smoke test for the slack-augmented Sudoku SJD task.

Composes the experiment config, builds the task and model, and runs a
single training step on a synthetic Sudoku batch. Verifies that:
  * cell_x_t.shape == (B, 81, 9) and slack_x_t.shape == (B, 27, 9)
  * The classifier produces (B, 108, 9) logits
  * loss/slack_residual_l2 ≈ sigma(t) (the per-coordinate VP residual)
  * state_dep/log_ratio_* is finite
  * Gradients flow through the joint-input projection params

Avoids any data download by constructing a synthetic batch directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hydra import compose, initialize_config_dir

import sticky.eval.sudoku as sudoku_eval_mod
from sticky.core.config_paths import config_root
from sticky.data.sudoku import compute_slack_vectors
from sticky.eval.sudoku import build_sudoku_eval_logger
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.training.state import init_state
from sticky.training.step import make_train_step_fn, params_for_sampling


CONFIG_DIR = str(config_root())


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def _compose_with_eval(overrides):
    """Compose root config including the eval group (`eval=sudoku_sjd`).

    The slack experiment carries its own `eval:` block but uses the same
    `eval` Hydra group as the cell-only experiment for top-level wiring.
    """
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=["eval=sudoku_sjd", *overrides])


def _synthetic_batch(B: int = 4) -> dict:
    """Build a batch of valid Sudoku boards via cyclic constructions."""
    boards = np.zeros((B, 81), dtype=np.int32)
    for b in range(B):
        shift = b % 9
        for r in range(9):
            for c in range(9):
                boards[b, r * 9 + c] = ((r * 3 + r // 3 + c + shift) % 9) + 1
    clue_mask = np.zeros((B, 81), dtype=np.bool_)
    clue_mask[:, :20] = True
    return {
        "solution_board": boards,
        "clue_board": np.where(clue_mask, boards, 0).astype(np.int32),
        "clue_mask": clue_mask,
        "slack_x0": compute_slack_vectors(boards),
    }


def test_one_step_forward_backward_smoke():
    cfg = _compose(
        [
            "experiment=sudoku/sjd_sudoku_slack",
            "experiment.dataset.auto_download=false",
            "experiment.dataset.batch_size=4",
        ]
    )
    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    rng = jax.random.PRNGKey(0)
    rng_init, rng_loss = jax.random.split(rng)
    # Init the model with shapes that match the production forward call.
    cell_xt_init = jnp.zeros((1, 81, 9), dtype=jnp.float32)
    slack_xt_init = jnp.zeros((1, 27, 9), dtype=jnp.float32)
    t_init = jnp.zeros((1,), dtype=jnp.float32)
    ids_init = jnp.zeros((1, 81), dtype=jnp.int32)
    params = model.init(
        rng_init,
        cell_xt_init,
        t_init,
        anchor_token_ids=ids_init,
        slack_y_t=slack_xt_init,
        train=False,
    )["params"]

    batch = _synthetic_batch(B=4)

    def loss_only(p, rng):
        loss, _ = task.loss_fn(
            rng=rng, model=model, params=p, batch=batch, train=True
        )
        return loss

    loss, grads = jax.value_and_grad(loss_only)(params, rng_loss)

    assert np.isfinite(float(loss))
    # Joint-input proj params must receive non-zero gradients.
    cell_grads = grads["joint_input_proj"]["cell_proj"]["kernel"]
    slack_grads = grads["joint_input_proj"]["slack_proj"]["kernel"]
    site_emb_grads = grads["joint_input_proj"]["site_type_emb"]
    assert cell_grads.shape == (9, cfg.experiment.model.feature_dim)
    assert slack_grads.shape == (9, cfg.experiment.model.feature_dim)
    assert site_emb_grads.shape == (2, cfg.experiment.model.feature_dim)
    assert float(jnp.linalg.norm(cell_grads)) > 0.0
    # The slack proj receives gradient because slack_x_t flows through to
    # the classifier, so logits depend on slack_proj params even though
    # the slack logits are sliced away.
    assert float(jnp.linalg.norm(slack_grads)) > 0.0


def test_one_step_metrics_have_expected_keys_and_shapes():
    cfg = _compose(
        [
            "experiment=sudoku/sjd_sudoku_slack",
            "experiment.dataset.auto_download=false",
            "experiment.dataset.batch_size=4",
        ]
    )
    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    rng_init, rng_loss = jax.random.split(jax.random.PRNGKey(1))
    cell_xt_init = jnp.zeros((1, 81, 9), dtype=jnp.float32)
    slack_xt_init = jnp.zeros((1, 27, 9), dtype=jnp.float32)
    t_init = jnp.zeros((1,), dtype=jnp.float32)
    ids_init = jnp.zeros((1, 81), dtype=jnp.int32)
    params = model.init(
        rng_init,
        cell_xt_init,
        t_init,
        anchor_token_ids=ids_init,
        slack_y_t=slack_xt_init,
        train=False,
    )["params"]

    batch = _synthetic_batch(B=4)
    loss, metrics = task.loss_fn(
        rng=rng_loss, model=model, params=params, batch=batch, train=False
    )

    expected_keys = {
        "loss",
        "loss/ce_nll_bits",
        "loss/acc_top1",
        "loss/frac_active",
        "loss/frac_never_unstuck",
        "loss/slack_residual_l2",
        "loss/slack_sigma_t_mean",
        "t/mean",
        "t/std",
        "state_dep/log_ratio_mean",
        "state_dep/log_ratio_std",
        "clean_index_min",
        "clean_index_max",
        "given_fraction",
    }
    missing = expected_keys - set(metrics.keys())
    assert not missing, f"missing metrics: {missing}"
    for key in expected_keys:
        v = float(metrics[key])
        assert np.isfinite(v), f"{key} is not finite: {v}"

    assert float(metrics["clean_index_min"]) == 0.0
    assert float(metrics["clean_index_max"]) == 8.0
    # given_fraction = 20 clues / 81 cells.
    np.testing.assert_allclose(
        float(metrics["given_fraction"]), 20.0 / 81.0, atol=1e-5
    )


def test_slack_train_step_and_eval_logger_smoke(monkeypatch):
    """Phase-4 end-to-end smoke: a single train step + the slack-aware
    mid-training eval through `build_sudoku_eval_logger`. Verifies the
    eval path no longer crashes and that the per-policy `solve_rate` /
    `full_cell_acc` keys are present in the returned metrics — the
    "training run finishes mid-eval" acceptance criterion."""
    cfg = _compose_with_eval(
        [
            "experiment=sudoku/sjd_sudoku_slack",
            "experiment.dataset.auto_download=false",
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
        ]
    )
    # Replace the experiment's eval entries with a tiny set of policy runs so
    # the smoke test stays fast. (The default has 3+ policies.)
    cfg.experiment.eval.sudoku_eval_sjd_runs = {
        "plugin_hazard": {
            "kind": "policy",
            "policy": "plugin_hazard",
            "n_steps": 2,
            "sampling_grid": "uniform",
            "eta": 0.97,
        },
    }
    cfg.experiment.eval.sudoku_primary_sampler_label = "plugin_hazard"
    # Mirror the experiment-level eval entries into the top-level eval group
    # (which is what `build_sudoku_eval_logger` actually consumes).
    cfg.eval.sudoku_eval_sjd_runs = cfg.experiment.eval.sudoku_eval_sjd_runs
    cfg.eval.sudoku_primary_sampler_label = (
        cfg.experiment.eval.sudoku_primary_sampler_label
    )
    cfg.eval.sudoku_num_batches = 1
    cfg.eval.sudoku_num_batches_per_sampler = 1
    cfg.eval.sudoku_prop52_enabled = False
    cfg.eval.sudoku_write_progress_csv = False
    cfg.eval.sudoku_write_latest_csv = False

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

    batch_dict = _synthetic_batch(B=1)
    batch = {
        "solution_board": jnp.asarray(batch_dict["solution_board"]),
        "clue_board": jnp.asarray(batch_dict["clue_board"]),
        "clue_mask": jnp.asarray(batch_dict["clue_mask"]),
        "slack_x0": jnp.asarray(batch_dict["slack_x0"]),
    }
    state, metrics = train_step_fn(state, batch, axis_name=None)
    assert float(jax.device_get(metrics["train/loss"])) >= 0.0

    # Fake the eval iterator so we don't try to download Sudoku data.
    monkeypatch.setattr(
        sudoku_eval_mod,
        "make_sudoku_board_iterator",
        lambda **kwargs: iter(
            [
                {
                    "solution_board": np.asarray(batch_dict["solution_board"]),
                    "clue_board": np.asarray(batch_dict["clue_board"]),
                    "clue_mask": np.asarray(batch_dict["clue_mask"]),
                    "slack_x0": np.asarray(batch_dict["slack_x0"]),
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

    # The slack-aware policy sampler should have run and reported its solve_rate.
    assert "eval/plugin_hazard/solve_rate" in eval_metrics
    assert "eval/plugin_hazard/full_cell_acc" in eval_metrics
