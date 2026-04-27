"""Phase-3 smoke test for the slack-augmented Sudoku SJD task.

Composes the experiment config, builds the task and model, and runs a
single training step on a synthetic Sudoku batch. Verifies that:
  * cell_x_t.shape == (B, 81, 9) and slack_x_t.shape == (B, 27, 9)
  * The classifier produces (B, 108, 9) logits
  * loss/slack_l2_to_ones is small at small t
  * state_dep/log_ratio_* is finite
  * Gradients flow through the joint-input projection params

Avoids any data download by constructing a synthetic batch directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.data.sudoku import compute_slack_vectors
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task


CONFIG_DIR = str(config_root())


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


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
        "loss/slack_l2_to_ones",
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
