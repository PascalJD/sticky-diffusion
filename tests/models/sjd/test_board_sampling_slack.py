"""Tests for the slack-aware Sudoku SJD sampler.

`conditional_generate_with_slack` is exercised in three layers:
  1. Shape + clue-pinning: a fully-known clue mask must be returned verbatim.
  2. Slack VP terminal sanity: with project_after_step=False and a trivial
     zero-logits model, after one step the slack state must match the
     analytical reverse-VP step on the data marginal mean alpha(t)*1.
  3. Projection sanity: with project_after_step=True, after a cell becomes
     fully committed (one-hot of digit v), the row/col/box slacks
     containing it must reflect that one-hot in their v-th coordinate.
  4. End-to-end: compose experiment=sudoku/sjd_sudoku_slack and run a
     small `plugin_hazard` sample on a synthetic batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.data.sudoku import compute_slack_vectors
from sticky.data.sudoku.slack import GROUPS, compute_slack_from_cells
from sticky.models.factory import build_model
from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.board_sampling import conditional_generate_with_slack
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta
from sticky.tasks.factory import build_task


CONFIG_DIR = str(config_root())


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def _valid_solution(B: int = 2) -> np.ndarray:
    boards = np.zeros((B, 81), dtype=np.int32)
    for b in range(B):
        shift = b % 9
        for r in range(9):
            for c in range(9):
                boards[b, r * 9 + c] = ((r * 3 + r // 3 + c + shift) % 9) + 1
    return boards


# ----- a tiny fake model that emits zero joint logits ---------------------


class _ZeroLogitsSlackModel:
    """A model.apply stand-in that returns (B, 108, 9) zero logits and an
    identity anchor table — bypasses Flax init entirely so we can exercise
    the sampler's sde / VP / projection logic without a real network."""

    def __init__(self, anchor_dim: int = 9):
        self._table = jnp.eye(anchor_dim, dtype=jnp.float32)

    def apply(self, variables, *args, method=None, **kwargs):
        del variables
        if method is not None and getattr(method, "__name__", "") == "anchor_table":
            return self._table
        # Forward call: args[0] is y_cells (B, 81, 9). slack_y_t in kwargs.
        y_cells = jnp.asarray(args[0], dtype=jnp.float32)
        B = int(y_cells.shape[0])
        return jnp.zeros((B, 108, 9), dtype=jnp.float32), {}


def _schedule():
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=2.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.97, std_floor=1e-3)
    return beta, hazard, jump


# ---------------- 1. shape / clue-pinning ----------------


def test_fully_known_clue_mask_is_returned_verbatim():
    beta, hazard, jump = _schedule()
    boards = _valid_solution(B=2)
    known_tokens = jnp.asarray(boards, dtype=jnp.int32)
    known_mask = jnp.ones_like(known_tokens, dtype=jnp.bool_)
    model = _ZeroLogitsSlackModel()
    pred = conditional_generate_with_slack(
        rng=jax.random.PRNGKey(0),
        params={},
        model=model,
        anchors=AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32)),
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=4,
        policy="linear_survival",
        sampling_grid="uniform",
    )
    assert pred.shape == (2, 81)
    np.testing.assert_array_equal(np.asarray(pred), boards)


def test_returns_diagnostics_when_requested():
    beta, hazard, jump = _schedule()
    boards = _valid_solution(B=2)
    known_tokens = jnp.asarray(boards, dtype=jnp.int32)
    known_mask = jnp.ones_like(known_tokens, dtype=jnp.bool_)
    model = _ZeroLogitsSlackModel()
    pred, diag = conditional_generate_with_slack(
        rng=jax.random.PRNGKey(0),
        params={},
        model=model,
        anchors=AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32)),
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=4,
        policy="linear_survival",
        sampling_grid="uniform",
        return_diagnostics=True,
    )
    assert pred.shape == (2, 81)
    assert "slack_residual_mean_total" in diag
    assert "slack_residual_count_total" in diag
    assert "final_masked_unknown_total" in diag


# ---------------- 2. slack VP terminal sanity ----------------


def test_slack_step_matches_analytical_reverse_vp_step():
    """With project_after_step=False and zero cell logits, after one step
    the slack state equals the analytical reverse VP step on alpha(t)*1.

    With zero logits + simplex_vertex anchors, mu = 1/9 for every coordinate
    and every cell, so the cell drift is symmetric and the cell SDE produces
    a bounded random update — but slack dynamics are independent of cells in
    this implementation. The slack step depends only on (alpha, sigma, beta,
    dt, y_slacks) and the slack RNG subkey, all of which are deterministic
    given a fixed seed.
    """
    beta, hazard, jump = _schedule()
    B = 1
    n_steps = 1
    init_std = 0.7
    rng = jax.random.PRNGKey(123)

    # Reproduce the sampler's RNG split + slack init analytically.
    rng_after_init, init_cell_key, init_slack_key = jax.random.split(rng, 3)
    y_slacks_init = init_std * jax.random.normal(
        init_slack_key, shape=(B, 27, 9), dtype=jnp.float32
    )

    # Run sampler: empty clue mask so no clamping interferes.
    known_tokens = jnp.zeros((B, 81), dtype=jnp.int32)
    known_mask = jnp.zeros((B, 81), dtype=jnp.bool_)

    model = _ZeroLogitsSlackModel()
    _, diag = conditional_generate_with_slack(
        rng=rng,
        params={},
        model=model,
        anchors=AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32)),
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=n_steps,
        policy="linear_survival",
        sampling_grid="uniform",
        init_std=init_std,
        project_slacks_after_step=False,
        return_diagnostics=True,
    )

    # Analytical: at the only step, t=T=1, s=0, dt=1.
    # alpha(t), sigma(t) at t=1.
    t_img = jnp.full((B,), 1.0, dtype=jnp.float32)
    alpha_t, sigma_t = alpha_sigma(beta, t_img)
    sigma2_t = jnp.maximum(sigma_t ** 2, 1e-12)
    beta_t = beta(t_img)
    dt = 1.0

    slack_clean_mean = jnp.ones((9,), dtype=jnp.float32)
    slack_marginal_mean = alpha_t[:, None, None] * slack_clean_mean[None, None, :]
    slack_score = (slack_marginal_mean - y_slacks_init) / sigma2_t[:, None, None]
    slack_drift = (
        0.5 * beta_t[:, None, None] * y_slacks_init
        + beta_t[:, None, None] * slack_score
    )

    # Replay the sampler's RNG splits to obtain the slack noise key.
    rng_loop = rng_after_init
    rng_loop, _cell_noise, slack_noise_key, _count, _score = jax.random.split(rng_loop, 5)
    slack_noise = jax.random.normal(
        slack_noise_key, shape=(B, 27, 9), dtype=jnp.float32
    )
    expected_y_slacks = (
        y_slacks_init
        + slack_drift * dt
        + jnp.sqrt(jnp.maximum(beta_t[:, None, None] * dt, 0.0)) * slack_noise
    )

    # The sampler doesn't expose the post-step y_slacks directly. Use the
    # diagnostic: residual = ||y_slacks - alpha_t * 1||. We compute the
    # analytical residual at s=0 (post-step), which corresponds to the
    # alpha at the "current" t=1 (since the diag is recorded inside the
    # loop using alpha_t_b from the score-time t).
    expected_residual = jnp.linalg.norm(
        expected_y_slacks - alpha_t[:, None, None] * slack_clean_mean[None, None, :],
        axis=-1,
    )
    measured = float(diag["slack_residual_mean_total"]) / float(
        diag["slack_residual_count_total"]
    )
    np.testing.assert_allclose(
        measured, float(jnp.mean(expected_residual)), rtol=1e-4, atol=1e-4
    )


# ---------------- 3. projection sanity ----------------


def test_compute_slack_from_cells_picks_up_committed_one_hot():
    """When a cell becomes one-hot of digit v, the row/col/box slack groups
    containing that cell must show v in their v-th coordinate. This is the
    invariant the sampler's post-step projection relies on; we test the
    underlying readout directly with a hand-built mid-trajectory state."""
    cell_state = np.zeros((1, 81, 9), dtype=np.float32)
    # Make all 9 cells in row 0 one-hot of distinct digits 0..8 (a valid row).
    for c in range(9):
        cell_state[0, c, c] = 1.0
    # The other 72 cells stay zero (just to keep the test focused).
    slack = compute_slack_from_cells(cell_state)  # (1, 27, 9)
    # Row 0 (group 0) should be ones (one of each digit).
    np.testing.assert_array_equal(
        np.asarray(slack[0, 0]), np.ones(9, dtype=np.float32)
    )
    # Column 0 (group 9) only has cell (0,0) committed; expect e_0.
    expected_col0 = np.zeros(9, dtype=np.float32)
    expected_col0[0] = 1.0
    np.testing.assert_array_equal(np.asarray(slack[0, 9]), expected_col0)
    # Box 0 (group 18) contains (0,0)=0, (0,1)=1, (0,2)=2 (committed)
    # plus 6 zero cells; expect [1,1,1,0,0,0,0,0,0].
    expected_box0 = np.zeros(9, dtype=np.float32)
    expected_box0[0:3] = 1.0
    np.testing.assert_array_equal(np.asarray(slack[0, 18]), expected_box0)


def test_groups_table_indexes_cells_correctly():
    """Sanity: GROUPS[g, k] gives the row-major index of the k-th cell in
    group g. Spot-check the box for cell (4, 4) — that's box 4 which should
    contain rows 3..5, cols 3..5."""
    box4 = np.asarray(GROUPS[18 + 4])
    expected = np.array(
        [3 * 9 + 3, 3 * 9 + 4, 3 * 9 + 5,
         4 * 9 + 3, 4 * 9 + 4, 4 * 9 + 5,
         5 * 9 + 3, 5 * 9 + 4, 5 * 9 + 5],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(box4, expected)


# ---------------- 4. plugin_hazard end-to-end ----------------


def test_plugin_hazard_end_to_end_on_synthetic_batch():
    """Smoke-test the full pipeline: build the slack model from the
    experiment config, init it, and run an 8-step plugin_hazard sample on
    a synthetic batch with all clues unknown. Output must be a Sudoku-shaped
    int array with values in 1..9."""
    cfg = _compose(
        [
            "experiment=sudoku/sjd_sudoku_slack",
            "experiment.dataset.auto_download=false",
            "experiment.dataset.batch_size=2",
        ]
    )
    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    rng_init, rng_sample = jax.random.split(jax.random.PRNGKey(7))
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

    boards = _valid_solution(B=2)
    known_tokens = jnp.asarray(boards, dtype=jnp.int32)
    known_mask = jnp.zeros_like(known_tokens, dtype=jnp.bool_)  # nothing pinned

    a_table = model.apply({"params": params}, method=model.anchor_table)
    anchors = AnchorTable(table_float=a_table)

    pred = conditional_generate_with_slack(
        rng=rng_sample,
        params=params,
        model=model,
        anchors=anchors,
        beta=task.beta,
        hazard=task.hazard,
        jump=task.jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=8,
        policy="plugin_hazard",
        sampling_grid="uniform",
        eta=0.97,
        init_std=1.0,
        project_slacks_after_step=True,
    )
    assert pred.shape == (2, 81)
    assert int(jnp.min(pred)) >= 1
    assert int(jnp.max(pred)) <= 9


def test_project_after_step_flag_changes_slack_residual_diagnostics():
    """The two ablation arms should produce visibly different slack residual
    averages: with projection on, slack is reset to a (mostly-clean) value
    every step; with projection off, slack diffuses freely under VP."""
    beta, hazard, jump = _schedule()
    boards = _valid_solution(B=4)
    known_tokens = jnp.asarray(boards, dtype=jnp.int32)
    known_mask = jnp.zeros_like(known_tokens, dtype=jnp.bool_)
    model = _ZeroLogitsSlackModel()
    common = dict(
        params={},
        model=model,
        anchors=AnchorTable(table_float=jnp.eye(9, dtype=jnp.float32)),
        beta=beta,
        hazard=hazard,
        jump=jump,
        known_tokens=known_tokens,
        known_token_mask=known_mask,
        n_steps=10,
        policy="linear_survival",
        sampling_grid="uniform",
        return_diagnostics=True,
    )
    _, diag_off = conditional_generate_with_slack(
        rng=jax.random.PRNGKey(0),
        project_slacks_after_step=False,
        **common,
    )
    _, diag_on = conditional_generate_with_slack(
        rng=jax.random.PRNGKey(0),
        project_slacks_after_step=True,
        **common,
    )
    res_off = float(diag_off["slack_residual_mean_total"]) / max(
        float(diag_off["slack_residual_count_total"]), 1.0
    )
    res_on = float(diag_on["slack_residual_mean_total"]) / max(
        float(diag_on["slack_residual_count_total"]), 1.0
    )
    assert res_off != pytest.approx(res_on, abs=1e-6), (
        f"Expected projection on/off to differ; got res_off={res_off}, res_on={res_on}"
    )
