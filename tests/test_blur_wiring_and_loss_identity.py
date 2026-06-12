"""Wiring + training-loss-identity tests for the W-sweep experiment.

Resurrects the factory-wiring coverage deleted in the publish cleanup
(e29c34c removed tests/tasks/test_sudoku_blur_wiring.py) and adds the
training-loss OFF-regression for the convex kernel:

  (a) Wiring matrix — composing experiment=sudoku/sjd_sudoku_wsweep through
      the real Hydra path and building the forward schedule via
      `_build_forward_schedule(cfg.experiment)` attaches the (81, 81) convex
      kernel: exact identity at the rho=0.0 default, the documented
      W = (1-rho)*I + rho*B structure at rho=0.35, and `blur_kernel is None`
      for the legacy sjd_sudoku experiment.
  (b) Frozen-decision audit — eta=1.0, frozen anchors, headline-selector
      checkpoint metric, the four-entry eval grid (incl. the wscore column),
      and the two CLI sweep knobs (rho, seed) compose and land in the cfg.
  (c) Training-loss bit-identity — on a fixed batch and fixed rng, the
      corruption + CE-loss path (the part blur touches, via
      `jump.apply_blur` inside `sample_pair`) is bit-identical between
      blur_kernel=None and the rho=0.0 convex kernel, and DIFFERS at
      rho=0.35 (so the identity assertion has power).

CPU-only; tiny synthetic anchors/batches; no dataset iteration, no model.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sticky.models.sjd.blur import sudoku_constraint_kernel_convex
from sticky.models.sjd.corruption import sample_pair
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta
from sticky.tasks.factory import _build_forward_schedule

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config")
)


def _compose(overrides: list[str]):
    """Compose the root config through the real Hydra path (the pattern used
    by test_blur_score_eval.py). Experiment content lands under cfg.experiment;
    the eval grid is rerouted to the global cfg.eval package."""
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            return compose(config_name="config", overrides=list(overrides))
    finally:
        GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# (a) Wiring matrix
# ---------------------------------------------------------------------------

def test_wsweep_default_rho0_attaches_exact_identity_kernel():
    """rho=0.0 (the config default) must yield W == I bitwise: the convex
    blend (1-0)*I + 0*B is exact in IEEE float."""
    cfg = _compose(["experiment=sudoku/sjd_sudoku_wsweep"])
    schedule = _build_forward_schedule(cfg.experiment)
    kernel = schedule.jump.blur_kernel
    assert kernel is not None, "wsweep must attach a kernel even at rho=0"
    assert tuple(kernel.shape) == (81, 81)
    np.testing.assert_array_equal(np.asarray(kernel), np.eye(81))
    # The schedule-level surfacing property agrees with the jump attribute.
    assert schedule.blur_kernel is kernel


def test_wsweep_rho035_kernel_structure():
    """experiment.forward.blur.rho=0.35 must produce a non-identity kernel
    with row sums 1 and the convex structure W = 0.65*I + 0.35*B (B_ii = 0,
    so the diagonal is exactly 0.65)."""
    cfg = _compose([
        "experiment=sudoku/sjd_sudoku_wsweep",
        "experiment.forward.blur.rho=0.35",
    ])
    schedule = _build_forward_schedule(cfg.experiment)
    kernel = np.asarray(schedule.jump.blur_kernel)
    assert kernel.shape == (81, 81)
    assert not np.array_equal(kernel, np.eye(81))
    np.testing.assert_allclose(kernel.sum(axis=1), np.ones(81), atol=1e-6)
    # Diagonal: (1 - rho) * 1 + rho * B_ii = 0.65 + 0 exactly.
    np.testing.assert_array_equal(
        np.diagonal(kernel), np.full(81, np.float32(1.0 - 0.35))
    )
    # Full structure check: rho=1.0 of the same builder is pure B, so the
    # composed kernel must be the convex blend of I and that B.
    B = np.asarray(sudoku_constraint_kernel_convex(sigma=1.5, rho=1.0))
    np.testing.assert_array_equal(np.diagonal(B), np.zeros(81))
    expected = (
        np.float32(0.65) * np.eye(81, dtype=np.float32)
        + np.float32(0.35) * B.astype(np.float32)
    )
    np.testing.assert_allclose(kernel, expected, rtol=0.0, atol=1e-7)


def test_legacy_sjd_sudoku_has_no_kernel():
    cfg = _compose(["experiment=sudoku/sjd_sudoku"])
    schedule = _build_forward_schedule(cfg.experiment)
    assert schedule.jump.blur_kernel is None
    assert schedule.blur_kernel is None


# ---------------------------------------------------------------------------
# (b) Frozen-decision audit
# ---------------------------------------------------------------------------

def test_wsweep_frozen_decisions():
    cfg = _compose(["experiment=sudoku/sjd_sudoku_wsweep"])
    exp = cfg.experiment
    assert float(exp.forward.jump.eta) == 1.0
    assert bool(exp.model.anchor.learnable) is False
    assert (
        str(exp.training.best_checkpoint_metric)
        == "eval/linear_topk_probability/solve_rate"
    )
    # objective=ce is selected via hazard_weighting=none (elbo_eta1 would
    # reject the attached kernel; see sjd_base.py).
    assert str(exp.forward.hazard_weighting.mode) == "none"
    # Blur group is the real convex kernel config at the sweep default.
    assert bool(exp.forward.blur.enabled) is True
    assert str(exp.forward.blur.normalization) == "convex"
    assert float(exp.forward.blur.rho) == 0.0
    # The paired eval grid lands at the GLOBAL eval package with all four
    # entries, incl. the wscore column with blur_score=true.
    runs = cfg.eval.sudoku_eval_sjd_runs
    assert {
        "linear_survival",
        "linear_topk_probability",
        "predictor_only",
        "linear_topk_probability_wscore",
    } <= set(runs.keys())
    assert bool(runs.linear_topk_probability_wscore.blur_score) is True
    assert abs(float(runs.linear_topk_probability_wscore.eta) - 1.0) < 1e-12


def test_wsweep_cli_sweep_knobs_compose_and_land():
    cfg = _compose([
        "experiment=sudoku/sjd_sudoku_wsweep",
        "experiment.forward.blur.rho=0.2",
        "experiment.training.seed=1",
        "wandb.run_name=sjd_sudoku_wsweep_rho0.2_seed1",
    ])
    assert abs(float(cfg.experiment.forward.blur.rho) - 0.2) < 1e-12
    assert int(cfg.experiment.training.seed) == 1
    assert str(cfg.wandb.run_name) == "sjd_sudoku_wsweep_rho0.2_seed1"


# ---------------------------------------------------------------------------
# (c) Training-loss bit-identity (the OFF regression)
# ---------------------------------------------------------------------------
#
# Driven directly at the corruption + loss level with tiny synthetic inputs:
# anchor table (9, 4), batch (2, 81) indices, and a stub linear classifier.
# This is exactly the part of the training path that blur touches —
# sample_pair calls jump.apply_blur(x0_anchor) for the unstuck-mixture mean,
# and ce_allocation_loss calls sample_pair. No data download, no 6M model.

_D = 4  # anchor dim


def _loss_harness():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0, eps=1e-5)
    rng = np.random.default_rng(0)
    a_table = jnp.asarray(rng.standard_normal((9, _D)), dtype=jnp.float32)
    x0_idx = jnp.asarray(rng.integers(0, 9, size=(2, 81)), dtype=jnp.int32)
    x0_anchor = jnp.take(a_table, x0_idx, axis=0)  # (2, 81, 4)
    proj = jnp.asarray(rng.standard_normal((_D, 9)), dtype=jnp.float32)

    def apply_fn(params, x_in, t_img):
        del params, t_img
        return jnp.einsum("bnd,dk->bnk", x_in, proj), {}

    return beta, hazard, a_table, x0_idx, x0_anchor, apply_fn


def _jump_with(beta, kernel):
    return VPMatchedGaussianJump(
        beta=beta, eta=1.0, std_floor=1e-3, clip=None, blur_kernel=kernel
    )


def test_sample_pair_bit_identical_none_vs_identity_kernel():
    beta, hazard, _, _, x0_anchor, _ = _loss_harness()
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    identity = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.0)
    np.testing.assert_array_equal(np.asarray(identity), np.eye(81))

    xt_none, mask_none = sample_pair(
        key, x0_anchor, t, beta, hazard, _jump_with(beta, None)
    )
    xt_id, mask_id = sample_pair(
        key, x0_anchor, t, beta, hazard, _jump_with(beta, identity)
    )
    assert np.array_equal(np.asarray(xt_none), np.asarray(xt_id))
    assert np.array_equal(np.asarray(mask_none), np.asarray(mask_id))


def test_sample_pair_rho035_kernel_differs():
    """Power check: a real kernel at the same rng must change x_t (only the
    unstuck-mixture mean is blurred, so the never-unstuck mask is unchanged)."""
    beta, hazard, _, _, x0_anchor, _ = _loss_harness()
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    kernel = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.35)

    xt_none, mask_none = sample_pair(
        key, x0_anchor, t, beta, hazard, _jump_with(beta, None)
    )
    xt_blur, mask_blur = sample_pair(
        key, x0_anchor, t, beta, hazard, _jump_with(beta, kernel)
    )
    assert not np.array_equal(np.asarray(xt_none), np.asarray(xt_blur))
    # The unstick decision is drawn before the mean is built; blur cannot
    # change which sites unstick.
    assert np.array_equal(np.asarray(mask_none), np.asarray(mask_blur))


def test_ce_loss_bit_identical_none_vs_identity_kernel():
    """The full ce training-loss value on a fixed batch and fixed rng must be
    bit-identical between blur_kernel=None and the rho=0.0 identity kernel."""
    beta, hazard, a_table, x0_idx, x0_anchor, apply_fn = _loss_harness()
    identity = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.0)
    key = jax.random.PRNGKey(7)

    def run(kernel):
        loss, metrics = ce_allocation_loss(
            key=key,
            params={},
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            T=1.0,
            jump=_jump_with(beta, kernel),
            anchor_table=a_table,
        )
        return np.asarray(loss), metrics

    loss_none, metrics_none = run(None)
    loss_id, metrics_id = run(identity)
    assert np.array_equal(loss_none, loss_id), (
        f"ce loss must be bit-identical: {loss_none!r} vs {loss_id!r}"
    )
    # Every whitelisted training metric rides the same corruption draw, so
    # the whole dict must match bitwise as well.
    assert set(metrics_none.keys()) == set(metrics_id.keys())
    for name in metrics_none:
        a = np.asarray(metrics_none[name])
        b = np.asarray(metrics_id[name])
        assert np.array_equal(a, b, equal_nan=True), name


def test_ce_loss_rho035_kernel_changes_loss():
    beta, hazard, a_table, x0_idx, x0_anchor, apply_fn = _loss_harness()
    kernel = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.35)
    key = jax.random.PRNGKey(7)

    def run(k):
        loss, _ = ce_allocation_loss(
            key=key,
            params={},
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            T=1.0,
            jump=_jump_with(beta, k),
            anchor_table=a_table,
        )
        return float(loss)

    assert run(None) != run(kernel)
