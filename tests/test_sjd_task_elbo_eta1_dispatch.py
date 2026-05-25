"""Dispatch tests: SJDTaskBase.loss_fn routes to elbo_eta1_loss when
``objective='elbo_eta1'`` and to ce_allocation_loss otherwise."""
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from sticky.models.sjd.schedule import ForwardSchedule
from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.tasks.sudoku_inpaint_sjd import SudokuInpaintSJDTask

B, L = 2, 81


def _make_forward(eta: float = 1.0, blur_kernel=None):
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=eta, blur_kernel=blur_kernel)
    return ForwardSchedule(beta=beta, hazard=hazard, jump=jump, T=1.0)


def _make_task(*, objective="ce", learn_log_w=False, forward=None, **kwargs):
    if forward is None:
        forward = _make_forward()
    return SudokuInpaintSJDTask(
        data_dir=None,
        train_file="train.npy",
        test_file="test.npy",
        batch_size=2,
        eval_batch_size=2,
        data_shape=(81,),
        vocab_size=9,
        num_classes=9,
        forward=forward,
        log_state_dependency=False,
        objective=objective,
        learn_log_w=learn_log_w,
        **kwargs,
    )


class _FakeModel:
    """Stand-in for a real Flax SJD model. Returns zero logits and exposes
    a fake `anchor_log_w` method for the elbo path."""

    def embed(self, token_ids):
        return jnp.zeros(token_ids.shape + (8,))

    def anchor_log_w(self):
        return jnp.zeros((9,))  # vocab_size=9

    def apply(self, variables, *args, method=None, **kwargs):
        if method is not None:
            if method is self.anchor_log_w or method == self.anchor_log_w:
                return jnp.zeros((9,))
            token_ids = args[0]
            return jnp.zeros(token_ids.shape + (8,))
        xt = args[0]
        return jnp.zeros(xt.shape + (9,)), {}


def _make_batch():
    rng = jax.random.PRNGKey(0)
    solution_board = jax.random.randint(rng, shape=(B, L), minval=1, maxval=10)
    clue_mask = jax.random.bernoulli(rng, p=0.3, shape=(B, L))
    return {"solution_board": solution_board, "clue_mask": clue_mask}


def test_objective_ce_routes_to_ce_allocation_loss():
    """Default objective='ce' must dispatch to ce_allocation_loss."""
    task = _make_task(objective="ce")
    batch = _make_batch()

    fake_loss = jnp.array(1.23)
    fake_metrics = {"ce": jnp.array(0.5)}

    with patch(
        "sticky.tasks.sjd_base.ce_allocation_loss",
        return_value=(fake_loss, fake_metrics),
    ) as ce_mock, patch(
        "sticky.tasks.sjd_base.elbo_eta1_loss",
        return_value=(jnp.array(9.99), {"loss/ce": jnp.array(9.99)}),
    ) as elbo_mock:
        loss, _ = task.loss_fn(
            rng=jax.random.PRNGKey(0), model=_FakeModel(), params={}, batch=batch, train=False
        )

    assert ce_mock.called, "ce_allocation_loss not called for objective='ce'"
    assert not elbo_mock.called, "elbo_eta1_loss should not be called for objective='ce'"
    assert abs(float(loss) - 1.23) < 1e-5


def test_objective_elbo_eta1_routes_to_elbo_loss():
    """objective='elbo_eta1' must dispatch to elbo_eta1_loss."""
    task = _make_task(objective="elbo_eta1", learn_log_w=True)
    batch = _make_batch()

    fake_loss = jnp.array(2.34)
    fake_metrics = {"loss/ce": jnp.array(2.0), "loss/rb": jnp.array(0.3)}

    with patch(
        "sticky.tasks.sjd_base.ce_allocation_loss",
        return_value=(jnp.array(9.99), {"ce": jnp.array(9.99)}),
    ) as ce_mock, patch(
        "sticky.tasks.sjd_base.elbo_eta1_loss",
        return_value=(fake_loss, fake_metrics),
    ) as elbo_mock:
        loss, _ = task.loss_fn(
            rng=jax.random.PRNGKey(0), model=_FakeModel(), params={}, batch=batch, train=False
        )

    assert elbo_mock.called, "elbo_eta1_loss not called for objective='elbo_eta1'"
    assert not ce_mock.called, "ce_allocation_loss should not be called for objective='elbo_eta1'"
    assert abs(float(loss) - 2.34) < 1e-5


def test_objective_elbo_eta1_requires_learn_log_w():
    """Constructing a task with objective='elbo_eta1' and learn_log_w=False
    must raise — the whole point of this mode is to learn w(a)."""
    with pytest.raises(ValueError, match="learn_log_w"):
        _make_task(objective="elbo_eta1", learn_log_w=False)


def test_objective_elbo_eta1_rejects_eta_neq_1():
    """Scope guardrail: elbo_eta1 path is written for eta=1 only; eta<1 would
    silently give wrong gradients because lam_hat is no longer y-free."""
    fw_eta_07 = _make_forward(eta=0.7)
    with pytest.raises(ValueError, match="eta"):
        _make_task(objective="elbo_eta1", learn_log_w=True, forward=fw_eta_07)


def test_objective_unknown_raises():
    """Unknown objective must raise to catch typos early."""
    with pytest.raises(ValueError, match="objective"):
        _make_task(objective="bogus")
