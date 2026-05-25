# Strategy: monkey-patch ce_allocation_loss to avoid a real Flax model stub.
from unittest.mock import patch

import jax
import jax.numpy as jnp

from sticky.models.sjd.schedule import ForwardSchedule
from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.tasks.sudoku_inpaint_sjd import SudokuInpaintSJDTask

B, L = 2, 81  # Sudoku: batch=2, seq_len=81


def _make_task():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    forward = ForwardSchedule(beta=beta, hazard=hazard, jump=jump, T=1.0)
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
        log_state_dependency=False,  # skip anchor_table lookup
    )


def test_loss_fn_returns_tuple():
    """loss_fn must return (scalar_loss, metrics_dict) for value_and_grad has_aux=True."""
    task = _make_task()

    rng = jax.random.PRNGKey(0)
    # Digits 1..9 for solution, clue_mask as bool
    solution_board = jax.random.randint(rng, shape=(B, L), minval=1, maxval=10)
    clue_mask = jax.random.bernoulli(rng, p=0.3, shape=(B, L))
    batch = {"solution_board": solution_board, "clue_mask": clue_mask}

    fake_loss = jnp.array(1.23)
    fake_metrics = {"ce": jnp.array(0.5)}

    class _FakeModel:
        def embed(self, token_ids):
            return jnp.zeros(token_ids.shape + (8,))

        def apply(self, variables, *args, method=None, **kwargs):
            if method is not None:
                # embed call: return anchor vectors (*shape, d)
                token_ids = args[0]
                return jnp.zeros(token_ids.shape + (8,))
            # __call__: return (logits, {})
            xt = args[0]
            return jnp.zeros(xt.shape + (9,)), {}

    model = _FakeModel()
    params = {}

    with patch(
        "sticky.tasks.sjd_base.ce_allocation_loss",
        return_value=(fake_loss, fake_metrics),
    ):
        result = task.loss_fn(rng=rng, model=model, params=params, batch=batch, train=False)

    assert isinstance(result, tuple) and len(result) == 2, (
        f"loss_fn must return a 2-tuple, got {type(result)}"
    )
    loss, metrics = result
    assert isinstance(loss, jnp.ndarray), f"loss must be jnp.ndarray, got {type(loss)}"
    assert loss.ndim == 0, f"loss must be scalar, got shape {loss.shape}"
    assert isinstance(metrics, dict), f"metrics must be dict, got {type(metrics)}"
    assert "loss" in metrics, f"'loss' key missing from metrics: {metrics.keys()}"
