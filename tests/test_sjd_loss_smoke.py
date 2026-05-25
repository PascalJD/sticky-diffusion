# Smoke test: ce_allocation_loss returns a finite scalar + metrics dict.
import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss

B, L, V, d = 2, 4, 8, 8  # batch=2, seq_len=4, vocab=8, embed_dim=8


def _make_schedules():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)
    return beta, hazard, jump


def test_ce_allocation_loss_smoke():
    beta, hazard, jump = _make_schedules()

    rng = jax.random.PRNGKey(1)
    x0_anchor = jax.random.normal(rng, shape=(B, L, d))
    x0_idx = jax.random.randint(rng, shape=(B, L), minval=0, maxval=V)

    def apply_fn(params, x_in, t_img):
        logits = jax.random.normal(jax.random.PRNGKey(42), shape=(B, L, V))
        return logits, {}

    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(0),
        params={},
        apply_fn=apply_fn,
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        T=1.0,
        jump=jump,
    )

    assert isinstance(loss, jnp.ndarray), f"expected jnp.ndarray, got {type(loss)}"
    assert loss.ndim == 0, f"expected scalar, got shape {loss.shape}"
    assert bool(jnp.isfinite(loss)), f"loss is not finite: {loss}"
    assert isinstance(metrics, dict), f"expected dict, got {type(metrics)}"
    assert "loss" in metrics
