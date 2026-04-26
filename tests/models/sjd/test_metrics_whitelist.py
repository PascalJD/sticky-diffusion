"""Single source of truth for the metric set emitted by `ce_allocation_loss`.

After Prompt B, the per-step training metrics dict must contain *exactly*
these nine keys, in this order. The test enforces that no rogue metric leaks
into W&B / console logging.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


EXPECTED_TRAIN_METRICS = {
    "loss",
    "loss/ce_nll_bits",
    "loss/acc_top1",
    "loss/frac_active",
    "loss/frac_never_unstuck",
    "t/mean",
    "t/std",
    "state_dep/log_ratio_mean",
    "state_dep/log_ratio_std",
}


def _kwargs(B: int = 8, seq_len: int = 6, V: int = 4, d: int = 3):
    table = jax.random.normal(
        jax.random.PRNGKey(0), (V, d), dtype=jnp.float32
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(1), (B, seq_len), 0, V
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]

    def apply_fn(params, xt, t_img):
        del params, xt, t_img
        return jnp.zeros((B, seq_len, V), dtype=jnp.float32), {}

    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    return dict(
        apply_fn=apply_fn,
        params={},
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=make_hazard_linear_time(beta, kappa=1.5),
        jump=VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3),
        anchor_table=table,
        T=1.0,
    )


def test_metrics_whitelist():
    """ce_allocation_loss returns exactly the whitelisted training metrics."""
    _loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(0), **_kwargs()
    )
    actual = set(metrics.keys())
    extra = actual - EXPECTED_TRAIN_METRICS
    missing = EXPECTED_TRAIN_METRICS - actual
    assert actual == EXPECTED_TRAIN_METRICS, (
        f"unexpected metrics: {sorted(extra)}\n"
        f"missing: {sorted(missing)}"
    )


def test_metrics_whitelist_alpha_deriv():
    """alpha_deriv loss-weighting must not add any metrics outside the
    whitelist (the legacy `CE/loss_weight_*` diagnostics were removed)."""
    _loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(0), loss_weighting="alpha_deriv", **_kwargs()
    )
    assert set(metrics.keys()) == EXPECTED_TRAIN_METRICS


def test_metrics_whitelist_antithetic():
    """Antithetic time-sampling must not change the emitted metric set."""
    _loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(0), time_sampling="antithetic", **_kwargs()
    )
    assert set(metrics.keys()) == EXPECTED_TRAIN_METRICS
