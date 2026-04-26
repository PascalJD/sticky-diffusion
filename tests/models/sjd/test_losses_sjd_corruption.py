from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


def _trivial_apply_fn(B: int, seq_len: int, V: int):
    """Constant-zero logits → uniform probs → per-site nll = log(V)."""
    def apply_fn(params, xt, t_img):
        del params, xt, t_img
        return jnp.zeros((B, seq_len, V), dtype=jnp.float32), {}
    return apply_fn


def _trivial_kwargs(
    *, B=8, seq_len=6, V=4, d=3, kappa=2.0, eta=0.5, seed=1,
):
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=kappa)
    jump = VPMatchedGaussianJump(beta=beta, eta=eta, std_floor=1e-3)
    table = jax.random.normal(
        jax.random.PRNGKey(seed), (V, d), dtype=jnp.float32
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(seed + 1), (B, seq_len), 0, V
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]
    return dict(
        apply_fn=_trivial_apply_fn(B, seq_len, V),
        params={},
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
    )


def test_sjd_corruption_handles_never_unstuck():
    """Three checks, all intrinsic to the SJD corruption (no VP comparison):
    (a) E[loss/frac_never_unstuck] over seeds matches E_t~U(0,1)[surv(t)].
    (b) E[loss/frac_active] matches E_t~U(0,1)[1 - surv(t)] — the unmasked
        loss fraction equals (1 - S(t)), which is what the loss should
        average over.
    (c) When surv(t) ~ 1 everywhere (kappa -> 0), every site is masked and
        the loss is exactly 0.
    """
    # E_t~U(0,T)[surv(t)] = (1 - exp(-kappa))/kappa for linear_time, T=1.
    # kappa = 1.6 gives ~0.5.
    kwargs = _trivial_kwargs(
        B=256, seq_len=4, V=4, d=3, kappa=1.6, eta=0.5, seed=20
    )
    expected_mean_surv = float((1.0 - jnp.exp(-1.6)) / 1.6)
    expected_mean_uncommitted = 1.0 - expected_mean_surv

    fracs_never_unstuck = []
    fracs_uncommitted = []
    for seed in range(10):
        loss, metrics = ce_allocation_loss(
            key=jax.random.PRNGKey(seed), **kwargs
        )
        assert np.isfinite(float(loss))
        fracs_never_unstuck.append(float(metrics["loss/frac_never_unstuck"]))
        fracs_uncommitted.append(float(metrics["loss/frac_active"]))

    np.testing.assert_allclose(
        np.mean(fracs_never_unstuck), expected_mean_surv, atol=2e-2
    )
    np.testing.assert_allclose(
        np.mean(fracs_uncommitted), expected_mean_uncommitted, atol=2e-2
    )

    # (c) Vanishing-hazard limit: surv(t) ~ 1 → all sites masked → loss = 0.
    super_kwargs = _trivial_kwargs(
        B=8, seq_len=6, V=4, d=3, kappa=1e-6, eta=0.5, seed=21
    )
    loss_zero, metrics_zero = ce_allocation_loss(
        key=jax.random.PRNGKey(0), **super_kwargs
    )
    assert float(metrics_zero["loss/frac_never_unstuck"]) > 0.99
    np.testing.assert_allclose(float(loss_zero), 0.0, atol=1e-6)
