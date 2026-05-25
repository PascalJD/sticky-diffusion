# Smoke test: sample_pair returns (xt, never_unstuck_mask) with correct shapes/dtypes.
import jax
import jax.numpy as jnp

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.corruption import sample_pair

B, L, d = 2, 4, 8  # batch=2, seq_len=4, embed_dim=8


def test_sample_pair_shapes_and_dtypes():
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(beta=beta)

    x0_anchor = jax.random.normal(jax.random.PRNGKey(1), shape=(B, L, d))
    t = jnp.array([0.3, 0.7])

    xt, never_unstuck_mask = sample_pair(
        key=jax.random.PRNGKey(0),
        x0_anchor=x0_anchor,
        t=t,
        beta=beta,
        hazard=hazard,
        jump=jump,
    )

    # xt has same shape as x0_anchor
    assert xt.shape == (B, L, d), f"expected ({B},{L},{d}), got {xt.shape}"
    assert xt.dtype == jnp.float32, f"expected float32, got {xt.dtype}"

    # never_unstuck_mask has shape (B, *site_shape) — site_shape here is (L,)
    assert never_unstuck_mask.shape == (B, L), f"expected ({B},{L}), got {never_unstuck_mask.shape}"
    assert never_unstuck_mask.dtype == jnp.bool_, f"expected bool, got {never_unstuck_mask.dtype}"

    # stuck sites should have xt == x0_anchor exactly
    mask_b = never_unstuck_mask[..., None]  # (B, L, 1)
    stuck_ok = jnp.all(jnp.where(mask_b, xt == x0_anchor, True))
    assert bool(stuck_ok), "stuck sites should have xt == x0_anchor"
