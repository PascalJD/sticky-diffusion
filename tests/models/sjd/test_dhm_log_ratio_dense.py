"""Numerical agreement: dense-L dhm_log_ratio matches the per-anchor vmap path.

The dense-L rewrite avoids the (L, B, *site, d) intermediate produced by the
old vmap-over-anchors implementation (which OOMed on text-vocab L). The two
algorithms compute algebraically equivalent quantities via different
LSE schedules, so the comparison is at float32 numerical tolerance, not
bit-exact.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.corruption import mixture_logpdf
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.plugin_intensity import dhm_log_ratio
from sticky.models.sjd.sdes import make_beta


def _dhm_log_ratio_vmap_legacy(
    *,
    y,
    t_img,
    anchors,
    beta,
    hazard,
    jump,
    log_ratio_clip=10.0,
    tau_grid_size=32,
):
    """Pre-fix per-anchor vmap implementation. Numerical oracle ONLY."""
    y = jnp.asarray(y, dtype=jnp.float32)
    a_table = jnp.asarray(anchors.table_float, dtype=jnp.float32)
    _eff = getattr(anchors, "effective_log_w", None)
    if _eff is None:
        _eff = jnp.zeros((int(a_table.shape[0]),), dtype=jnp.float32)
    log_w_table = jnp.asarray(_eff, dtype=jnp.float32)

    def per_anchor(a_l, log_w_l):
        a_b = jnp.broadcast_to(a_l, y.shape)
        log_r = jump.logpdf(y, a_b, t_img)
        log_p = mixture_logpdf(
            y=y,
            anchor=a_b,
            t=t_img,
            beta=beta,
            hazard=hazard,
            jump=jump,
            tau_grid_size=int(tau_grid_size),
            log_w_anchor=log_w_l,
        )
        return log_r - log_p

    lr = jax.vmap(per_anchor, in_axes=(0, 0))(a_table, log_w_table)
    lr = jnp.moveaxis(lr, 0, -1)
    if log_ratio_clip is not None:
        lr = jnp.clip(lr, -float(log_ratio_clip), float(log_ratio_clip))
    return lr


def _setup(L=8, d=4, B=2, site_shape=(3,), seed=0):
    beta = make_beta(beta_min=0.1, beta_max=8.0, T=1.0)
    hazard = make_hazard_poly_alpha(beta, p=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7, std_floor=1e-3)
    key = jax.random.PRNGKey(seed)
    k_a, k_y, k_t = jax.random.split(key, 3)
    a_table = jax.random.normal(k_a, (L, d), dtype=jnp.float32)
    y = jax.random.normal(k_y, (B,) + site_shape + (d,), dtype=jnp.float32)
    t = jax.random.uniform(k_t, (B,), minval=0.1, maxval=0.9)
    return beta, hazard, jump, a_table, y, t


def test_dhm_log_ratio_matches_vmap_uniform_log_w():
    beta, hazard, jump, a_table, y, t = _setup()
    anchors = AnchorTable(table_float=a_table, log_w=None)
    new = dhm_log_ratio(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    legacy = _dhm_log_ratio_vmap_legacy(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    assert new.shape == legacy.shape
    assert new.dtype == legacy.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(legacy), rtol=1e-5, atol=1e-5,
    )


def test_dhm_log_ratio_matches_vmap_freq_weighted():
    beta, hazard, jump, a_table, y, t = _setup(L=8, d=4, B=2, site_shape=(3,))
    # Non-trivial w(a) in [0.1, 10] mirrors test_frequency_hazard.py:144-146.
    log_w = jnp.linspace(
        jnp.log(jnp.asarray(0.1, dtype=jnp.float32)),
        jnp.log(jnp.asarray(10.0, dtype=jnp.float32)),
        a_table.shape[0],
        dtype=jnp.float32,
    )
    anchors = AnchorTable(table_float=a_table, log_w=log_w)
    new = dhm_log_ratio(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    legacy = _dhm_log_ratio_vmap_legacy(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=10.0,
    )
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(legacy), rtol=1e-5, atol=1e-5,
    )


def test_dhm_log_ratio_matches_vmap_no_clip():
    beta, hazard, jump, a_table, y, t = _setup(seed=7)
    anchors = AnchorTable(table_float=a_table, log_w=None)
    new = dhm_log_ratio(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=None,
    )
    legacy = _dhm_log_ratio_vmap_legacy(
        y=y, t_img=t, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, log_ratio_clip=None,
    )
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(legacy), rtol=1e-5, atol=1e-5,
    )
