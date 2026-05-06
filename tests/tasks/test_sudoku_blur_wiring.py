"""Tests for Task B7/B8 (Phase 1.B): sjd_sudoku_blur factory wiring.

These tests verify the end-to-end wiring between the Hydra
`experiment=sudoku/sjd_sudoku_blur` config and the `_sjd_schedule_kwargs`
factory helper: that the constraint kernel is correctly attached (or absent)
to the `jump` object produced by the factory, and that downstream calls using
that jump behave correctly.
"""
from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydra import compose, initialize_config_dir
from sticky.tasks.factory import _sjd_schedule_kwargs
from sticky.models.sjd.corruption import sample_pair, mixture_logpdf
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


def _config_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config")
    )


def _make_no_blur_cfg():
    """Minimal OmegaConf config that satisfies _sjd_schedule_kwargs with blur disabled."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "forward": {
            "beta": {
                "_target_": "sticky.models.sjd.sdes.make_beta",
                "beta_min": 0.1,
                "beta_max": 20.0,
                "T": 1.0,
            },
            "hazard": {
                "_target_": "sticky.models.sjd.hazard.make_hazard_linear_time",
                "kappa": 1.5,
            },
            "jump": {
                "_target_": "sticky.models.sjd.jump.VPMatchedGaussianJump",
                "eta": 0.97,
                "std_floor": 1e-3,
                "clip": None,
            },
            "blur": {
                "enabled": False,
                "kind": None,
            },
        },
        "dataset": {"vocab_size": 9, "data_shape": [81]},
        "training": {},
        "sampler": {},
    })


# ---------------------------------------------------------------------------
# Test 1: No-blur path produces blur_kernel is None
# ---------------------------------------------------------------------------

def test_no_blur_path_kernel_is_none():
    """_sjd_schedule_kwargs with blur disabled (or missing) must leave
    jump.blur_kernel as None (Python identity None, not just falsy)."""
    cfg = _make_no_blur_cfg()
    out = _sjd_schedule_kwargs(cfg)
    assert out["jump"].blur_kernel is None


# ---------------------------------------------------------------------------
# Test 2: Constraint kernel is attached when experiment=sudoku/sjd_sudoku_blur
# ---------------------------------------------------------------------------

def test_sjd_sudoku_blur_experiment_attaches_constraint_kernel():
    """Composing experiment=sudoku/sjd_sudoku_blur must produce a jump with
    a (81, 81) constraint kernel whose diagonal is all-ones and whose
    [0, 80] entry is zero (cells (0,0) and (8,8) share no constraint group
    under exclusive-box semantics)."""
    config_dir = _config_dir()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config.yaml",
            overrides=["experiment=sudoku/sjd_sudoku_blur"],
        )

    out = _sjd_schedule_kwargs(cfg.experiment)
    kernel = out["jump"].blur_kernel

    assert kernel is not None, "blur_kernel should be attached for sjd_sudoku_blur"
    assert isinstance(kernel, jnp.ndarray), "blur_kernel must be a jnp.ndarray"
    assert kernel.shape == (81, 81), f"expected (81, 81), got {kernel.shape}"

    diag = np.asarray(jnp.diagonal(kernel))
    np.testing.assert_allclose(diag, np.ones(81), atol=1e-6,
                               err_msg="Kernel diagonal must be all-ones after rescaling")

    assert float(kernel[0, 80]) == 0.0, (
        "W[0, 80] must be 0: cells (0,0) and (8,8) share no row, col, or box"
    )


# ---------------------------------------------------------------------------
# Test 3: Smoke step — mixture_logpdf with blurred jump doesn't crash
# ---------------------------------------------------------------------------

def test_mixture_logpdf_with_blurred_jump_is_finite():
    """Smoke test: calling mixture_logpdf with a blur-kernel-bearing jump must
    not raise and must produce finite outputs.

    Scope note: a full single-training-step would require building the actual
    model (GPT2-like backbone + anchor table). Instead we exercise the SJD
    inference path (mixture_logpdf) directly on 4-example minibatches — this
    covers the kernel-attached jump through the loss-relevant code path
    without duplicating model setup. Per the spec, this substitution is
    documented explicitly here.
    """
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump_base = VPMatchedGaussianJump(beta=beta, eta=0.97)

    from sticky.models.sjd.blur import sudoku_constraint_kernel
    kernel = sudoku_constraint_kernel(sigma=1.5)
    jump = dataclasses.replace(jump_base, blur_kernel=kernel)

    B = 4
    N = 81
    d = 3  # anchor embedding dim (toy)
    key = jax.random.PRNGKey(42)
    k_y, k_a = jax.random.split(key)
    y = jax.random.normal(k_y, (B, N, d))
    anchor = jax.random.normal(k_a, (B, N, d))
    t = jnp.full((B,), 0.5, dtype=jnp.float32)

    lp = mixture_logpdf(y, anchor, t, beta, hazard, jump,
                        tau_grid_size=17, tau_grid="simpson")

    assert lp.shape == (B, N), f"expected shape ({B}, {N}), got {lp.shape}"
    assert bool(jnp.all(jnp.isfinite(lp))), "mixture_logpdf must be finite with blur kernel"


# ---------------------------------------------------------------------------
# Test 4: Bit-exact determinism on no-blur path
# ---------------------------------------------------------------------------

def test_no_blur_sample_pair_is_deterministic():
    """Two VPMatchedGaussianJump instances with blur_kernel=None (the default)
    called on the same PRNG key must produce bit-exact outputs.

    Limitation: this is not a snapshot test against a pre-Phase-1 stored
    array, because no such snapshot was captured before the factory changes
    landed. What it does guarantee is that the no-blur path (blur_kernel=None)
    is deterministic — any future change that breaks this invariant will cause
    a failure here. The backward-compat guarantee is established by construction
    in test_blur.py::test_apply_blur_is_python_identity_when_disabled (apply_blur
    returns the same Python object when blur_kernel is None, producing no
    numerical change at all).
    """
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)

    jump_a = VPMatchedGaussianJump(beta=beta, eta=0.97)
    jump_b = VPMatchedGaussianJump(beta=beta, eta=0.97)

    assert jump_a.blur_kernel is None
    assert jump_b.blur_kernel is None

    B, N, d = 4, 81, 3
    key = jax.random.PRNGKey(99)
    k_x0, k_t, k_pair = jax.random.split(key, 3)
    x0 = jax.random.normal(k_x0, (B, N, d))
    t = jax.random.uniform(k_t, (B,), minval=0.05, maxval=0.95)

    xt_a, mask_a = sample_pair(k_pair, x0, t, beta, hazard, jump_a)
    xt_b, mask_b = sample_pair(k_pair, x0, t, beta, hazard, jump_b)

    np.testing.assert_array_equal(
        np.asarray(xt_a), np.asarray(xt_b),
        err_msg="No-blur sample_pair must be bit-exact across equivalent jump instances",
    )
    np.testing.assert_array_equal(
        np.asarray(mask_a), np.asarray(mask_b),
        err_msg="No-blur never_unstuck mask must be bit-exact across equivalent jump instances",
    )
