"""Tests for the gated, default-OFF blur-aware score in conditional_generate.

Covers:
  Unit  - blur_score_mean semantics (committed anchors vs classifier mean,
          identity kernel exactness, uncommitted pass-through).
  T8    - OFF-path bit-exactness through conditional_generate: an attached
          kernel is completely ignored when blur_score=False, and
          blur_score=True with no kernel is a bit-exact no-op.
  T9    - ON-path correctness: identity kernel reproduces the legacy output,
          a real kernel changes the output, and eta != 1 raises ValueError
          (with or without a kernel attached).

CPU-only, no GPU required.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.blur import sudoku_constraint_kernel_convex
from sticky.models.sjd.board_sampling import blur_score_mean, conditional_generate
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


# ---------------------------------------------------------------------------
# Unit tests on blur_score_mean (no sampler)
# ---------------------------------------------------------------------------

def _unit_inputs(seed: int = 3, n_sites: int = 5, d: int = 3):
    rng = np.random.default_rng(seed)
    a_table = rng.standard_normal((9, d)).astype(np.float32)
    probs_mean = rng.standard_normal((1, n_sites, d)).astype(np.float32)
    return rng, a_table, probs_mean


def test_blur_score_mean_committed_uses_anchor_not_probs():
    """At a committed site whose classifier mean points to a DIFFERENT digit,
    the result must be W @ Ehat built from the committed anchor."""
    rng, a_table, probs_mean = _unit_inputs()
    n_sites = probs_mean.shape[1]

    committed_mask = np.zeros((1, n_sites), dtype=bool)
    committed_mask[0, 0] = True
    committed_idx = np.full((1, n_sites), -1, dtype=np.int32)
    committed_idx[0, 0] = 2
    # Classifier mean at the committed site deliberately points to a different
    # digit's anchor (digit index 5, not the committed 2).
    probs_mean[0, 0] = a_table[5]

    raw = rng.random((n_sites, n_sites)).astype(np.float32)
    kernel = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    out = np.asarray(
        blur_score_mean(
            probs_mean=jnp.asarray(probs_mean),
            committed_mask=jnp.asarray(committed_mask),
            committed_idx=jnp.asarray(committed_idx),
            a_table=jnp.asarray(a_table),
            kernel=jnp.asarray(kernel),
        )
    )

    # Independent numpy expectation: committed anchor wins at site 0.
    e_hat = probs_mean.copy()
    e_hat[0, 0] = a_table[2]
    expected = np.einsum("ij,bjd->bid", kernel, e_hat)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    # Sanity: it must NOT equal the contaminated version that lets the
    # classifier mean leak into the committed site.
    contaminated = np.einsum("ij,bjd->bid", kernel, probs_mean)
    assert not np.allclose(out, contaminated)


def test_blur_score_mean_identity_kernel_returns_ehat_exactly():
    _, a_table, probs_mean = _unit_inputs(seed=7)
    n_sites = probs_mean.shape[1]

    committed_mask = np.zeros((1, n_sites), dtype=bool)
    committed_mask[0, 1] = True
    committed_mask[0, 4] = True
    committed_idx = np.full((1, n_sites), -1, dtype=np.int32)
    committed_idx[0, 1] = 0
    committed_idx[0, 4] = 8

    kernel = np.eye(n_sites, dtype=np.float32)
    out = np.asarray(
        blur_score_mean(
            probs_mean=jnp.asarray(probs_mean),
            committed_mask=jnp.asarray(committed_mask),
            committed_idx=jnp.asarray(committed_idx),
            a_table=jnp.asarray(a_table),
            kernel=jnp.asarray(kernel),
        )
    )

    e_hat = probs_mean.copy()
    e_hat[0, 1] = a_table[0]
    e_hat[0, 4] = a_table[8]
    np.testing.assert_array_equal(out, e_hat)


def test_blur_score_mean_uncommitted_pass_through():
    """With no committed sites and an identity kernel, the result is exactly
    the classifier mean (probs_mean passes through to Ehat)."""
    _, a_table, probs_mean = _unit_inputs(seed=11)
    n_sites = probs_mean.shape[1]

    committed_mask = np.zeros((1, n_sites), dtype=bool)
    committed_idx = np.full((1, n_sites), -1, dtype=np.int32)
    kernel = np.eye(n_sites, dtype=np.float32)

    out = np.asarray(
        blur_score_mean(
            probs_mean=jnp.asarray(probs_mean),
            committed_mask=jnp.asarray(committed_mask),
            committed_idx=jnp.asarray(committed_idx),
            a_table=jnp.asarray(a_table),
            kernel=jnp.asarray(kernel),
        )
    )
    np.testing.assert_array_equal(out, probs_mean)


# ---------------------------------------------------------------------------
# Minimal CPU harness for conditional_generate
# ---------------------------------------------------------------------------

_D = 4  # anchor dim


class _StubModel:
    """Minimal stand-in for the classifier used by conditional_generate.

    _predict_logits calls model.apply({"params": params}, y, t=t_img,
    train=False) and unpacks (logits, aux) with logits (B, 81, 9). We mimic
    exactly that with a deterministic linear map of y.
    """

    def __init__(self, proj: np.ndarray):
        self._proj = jnp.asarray(proj, dtype=jnp.float32)  # (d, 9)

    def apply(self, variables, y, *, t, train=False):
        del variables, t, train
        logits = jnp.einsum("bnd,dk->bnk", y, self._proj)
        return logits, None


def _make_harness():
    beta = make_beta(0.1, 20.0, T=1.0)
    rng = np.random.default_rng(0)
    a_table = jnp.asarray(rng.standard_normal((9, _D)), dtype=jnp.float32)
    anchors = AnchorTable(table_float=a_table)
    model = _StubModel(rng.standard_normal((_D, 9)))
    params = {}
    return beta, anchors, model, params


def _make_known(batch: int = 2, n_clues: int = 25, seed: int = 1):
    rng = np.random.default_rng(seed)
    digits = rng.integers(1, 10, size=(batch, 81)).astype(np.int32)
    mask = np.zeros((batch, 81), dtype=bool)
    for b in range(batch):
        mask[b, rng.choice(81, size=n_clues, replace=False)] = True
    tokens = np.where(mask, digits, 0).astype(np.int32)
    return jnp.asarray(tokens), jnp.asarray(mask, dtype=jnp.bool_)


def _run(*, harness, blur_kernel, blur_score: bool, eta: float = 1.0, seed: int = 42):
    beta, anchors, model, params = harness
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0, blur_kernel=blur_kernel)
    tokens, mask = _make_known()
    out = conditional_generate(
        jax.random.PRNGKey(seed),
        params=params,
        model=model,
        anchors=anchors,
        beta=beta,
        hazard=None,  # unused by the linear_survival policy
        jump=jump,
        known_tokens=tokens,
        known_token_mask=mask,
        n_steps=3,
        policy="linear_survival",
        sampling_grid="uniform",
        eta=eta,
        blur_score=blur_score,
    )
    return np.asarray(out)


# ---------------------------------------------------------------------------
# T8 — OFF path bit-exactness
# ---------------------------------------------------------------------------

def test_t8_off_path_ignores_attached_kernel():
    """blur_score=False with a real convex kernel attached must be bit-exact
    with blur_score=False and no kernel at all."""
    harness = _make_harness()
    kernel = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.35)
    out_with_kernel = _run(harness=harness, blur_kernel=kernel, blur_score=False)
    out_no_kernel = _run(harness=harness, blur_kernel=None, blur_score=False)
    assert np.array_equal(out_with_kernel, out_no_kernel)


def test_t8_on_without_kernel_is_bit_exact_noop():
    """blur_score=True but blur_kernel=None falls back to the legacy path."""
    harness = _make_harness()
    out_on = _run(harness=harness, blur_kernel=None, blur_score=True, eta=1.0)
    out_off = _run(harness=harness, blur_kernel=None, blur_score=False, eta=1.0)
    assert np.array_equal(out_on, out_off)


# ---------------------------------------------------------------------------
# T9 — ON path correctness
# ---------------------------------------------------------------------------

def test_t9a_identity_kernel_matches_legacy():
    """rho=0.0 convex kernel is the exact identity; blur_score=True with it
    must reproduce the legacy (no-kernel, blur_score=False) output."""
    harness = _make_harness()
    identity = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.0)
    np.testing.assert_array_equal(np.asarray(identity), np.eye(81, dtype=np.float32))
    out_on = _run(harness=harness, blur_kernel=identity, blur_score=True, eta=1.0)
    out_legacy = _run(harness=harness, blur_kernel=None, blur_score=False, eta=1.0)
    # array_equal verified empirically on this platform (identity einsum is
    # bit-exact for these inputs); outputs are int32 digits.
    assert np.array_equal(out_on, out_legacy)


def test_t9b_real_kernel_changes_output():
    harness = _make_harness()
    kernel = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.5)
    out_on = _run(harness=harness, blur_kernel=kernel, blur_score=True, eta=1.0)
    out_off = _run(harness=harness, blur_kernel=kernel, blur_score=False, eta=1.0)
    assert not np.array_equal(out_on, out_off)


def test_t9c_eta_guard_raises():
    harness = _make_harness()
    kernel = sudoku_constraint_kernel_convex(sigma=1.5, rho=0.5)
    with pytest.raises(ValueError, match="exact only at eta=1"):
        _run(harness=harness, blur_kernel=kernel, blur_score=True, eta=0.97)
    # The guard fires even when no kernel is attached.
    with pytest.raises(ValueError, match="exact only at eta=1"):
        _run(harness=harness, blur_kernel=None, blur_score=True, eta=0.97)
