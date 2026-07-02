"""Unit tests for the W-aware score fix plumbing.

Covers, at the smallest scope each property allows:
  - the eta!=1 and argument-validation guards on classifier_induced_score;
  - the committed delta override against a hand-computed numpy reference
    (and against the contaminated no-override result);
  - eta=1 identity-kernel equivalence: the closed-form W branch vs the legacy
    tau-quadrature (which collapses to the same formula at eta=1) — the two
    are different code paths, hence allclose, not array_equal;
  - kernel_fingerprint determinism / shape / value sensitivity;
  - blur_score_mean delegating bit-exactly to blur.blurred_posterior_mean;
  - _attach_blur_kernel source preference (task array > cfg rebuild > none);
  - the sampler-spec plumbing round-trip for the blur_score key.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from sticky.models.sjd.blur import (
    blurred_posterior_mean,
    gaussian_position_kernel,
    kernel_fingerprint,
)
from sticky.models.sjd.board_sampling import blur_score_mean
from sticky.models.sjd.corruption import classifier_induced_score
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import alpha_sigma, make_beta
from sticky.training.sampling import (
    _attach_blur_kernel,
    _base_sampler_spec_from_cfg,
    _sjd_sampler_cfg_from_dict,
)

_L, _N, _D = 3, 3, 2


def _fixture(eta: float = 1.0, kernel=None):
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    jump = VPMatchedGaussianJump(
        beta=beta, eta=eta, std_floor=1e-3, blur_kernel=kernel
    )
    rng = np.random.default_rng(0)
    a_table = jnp.asarray(rng.standard_normal((_L, _D)), dtype=jnp.float32)
    anchors = SimpleNamespace(table_float=a_table)
    y = jnp.asarray(rng.normal(0.3, 0.8, size=(1, _N, _D)), dtype=jnp.float32)
    logits = jnp.asarray(rng.standard_normal((1, _N, _L)), dtype=jnp.float32)
    t = jnp.full((1,), 0.4, dtype=jnp.float32)
    return beta, hazard, jump, anchors, y, logits, t


def _row_stochastic_kernel(seed: int = 5):
    raw = np.random.default_rng(seed).uniform(0.05, 1.0, size=(_N, _N))
    return jnp.asarray(raw / raw.sum(axis=1, keepdims=True), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_score_eta_guard_raises_with_kernel():
    kernel = _row_stochastic_kernel()
    beta, hazard, jump, anchors, y, logits, t = _fixture(eta=0.97, kernel=kernel)
    with pytest.raises(ValueError, match="exact only at eta=1"):
        classifier_induced_score(
            y=y, t=t, anchor_logits=logits, anchors=anchors,
            beta=beta, hazard=hazard, jump=jump,
        )


def test_score_eta_lt_one_without_kernel_still_works():
    beta, hazard, jump, anchors, y, logits, t = _fixture(eta=0.7, kernel=None)
    out = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert out.shape == y.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_score_committed_args_must_come_in_pairs():
    kernel = _row_stochastic_kernel()
    beta, hazard, jump, anchors, y, logits, t = _fixture(kernel=kernel)
    committed = jnp.zeros((1, _N), dtype=bool)
    with pytest.raises(ValueError, match="both be provided"):
        classifier_induced_score(
            y=y, t=t, anchor_logits=logits, anchors=anchors,
            beta=beta, hazard=hazard, jump=jump, committed=committed,
        )


# ---------------------------------------------------------------------------
# Committed override vs numpy reference
# ---------------------------------------------------------------------------


def test_committed_override_matches_numpy_reference():
    kernel = _row_stochastic_kernel()
    beta, hazard, jump, anchors, y, logits, t = _fixture(kernel=kernel)
    j0, a_star = 1, 2
    # Contaminate the committed site's logits toward a different anchor.
    logits = logits.at[0, j0].set(jnp.asarray([10.0, 0.0, -10.0]))
    committed = jnp.zeros((1, _N), dtype=bool).at[0, j0].set(True)
    committed_idx = jnp.full((1, _N), -1, dtype=jnp.int32).at[0, j0].set(a_star)

    got = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
        committed=committed, committed_idx=committed_idx,
    )

    # Hand-computed: m = softmax(logits) @ E with row j0 replaced by E[a_star];
    # score = -(y - alpha * (W @ m)) / max(sigma^2, std_floor^2).
    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha, sigma = float(alpha_t[0]), float(sigma_t[0])
    e_np = np.asarray(anchors.table_float, dtype=np.float64)
    probs = np.exp(np.asarray(logits[0], dtype=np.float64))
    probs /= probs.sum(axis=-1, keepdims=True)
    m = probs @ e_np
    m[j0] = e_np[a_star]
    v = max(sigma * sigma, 1e-3 * 1e-3)
    want = -(np.asarray(y[0], dtype=np.float64)
             - alpha * (np.asarray(kernel, dtype=np.float64) @ m)) / v
    np.testing.assert_allclose(np.asarray(got)[0], want, rtol=1e-5, atol=1e-6)

    # Without the override, the contaminated logits must give a different
    # score at the neighbors (the leak the fix removes).
    got_no_override = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert not np.allclose(
        np.asarray(got_no_override)[0], np.asarray(got)[0], rtol=1e-5, atol=1e-6
    )


def test_w_branch_honors_posterior_temperature():
    """The W branch must consume the TEMPERED classifier posterior, exactly
    like the legacy path (symmetric_temperature plumbing). Guards against a
    refactor that recomputes an untempered softmax inside the branch."""
    kernel = _row_stochastic_kernel()
    beta, hazard, jump, anchors, y, logits, t = _fixture(kernel=kernel)
    temperature = 2.5
    got = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
        posterior_temperature=temperature,
    )
    alpha_t, sigma_t = alpha_sigma(beta, t)
    alpha, sigma = float(alpha_t[0]), float(sigma_t[0])
    e_np = np.asarray(anchors.table_float, dtype=np.float64)
    logits_np = np.asarray(logits[0], dtype=np.float64) / temperature
    probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    m = probs @ e_np
    v = max(sigma * sigma, 1e-6)
    want = -(np.asarray(y[0], dtype=np.float64)
             - alpha * (np.asarray(kernel, dtype=np.float64) @ m)) / v
    np.testing.assert_allclose(np.asarray(got)[0], want, rtol=1e-5, atol=1e-6)
    # And the tempered score must differ from the untempered one.
    got_t1 = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    assert not np.allclose(np.asarray(got_t1), np.asarray(got), rtol=1e-5)


def test_w_branch_per_example_time_broadcast():
    """B=2 with distinct t: alpha_t / v_b must broadcast per example, not
    across sites (a transposed broadcast would give wrong numbers, not a
    shape error). Reference computed per example independently."""
    kernel = _row_stochastic_kernel()
    beta, hazard, jump, anchors, y1, logits1, _ = _fixture(kernel=kernel)
    rng = np.random.default_rng(11)
    y = jnp.concatenate(
        [y1, jnp.asarray(rng.normal(0.1, 0.7, size=(1, _N, _D)), dtype=jnp.float32)]
    )
    logits = jnp.concatenate(
        [logits1, jnp.asarray(rng.standard_normal((1, _N, _L)), dtype=jnp.float32)]
    )
    t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    got = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump,
    )
    alpha_t, sigma_t = alpha_sigma(beta, t)
    e_np = np.asarray(anchors.table_float, dtype=np.float64)
    k_np = np.asarray(kernel, dtype=np.float64)
    for b in range(2):
        probs = np.exp(np.asarray(logits[b], dtype=np.float64))
        probs /= probs.sum(axis=-1, keepdims=True)
        m = probs @ e_np
        v = max(float(sigma_t[b]) ** 2, 1e-6)
        want = -(np.asarray(y[b], dtype=np.float64)
                 - float(alpha_t[b]) * (k_np @ m)) / v
        np.testing.assert_allclose(
            np.asarray(got)[b], want, rtol=1e-5, atol=1e-6
        )


def test_identity_kernel_matches_legacy_quadrature_at_eta1():
    identity = jnp.eye(_N, dtype=jnp.float32)
    beta, hazard, jump_w, anchors, y, logits, t = _fixture(kernel=identity)
    _, _, jump_none, _, _, _, _ = _fixture(kernel=None)
    got_w = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump_w,
    )
    got_none = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=anchors,
        beta=beta, hazard=hazard, jump=jump_none,
    )
    np.testing.assert_allclose(
        np.asarray(got_w), np.asarray(got_none), rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# kernel_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_none_and_determinism():
    assert kernel_fingerprint(None) is None
    k = _row_stochastic_kernel()
    assert kernel_fingerprint(k) == kernel_fingerprint(k)


def test_fingerprint_shape_and_value_sensitivity():
    fp4 = kernel_fingerprint(jnp.eye(4, dtype=jnp.float32))
    fp5 = kernel_fingerprint(jnp.eye(5, dtype=jnp.float32))
    assert fp4 != fp5
    k = _row_stochastic_kernel()
    k2 = k.at[0, 0].add(1e-6)
    assert kernel_fingerprint(k) != kernel_fingerprint(k2)


def test_fingerprint_matches_independent_hashlib():
    k = _row_stochastic_kernel()
    arr = np.ascontiguousarray(np.asarray(k, dtype=np.float32))
    h = hashlib.sha256()
    h.update(f"float32:{tuple(arr.shape)}".encode("utf-8"))
    h.update(arr.tobytes())
    assert kernel_fingerprint(k) == h.hexdigest()


# ---------------------------------------------------------------------------
# Delegation + attach helper + spec plumbing
# ---------------------------------------------------------------------------


def test_blur_score_mean_delegates_bit_exactly():
    rng = np.random.default_rng(9)
    kernel = _row_stochastic_kernel()
    probs_mean = jnp.asarray(rng.standard_normal((2, _N, _D)), dtype=jnp.float32)
    committed = jnp.asarray(rng.random((2, _N)) < 0.4)
    idx = jnp.where(
        committed, jnp.asarray(rng.integers(0, _L, size=(2, _N))), -1
    ).astype(jnp.int32)
    a_table = jnp.asarray(rng.standard_normal((_L, _D)), dtype=jnp.float32)
    kwargs = dict(
        probs_mean=probs_mean, committed_mask=committed, committed_idx=idx,
        a_table=a_table, kernel=kernel,
    )
    assert np.array_equal(
        np.asarray(blur_score_mean(**kwargs)),
        np.asarray(blurred_posterior_mean(**kwargs)),
    )


def test_attach_blur_kernel_prefers_task_array():
    kernel = _row_stochastic_kernel()
    beta = make_beta(0.1, 20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    task = SimpleNamespace(
        forward=SimpleNamespace(blur_kernel=kernel),
        spec=SimpleNamespace(data_shape=(_N,)),
    )
    cfg = OmegaConf.create({"forward": {"blur": {"enabled": False}}})
    out = _attach_blur_kernel(jump, cfg=cfg, task=task)
    assert out.blur_kernel is kernel


def test_attach_blur_kernel_rebuilds_from_cfg():
    beta = make_beta(0.1, 20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    task = SimpleNamespace(spec=SimpleNamespace(data_shape=(6,)))  # no .forward
    cfg = OmegaConf.create(
        {
            "forward": {
                "blur": {
                    "enabled": True,
                    "kind": "gaussian_1d",
                    "sigma": 1.0,
                    "include_self": True,
                }
            }
        }
    )
    out = _attach_blur_kernel(jump, cfg=cfg, task=task)
    want = gaussian_position_kernel(seq_len=6, sigma=1.0, include_self=True)
    assert out.blur_kernel is not None
    np.testing.assert_array_equal(np.asarray(out.blur_kernel), np.asarray(want))


def test_attach_blur_kernel_disabled_returns_same_jump_object():
    beta = make_beta(0.1, 20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    task = SimpleNamespace(spec=SimpleNamespace(data_shape=(6,)))
    cfg = OmegaConf.create({"forward": {"blur": {"enabled": False}}})
    assert _attach_blur_kernel(jump, cfg=cfg, task=task) is jump


def test_sampler_spec_blur_score_round_trip():
    spec = _base_sampler_spec_from_cfg(OmegaConf.create({}), sample_timesteps=8)
    assert spec["blur_score"] is True
    assert _sjd_sampler_cfg_from_dict({"n_steps": 8}).blur_score is True
    assert (
        _sjd_sampler_cfg_from_dict({"n_steps": 8, "blur_score": False}).blur_score
        is False
    )
    spec_off = _base_sampler_spec_from_cfg(
        OmegaConf.create({"blur_score": False}), sample_timesteps=8
    )
    assert spec_off["blur_score"] is False
