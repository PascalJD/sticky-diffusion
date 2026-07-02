"""Value-space blending (embed-the-blur): mu = E(B v) forward + plug-in score.

Covers the blur_space knob end to end:
  - sinusoidal_value_embedding matches the frozen sin8 table construction
    (column order, freqs {1..4} of v/255) and is norm-2 for EVERY real v;
  - blur_values == the blur_means position contraction on scalar fields,
    and row-stochastic kernels keep (B v) inside [0, 255];
  - sample_pair blur_space='value': hand-computed unstuck mean, never-unstuck
    passthrough, x0_idx-required guard, and the x0_idx no-op on the
    embedding / no-blur paths (bit-exact legacy);
  - classifier_induced_score value branch: hand-computed plug-in reference,
    committed-value delta override, L != 256 and eta != 1 guards;
  - q_t_sample (elbo_eta1) rejects value space;
  - _attach_blur_kernel propagates blur_space from the task's ForwardSchedule;
  - factory._validate_value_space_blur accepts the frozen sin8 contract and
    rejects additive normalization / learnable anchors / a wrong table.

CPU-pinned via conftest.py; bitwise assertions use np.array_equal.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.blur import (
    SIN8_VALUE_PERIOD,
    blur_values,
    blurred_value_center,
    gaussian_2d_position_kernel,
    sinusoidal_value_embedding,
)
from sticky.models.sjd.corruption import classifier_induced_score, sample_pair
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.schedule import ForwardSchedule
from sticky.models.sjd.sdes import alpha_sigma, make_beta

_H = _W = 2
_C = 1
_B = 2
_L = 256
_D = 8


def _make_sin8_table_f64() -> np.ndarray:
    """The frozen-table construction (gen_cifar_sin8.make_sin8): float64
    sin/cos at freqs {1,2,3,4} of v/255, interleaved, cast to float32."""
    v = np.arange(256, dtype=np.float64) / 255.0
    cols = []
    for f in (1, 2, 3, 4):
        cols.append(np.sin(2.0 * np.pi * f * v))
        cols.append(np.cos(2.0 * np.pi * f * v))
    return np.stack(cols, axis=1).astype(np.float32)


def _harness(sigma_b: float = 1.0, eta: float = 1.0, blur_space: str = "value"):
    beta = make_beta(0.1, 20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=3.0)
    kernel = gaussian_2d_position_kernel(_H, _W, sigma=sigma_b, normalization="rowstoch")
    jump = VPMatchedGaussianJump(
        beta=beta, eta=eta, std_floor=1e-3,
        blur_kernel=kernel, blur_space=blur_space,
    )
    return beta, hazard, jump, kernel


def _sin8_a_table() -> jnp.ndarray:
    return jnp.asarray(_make_sin8_table_f64())


class _Anchors:
    def __init__(self, table):
        self.table_float = table


# ---------------------------------------------------------------------------
# sinusoidal_value_embedding
# ---------------------------------------------------------------------------

def test_value_embedding_matches_frozen_table_construction():
    table = _make_sin8_table_f64()  # float64->float32, the npz construction
    formula = np.asarray(sinusoidal_value_embedding(jnp.arange(256, dtype=jnp.float32)))
    assert formula.shape == (256, 8)
    # float32 angle rounding at freq 4 gives ~2e-6; a wrong column order,
    # freq set, or period is O(1) off.
    assert np.max(np.abs(table - formula)) < 1e-5


def test_value_embedding_norm_two_everywhere_and_periodic():
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.uniform(0.0, 255.0, size=(257,)), dtype=jnp.float32)
    e = np.asarray(sinusoidal_value_embedding(v))
    norms = np.linalg.norm(e, axis=-1)
    np.testing.assert_allclose(norms, 2.0, rtol=0, atol=1e-5)
    # The known endpoint aliasing: E(0) == E(255) (255-periodic embedding).
    e0 = np.asarray(sinusoidal_value_embedding(jnp.asarray([0.0, 255.0])))
    assert np.max(np.abs(e0[0] - e0[1])) < 1e-5


def test_value_embedding_rejects_bad_args():
    with pytest.raises(ValueError):
        sinusoidal_value_embedding(jnp.zeros((2,)), num_freqs=0)
    with pytest.raises(ValueError):
        sinusoidal_value_embedding(jnp.zeros((2,)), period=0.0)


# ---------------------------------------------------------------------------
# blur_values
# ---------------------------------------------------------------------------

def test_blur_values_matches_manual_einsum_image_layout():
    rng = np.random.default_rng(1)
    kernel = gaussian_2d_position_kernel(_H, _W, sigma=1.0, normalization="rowstoch")
    v = jnp.asarray(rng.integers(0, 256, size=(_B, _H, _W, _C)), dtype=jnp.float32)
    got = np.asarray(blur_values(v, kernel))
    ref = np.einsum(
        "ij,bjc->bic", np.asarray(kernel), np.asarray(v).reshape(_B, _H * _W, _C)
    ).reshape(_B, _H, _W, _C)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
    # Row-stochastic B: convex combination stays inside [0, 255].
    assert got.min() >= 0.0 and got.max() <= 255.0


def test_blur_values_matches_manual_einsum_seq_layout():
    rng = np.random.default_rng(2)
    from sticky.models.sjd.blur import gaussian_position_kernel

    kernel = gaussian_position_kernel(6, sigma=1.0, include_self=False)
    v = jnp.asarray(rng.uniform(0, 255, size=(_B, 6)), dtype=jnp.float32)
    got = np.asarray(blur_values(v, kernel))
    ref = np.einsum("ij,bj->bi", np.asarray(kernel), np.asarray(v))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# sample_pair forward
# ---------------------------------------------------------------------------

def test_sample_pair_value_space_unstuck_mean_hand_computed():
    """Force unstuck everywhere (t large, hazard fast) and eps ~ known: check
    xt == alpha(t) * E((B v)) + std * eps by reconstructing mean/std."""
    beta, hazard, jump, kernel = _harness(sigma_b=1.0)
    a_table = _sin8_a_table()
    rng = np.random.default_rng(3)
    x0_idx = jnp.asarray(rng.integers(0, 256, size=(_B, _H, _W, _C)), dtype=jnp.int32)
    x0_anchor = a_table[x0_idx]
    t = jnp.asarray([0.9, 0.7], dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    xt, never = sample_pair(key, x0_anchor, t, beta, hazard, jump, x0_idx=x0_idx)
    assert xt.shape == x0_anchor.shape

    # Reconstruct the branch: same splits as sample_pair.
    k_unstuck, k_tau, k_normal = jax.random.split(key, 3)
    site_shape = (_H, _W, _C)
    cdf_t = hazard.cdf(t)
    cdf_b = jnp.broadcast_to(cdf_t.reshape((_B, 1, 1, 1)), (_B,) + site_shape)
    u_tau = jax.random.uniform(k_tau, shape=(_B,) + site_shape, dtype=jnp.float32)
    tau = hazard.inv_cdf(u_tau * cdf_b)
    t_b = jnp.broadcast_to(t.reshape((_B, 1, 1, 1)), (_B,) + site_shape)

    from sticky.models.sjd.convolution import mixture_component_mean_std

    center = sinusoidal_value_embedding(
        blur_values(x0_idx.astype(jnp.float32), kernel),
        num_freqs=4, period=SIN8_VALUE_PERIOD,
    )
    mean, std = mixture_component_mean_std(
        anchor=center, t=t_b, tau=tau, beta=beta, eta=1.0, std_floor=1e-3
    )
    eps = jax.random.normal(k_normal, shape=x0_anchor.shape, dtype=jnp.float32)
    expected_unstuck = mean + std * eps
    expected = jnp.where(never[..., None], x0_anchor, expected_unstuck)
    np.testing.assert_array_equal(np.asarray(xt), np.asarray(expected))


def test_sample_pair_value_space_requires_x0_idx():
    beta, hazard, jump, _ = _harness()
    a_table = _sin8_a_table()
    x0_idx = jnp.zeros((_B, _H, _W, _C), dtype=jnp.int32)
    x0_anchor = a_table[x0_idx]
    t = jnp.full((_B,), 0.5, dtype=jnp.float32)
    with pytest.raises(ValueError, match="requires x0_idx"):
        sample_pair(jax.random.PRNGKey(0), x0_anchor, t, beta, hazard, jump)


def test_sample_pair_x0_idx_noop_on_embedding_and_none_paths():
    """Passing x0_idx must be bit-exact vs not passing it when blur_space is
    'embedding' (kernel attached) and when blur is disabled."""
    a_table = _sin8_a_table()
    rng = np.random.default_rng(4)
    x0_idx = jnp.asarray(rng.integers(0, 256, size=(_B, _H, _W, _C)), dtype=jnp.int32)
    x0_anchor = a_table[x0_idx]
    t = jnp.asarray([0.4, 0.8], dtype=jnp.float32)
    key = jax.random.PRNGKey(7)

    for space, with_kernel in (("embedding", True), ("embedding", False)):
        beta, hazard, jump, kernel = _harness(blur_space=space)
        if not with_kernel:
            jump = dataclasses.replace(jump, blur_kernel=None)
        xt_a, nv_a = sample_pair(key, x0_anchor, t, beta, hazard, jump)
        xt_b, nv_b = sample_pair(
            key, x0_anchor, t, beta, hazard, jump, x0_idx=x0_idx
        )
        np.testing.assert_array_equal(np.asarray(xt_a), np.asarray(xt_b))
        np.testing.assert_array_equal(np.asarray(nv_a), np.asarray(nv_b))


def test_sample_pair_value_space_tiny_sigma_is_near_none():
    """sigma_B -> 0 makes B == I to float precision, so the value-space
    corruption must approach the blur=none corruption (same keys). The only
    residual is E_continuous(v) vs the table's float64-cast rows (~2e-6),
    scaled by alpha(t) <= 1."""
    a_table = _sin8_a_table()
    rng = np.random.default_rng(5)
    x0_idx = jnp.asarray(rng.integers(0, 256, size=(_B, _H, _W, _C)), dtype=jnp.int32)
    x0_anchor = a_table[x0_idx]
    t = jnp.asarray([0.3, 0.6], dtype=jnp.float32)
    key = jax.random.PRNGKey(11)

    beta, hazard, jump_v, _ = _harness(sigma_b=0.05)
    xt_v, _ = sample_pair(key, x0_anchor, t, beta, hazard, jump_v, x0_idx=x0_idx)
    jump_n = dataclasses.replace(jump_v, blur_kernel=None)
    xt_n, _ = sample_pair(key, x0_anchor, t, beta, hazard, jump_n, x0_idx=x0_idx)
    assert float(jnp.max(jnp.abs(xt_v - xt_n))) < 1e-5


# ---------------------------------------------------------------------------
# classifier_induced_score value branch
# ---------------------------------------------------------------------------

def _score_inputs(seed: int = 6):
    rng = np.random.default_rng(seed)
    y = jnp.asarray(rng.normal(size=(_B, _H, _W, _C, _D)), dtype=jnp.float32)
    t = jnp.asarray([0.55, 0.75], dtype=jnp.float32)
    logits = jnp.asarray(rng.normal(size=(_B, _H, _W, _C, _L)), dtype=jnp.float32)
    return y, t, logits


def test_value_space_score_matches_hand_computed_plugin():
    beta, hazard, jump, kernel = _harness()
    a_table = _sin8_a_table()
    y, t, logits = _score_inputs()

    committed = jnp.zeros((_B, _H, _W, _C), dtype=bool).at[0, 0, 0, 0].set(True)
    committed_idx = (
        -jnp.ones((_B, _H, _W, _C), dtype=jnp.int32).at[0, 0, 0, 0].set(200)
    )

    got = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
        beta=beta, hazard=hazard, jump=jump,
        committed=committed, committed_idx=committed_idx,
    )

    # Hand-computed plug-in.
    probs = jax.nn.softmax(logits, axis=-1)
    v_mean = jnp.einsum("bhwcl,l->bhwc", probs, jnp.arange(_L, dtype=jnp.float32))
    v_hat = jnp.where(committed, committed_idx.astype(jnp.float32), v_mean)
    mu = sinusoidal_value_embedding(
        blur_values(v_hat, kernel), num_freqs=4, period=SIN8_VALUE_PERIOD
    )
    alpha_t, sigma_t = alpha_sigma(beta, t)
    v_b = jnp.maximum(jnp.square(sigma_t), jnp.square(jnp.asarray(1e-3)))
    expected = (
        -(y - alpha_t[:, None, None, None, None] * mu)
        / v_b[:, None, None, None, None]
    )
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_value_space_score_committed_delta_changes_center():
    """Committing a site to a different value must move nearby scores (the
    committed VALUE enters through B)."""
    beta, hazard, jump, _ = _harness()
    a_table = _sin8_a_table()
    y, t, logits = _score_inputs(seed=8)

    base = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
        beta=beta, hazard=hazard, jump=jump,
    )
    committed = jnp.zeros((_B, _H, _W, _C), dtype=bool).at[0, 0, 0, 0].set(True)
    committed_idx = (
        -jnp.ones((_B, _H, _W, _C), dtype=jnp.int32).at[0, 0, 0, 0].set(255)
    )
    with_commit = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
        beta=beta, hazard=hazard, jump=jump,
        committed=committed, committed_idx=committed_idx,
    )
    diff = np.abs(np.asarray(base) - np.asarray(with_commit))
    assert diff[0].max() > 0.0  # example 0 sees the committed delta
    assert diff[1].max() == 0.0  # example 1 untouched


def test_value_space_score_rejects_non_256_vocab():
    beta, hazard, jump, _ = _harness()
    rng = np.random.default_rng(9)
    L_bad = 16
    a_table = jnp.asarray(rng.normal(size=(L_bad, _D)), dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=(_B, _H, _W, _C, _D)), dtype=jnp.float32)
    logits = jnp.asarray(
        rng.normal(size=(_B, _H, _W, _C, L_bad)), dtype=jnp.float32
    )
    t = jnp.full((_B,), 0.5, dtype=jnp.float32)
    with pytest.raises(ValueError, match="256-value"):
        classifier_induced_score(
            y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
            beta=beta, hazard=hazard, jump=jump,
        )


def test_value_space_score_rejects_eta_not_one():
    beta, hazard, jump, _ = _harness(eta=0.7)
    a_table = _sin8_a_table()
    y, t, logits = _score_inputs(seed=10)
    with pytest.raises(ValueError, match="eta=1"):
        classifier_induced_score(
            y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
            beta=beta, hazard=hazard, jump=jump,
        )


# ---------------------------------------------------------------------------
# elbo guard, blur_space propagation
# ---------------------------------------------------------------------------

def test_q_t_sample_rejects_value_space():
    from sticky.models.sjd.sjd_elbo_loss import q_t_sample

    beta, _, jump, _ = _harness()
    x0_anchor = jnp.zeros((_B, _H, _W, _C, _D), dtype=jnp.float32)
    t = jnp.full((_B,), 0.5, dtype=jnp.float32)
    with pytest.raises(ValueError, match="blur_space='value'"):
        q_t_sample(
            key=jax.random.PRNGKey(0), x0_anchor=x0_anchor, t_img=t,
            beta=beta, jump=jump,
        )


def test_forward_schedule_with_blur_carries_space():
    beta, hazard, jump, kernel = _harness()
    base = dataclasses.replace(jump, blur_kernel=None, blur_space="embedding")
    sched = ForwardSchedule(beta=beta, hazard=hazard, jump=base, T=1.0)
    assert sched.blur_space == "embedding"
    sched_v = sched.with_blur(kernel, blur_space="value")
    assert sched_v.blur_space == "value"
    assert sched_v.jump.blur_space == "value"
    assert sched_v.blur_kernel is kernel


def test_attach_blur_kernel_propagates_space_from_task_forward():
    from sticky.training.sampling import _attach_blur_kernel

    beta, hazard, jump, kernel = _harness()
    fresh = dataclasses.replace(jump, blur_kernel=None, blur_space="embedding")

    class _Task:
        forward = ForwardSchedule(
            beta=beta, hazard=hazard, jump=jump, T=1.0
        )  # jump carries kernel + blur_space="value"

    out = _attach_blur_kernel(fresh, cfg={}, task=_Task())
    assert out.blur_space == "value"
    assert out.blur_kernel is kernel

    class _NoBlurTask:
        forward = ForwardSchedule(
            beta=beta, hazard=hazard,
            jump=dataclasses.replace(jump, blur_kernel=None), T=1.0,
        )

    class _EmptyCfg(dict):
        pass

    same = _attach_blur_kernel(fresh, cfg=_EmptyCfg(), task=_NoBlurTask())
    assert same is fresh  # no kernel -> strict bypass, same object


# ---------------------------------------------------------------------------
# factory validation
# ---------------------------------------------------------------------------

def _value_cfg(tmp_path, table: np.ndarray, **overrides):
    from omegaconf import OmegaConf

    npz = tmp_path / "sin8.npz"
    np.savez(npz, wte=table)
    cfg = {
        "dataset": {"vocab_size": 256},
        "model": {
            "anchor": {
                "family": "pretrained",
                "dim": 8,
                "learnable": False,
                "normalize_at_use": False,
                "pretrained_path": str(npz),
                "transform": {
                    "center_columns": False,
                    "whiten": False,
                    "equalize_row_norms": False,
                    "target_row_norm": None,
                    "scale": 1.0,
                },
            }
        },
    }
    blur = {
        "enabled": True,
        "kind": "gaussian_2d",
        "sigma": 1.0,
        "normalization": "rowstoch",
        "blur_space": "value",
    }
    blur.update(overrides.pop("blur", {}))
    for k, v in overrides.items():
        cfg[k] = v
    return OmegaConf.create(cfg), OmegaConf.create(blur)


def test_factory_value_validation_accepts_frozen_sin8(tmp_path):
    from sticky.tasks.factory import _validate_value_space_blur

    cfg, blur = _value_cfg(tmp_path, _make_sin8_table_f64())
    _validate_value_space_blur(cfg, blur)  # should not raise


def test_factory_value_validation_rejects_additive(tmp_path):
    from sticky.tasks.factory import _validate_value_space_blur

    cfg, blur = _value_cfg(
        tmp_path, _make_sin8_table_f64(), blur={"normalization": "additive"}
    )
    with pytest.raises(ValueError, match="rowstoch"):
        _validate_value_space_blur(cfg, blur)


def test_factory_value_validation_rejects_learnable_anchor(tmp_path):
    from sticky.tasks.factory import _validate_value_space_blur

    cfg, blur = _value_cfg(tmp_path, _make_sin8_table_f64())
    cfg.model.anchor.learnable = True
    with pytest.raises(ValueError, match="frozen sin8"):
        _validate_value_space_blur(cfg, blur)


def test_factory_value_validation_rejects_wrong_table(tmp_path):
    from sticky.tasks.factory import _validate_value_space_blur

    bad = _make_sin8_table_f64()
    bad[:, [0, 1]] = bad[:, [1, 0]]  # swapped sin/cos columns at freq 1
    cfg, blur = _value_cfg(tmp_path, bad)
    with pytest.raises(ValueError, match="does not match"):
        _validate_value_space_blur(cfg, blur)


def test_factory_value_validation_rejects_wrong_vocab(tmp_path):
    from sticky.tasks.factory import _validate_value_space_blur

    cfg, blur = _value_cfg(tmp_path, _make_sin8_table_f64())
    cfg.dataset.vocab_size = 10
    with pytest.raises(ValueError, match="vocab_size=256"):
        _validate_value_space_blur(cfg, blur)


# ---------------------------------------------------------------------------
# reverse_sample smoke (stub model, value space end to end)
# ---------------------------------------------------------------------------

def test_reverse_sample_value_space_smoke_and_legacy_strip():
    from sticky.models.sjd import sampling as sjd_sampling
    from sticky.models.sjd.anchors import AnchorTable
    from sticky.models.sjd.sampler import SamplerConfig

    beta, hazard, jump, kernel = _harness()
    a_table = _sin8_a_table()
    rng = np.random.default_rng(12)
    proj = jnp.asarray(rng.normal(size=(_D, _L)) * 0.05, dtype=jnp.float32)

    class _Stub:
        def apply(self, variables, y, *args, **kwargs):
            return jnp.einsum("...d,dk->...k", y, proj), None

    cfg = SamplerConfig(T=1.0, n_steps=4, alloc_mode="sample", blur_score=True)
    anchors = AnchorTable(table_float=a_table)

    res_v = sjd_sampling.simple_generate(
        rng=jax.random.PRNGKey(3), params={}, model=_Stub(), anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, cfg=cfg,
        batch_size=_B, shape=(_H, _W, _C),
    )
    assert res_v.k_filled.shape == (_B, _H, _W, _C)
    assert int(res_v.k_filled.min()) >= 0

    # blur_score=False strips the kernel -> bitwise identical to kernel=None.
    cfg_off = dataclasses.replace(cfg, blur_score=False)
    res_off = sjd_sampling.simple_generate(
        rng=jax.random.PRNGKey(3), params={}, model=_Stub(), anchors=anchors,
        beta=beta, hazard=hazard, jump=jump, cfg=cfg_off,
        batch_size=_B, shape=(_H, _W, _C),
    )
    jump_none = dataclasses.replace(jump, blur_kernel=None)
    res_none = sjd_sampling.simple_generate(
        rng=jax.random.PRNGKey(3), params={}, model=_Stub(), anchors=anchors,
        beta=beta, hazard=hazard, jump=jump_none, cfg=cfg,
        batch_size=_B, shape=(_H, _W, _C),
    )
    np.testing.assert_array_equal(
        np.asarray(res_off.k_filled), np.asarray(res_none.k_filled)
    )

    # And the value-space score is a genuinely different drift than the
    # embedding-space score on the same state (discrete commits can coincide
    # at these step counts, so assert on the continuous quantity).
    jump_e = dataclasses.replace(jump, blur_space="embedding")
    y, t, logits = _score_inputs(seed=13)
    s_v = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
        beta=beta, hazard=hazard, jump=jump,
    )
    s_e = classifier_induced_score(
        y=y, t=t, anchor_logits=logits, anchors=_Anchors(a_table),
        beta=beta, hazard=hazard, jump=jump_e,
    )
    assert float(jnp.max(jnp.abs(s_v - s_e))) > 1e-3
