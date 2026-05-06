from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sticky.models.sjd.blur import gaussian_position_kernel, sudoku_constraint_kernel, blur_means, build_blur_kernel
from sticky.models.sjd.corruption import sample_pair
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sdes import make_beta


def test_gaussian_position_kernel_small_sigma_is_identity():
    """sigma -> 0 with self-inclusion should recover I_L within 1e-3."""
    L = 16
    W = gaussian_position_kernel(L, sigma=1e-3, include_self=True)
    np.testing.assert_allclose(np.asarray(W), np.eye(L), atol=1e-3)


def test_gaussian_position_kernel_diagonal_is_one():
    """Every diagonal entry must be exactly 1 after rescaling."""
    W = gaussian_position_kernel(32, sigma=1.5, include_self=True)
    diag = np.asarray(jnp.diagonal(W))
    np.testing.assert_allclose(diag, np.ones(32), atol=1e-6)


def test_gaussian_position_kernel_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_position_kernel(16, sigma=0.0)
    with pytest.raises(ValueError):
        gaussian_position_kernel(16, sigma=-1.0)


def test_gaussian_position_kernel_no_self_is_finite_and_row_stochastic():
    """include_self=False must NOT produce NaN/Inf. Diagonal is zero by design;
    rows sum to 1 because softmax is row-stochastic and no rescale is applied."""
    W = gaussian_position_kernel(8, sigma=1.0, include_self=False)
    W_np = np.asarray(W)
    assert np.isfinite(W_np).all(), "include_self=False produced non-finite entries"
    np.testing.assert_allclose(np.diagonal(W_np), np.zeros(8), atol=1e-6)
    np.testing.assert_allclose(W_np.sum(axis=-1), np.ones(8), atol=1e-6)


def test_sudoku_kernel_small_sigma_is_identity():
    W = sudoku_constraint_kernel(sigma=1e-3)
    np.testing.assert_allclose(np.asarray(W), np.eye(81), atol=1e-3)


def test_sudoku_kernel_no_share_is_zero():
    """Cells (0,0) and (8,8) share no row/col/box -> W[0, 80] == 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 80]) == 0.0


def test_sudoku_kernel_row_neighbor_nonzero():
    """Cells (0,0) and (0,1) share row 0 -> W[0, 1] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 1]) > 0.0


def test_sudoku_kernel_col_neighbor_nonzero():
    """Cells (0,0) and (1,0) share col 0 -> W[0, 9] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 9]) > 0.0


def test_sudoku_kernel_box_only_nonzero():
    """Cells (0,0) and (1,1) share top-left box but no row/col -> W[0, 10] > 0."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 10]) > 0.0


def test_sudoku_kernel_distance_monotone_within_row():
    """Closer column dominates: W[0, 1] > W[0, 8] (both row-mates of (0,0))."""
    W = sudoku_constraint_kernel(sigma=1.5)
    assert float(W[0, 1]) > float(W[0, 8])


def test_sudoku_kernel_disable_row_zeros_row_neighbor():
    """include_row=False -> W[0, 1] == 0 (no longer share any active group)."""
    W = sudoku_constraint_kernel(sigma=1.5, include_row=False)
    assert float(W[0, 1]) == 0.0


def test_blur_means_identity_kernel_is_passthrough():
    """blur_means(E, I_N) == E to float32 precision."""
    B, N, d = 3, 16, 8
    e = jax.random.normal(jax.random.PRNGKey(0), (B, N, d))
    out = blur_means(e, jnp.eye(N, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(out), np.asarray(e), atol=1e-6)


def test_blur_means_rejects_2d_site_shape():
    """Phase 1 only supports 1D site_shape; 2D images must raise."""
    e = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 3, 16))
    kernel = jnp.eye(32 * 32, dtype=jnp.float32)
    with pytest.raises(NotImplementedError):
        blur_means(e, kernel)


def test_blur_means_rejects_kernel_shape_mismatch():
    e = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 8))
    bad_kernel = jnp.eye(15, dtype=jnp.float32)
    with pytest.raises((ValueError, AssertionError)):
        blur_means(e, bad_kernel)


def test_build_blur_kernel_disabled_returns_none():
    cfg = {"enabled": False, "kind": None, "sigma": 1.0}
    assert build_blur_kernel(cfg, seq_len=16) is None
    assert build_blur_kernel(None, seq_len=16) is None  # covers the blur_cfg-is-None branch


def test_build_blur_kernel_gaussian_1d():
    cfg = {"enabled": True, "kind": "gaussian_1d", "sigma": 1.0, "include_self": True}
    W = build_blur_kernel(cfg, seq_len=16)
    assert W.shape == (16, 16)
    np.testing.assert_allclose(np.diagonal(np.asarray(W)), np.ones(16), atol=1e-6)


def test_build_blur_kernel_gaussian_1d_requires_seq_len():
    cfg = {"enabled": True, "kind": "gaussian_1d", "sigma": 1.0, "include_self": True}
    with pytest.raises(ValueError):
        build_blur_kernel(cfg, seq_len=None)


def test_build_blur_kernel_sudoku_ignores_seq_len():
    cfg = {
        "enabled": True, "kind": "sudoku_constraint", "sigma": 1.5,
        "include_row": True, "include_col": True, "include_box": True,
    }
    W = build_blur_kernel(cfg, seq_len=None)
    assert W.shape == (81, 81)


def test_build_blur_kernel_unknown_kind_raises():
    cfg = {"enabled": True, "kind": "ring_kernel", "sigma": 1.0}
    with pytest.raises(ValueError):
        build_blur_kernel(cfg, seq_len=16)


def test_build_blur_kernel_kind_required_when_enabled():
    cfg = {"enabled": True, "kind": None, "sigma": 1.0}
    with pytest.raises(ValueError, match="kind"):
        build_blur_kernel(cfg, seq_len=16)


def test_build_blur_kernel_gaussian_1d_rejects_zero_seq_len():
    cfg = {"enabled": True, "kind": "gaussian_1d", "sigma": 1.0, "include_self": True}
    with pytest.raises(ValueError, match="seq_len"):
        build_blur_kernel(cfg, seq_len=0)


def test_public_reexports():
    from sticky.models.sjd import (
        gaussian_position_kernel as g,
        sudoku_constraint_kernel as s,
        blur_means as b,
    )
    assert callable(g) and callable(s) and callable(b)


def test_apply_blur_is_python_identity_when_disabled():
    """No blur_kernel -> apply_blur returns the SAME object (Python identity).
    This guarantees zero numerical drift in the no-blur path."""
    from sticky.models.sjd.jump import VPMatchedGaussianJump
    from sticky.models.sjd.sdes import make_beta
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    e = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 8))
    assert jump.apply_blur(e) is e


def test_apply_blur_uses_kernel_when_set():
    import dataclasses
    from sticky.models.sjd.jump import VPMatchedGaussianJump
    from sticky.models.sjd.sdes import make_beta
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    kernel = jnp.eye(16, dtype=jnp.float32)
    jump_blur = dataclasses.replace(jump, blur_kernel=kernel)
    e = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 8))
    out = jump_blur.apply_blur(e)
    # Identity kernel should produce identical values (but not necessarily same object).
    np.testing.assert_allclose(np.asarray(out), np.asarray(e), atol=1e-6)


def test_jump_with_kernel_is_hashable():
    """Frozen dataclass with a JAX-array field must be hashable so it can be
    passed as a jit static_argnames target or used as a dict key without crash."""
    import dataclasses
    from sticky.models.sjd.jump import VPMatchedGaussianJump
    from sticky.models.sjd.sdes import make_beta
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    jump_blur = dataclasses.replace(jump, blur_kernel=jnp.eye(8, dtype=jnp.float32))
    # Should not raise.
    h = hash(jump_blur)
    assert isinstance(h, int)


def test_jump_with_kernel_eq_does_not_crash():
    """Comparing two kernel-bearing jumps must not raise the JAX boolean-coercion
    ValueError. With eq=False, equality is identity-based — unequal unless `is`."""
    import dataclasses
    from sticky.models.sjd.jump import VPMatchedGaussianJump
    from sticky.models.sjd.sdes import make_beta
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    kernel = jnp.eye(8, dtype=jnp.float32)
    jump_a = dataclasses.replace(jump, blur_kernel=kernel)
    jump_b = dataclasses.replace(jump, blur_kernel=kernel)
    # Different objects, identity-based equality -> not equal. The point is no crash.
    result = (jump_a == jump_b)
    assert result is False  # object.__eq__ returns False for distinct instances
    assert (jump_a == jump_a) is True


def test_sample_pair_no_blur_unchanged():
    """Determinism check: two calls to sample_pair with equivalent (replace'd)
    no-blur jumps and the same PRNG key produce identical outputs. This is NOT
    a snapshot test against pre-Task-7 behavior — that backward-compat
    guarantee is established by-construction in
    test_apply_blur_is_python_identity_when_disabled (apply_blur returns the
    same Python object when blur_kernel is None)."""
    import dataclasses
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)

    B, N, d = 4, 16, 3
    key = jax.random.PRNGKey(7)
    k_anchor, k_t, k_pair = jax.random.split(key, 3)
    x0 = jax.random.normal(k_anchor, (B, N, d))
    t = jax.random.uniform(k_t, (B,), minval=0.05, maxval=0.95)

    xt_a, mask_a = sample_pair(k_pair, x0, t, beta, hazard, jump)
    # Replace with same default jump (still blur_kernel=None) and call again.
    jump2 = dataclasses.replace(jump)
    xt_b, mask_b = sample_pair(k_pair, x0, t, beta, hazard, jump2)
    np.testing.assert_array_equal(np.asarray(xt_a), np.asarray(xt_b))
    np.testing.assert_array_equal(np.asarray(mask_a), np.asarray(mask_b))


def test_sample_pair_with_gaussian_blur_runs_and_preserves_never_unstuck():
    """Smoke: gaussian_position_kernel doesn't break sample_pair, and the
    empirical never-unstuck mass still matches hazard.surv(t)."""
    import dataclasses
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    jump = dataclasses.replace(
        jump, blur_kernel=gaussian_position_kernel(16, sigma=1.0)
    )

    N_samples = 4096
    L, d = 16, 3
    a = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.broadcast_to(a, (N_samples, L, d))
    t_val = 0.6
    t = jnp.full((N_samples,), t_val, dtype=jnp.float32)

    xt, mask = sample_pair(jax.random.PRNGKey(7), x0, t, beta, hazard, jump)
    assert jnp.isfinite(xt).all()
    assert xt.shape == (N_samples, L, d)
    assert mask.shape == (N_samples, L)

    expected = float(hazard.surv(jnp.asarray(t_val, dtype=jnp.float32)))
    actual = float(jnp.mean(mask.astype(jnp.float32)))
    np.testing.assert_allclose(actual, expected, atol=2e-2)


def test_sample_pair_with_sudoku_blur_runs_and_preserves_never_unstuck():
    """Smoke: sudoku_constraint_kernel works with sample_pair on 81-cell anchors."""
    import dataclasses
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.7)
    jump = dataclasses.replace(
        jump, blur_kernel=sudoku_constraint_kernel(sigma=1.5)
    )

    N_samples = 4096
    L, d = 81, 3
    a = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.broadcast_to(a, (N_samples, L, d))
    t_val = 0.6
    t = jnp.full((N_samples,), t_val, dtype=jnp.float32)

    xt, mask = sample_pair(jax.random.PRNGKey(11), x0, t, beta, hazard, jump)
    assert jnp.isfinite(xt).all()
    assert xt.shape == (N_samples, L, d)
    assert mask.shape == (N_samples, L)

    expected = float(hazard.surv(jnp.asarray(t_val, dtype=jnp.float32)))
    actual = float(jnp.mean(mask.astype(jnp.float32)))
    np.testing.assert_allclose(actual, expected, atol=2e-2)


def _make_factory_cfg(blur_block: dict | None, *, data_shape=(16,)) -> "OmegaConf":
    """Build a minimal Hydra-like config that satisfies _sjd_schedule_kwargs.

    Phase 1.B: blur lives at the orthogonal `forward.blur` group, NOT under
    `forward.jump`. seq_len is derived from `dataset.data_shape` inside the
    factory helper.
    """
    from omegaconf import OmegaConf
    forward = {
        "beta": {
            "_target_": "sticky.models.sjd.sdes.make_beta",
            "beta_min": 0.1, "beta_max": 20.0, "T": 1.0,
        },
        "hazard": {
            "_target_": "sticky.models.sjd.hazard.make_hazard_linear_time",
            "kappa": 1.5,
        },
        "jump": {
            "_target_": "sticky.models.sjd.jump.VPMatchedGaussianJump",
            "eta": 0.7,
            "std_floor": 1e-3,
            "clip": None,
        },
    }
    if blur_block is not None:
        forward["blur"] = blur_block
    return OmegaConf.create({
        "forward": forward,
        "dataset": {"vocab_size": 10, "data_shape": list(data_shape)},
        "training": {},
        "sampler": {},
    })


def test_factory_strips_blur_and_attaches_kernel():
    """End-to-end: an enabled blur cfg at forward.blur attaches a kernel to the
    jump after instantiate (the jump schema itself never sees `blur`)."""
    from sticky.tasks.factory import _sjd_schedule_kwargs

    cfg = _make_factory_cfg({
        "enabled": True,
        "kind": "gaussian_1d",
        "sigma": 1.0,
        "include_self": True,
    })
    out = _sjd_schedule_kwargs(cfg)
    assert out["jump"].blur_kernel is not None
    assert out["jump"].blur_kernel.shape == (16, 16)


def test_factory_no_blur_yields_none_kernel():
    """Explicit `forward.blur` with enabled=False produces no kernel."""
    from sticky.tasks.factory import _sjd_schedule_kwargs

    cfg = _make_factory_cfg({
        "enabled": False, "kind": None, "sigma": 1.0,
        "include_self": True, "include_row": True,
        "include_col": True, "include_box": True,
    })
    out = _sjd_schedule_kwargs(cfg)
    assert out["jump"].blur_kernel is None


def test_factory_missing_blur_yields_none_kernel():
    """Missing `forward.blur` (the typical baseline path) produces no kernel."""
    from sticky.tasks.factory import _sjd_schedule_kwargs

    cfg = _make_factory_cfg(blur_block=None)
    out = _sjd_schedule_kwargs(cfg)
    assert out["jump"].blur_kernel is None


def test_factory_rejects_blur_on_image_task():
    """Phase 1 doesn't support blur for 2D image (CIFAR10) tasks. The factory
    must fail loudly rather than silently attach a kernel of the wrong shape."""
    from omegaconf import OmegaConf
    from sticky.tasks.factory import _build_tfds_sjd_task

    # Minimal cfg: only forward.blur is needed; the guard fires before
    # _tfds_image_dataset_kwargs or _sjd_schedule_kwargs are reached.
    cfg = OmegaConf.create({
        "forward": {
            "blur": {
                "enabled": True,
                "kind": "sudoku_constraint",
                "sigma": 1.5,
                "include_row": True,
                "include_col": True,
                "include_box": True,
            },
        },
    })
    with pytest.raises(ValueError, match="2D image"):
        _build_tfds_sjd_task(cfg, task_name="sjd_cifar10")
