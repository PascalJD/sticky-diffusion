from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


def _fixture(vocab_size: int = 4, seq_len: int = 6, B: int = 8, d: int = 3):
    table = jax.random.normal(
        jax.random.PRNGKey(0), (vocab_size, d), dtype=jnp.float32
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(1), (B, seq_len), 0, vocab_size
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]

    def apply_fn(params, xt, t_img):
        del params, xt, t_img
        return jnp.zeros((B, seq_len, vocab_size), dtype=jnp.float32), {}

    return dict(
        apply_fn=apply_fn,
        params={},
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=make_beta(beta_min=0.1, beta_max=20.0, T=1.0),
        hazard=None,
        T=1.0,
    )


def test_uniform_defaults_regression():
    kwargs = _fixture()
    key = jax.random.PRNGKey(42)
    loss_default, metrics_default = ce_allocation_loss(key=key, **kwargs)
    loss_explicit, metrics_explicit = ce_allocation_loss(
        key=key,
        time_sampling="uniform",
        loss_weighting="uniform",
        **kwargs,
    )
    np.testing.assert_array_equal(
        np.asarray(loss_default), np.asarray(loss_explicit)
    )
    for k in metrics_default:
        np.testing.assert_array_equal(
            np.asarray(metrics_default[k]), np.asarray(metrics_explicit[k])
        )


def test_antithetic_pairs_times():
    B = 8
    T = 1.0
    kwargs = _fixture(B=B)
    key = jax.random.PRNGKey(7)

    key_t, _, _ = jax.random.split(key, 3)
    half_B = B // 2
    t_half = jax.random.uniform(key_t, shape=(half_B,), minval=0.0, maxval=T)
    expected_t = jnp.concatenate([t_half, T - t_half], axis=0)

    _, metrics = ce_allocation_loss(
        key=key, time_sampling="antithetic", **kwargs
    )

    np.testing.assert_allclose(
        float(metrics["CE/time_mean"]), float(jnp.mean(expected_t)), rtol=1e-5
    )
    np.testing.assert_allclose(
        float(metrics["CE/time_std"]), float(jnp.std(expected_t)), rtol=1e-5
    )


def test_antithetic_handles_odd_batch():
    kwargs = _fixture(B=31)
    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(13), time_sampling="antithetic", **kwargs
    )
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["CE/time_mean"]))


def test_alpha_deriv_weights_positive_finite():
    kwargs = _fixture(B=16)
    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(3),
        loss_weighting="alpha_deriv",
        **kwargs,
    )
    assert bool(jnp.isfinite(loss).all())
    assert "CE/loss_weight_mean" in metrics
    assert "CE/loss_weight_std" in metrics
    assert float(metrics["CE/loss_weight_mean"]) > 0.0
    assert bool(jnp.isfinite(metrics["CE/loss_weight_mean"]))
    assert bool(jnp.isfinite(metrics["CE/loss_weight_std"]))


def test_combined_features_scalar_finite():
    kwargs = _fixture(B=8)
    loss, _ = ce_allocation_loss(
        key=jax.random.PRNGKey(11),
        time_sampling="antithetic",
        loss_weighting="alpha_deriv",
        **kwargs,
    )
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert bool(jnp.isfinite(loss))


def test_uniform_has_no_loss_weight_metric():
    _, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(5), **_fixture()
    )
    assert "CE/loss_weight_mean" not in metrics
    assert "CE/loss_weight_std" not in metrics
    assert "CE/time_mean" in metrics
    assert "CE/time_std" in metrics


class _ConstantHazard:
    """Forward hazard whose survival is a fixed scalar regardless of t."""

    def __init__(self, surv_value: float):
        self._surv = float(surv_value)

    def surv(self, t):
        return jnp.full_like(jnp.asarray(t, dtype=jnp.float32), self._surv)


def _order_fixture(B=4, seq_len=6, vocab_size=4, d=3):
    kwargs = _fixture(vocab_size=vocab_size, seq_len=seq_len, B=B, d=d)
    kwargs["hazard"] = _ConstantHazard(0.5)
    return kwargs


def test_order_conditioning_none_with_rank_matches_baseline():
    kwargs = _order_fixture()
    B, seq_len = kwargs["x0_idx"].shape
    rng = np.random.default_rng(0)
    rank = jnp.asarray(rng.uniform(size=(B, seq_len)).astype(np.float32))
    key = jax.random.PRNGKey(17)
    loss_base, metrics_base = ce_allocation_loss(key=key, **kwargs)
    loss_order_none, metrics_order_none = ce_allocation_loss(
        key=key,
        solver_rank=rank,
        order_conditioning="none",
        **kwargs,
    )
    np.testing.assert_array_equal(np.asarray(loss_base), np.asarray(loss_order_none))
    assert set(metrics_base.keys()) == set(metrics_order_none.keys())
    for k in metrics_base:
        np.testing.assert_array_equal(
            np.asarray(metrics_base[k]), np.asarray(metrics_order_none[k])
        )


def test_order_conditioning_uniform_weight_matches_baseline():
    kwargs = _order_fixture()
    B, seq_len = kwargs["x0_idx"].shape
    rng = np.random.default_rng(1)
    rank = jnp.asarray(rng.uniform(size=(B, seq_len)).astype(np.float32))
    key = jax.random.PRNGKey(29)
    loss_base, _ = ce_allocation_loss(key=key, **kwargs)
    loss_order, metrics_order = ce_allocation_loss(
        key=key,
        solver_rank=rank,
        order_conditioning="solver",
        order_w_min=1.0,
        order_w_max=1.0,
        **kwargs,
    )
    # w_i == 1 everywhere -> S^1 == S -> identical draws and loss.
    np.testing.assert_allclose(
        float(loss_base), float(loss_order), rtol=0.0, atol=1e-6
    )
    assert "CE/order_w_mean" in metrics_order
    assert abs(float(metrics_order["CE/order_w_mean"]) - 1.0) < 1e-6


def test_mode_none_matches_legacy_defaults():
    """forward_site_hazard_mode='none' with all teacher kwargs defaulted must
    produce bit-identical outputs to the legacy call."""
    kwargs = _order_fixture()
    key = jax.random.PRNGKey(101)
    loss_legacy, metrics_legacy = ce_allocation_loss(key=key, **kwargs)
    loss_new, metrics_new = ce_allocation_loss(
        key=key,
        forward_site_hazard_mode="none",
        **kwargs,
    )
    np.testing.assert_array_equal(np.asarray(loss_legacy), np.asarray(loss_new))
    assert set(metrics_legacy.keys()) == set(metrics_new.keys())
    for k in metrics_legacy:
        np.testing.assert_array_equal(
            np.asarray(metrics_legacy[k]), np.asarray(metrics_new[k])
        )


def _teacher_fixture(vocab_size: int = 4, seq_len: int = 6, B: int = 4, d: int = 3):
    kwargs = _fixture(vocab_size=vocab_size, seq_len=seq_len, B=B, d=d)
    kwargs["hazard"] = _ConstantHazard(0.5)
    # Teacher anchors identical to student's so shapes/dtypes line up.
    kwargs["x0_anchor_teacher"] = kwargs["x0_anchor"]

    # Per-position sharpness so confidence varies deterministically across sites.
    sharpness = jnp.linspace(0.0, 5.0, seq_len, dtype=jnp.float32)

    def teacher_apply_fn(p, xt, t_img):
        del p, t_img
        # Peak at position-dependent class with position-dependent temperature.
        target = jnp.arange(seq_len, dtype=jnp.int32) % vocab_size
        one_hot = jax.nn.one_hot(target, vocab_size, dtype=jnp.float32)  # (seq_len, V)
        logits = one_hot * sharpness[:, None]  # (seq_len, V)
        logits = jnp.broadcast_to(logits[None, :, :], (B, seq_len, vocab_size))
        return logits, {}

    kwargs["teacher_apply_fn"] = teacher_apply_fn
    kwargs["teacher_params"] = {}
    return kwargs


def test_teacher_margin_shapes_and_finite():
    kwargs = _teacher_fixture()
    loss, metrics = ce_allocation_loss(
        key=jax.random.PRNGKey(7),
        forward_site_hazard_mode="teacher_margin",
        teacher_w_min=0.5,
        teacher_w_max=2.0,
        **kwargs,
    )
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    for k in (
        "CE/teacher_conf_mean",
        "CE/teacher_conf_std",
        "CE/teacher_conf_q25",
        "CE/teacher_conf_q75",
        "CE/teacher_w_mean",
        "CE/teacher_w_std",
        "CE/teacher_probe_committed_fraction",
        "CE/teacher_final_committed_fraction",
    ):
        assert k in metrics, f"missing metric {k}"
        assert bool(jnp.isfinite(metrics[k]))


def test_teacher_margin_requires_teacher_inputs():
    kwargs = _teacher_fixture()
    # Strip one required input at a time; each case should raise.
    for missing in ("teacher_params", "teacher_apply_fn", "x0_anchor_teacher"):
        broken = dict(kwargs)
        broken[missing] = None
        try:
            ce_allocation_loss(
                key=jax.random.PRNGKey(0),
                forward_site_hazard_mode="teacher_margin",
                **broken,
            )
        except ValueError:
            continue
        else:
            raise AssertionError(f"expected ValueError when {missing} is None")


def test_teacher_margin_gradient_does_not_flow_through_teacher():
    """Grad through the teacher probe must be blocked by stop_gradient."""
    kwargs = _teacher_fixture()

    def loss_fn(teacher_scalar):
        def teacher_apply(p, xt, t_img):
            del p, t_img
            B_, seq = xt.shape[:2]
            target = jnp.arange(seq, dtype=jnp.int32) % 4
            one_hot = jax.nn.one_hot(target, 4, dtype=jnp.float32)
            sharp = jnp.linspace(0.0, 5.0, seq, dtype=jnp.float32) * teacher_scalar
            logits = one_hot * sharp[:, None]
            logits = jnp.broadcast_to(logits[None, :, :], (B_, seq, 4))
            return logits, {}

        kw = dict(kwargs)
        kw["teacher_apply_fn"] = teacher_apply
        loss, _ = ce_allocation_loss(
            key=jax.random.PRNGKey(0),
            forward_site_hazard_mode="teacher_margin",
            **kw,
        )
        return loss

    grad = jax.grad(loss_fn)(jnp.asarray(3.14, dtype=jnp.float32))
    assert float(grad) == 0.0


def test_order_conditioning_commits_easier_cells_more_often():
    # All-zero rank cells should see p_committed = S^{w_min}, which is
    # larger than p_committed = S^{w_max} at all-one rank cells when S < 1.
    kwargs = _order_fixture(B=256, seq_len=2)
    B, seq_len = kwargs["x0_idx"].shape
    rank = jnp.zeros((B, seq_len), dtype=jnp.float32)
    rank = rank.at[:, 0].set(0.0)  # easy
    rank = rank.at[:, 1].set(1.0)  # hard
    key = jax.random.PRNGKey(31)
    _, metrics = ce_allocation_loss(
        key=key,
        solver_rank=rank,
        order_conditioning="solver",
        order_w_min=0.2,
        order_w_max=5.0,
        **kwargs,
    )
    # With survival ~ 0.5 and big w_max=5, hard cells almost never commit;
    # with w_min=0.2, easy cells commit far more often.
    easy = float(metrics["CE/order_committed_easy"])
    hard = float(metrics["CE/order_committed_hard"])
    assert easy > hard + 0.3  # large gap expected by construction
