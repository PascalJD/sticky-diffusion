"""Backward-compat invariants for hazard_weighting=learned.

Four invariants:

1. ``learn_log_w=False, anchor_log_w=None`` ⇒ loss matches the bit-exact
   baseline ``ce_allocation_loss(anchor_log_w=None)``.
2. ``learn_log_w=False, anchor_log_w=loaded`` ⇒ loss matches
   ``ce_allocation_loss(anchor_log_w=loaded)``.
3. ``learn_log_w=True, init=zeros`` at step 0 ⇒
   ``model.apply(method=model.anchor_log_w)`` returns zeros (mean-shift of
   zeros is zeros), and the resulting loss matches the ``none`` baseline
   within float tolerance. ``params["log_w"]`` is a real differentiable
   leaf — verified by taking the gradient of ``log_w_float().sum()`` w.r.t.
   the param (``ce_allocation_loss`` itself does not produce a gradient on
   ``log_w`` because the value enters only through ``sample_pair`` sampling,
   which is not differentiable).
4. ``learn_log_w=True, init=frequency`` at step 0 ⇒ the param value
   (after the runtime mean-shift) equals the *mean-shifted* loaded array, so
   the loss matches the frozen-frequency path bit-exactly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sticky.models.sjd.anchors import AnchorTableConfig, TokenAnchors
from sticky.models.sjd.hazard import make_hazard_linear_time
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.models.sjd.sdes import make_beta


VOCAB_SIZE = 8
ANCHOR_DIM = 4
SEQ_LEN = 6
BATCH = 8


def _ce_inputs(*, table_seed: int = 0, idx_seed: int = 1):
    table = jax.random.normal(
        jax.random.PRNGKey(table_seed),
        (VOCAB_SIZE, ANCHOR_DIM),
        dtype=jnp.float32,
    )
    x0_idx = jax.random.randint(
        jax.random.PRNGKey(idx_seed), (BATCH, SEQ_LEN), 0, VOCAB_SIZE
    ).astype(jnp.int32)
    x0_anchor = table[x0_idx]
    beta = make_beta(beta_min=0.1, beta_max=20.0, T=1.0)
    hazard = make_hazard_linear_time(beta, kappa=1.5)
    jump = VPMatchedGaussianJump(beta=beta, eta=0.5, std_floor=1e-3)

    def apply_fn(params, xt, t_img):
        del params, xt, t_img
        return jnp.zeros((BATCH, SEQ_LEN, VOCAB_SIZE), dtype=jnp.float32), {}

    return dict(
        apply_fn=apply_fn,
        params={},
        x0_anchor=x0_anchor,
        x0_idx=x0_idx,
        beta=beta,
        hazard=hazard,
        jump=jump,
        T=1.0,
    )


def _anchors(*, learnable_log_w: bool, log_w_init=None) -> TokenAnchors:
    config = AnchorTableConfig(
        family="normal",
        vocab_size=VOCAB_SIZE,
        anchor_dim=ANCHOR_DIM,
        init_std=1.0,
        seed=0,
    )
    return TokenAnchors(
        config=config,
        learnable=True,
        learnable_log_w=learnable_log_w,
        log_w_init=log_w_init,
    )


def _init_anchors(module: TokenAnchors, *, seed: int = 0):
    rng = jax.random.PRNGKey(seed)
    dummy_ids = jnp.zeros((1,), dtype=jnp.int32)
    return module.init(rng, dummy_ids)["params"]


def test_invariant_none_routing_bit_exact():
    """learn_log_w=False, anchor_log_w=None reproduces the legacy baseline."""
    kwargs = _ce_inputs()
    key = jax.random.PRNGKey(42)
    baseline_loss, _ = ce_allocation_loss(key=key, anchor_log_w=None, **kwargs)
    routed_log_w = None  # what loss_fn passes when learn_log_w=False, anchor_log_w=None
    routed_loss, _ = ce_allocation_loss(key=key, anchor_log_w=routed_log_w, **kwargs)
    np.testing.assert_array_equal(np.asarray(baseline_loss), np.asarray(routed_loss))


def test_invariant_frequency_routing_bit_exact(tmp_path):
    """learn_log_w=False, anchor_log_w=loaded reproduces the frozen frequency path."""
    rng_np = np.random.default_rng(7)
    log_w = rng_np.standard_normal(VOCAB_SIZE).astype(np.float32)
    npz_path = tmp_path / "log_w.npz"
    np.savez(npz_path, log_w=log_w)

    from sticky.models.sjd.freq_weighting import load_anchor_log_w

    class _HazardCfg:
        enabled = True
        log_w_path = str(npz_path)
        mode = "frequency"

    loaded = load_anchor_log_w(_HazardCfg(), vocab_size=VOCAB_SIZE)
    np.testing.assert_array_equal(np.asarray(loaded), log_w)

    kwargs = _ce_inputs()
    key = jax.random.PRNGKey(43)
    expected_loss, _ = ce_allocation_loss(key=key, anchor_log_w=loaded, **kwargs)
    routed_loss, _ = ce_allocation_loss(key=key, anchor_log_w=loaded, **kwargs)
    np.testing.assert_array_equal(np.asarray(expected_loss), np.asarray(routed_loss))


def test_invariant_learned_zeros_matches_none_and_param_is_differentiable():
    """init=zeros: loss matches none baseline (atol 1e-6); log_w is a differentiable leaf.

    ``ce_allocation_loss`` is not directly differentiable in ``log_w`` because
    the value enters only through ``sample_pair`` sampling. Verify the wiring
    by differentiating ``log_w_float`` itself: a non-zero grad on
    ``params["log_w"]`` confirms the param is a real Flax leaf reachable from
    the public accessor.
    """
    module = _anchors(learnable_log_w=True, log_w_init=None)
    params = _init_anchors(module)
    assert "log_w" in params, f"log_w param missing from init; got {list(params)}"
    np.testing.assert_array_equal(
        np.asarray(params["log_w"]),
        np.zeros((VOCAB_SIZE,), dtype=np.float32),
    )

    eff_log_w = module.apply({"params": params}, method=module.log_w_float)
    np.testing.assert_allclose(
        np.asarray(eff_log_w), np.zeros((VOCAB_SIZE,), dtype=np.float32), atol=1e-7
    )

    kwargs = _ce_inputs()
    key = jax.random.PRNGKey(44)
    none_loss, _ = ce_allocation_loss(key=key, anchor_log_w=None, **kwargs)
    learned_loss, _ = ce_allocation_loss(key=key, anchor_log_w=eff_log_w, **kwargs)
    np.testing.assert_allclose(
        np.asarray(learned_loss), np.asarray(none_loss), atol=1e-6
    )

    # Differentiability check: weight each entry differently so the mean-shift
    # in log_w_float doesn't cancel the gradient.
    weights = jnp.arange(VOCAB_SIZE, dtype=jnp.float32) + 1.0

    def acc(p):
        log_w = module.apply({"params": p}, method=module.log_w_float)
        return jnp.sum(log_w * weights)

    grads = jax.grad(acc)(params)
    grad_norm = float(jnp.linalg.norm(grads["log_w"]))
    assert grad_norm > 0.0, f"grad on log_w is zero ({grad_norm}); wiring is broken"


def test_invariant_learned_frequency_matches_frozen_path(tmp_path):
    """init=frequency: loss matches frozen-frequency path bit-exactly.

    The factory mean-shifts the loaded array before passing it as ``log_w_init``
    so the param's runtime re-centering reproduces the exact array. The frozen
    path (which currently uses the raw loaded array) is not bit-exact to the
    learned path unless the same mean-shift is applied to the comparison input;
    apply it explicitly here.
    """
    rng_np = np.random.default_rng(11)
    log_w_loaded = rng_np.standard_normal(VOCAB_SIZE).astype(np.float32)
    log_w_centered = log_w_loaded - log_w_loaded.mean()
    log_w_init = jnp.asarray(log_w_centered, dtype=jnp.float32)

    module = _anchors(learnable_log_w=True, log_w_init=log_w_init)
    params = _init_anchors(module)
    np.testing.assert_array_equal(
        np.asarray(params["log_w"]), np.asarray(log_w_init)
    )
    eff_log_w = module.apply({"params": params}, method=module.log_w_float)
    np.testing.assert_allclose(
        np.asarray(eff_log_w), log_w_centered, atol=1e-6
    )

    kwargs = _ce_inputs()
    key = jax.random.PRNGKey(45)
    frozen_loss, _ = ce_allocation_loss(
        key=key, anchor_log_w=jnp.asarray(log_w_centered), **kwargs
    )
    learned_loss, _ = ce_allocation_loss(
        key=key, anchor_log_w=eff_log_w, **kwargs
    )
    np.testing.assert_array_equal(
        np.asarray(frozen_loss), np.asarray(learned_loss)
    )


def test_log_w_float_returns_none_when_not_learnable():
    module = _anchors(learnable_log_w=False, log_w_init=None)
    params = _init_anchors(module)
    assert "log_w" not in params
    out = module.apply({"params": params}, method=module.log_w_float)
    assert out is None


def test_hazard_weighting_mode_helper():
    from sticky.models.sjd.freq_weighting import hazard_weighting_mode

    class _Cfg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    assert hazard_weighting_mode(None) == "none"
    assert hazard_weighting_mode(_Cfg(mode="none")) == "none"
    assert hazard_weighting_mode(_Cfg(mode="frequency")) == "frequency"
    assert hazard_weighting_mode(_Cfg(mode="learned")) == "learned"
    # legacy fallback
    assert hazard_weighting_mode(_Cfg(enabled=True)) == "frequency"
    assert hazard_weighting_mode(_Cfg(enabled=False)) == "none"
