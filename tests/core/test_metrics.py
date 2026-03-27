from __future__ import annotations

import sys
from types import ModuleType

import jax.numpy as jnp

from sticky.core.metrics import scale_loss_metrics_to_bits


def _import_md4_utils():
    created_modules: list[str] = []

    def _ensure_module(name: str, *, attr_name: str | None = None, attr_value=None):
        if name in sys.modules:
            return sys.modules[name]
        module = ModuleType(name)
        if attr_name is not None:
            setattr(module, attr_name, attr_value)
        sys.modules[name] = module
        created_modules.append(name)
        return module

    _ensure_module("distrax", attr_name="Distribution", attr_value=object)
    matplotlib_mod = _ensure_module("matplotlib")
    pyplot_mod = _ensure_module("matplotlib.pyplot")
    if not hasattr(matplotlib_mod, "pyplot"):
        matplotlib_mod.pyplot = pyplot_mod
    orbax_mod = _ensure_module("orbax")
    checkpoint_mod = _ensure_module("orbax.checkpoint")
    if not hasattr(checkpoint_mod, "CheckpointManager"):
        checkpoint_mod.CheckpointManager = object
    if not hasattr(orbax_mod, "checkpoint"):
        orbax_mod.checkpoint = checkpoint_mod
    _ensure_module("seaborn")

    try:
        from sticky.models.md4 import utils as md4_utils

        return md4_utils
    finally:
        for name in reversed(created_modules):
            sys.modules.pop(name, None)


def test_scale_loss_metrics_to_bits_scales_only_loss_keys():
    metrics = {
        "loss": jnp.asarray(12.0, dtype=jnp.float32),
        "loss_diff": jnp.asarray(6.0, dtype=jnp.float32),
        "loss_aux": jnp.asarray(3.0, dtype=jnp.float32),
        "mask_frac": jnp.asarray(0.25, dtype=jnp.float32),
        "sigma_mean": jnp.asarray(1.5, dtype=jnp.float32),
        "t_mean": jnp.asarray(0.75, dtype=jnp.float32),
    }

    scaled = scale_loss_metrics_to_bits(metrics, (2, 3))
    expected_scale = 1.0 / (6.0 * jnp.log(2.0))

    assert jnp.allclose(scaled["loss"], metrics["loss"] * expected_scale)
    assert jnp.allclose(scaled["loss_diff"], metrics["loss_diff"] * expected_scale)
    assert jnp.allclose(scaled["loss_aux"], metrics["loss_aux"] * expected_scale)
    assert jnp.array_equal(scaled["mask_frac"], metrics["mask_frac"])
    assert jnp.array_equal(scaled["sigma_mean"], metrics["sigma_mean"])
    assert jnp.array_equal(scaled["t_mean"], metrics["t_mean"])


def test_md4_loss2bpt_is_a_compatibility_wrapper():
    md4_utils = _import_md4_utils()
    metrics = {
        "loss": jnp.asarray(8.0, dtype=jnp.float32),
        "loss_prior": jnp.asarray(2.0, dtype=jnp.float32),
        "selected_count_total": jnp.asarray(4.0, dtype=jnp.float32),
    }

    direct = scale_loss_metrics_to_bits(metrics, (4,))
    wrapped = md4_utils.loss2bpt(metrics, (4,))

    assert set(wrapped) == set(direct)
    for key in direct:
        assert jnp.allclose(wrapped[key], direct[key])
