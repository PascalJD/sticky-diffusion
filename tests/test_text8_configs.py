"""Text8 FIXED experiment resolves with the right invariants and ~8M params."""
import os

import jax
import jax.numpy as jnp
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sticky.models.factory import build_model
from sticky.rng import make_rng
from sticky.tasks.factory import build_task
from sticky.training.state import init_state

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


def _compose():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=text8/sjd_text8_fixed"],
        )
    return cfg.experiment


def test_fixed_forward_invariants():
    exp = _compose()
    assert str(exp.forward.hazard_weighting.mode) == "none"
    assert abs(float(exp.forward.jump.eta) - 1.0) < 1e-9
    assert bool(exp.forward.blur.enabled) is False
    assert abs(float(exp.forward.hazard.p) - 1.0) < 1e-9
    GlobalHydra.instance().clear()


def test_data_and_optim_match_candi():
    exp = _compose()
    assert int(exp.dataset.seq_len) == 128
    assert int(exp.dataset.vocab_size) == 27
    assert abs(float(exp.optim.learning_rate) - 1.0e-3) < 1e-12
    assert int(exp.optim.warmup_steps) == 0
    assert float(exp.optim.weight_decay) == 0.0
    assert abs(float(exp.training.ema_rate) - 0.9999) < 1e-12
    GlobalHydra.instance().clear()


def test_param_count_is_candi_tiny_scale():
    exp = _compose()
    task = build_task(exp)
    model = build_model(
        exp,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    rng = make_rng(int(exp.training.seed))
    state, _ = init_state(exp, model, rng)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
    # CANDI "tiny" is ~8M. The repo transformer derives its SwiGLU FFN width
    # internally (not CANDI's explicit mlp_ratio=4), so accept a band around 8M
    # and surface the exact count for the pre-gate fidelity check.
    print(f"[text8] FIXED model param count = {n_params:,}")
    assert 4_000_000 < n_params < 14_000_000, f"unexpected param count {n_params}"
    GlobalHydra.instance().clear()
