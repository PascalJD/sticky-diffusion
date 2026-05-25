"""Verify that hazard_weighting=learned_e2e is recognised and threaded
through to SJDTaskBase as objective='elbo_eta1' with the right kwargs."""
import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sticky.models.sjd.freq_weighting import hazard_weighting_mode
from sticky.tasks.factory import _sjd_dhm_kwargs

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config")
)


def _compose_sudoku_with_e2e_hw():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=sudoku/sjd_sudoku",
                "forward/hazard_weighting@experiment.forward.hazard_weighting=learned_e2e",
            ],
        )
    return cfg.experiment


def test_learned_e2e_mode_recognised():
    cfg = _compose_sudoku_with_e2e_hw()
    mode = hazard_weighting_mode(cfg.forward.hazard_weighting)
    assert mode == "learned_e2e", f"expected mode='learned_e2e', got {mode!r}"


def test_dhm_kwargs_dispatch_e2e_objective():
    cfg = _compose_sudoku_with_e2e_hw()
    kwargs = _sjd_dhm_kwargs(cfg)
    assert kwargs["objective"] == "elbo_eta1", (
        f"expected objective='elbo_eta1', got {kwargs.get('objective')!r}"
    )
    assert kwargs["learn_log_w"] is True
    assert kwargs["anchor_log_w"] is None  # learned via params, not preloaded
    assert kwargs["rb_weight"] == 1.0
    assert kwargs["rb_share_sample"] is True
    assert kwargs["prior_strength"] == 0.0
    GlobalHydra.instance().clear()


def test_dhm_kwargs_default_ce_objective():
    """Baseline: an experiment without hazard_weighting=learned_e2e gets
    objective='ce' (the existing path), preserving bit-exact backward compat."""
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["experiment=sudoku/sjd_sudoku"],
            )
        kwargs = _sjd_dhm_kwargs(cfg.experiment)
        assert kwargs["objective"] == "ce", (
            f"expected objective='ce' for default config, got {kwargs.get('objective')!r}"
        )
    finally:
        GlobalHydra.instance().clear()
