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
    # learned_e2e.yaml ships a weak Omega(w) prior (lambda_Omega=0.01) per
    # corrected appendix alg:sjd-e2e-eta1 note (b); the clamp on log_w is
    # the load-bearing safety, the prior just nudges back to w=1.
    assert kwargs["prior_strength"] == 0.01
    # Default for the World-1-vs-World-2 score-term gating flag: False
    # preserves the v3 ELBO behavior (current published objective). The
    # CE-only condition flips it to True via an experiment-level override.
    assert kwargs["score_w_stop_gradient"] is False
    # Default for the L_RB theta-gradient routing flag: False preserves the
    # full-ELBO theta path (theta trains on L_CE + L_RB). frozenw-clean /
    # CE-clean flip it to True via an experiment-level override.
    assert kwargs["rb_theta_stop_gradient"] is False
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


def test_sudoku_world_test_three_experiments_compose_distinctly():
    """The three Sudoku World-1-vs-World-2 experiment YAMLs must compose to
    three distinct (objective, learn_log_w, score_w_stop_gradient) tuples.
    Catches Hydra/OmegaConf override mistakes before any training launch."""
    expected = {
        "sudoku/sjd_sudoku_e2e_baseline": ("ce", False, False),
        "sudoku/sjd_sudoku_e2e":          ("elbo_eta1", True, False),
        "sudoku/sjd_sudoku_e2e_ce":       ("elbo_eta1", True, True),
    }
    for exp_name, (obj, learn, score_sg) in expected.items():
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"experiment={exp_name}"],
                )
            kwargs = _sjd_dhm_kwargs(cfg.experiment)
            assert kwargs["objective"] == obj, (
                f"{exp_name}: objective expected {obj!r}, "
                f"got {kwargs['objective']!r}"
            )
            assert kwargs["learn_log_w"] is learn, (
                f"{exp_name}: learn_log_w expected {learn}, "
                f"got {kwargs['learn_log_w']}"
            )
            assert kwargs["score_w_stop_gradient"] is score_sg, (
                f"{exp_name}: score_w_stop_gradient expected {score_sg}, "
                f"got {kwargs['score_w_stop_gradient']}"
            )
        finally:
            GlobalHydra.instance().clear()
