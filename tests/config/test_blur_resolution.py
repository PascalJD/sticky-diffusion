from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir


def _config_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config")
    )


@pytest.mark.parametrize("experiment", [
    "sudoku/sjd_sudoku",
    "sudoku/sjd_sudoku_blur",
    "openwebtext/sjd_openwebtext",
    "openwebtext/sjd_openwebtext_blur",
    "cifar10/sjd_cifar10",
    "imagenet64/sjd_imagenet64",
])
def test_experiment_resolves_with_blur_key(experiment):
    """Every SJD-family experiment must expose cfg.experiment.forward.blur after
    composition (either as 'none' or as the kernel-on variant). The factory's
    cfg.forward.get('blur', None) call only finds it if the experiment composes
    the forward/blur group."""
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
        forward = cfg.experiment.forward
        assert "blur" in forward, f"{experiment}: cfg.experiment.forward.blur missing"
        assert "enabled" in forward.blur
        assert "kind" in forward.blur


def test_owt_blur_experiment_has_kernel_kind():
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=openwebtext/sjd_openwebtext_blur"],
        )
        blur = cfg.experiment.forward.blur
        assert blur.enabled is True
        assert blur.kind == "gaussian_1d"
        assert blur.sigma == 1.5


def test_owt_baseline_has_blur_disabled():
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=openwebtext/sjd_openwebtext"],
        )
        assert cfg.experiment.forward.blur.enabled is False


def test_sudoku_blur_experiment_has_constraint_kernel():
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment=sudoku/sjd_sudoku_blur"],
        )
        blur = cfg.experiment.forward.blur
        assert blur.enabled is True
        assert blur.kind == "sudoku_constraint"
