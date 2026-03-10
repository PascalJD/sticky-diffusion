from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_root_anchor_override_targets_nested_anchor_package():
    cfg = _compose(
        [
            "experiment=sjd_anchor_study_cifar10",
            "eval=sjd_anchor_study_cifar10",
            "model/anchor@experiment.model.anchor=normal_64",
        ]
    )

    assert cfg.experiment.model.anchor.family == "normal"
    assert cfg.experiment.model.anchor.dim == 64


def test_anchor_study_preset_experiment_composes():
    cfg = _compose(
        [
            "experiment=sjd_anchor_study/fixed_normal_64",
            "eval=sjd_anchor_study_cifar10",
        ]
    )

    assert cfg.experiment.task.name == "sjd_anchor_study_cifar10_fixed_normal_64"
    assert cfg.experiment.model.anchor.family == "normal"
    assert cfg.experiment.model.anchor.dim == 64
    assert cfg.experiment.model.anchor.learnable is False
