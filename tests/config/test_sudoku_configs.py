from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from sticky.models.factory import build_model
from sticky.tasks.factory import build_task


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def test_sjd_sudoku_train_and_eval_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=sjd_sudoku", "eval=sjd_sudoku"],
    )

    assert cfg.experiment.task.name == "sjd_sudoku"
    assert cfg.experiment.dataset.batch_size == 256
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.model.n_layers == 3
    assert cfg.experiment.model.num_heads == 12
    assert cfg.experiment.model.feature_dim == 32
    assert cfg.experiment.model.anchor.dim == 64
    assert cfg.experiment.model.anchor.learnable is False
    assert cfg.experiment.optim.learning_rate == 3.0e-4
    assert cfg.experiment.optim.warmup_steps == 4000
    assert cfg.experiment.optim.grad_clip_norm == 1.0
    assert cfg.experiment.sampler.n_steps == 50
    assert cfg.experiment.sampler.logit_temperature == 0.8
    assert cfg.experiment.sampler.intensity_mode == "full"
    assert cfg.experiment.forward.jump.eta == 0.6
    assert cfg.experiment.training.best_checkpoint_metric == "eval/solve_rate"
    assert cfg.eval.mode == "sudoku"

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert task.spec.name == "sjd_sudoku"
    assert model.anchor_config.anchor_dim == 64


def test_sjd_sudoku_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=sjd_sudoku", "eval=sjd_sudoku"],
    )

    assert cfg.experiment.task.name == "sjd_sudoku"
    assert cfg.eval.mode == "sudoku"
    assert cfg.offline_eval.checkpoint_source == "best"
