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


def test_mdlm_sudoku_train_and_eval_configs_compose():
    for experiment, sampler_method in (
        ("mdlm_sudoku_uniform", "uniform"),
        ("mdlm_sudoku_top_prob_margin", "top_prob_margin"),
    ):
        cfg = _compose(
            config_name="config.yaml",
            overrides=[f"experiment={experiment}", "eval=sudoku_mdlm"],
        )

        assert cfg.experiment.task.name == "mdlm_sudoku"
        assert cfg.experiment.dataset.batch_size == 256
        assert cfg.experiment.model.name == "mdlm"
        assert cfg.experiment.model.sequence_backbone == "gpt2_like"
        assert cfg.experiment.model.n_layers == 3
        assert cfg.experiment.model.num_heads == 12
        assert cfg.experiment.model.feature_dim == 32
        assert cfg.experiment.model.sequence_mlp_hidden_dim == 1792
        assert cfg.experiment.model.sequence_max_length == 243
        assert cfg.experiment.model.sequence_causal is False
        assert cfg.experiment.model.time_features == "none"
        assert cfg.experiment.model.noise_schedule_type == "loglinear"
        assert cfg.experiment.sampler.method == sampler_method
        assert cfg.experiment.sampler.n_steps == 50
        assert cfg.experiment.sampler.sampling_grid == "loglinear"
        assert cfg.experiment.sampler.oracle_noise_type == "none"
        assert cfg.experiment.sampler.oracle_noise_scale == 0.0
        assert cfg.experiment.training.best_checkpoint_metric == "eval/solve_rate"
        assert cfg.experiment.training.best_update_on_equal is True
        assert cfg.eval.mode == "sudoku"

        task = build_task(cfg.experiment)
        model = build_model(
            cfg.experiment,
            data_shape=task.spec.data_shape,
            vocab_size=task.spec.vocab_size,
        )

        assert task.spec.name == "mdlm_sudoku"
        assert model.sequence_backbone == "gpt2_like"
        assert model.sequence_mlp_hidden_dim == 1792
        assert model.sequence_max_length == 243
        assert model.sequence_causal is False
        assert model.oracle_noise_type == "none"
        assert model.oracle_noise_scale == 0.0


def test_mdlm_sudoku_tfw_config_composes():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=mdlm_sudoku_tfw_top_prob_margin", "eval=sudoku_mdlm"],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.dataset.batch_size == 128
    assert cfg.experiment.dataset.eval_batch_size == 128
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_mlp_hidden_dim == 1792
    assert cfg.experiment.model.timesteps == 50
    assert cfg.experiment.model.time_features == "none"
    assert cfg.experiment.model.noise_schedule_type == "loglinear"
    assert cfg.experiment.optim.learning_rate == 1.0e-3
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert cfg.experiment.sampler.sampling_grid == "loglinear"
    assert cfg.experiment.sampler.oracle_noise_type == "gumbel"
    assert cfg.experiment.sampler.oracle_noise_scale == 0.5
    assert cfg.experiment.training.num_train_epochs == 300
    assert cfg.experiment.training.best_checkpoint_metric == "eval/solve_rate"
    assert cfg.experiment.training.best_checkpoint_mode == "max"
    assert cfg.experiment.training.best_update_on_equal is True
    assert cfg.eval.mode == "sudoku"

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    assert task.spec.name == "mdlm_sudoku"
    assert model.sampler == "top_prob_margin"
    assert model.oracle_noise_type == "gumbel"
    assert model.oracle_noise_scale == 0.5


def test_mdlm_sudoku_overfit_configs_compose():
    for experiment, max_examples in (
        ("mdlm_sudoku_overfit_512", 512),
        ("mdlm_sudoku_overfit_2048", 2048),
    ):
        cfg = _compose(
            config_name="config.yaml",
            overrides=[f"experiment={experiment}", "eval=sudoku_mdlm"],
        )

        assert cfg.experiment.task.name == "mdlm_sudoku"
        assert cfg.experiment.dataset.max_train_examples == max_examples
        assert cfg.experiment.dataset.max_test_examples == max_examples
        assert cfg.experiment.training.num_train_epochs == 100
        assert cfg.experiment.training.best_checkpoint_metric == "eval/solve_rate"


def test_mdlm_sudoku_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=mdlm_sudoku_uniform", "eval=sudoku_mdlm"],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.eval.mode == "sudoku"
    assert cfg.offline_eval.checkpoint_source == "best"


def test_root_defaults_now_target_tfw_top_margin_mdlm_sudoku():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert cfg.experiment.sampler.oracle_noise_type == "gumbel"
    assert cfg.experiment.sampler.oracle_noise_scale == 0.5
    assert cfg.experiment.training.num_train_epochs == 300
    assert cfg.eval.mode == "sudoku"
