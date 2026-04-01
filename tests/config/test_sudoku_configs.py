from __future__ import annotations

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task


CONFIG_DIR = str(config_root())


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def test_mdm_sudoku_inpaint_train_and_eval_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/mdm_sudoku_inpaint_tfw_sanity",
            "eval=sudoku_inpaint_discrete",
        ],
    )

    assert cfg.experiment.task.name == "mdm_sudoku_inpaint"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.dataset.batch_size == 128
    assert cfg.experiment.dataset.eval_batch_size == 128
    assert cfg.experiment.model.name == "mdm_inpaint"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_max_length == 81
    assert cfg.experiment.model.timesteps == 50
    assert cfg.experiment.optim.name == "adamw"
    assert cfg.experiment.optim.learning_rate == 1.0e-3
    assert cfg.experiment.training.name == "sudoku_mdm_inpaint_tfw"
    assert cfg.experiment.training.num_train_epochs == 300
    assert (
        cfg.experiment.training.best_checkpoint_metric
        == "eval/top_prob_margin/solve_rate"
    )
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_run_all_sampler_modes is True
    assert cfg.eval.sudoku_primary_sampler_label == "top_prob_margin"

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert task.spec.name == "mdm_sudoku_inpaint"
    assert task.spec.data_shape == (81,)
    assert task.spec.vocab_size == 10
    assert model.sequence_max_length == 81


def test_mdm_sudoku_inpaint_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=[
            "experiment=sudoku/mdm_sudoku_inpaint_tfw_sanity",
            "eval=sudoku_inpaint_discrete",
        ],
    )

    assert cfg.experiment.task.name == "mdm_sudoku_inpaint"
    assert cfg.eval.mode == "sudoku"
    assert cfg.offline_eval.checkpoint_source == "best"


def test_root_defaults_target_board_benchmark():
    cfg = _compose(config_name="config.yaml", overrides=[])

    assert cfg.experiment.task.name == "mdm_sudoku_inpaint"
    assert cfg.eval.mode == "sudoku"
