from __future__ import annotations

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task

CONFIG_DIR = str(config_root())


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def test_sjd_sudoku_slack_experiment_composes():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku_slack",
        ],
    )

    assert cfg.experiment.task.name == "sjd_sudoku_inpaint_slack"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_max_length == 108
    assert cfg.experiment.model.enable_joint_input is True
    assert cfg.experiment.model.anchor.family == "simplex_vertex"
    assert cfg.experiment.model.anchor.dim == 9
    assert cfg.experiment.model.anchor.learnable is False
    assert cfg.experiment.model.anchor.transform.equalize_row_norms is False
    assert cfg.experiment.training.name == "sudoku_sjd"
    # Phase 4: eval is re-enabled; the slack-aware sampler runs end-to-end.
    assert cfg.experiment.training.eval_every_steps == 5000
    assert cfg.experiment.training.checkpoint_every_steps == 5000
    # Slack-axis sampler knobs are now plumbed through.
    assert cfg.experiment.sampler.slack.project_after_step is True
    # The eval mix is policy-only (no slack-aware predictor_only sampler yet).
    assert "predictor_only" in cfg.experiment.eval.sudoku_eval_sjd_runs
    assert (
        cfg.experiment.eval.sudoku_eval_sjd_runs.predictor_only.kind == "policy"
    )


def test_slack_task_and_model_build_from_composed_config():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku_slack",
            # Don't auto-download data during the test:
            "experiment.dataset.auto_download=false",
        ],
    )
    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert task.spec.name == "sjd_sudoku_inpaint_slack"
    assert task.spec.vocab_size == 9
    assert task.spec.data_shape == (81,)
    assert model.sequence_max_length == 108
    assert model.enable_joint_input is True
    assert model.vocab_size == 9
