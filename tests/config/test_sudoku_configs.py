from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.eval.sudoku import build_sudoku_eval_logger
from sticky.models.factory import build_model
from sticky.tasks.factory import build_task


CONFIG_DIR = str(config_root())


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def _disabled_test_mdlm_sudoku_train_and_eval_configs_compose():
    """Removed in the config cleanup: non-SJD Sudoku configs were deleted."""
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/mdlm_sudoku",
            "eval=sudoku_mdlm",
        ],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.dataset.batch_size == 128
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_max_length == 81
    assert cfg.experiment.optim.learning_rate == 1.0e-3
    assert cfg.experiment.training.name == "sudoku_mdlm"
    assert cfg.experiment.training.num_train_epochs == 300
    assert cfg.experiment.training.best_checkpoint_metric == "eval/deterministic/solve_rate"
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_run_all_sampler_modes is True
    assert cfg.eval.sudoku_primary_sampler_label == "deterministic"
    assert set(cfg.eval.sudoku_eval_samplers.keys()) == {"deterministic"}

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg.experiment,
        eval_cfg=cfg.eval,
        task=task,
        model=model,
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    assert task.spec.name == "mdlm_sudoku"
    assert task.spec.data_shape == (81,)
    assert task.spec.vocab_size == 10
    assert model.sequence_max_length == 81
    assert callable(maybe_eval)


def test_sjd_sudoku_train_and_eval_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku",
            "eval=sudoku_sjd",
        ],
    )

    assert cfg.experiment.task.name == "sjd_sudoku_inpaint"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_max_length == 81
    assert cfg.experiment.model.anchor.family == "thermometer"
    assert cfg.experiment.model.anchor.dim == 64
    assert cfg.experiment.model.anchor.learnable is False
    assert cfg.experiment.model.anchor.transform.equalize_row_norms is True
    assert cfg.experiment.training.name == "sudoku_sjd"
    assert cfg.experiment.training.best_checkpoint_metric == "eval/predictor_only/solve_rate"
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_primary_sampler_label == "predictor_only"
    assert cfg.eval.sudoku_write_progress_csv is True
    assert cfg.eval.sudoku_progress_csv_path == "metrics/sudoku_sjd_progress.csv"
    assert cfg.eval.sudoku_write_latest_csv is True
    assert cfg.eval.sudoku_latest_csv_path == "metrics/sudoku_sjd_latest.csv"
    assert set(cfg.eval.sudoku_eval_sjd_runs.keys()) == {
        "linear_survival",
        "cosine_survival",
        "linear_topk_probability",
        "plugin_hazard_eta_0p97",
        "predictor_only",
    }

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    maybe_eval = build_sudoku_eval_logger(
        cfg=cfg.experiment,
        eval_cfg=cfg.eval,
        task=task,
        model=model,
        wandb_mod=None,
        eval_every=1,
        log_at_step_zero=False,
    )
    assert task.spec.name == "sjd_sudoku_inpaint"
    assert task.spec.vocab_size == 9
    assert callable(maybe_eval)


def _disabled_test_sjd_sudoku_report_config_composes():
    """Removed in the config cleanup: sudoku_sjd_report eval profile deleted."""
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku",
            "eval=sudoku_sjd_report",
        ],
    )

    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_primary_sampler_label == "predictor_only"
    assert cfg.eval.sudoku_prop52_enabled is True
    assert cfg.eval.sudoku_write_progress_csv is True
    assert cfg.eval.sudoku_write_latest_csv is True
    assert "linear_survival" in cfg.eval.sudoku_eval_sjd_runs
    assert "cosine_survival" in cfg.eval.sudoku_eval_sjd_runs
    assert "linear_topk_probability" in cfg.eval.sudoku_eval_sjd_runs
    plugin_labels = sorted(
        label
        for label, spec in cfg.eval.sudoku_eval_sjd_runs.items()
        if spec.get("kind") == "policy" and spec.get("policy") == "plugin_hazard"
    )
    assert plugin_labels == ["plugin_hazard_eta_0p97"]
    assert "predictor_only" in cfg.eval.sudoku_eval_sjd_runs


def test_root_defaults_target_board_benchmark():
    cfg = _compose(config_name="config.yaml", overrides=[])

    assert cfg.experiment.task.name == "sjd_sudoku_inpaint"
    assert cfg.eval.mode == "sudoku"


def test_removed_sudoku_aliases_do_not_compose():
    removed = [
        "experiment=sudoku/mdlm_sudoku_deterministic",
        "experiment=sudoku/mdlm_sudoku_deterministic",
        "experiment=sudoku/mdlm_sudoku_deterministic",
        "experiment=sudoku/sjd_sudoku_policy_ablation",
        "eval=sudoku_sjd_pc",
        "eval=sudoku_sjd_policy_ablation",
    ]

    for override in removed:
        parts = [override]
        if override.startswith("experiment="):
            parts.append("eval=sudoku_mdlm" if "mdlm" in override else "eval=sudoku_sjd")
        else:
            parts.append("experiment=sudoku/sjd_sudoku")
        with pytest.raises(Exception):
            _compose(config_name="config.yaml", overrides=parts)


def test_readme_and_docs_reference_only_canonical_sudoku_grouped_configs():
    """Post-cleanup: only the SJD Sudoku experiment remains; non-SJD Sudoku
    configs (mdlm_sudoku, sudoku_sjd_report, ablation variants) were deleted
    and must not appear in user-facing docs."""
    root = Path(CONFIG_DIR).parent
    docs_text = (root / "docs" / "configs.md").read_text(encoding="utf-8")

    assert "experiment=sudoku/sjd_sudoku" in docs_text
    # Removed configs must not be advertised in docs.
    assert "experiment=sudoku/mdlm_sudoku" not in docs_text
    assert "experiment=sudoku/sjd_sudoku_policy_ablation" not in docs_text
    assert "eval=sudoku_sjd_report" not in docs_text
    assert "eval=sudoku_mdlm" not in docs_text
