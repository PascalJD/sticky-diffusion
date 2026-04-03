from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.eval.sudoku import build_sudoku_eval_logger
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
    assert cfg.experiment.model.name == "mdm"
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
    assert cfg.offline_eval.eval_prop52_only is False


def test_sjd_sudoku_policy_ablation_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku_policy_ablation",
            "eval=sudoku_sjd_policy_ablation",
        ],
    )

    assert cfg.experiment.task.name == "sjd_sudoku_inpaint"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.sequence_max_length == 81
    assert cfg.experiment.model.anchor.family == "ordered_normal"
    assert cfg.experiment.model.anchor.dim == 64
    assert cfg.experiment.training.name == "sudoku_sjd_policy_ablation"
    assert cfg.experiment.training.best_checkpoint_metric == "eval/plugin_hazard_eta_0p97/solve_rate"
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_primary_sampler_label == "plugin_hazard_eta_0p97"
    assert "plugin_hazard_eta_1p00" in cfg.eval.sudoku_eval_policies
    assert cfg.eval.sudoku_prop52_enabled is True
    assert cfg.eval.sudoku_prop52_num_bins == 20
    assert cfg.eval.sudoku_prop52_num_batches == 8

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert task.spec.name == "sjd_sudoku_inpaint"
    assert task.spec.vocab_size == 9
    assert model.sequence_max_length == 81


def test_sjd_sudoku_pc_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku_pc",
            "eval=sudoku_sjd_pc",
        ],
    )

    assert cfg.experiment.task.name == "sjd_sudoku_inpaint"
    assert cfg.experiment.dataset.name == "sudoku_shah_board"
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.sampler.pc_enabled is False
    assert cfg.experiment.training.best_checkpoint_metric == "eval/pc_margin_l1_s0p10/solve_rate"
    assert cfg.eval.sudoku_primary_sampler_label == "pc_margin_l1_s0p10"
    assert "pc_margin_l1_s0p10" in cfg.eval.sudoku_eval_sjd_samplers
    assert "predictor_only" in cfg.eval.sudoku_eval_sjd_samplers

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
    assert callable(maybe_eval)


def test_sjd_sudoku_pc_report_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=[
            "experiment=sudoku/sjd_sudoku_pc",
            "eval=sudoku_sjd_pc_report",
        ],
    )

    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_primary_sampler_label == "pc_margin_l1_s0p10"
    assert "pc_entropy_l4_s0p40" in cfg.eval.sudoku_eval_sjd_samplers
    assert "pc_margin_l4_s0p40" in cfg.eval.sudoku_eval_sjd_samplers


def test_root_defaults_target_board_benchmark():
    cfg = _compose(config_name="config.yaml", overrides=[])

    assert cfg.experiment.task.name == "mdm_sudoku_inpaint"
    assert cfg.eval.mode == "sudoku"


def test_legacy_sjd_sudoku_entrypoints_removed_and_readme_points_to_canonical_path():
    root = Path(CONFIG_DIR).parent
    readme_text = (root / "README.md").read_text(encoding="utf-8")

    assert "experiment=sudoku/sjd_sudoku_policy_ablation" in readme_text
    assert "eval=sudoku_sjd_policy_ablation" in readme_text
    assert "experiment=sudoku/sjd_sudoku eval=sjd_sudoku" not in readme_text
    assert not (Path(CONFIG_DIR) / "eval" / "sjd_sudoku.yaml").exists()
    assert not (Path(CONFIG_DIR) / "experiment" / "sudoku" / "sjd_sudoku.yaml").exists()
