from __future__ import annotations

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root
from sticky.models.factory import build_model
from sticky.rng import make_rng
from sticky.tasks.factory import build_task
from sticky.training.state import init_state


CONFIG_DIR = str(config_root())


def _compose(*, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def test_sjd_sudoku_train_and_eval_configs_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=sudoku/sjd_sudoku", "eval=sjd_sudoku"],
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
    assert cfg.eval.sudoku_run_all_sampler_modes is False

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert task.spec.name == "sjd_sudoku"
    assert model.anchor_config.anchor_dim == 64


def test_grouped_sudoku_experiment_paths_compose():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=sudoku/mdlm_sudoku_tfw_top_prob_margin", "eval=sudoku_mdlm"],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.sampler.method == "top_prob_margin"


def test_sjd_sudoku_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=sudoku/sjd_sudoku", "eval=sjd_sudoku"],
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
            overrides=[f"experiment=sudoku/{experiment}", "eval=sudoku_mdlm"],
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
        assert cfg.experiment.sampler.revealed_token_sample_mode == "sample"
        assert cfg.experiment.sampler.oracle_noise_type == "none"
        assert cfg.experiment.sampler.oracle_noise_scale == 0.0
        assert cfg.experiment.training.best_checkpoint_metric == "eval/solve_rate"
        assert cfg.experiment.training.best_update_on_equal is True
        assert cfg.eval.mode == "sudoku"
        assert cfg.eval.param_source == "ema"
        assert cfg.eval.sudoku_run_all_sampler_modes is False

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
        overrides=["experiment=sudoku/mdlm_sudoku_tfw_top_prob_margin", "eval=sudoku_mdlm"],
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
    assert cfg.experiment.sampler.categorical_sampling_policy == "exact"
    assert cfg.experiment.sampler.revealed_token_sample_mode == "sample"
    assert cfg.experiment.sampler.oracle_noise_type == "gumbel"
    assert cfg.experiment.sampler.oracle_noise_scale == 0.5
    assert cfg.experiment.training.num_train_epochs == 300
    assert cfg.experiment.training.best_checkpoint_metric == "eval/acc_complete_puzzle"
    assert cfg.experiment.training.best_checkpoint_mode == "max"
    assert cfg.experiment.training.best_update_on_equal is True
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.param_source == "ema"
    assert cfg.eval.sudoku_run_all_sampler_modes is False

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
    assert model.revealed_token_sample_mode == "sample"


def test_mdlm_sudoku_tfw_argmax_ablation_config_composes():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=sudoku/mdlm_sudoku_tfw_top_prob_margin_argmax", "eval=sudoku_mdlm"],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.dataset.batch_size == 128
    assert cfg.experiment.dataset.eval_batch_size == 128
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert cfg.experiment.sampler.categorical_sampling_policy == "exact"
    assert cfg.experiment.sampler.revealed_token_sample_mode == "argmax"
    assert cfg.experiment.sampler.cache_predictions is True
    assert cfg.experiment.training.best_checkpoint_metric == "eval/acc_complete_puzzle"

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    assert model.revealed_token_sample_mode == "argmax"
    assert model.cache_predictions is True


def test_mdlm_sudoku_overfit_configs_compose():
    for experiment, max_examples in (
        ("mdlm_sudoku_overfit_512", 512),
        ("mdlm_sudoku_overfit_2048", 2048),
    ):
        cfg = _compose(
            config_name="config.yaml",
            overrides=[f"experiment=sudoku/{experiment}", "eval=sudoku_mdlm"],
        )

        assert cfg.experiment.task.name == "mdlm_sudoku"
        assert cfg.experiment.dataset.max_train_examples == max_examples
        assert cfg.experiment.dataset.max_test_examples == max_examples
        assert cfg.experiment.training.num_train_epochs == 100
        assert cfg.experiment.training.best_checkpoint_metric == "eval/acc_complete_puzzle"


def test_mdlm_sudoku_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=sudoku/mdlm_sudoku_uniform", "eval=sudoku_mdlm"],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.eval.mode == "sudoku"
    assert cfg.offline_eval.checkpoint_source == "best"


def test_mdm_sudoku_tfw_config_scaffolding_composes():
    cfg = _compose(
        config_name="config.yaml",
        overrides=["experiment=sudoku/mdm_sudoku_tfw", "eval=sudoku_discrete"],
    )

    assert cfg.experiment.task.name == "mdm_sudoku"
    assert cfg.experiment.dataset.batch_size == 128
    assert cfg.experiment.dataset.eval_batch_size == 128
    assert cfg.experiment.model.name == "mdm"
    assert cfg.experiment.model.sequence_backbone == "gpt2_like"
    assert cfg.experiment.model.n_layers == 3
    assert cfg.experiment.model.num_heads == 12
    assert cfg.experiment.model.feature_dim == 32
    assert cfg.experiment.model.sequence_mlp_hidden_dim == 1536
    assert cfg.experiment.model.sequence_max_length == 245
    assert cfg.experiment.model.sequence_causal is False
    assert cfg.experiment.model.dropout_rate == 0.1
    assert cfg.experiment.model.cont_time is False
    assert cfg.experiment.model.timesteps == 50
    assert cfg.experiment.model.time_features == "none"
    assert cfg.experiment.model.token_reweighting is True
    assert cfg.experiment.model.alpha == 0.25
    assert cfg.experiment.model.gamma == 1.0
    assert cfg.experiment.model.time_reweighting == "linear"
    assert cfg.experiment.optim.learning_rate == 1.0e-3
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert cfg.experiment.sampler.n_steps == 50
    assert cfg.experiment.sampler.categorical_sampling_policy == "exact"
    assert cfg.experiment.sampler.decoding_style == "topk_remask"
    assert cfg.experiment.sampler.oracle_noise_type == "gumbel"
    assert cfg.experiment.sampler.oracle_noise_scale == 0.5
    assert cfg.experiment.training.num_train_epochs == 300
    assert (
        cfg.experiment.training.best_checkpoint_metric
        == "eval/top_prob_margin/acc_complete_puzzle"
    )
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.param_source == "ema"
    assert cfg.eval.sudoku_run_all_sampler_modes is True
    assert cfg.eval.sudoku_primary_sampler_label == "top_prob_margin"
    assert cfg.eval.sudoku_num_batches_per_sampler == cfg.eval.sudoku_num_batches
    assert list(cfg.eval.sudoku_eval_samplers.keys()) == [
        "vanilla",
        "top_probability",
        "top_prob_margin",
    ]
    assert [cfg.eval.sudoku_eval_samplers[label].sampler for label in cfg.eval.sudoku_eval_samplers] == [
        "mdm_sudoku_vanilla",
        "mdm_sudoku_top_probability",
        "mdm_sudoku_top_prob_margin",
    ]
    assert [cfg.eval.sudoku_eval_samplers[label].method for label in cfg.eval.sudoku_eval_samplers] == [
        "vanilla",
        "top_probability",
        "top_prob_margin",
    ]
    assert [cfg.eval.sudoku_eval_samplers[label].decoding_style for label in cfg.eval.sudoku_eval_samplers] == [
        "monotone_reveal",
        "topk_remask",
        "topk_remask",
    ]
    assert [cfg.eval.sudoku_eval_samplers[label].oracle_noise_type for label in cfg.eval.sudoku_eval_samplers] == [
        "none",
        "gumbel",
        "gumbel",
    ]
    assert [cfg.eval.sudoku_eval_samplers[label].oracle_noise_scale for label in cfg.eval.sudoku_eval_samplers] == [
        0.0,
        0.5,
        0.5,
    ]

    task = build_task(cfg.experiment)
    assert task.spec.name == "mdm_sudoku"
    assert task.spec.task_type == "text"
    assert task.spec.data_shape == (245,)
    assert task.spec.vocab_size == 12

    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    assert model.vocab_size == 12
    assert model.mask_token_id == 12
    assert model.sequence_backbone == "gpt2_like"
    assert model.sequence_mlp_hidden_dim == 1536
    assert model.sequence_max_length == 245
    assert model.sequence_causal is False
    assert model.time_features == "none"
    assert model.decoding_style == "topk_remask"
    assert model.token_reweighting is True
    assert model.alpha == 0.25
    assert model.gamma == 1.0
    assert model.time_reweighting == "linear"

    state, _ = init_state(
        cfg.experiment,
        model,
        make_rng(int(cfg.experiment.training.seed)),
    )
    assert int(state.step) == 0


def test_mdm_sudoku_offline_eval_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=sudoku/mdm_sudoku_tfw", "eval=sudoku_discrete"],
    )

    assert cfg.experiment.task.name == "mdm_sudoku"
    assert cfg.experiment.model.name == "mdm"
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert (
        cfg.experiment.training.best_checkpoint_metric
        == "eval/top_prob_margin/acc_complete_puzzle"
    )
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_run_all_sampler_modes is True
    assert cfg.eval.sudoku_primary_sampler_label == "top_prob_margin"
    assert list(cfg.eval.sudoku_eval_samplers.keys()) == [
        "vanilla",
        "top_probability",
        "top_prob_margin",
    ]
    assert cfg.offline_eval.checkpoint_source == "best"


def test_mdm_sudoku_offline_eval_ablation_config_composes():
    cfg = _compose(
        config_name="eval_checkpoint.yaml",
        overrides=["experiment=sudoku/mdm_sudoku_tfw", "eval=sudoku_discrete_mdm_ablation"],
    )

    assert cfg.experiment.task.name == "mdm_sudoku"
    assert cfg.eval.mode == "sudoku"
    assert cfg.eval.sudoku_run_all_sampler_modes is True
    assert cfg.eval.sudoku_primary_sampler_label == "top_prob_margin"
    assert list(cfg.eval.sudoku_eval_samplers.keys()) == [
        "vanilla",
        "top_probability_monotone",
        "top_prob_margin_monotone",
        "top_probability",
        "top_prob_margin",
    ]
    assert [
        cfg.eval.sudoku_eval_samplers[label].decoding_style
        for label in cfg.eval.sudoku_eval_samplers
    ] == [
        "monotone_reveal",
        "monotone_reveal",
        "monotone_reveal",
        "topk_remask",
        "topk_remask",
    ]
    assert cfg.offline_eval.checkpoint_source == "best"


def test_root_defaults_now_target_tfw_top_margin_mdlm_sudoku():
    cfg = _compose(
        config_name="config.yaml",
        overrides=[],
    )

    assert cfg.experiment.task.name == "mdlm_sudoku"
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.sampler.method == "top_prob_margin"
    assert cfg.experiment.sampler.categorical_sampling_policy == "exact"
    assert cfg.experiment.sampler.oracle_noise_type == "gumbel"
    assert cfg.experiment.sampler.oracle_noise_scale == 0.5
    assert cfg.experiment.training.num_train_epochs == 300
    assert cfg.eval.mode == "sudoku"
