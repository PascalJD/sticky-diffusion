from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_cadd_latent_defaults_to_gaussian():
    cfg = _compose(
        [
            "experiment=cadd_cifar10",
            "eval=cadd_cifar10",
        ]
    )

    assert cfg.experiment.model.cadd_latent.type == "gaussian"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"
    assert cfg.experiment.dataset.batch_size == 512
    assert cfg.experiment.dataset.eval_batch_size == 512
    assert cfg.experiment.dataset.augment.enabled is True
    assert cfg.experiment.dataset.augment.prob == 0.15
    assert cfg.experiment.optim.learning_rate == 1.0e-4
    assert cfg.experiment.optim.lr_schedule == "warmup_constant"
    assert cfg.experiment.optim.b2 == 0.999
    assert cfg.experiment.optim.weight_decay == 0.0
    assert cfg.experiment.model.corrector_enabled is True
    assert cfg.experiment.model.corrector_steps == 1
    assert cfg.experiment.model.corrector_remask_frac == 0.1
    assert cfg.experiment.training.num_train_steps == 400000
    assert cfg.experiment.training.best_checkpoint_metric == "eval/fid"
    assert cfg.experiment.training.best_checkpoint_mode == "min"
    assert cfg.eval.fid_every == 10000
    assert cfg.eval.fid_num_samples == 5000
    assert cfg.wandb.enabled is True


def test_cadd_latent_override_targets_nested_package():
    cfg = _compose(
        [
            "experiment=cadd_cifar10",
            "eval=cadd_cifar10",
            "model/cadd_latent@experiment.model.cadd_latent=flow_matching",
        ]
    )

    assert cfg.experiment.model.cadd_latent.type == "flow_matching"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"


def test_cadd_paper_profile_defaults_to_flow_matching():
    cfg = _compose(
        [
            "experiment=cadd_cifar10_paper",
            "eval=cadd_cifar10_paper",
        ]
    )

    assert cfg.experiment.task.name == "cadd_cifar10_paper"
    assert cfg.experiment.model.cadd_latent.type == "flow_matching"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"
    assert cfg.experiment.dataset.batch_size == 512
    assert cfg.experiment.dataset.eval_batch_size == 512
    assert cfg.experiment.dataset.augment.enabled is True
    assert cfg.experiment.dataset.augment.prob == 0.15
    assert cfg.experiment.dataset.augment.rotate is True
    assert cfg.experiment.dataset.augment.hflip is True
    assert cfg.experiment.optim.learning_rate == 1.0e-4
    assert cfg.experiment.optim.warmup_steps == 1000
    assert cfg.experiment.optim.lr_schedule == "warmup_constant"
    assert cfg.experiment.optim.b2 == 0.999
    assert cfg.experiment.optim.weight_decay == 0.0
    assert cfg.experiment.training.num_train_steps == 400000
    assert cfg.experiment.training.ema_rate == 0.9999
    assert cfg.experiment.training.best_checkpoint_metric == "eval/fid"
    assert cfg.experiment.training.best_checkpoint_mode == "min"
    assert cfg.experiment.model.feature_dim == 96
    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.ch_mult == [3, 4, 4]
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.adm_use_new_attention_order is False
    assert cfg.experiment.model.timesteps == 512
    assert cfg.experiment.model.tau_max == 2.5
    assert cfg.experiment.model.corrector_enabled is True
    assert cfg.eval.enabled is True
    assert cfg.eval.run_at_end is True
    assert cfg.eval.fid_enabled is True
    assert cfg.eval.fid_every == 10000
    assert cfg.eval.fid_num_samples == 5000
    assert cfg.eval.fid_batch_size == 128
    assert cfg.eval.is_enabled is True
    assert cfg.eval.is_every == 0


def test_cadd_paper_profile_keeps_gaussian_available_as_override():
    cfg = _compose(
        [
            "experiment=cadd_cifar10_paper",
            "eval=cadd_cifar10_paper",
            "model/cadd_latent@experiment.model.cadd_latent=gaussian",
        ]
    )

    assert cfg.experiment.model.cadd_latent.type == "gaussian"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"


def test_cadd_report_eval_profile_composes():
    cfg = _compose(
        [
            "experiment=cadd_cifar10_paper",
            "eval=cadd_cifar10_report",
        ]
    )

    assert cfg.eval.enabled is True
    assert cfg.eval.run_at_end is True
    assert cfg.eval.fid_enabled is True
    assert cfg.eval.fid_every == 0
    assert cfg.eval.fid_num_samples == 50000
    assert cfg.eval.fid_batch_size == 128
    assert cfg.eval.is_enabled is True
    assert cfg.eval.is_every == 0
