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
