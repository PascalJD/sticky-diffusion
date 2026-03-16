from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_sjd_paper_profile_composes_with_adm_recipe():
    cfg = _compose(
        [
            "experiment=sjd_cifar10_paper",
            "eval=sjd_cifar10_paper",
        ]
    )

    assert cfg.experiment.task.name == "sjd_cifar10_paper"
    assert cfg.experiment.model.name == "sjd"
    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 96
    assert cfg.experiment.model.ch_mult == [3, 4, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 4
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
    assert cfg.experiment.model.adm_num_heads == 4
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.adm_num_heads_upsample == -1
    assert cfg.experiment.model.adm_use_scale_shift_norm is True
    assert cfg.experiment.model.adm_use_conv_skip is False
    assert cfg.experiment.model.adm_use_new_attention_order is False
    assert cfg.experiment.dataset.batch_size == 512
    assert cfg.experiment.dataset.eval_batch_size == 512
    assert cfg.experiment.dataset.augment.enabled is True
    assert cfg.experiment.dataset.augment.prob == 0.15
    assert cfg.experiment.dataset.augment.rotate is True
    assert cfg.experiment.dataset.augment.hflip is True
    assert cfg.experiment.optim.name == "adamw"
    assert cfg.experiment.optim.learning_rate == 1.0e-4
    assert cfg.experiment.optim.warmup_steps == 1000
    assert cfg.experiment.optim.lr_schedule == "warmup_constant"
    assert cfg.experiment.optim.b2 == 0.999
    assert cfg.experiment.optim.weight_decay == 0.0
    assert cfg.experiment.training.name == "sjd_cifar10_paper"
    assert cfg.experiment.training.num_train_steps == 400000
    assert cfg.experiment.training.ema_rate == 0.9999
    assert cfg.experiment.training.checkpoint_every_steps == 10000
    assert cfg.experiment.training.log_images_every_steps == 10000
    assert cfg.experiment.training.sample_timesteps == 256
    assert cfg.experiment.sampler.n_steps == 256
    assert cfg.experiment.sampler.refresh_logits_after_em_step is True
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.forward.hazard._target_ == "sticky.models.sjd.hazard.make_hazard_poly_alpha"
    assert cfg.experiment.forward.hazard.p == 3.0
    assert cfg.eval.enabled is True
    assert cfg.eval.run_at_end is True
    assert cfg.eval.fid_enabled is True
    assert cfg.eval.fid_every == 10000
    assert cfg.eval.fid_num_samples == 5000
    assert cfg.eval.fid_batch_size == 128
    assert cfg.eval.is_enabled is True
    assert cfg.eval.is_every == 0


def test_sjd_paper_report_eval_profile_composes():
    cfg = _compose(
        [
            "experiment=sjd_cifar10_paper",
            "eval=sjd_cifar10_report",
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
