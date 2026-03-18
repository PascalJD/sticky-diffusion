from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_canonical_cifar10_experiments_compose():
    for experiment, model_name in (
        ("cadd_cifar10", "cadd"),
        ("ddpm_cifar10", "ddpm"),
        ("md4_cifar10", "md4"),
        ("sjd_cifar10", "sjd"),
    ):
        cfg = _compose([f"experiment={experiment}", "eval=cifar10"])

        assert cfg.experiment.task.name == experiment
        assert cfg.experiment.model.name == model_name
        assert cfg.experiment.training.sample_timesteps == cfg.experiment.sampler.n_steps
        assert cfg.eval.enabled is True
        assert cfg.eval.run_at_end is True
        assert cfg.eval.fid_every == 10000


def test_adm_image_models_share_canonical_architecture_bundle():
    expectations = {
        "cadd_cifar10": "adm_unet5d",
        "ddpm_cifar10": "adm_unet2d",
        "sjd_cifar10": "adm_unet5d",
    }

    for experiment, image_backbone in expectations.items():
        cfg = _compose([f"experiment={experiment}", "eval=cifar10"])

        assert cfg.experiment.model.feature_dim == 96
        assert cfg.experiment.model.ch_mult == [3, 4, 4]
        assert cfg.experiment.model.adm_num_res_blocks == 4
        assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
        assert cfg.experiment.model.adm_num_heads == 4
        assert cfg.experiment.model.adm_num_head_channels == 64
        assert cfg.experiment.model.adm_use_scale_shift_norm is True
        assert cfg.experiment.model.adm_use_new_attention_order is False
        assert cfg.experiment.model.image_backbone == image_backbone


def test_cadd_cifar10_keeps_gaussian_default_and_flow_matching_override():
    cfg = _compose(["experiment=cadd_cifar10", "eval=cifar10"])
    assert cfg.experiment.model.cadd_latent.type == "gaussian"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.sampler.temperature_schedule == "cosine_decay"
    assert cfg.experiment.sampler.corrector_remask_frac == 0.1
    assert cfg.experiment.training.sample_timesteps == 512

    override_cfg = _compose(
        [
            "experiment=cadd_cifar10",
            "eval=cifar10",
            "model/cadd_latent@experiment.model.cadd_latent=flow_matching",
        ]
    )
    assert override_cfg.experiment.model.cadd_latent.type == "flow_matching"


def test_sjd_cifar10_preserves_logging_and_canonical_anchor_overrides():
    cfg = _compose(["experiment=sjd_cifar10", "eval=cifar10"])

    assert cfg.experiment.training.sample_timesteps == 256
    assert cfg.experiment.training.log_state_dependency is True
    assert cfg.experiment.training.state_dep_log_ratio_clip == cfg.experiment.sampler.log_ratio_clip
    assert cfg.experiment.model.anchor.family == "ordered_normal"
    assert cfg.experiment.model.anchor.learnable is True

    override_cfg = _compose(
        [
            "experiment=sjd_cifar10",
            "eval=cifar10",
            "model/anchor@experiment.model.anchor=normal",
        ]
    )
    assert override_cfg.experiment.model.anchor.family == "normal"
    assert override_cfg.experiment.model.anchor.learnable is True


def test_md4_cifar10_uses_md4_architecture_bundle():
    cfg = _compose(["experiment=md4_cifar10", "eval=cifar10"])

    assert cfg.experiment.model.image_backbone == "unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 128
    assert cfg.experiment.model.ch_mult == [1]
    assert cfg.experiment.sampler.method == "ancestral"
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.sampler.topp == 0.98
    assert cfg.experiment.training.sample_timesteps == 256


def test_ddpm_cifar10_sampler_tracks_model_timesteps():
    cfg = _compose(["experiment=ddpm_cifar10", "eval=cifar10"])

    assert cfg.experiment.sampler.n_steps == cfg.experiment.model.timesteps
    assert cfg.experiment.training.sample_timesteps == 1000


def test_cifar10_report_eval_profile_composes_for_all_canonical_experiments():
    for experiment in ("cadd_cifar10", "ddpm_cifar10", "md4_cifar10", "sjd_cifar10"):
        cfg = _compose([f"experiment={experiment}", "eval=cifar10_report"])

        assert cfg.eval.enabled is True
        assert cfg.eval.run_at_end is True
        assert cfg.eval.fid_every == 0
        assert cfg.eval.fid_num_samples == 50000
        assert cfg.eval.is_enabled is True
