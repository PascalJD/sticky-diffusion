from __future__ import annotations

from hydra import compose, initialize_config_dir
from sticky.core.config_paths import config_root


CONFIG_DIR = str(config_root())


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_canonical_cifar10_experiments_compose():
    for experiment, model_name in (
        ("bitdiff_cifar10", "bitdiff"),
        ("candi_cifar10", "candi"),
        ("cadd_cifar10", "cadd"),
        ("ddpm_cifar10", "ddpm"),
        ("d3pm_absorb_cifar10", "d3pm"),
        ("d3pm_uniform_cifar10", "d3pm"),
        ("d3pm_gaussian_cifar10", "d3pm"),
        ("md4_cifar10", "md4"),
        ("mdlm_cifar10", "mdlm"),
        ("sjd_cifar10", "sjd"),
    ):
        cfg = _compose([f"experiment=cifar10/{experiment}"])

        assert cfg.experiment.task.name == experiment
        assert cfg.experiment.model.name == model_name
        assert cfg.experiment.training.sample_timesteps == cfg.experiment.sampler.n_steps
        assert cfg.eval.enabled is True
        assert cfg.eval.run_at_end is True
        assert cfg.eval.fid_every == 10000


def test_grouped_cifar10_experiment_paths_compose():
    cfg = _compose(["experiment=cifar10/md4_cifar10"])

    assert cfg.experiment.task.name == "md4_cifar10"
    assert cfg.experiment.model.name == "md4"
    assert cfg.experiment.training.sample_timesteps == 256


def test_adm_image_models_share_canonical_architecture_bundle():
    expectations = {
        "bitdiff_cifar10": "adm_unet5d",
        "candi_cifar10": "adm_unet5d",
        "cadd_cifar10": "adm_unet5d",
        "ddpm_cifar10": "adm_unet2d",
        "d3pm_absorb_cifar10": "adm_unet5d",
        "d3pm_uniform_cifar10": "adm_unet5d",
        "d3pm_gaussian_cifar10": "adm_unet5d",
        "mdlm_cifar10": "adm_unet5d",
        "sjd_cifar10": "adm_unet5d",
    }

    for experiment, image_backbone in expectations.items():
        cfg = _compose([f"experiment=cifar10/{experiment}"])

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
    cfg = _compose(["experiment=cifar10/cadd_cifar10"])
    assert cfg.experiment.model.cadd_latent.type == "gaussian"
    assert cfg.experiment.model.cadd_latent.continuous_schedule_type == "linear"
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.sampler.temperature_schedule == "cosine_decay"
    assert cfg.experiment.sampler.categorical_sampling_policy == "legacy_low"
    assert cfg.experiment.sampler.K == 3
    assert cfg.experiment.sampler.remask_refine_frac == 0.1
    assert cfg.experiment.training.sample_timesteps == 512

    override_cfg = _compose(
        [
            "experiment=cifar10/cadd_cifar10",
            "model/cadd_latent@experiment.model.cadd_latent=flow_matching",
        ]
    )
    assert override_cfg.experiment.model.cadd_latent.type == "flow_matching"


def test_candi_cifar10_uses_canonical_adm_image_bundle():
    cfg = _compose(["experiment=cifar10/candi_cifar10"])

    assert cfg.experiment.model.name == "candi"
    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 96
    assert cfg.experiment.model.ch_mult == [3, 4, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 4
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
    assert cfg.experiment.model.adm_num_heads == 4
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.representation == "embed"
    assert cfg.experiment.model.experimental is True
    assert cfg.experiment.model.classes == -1
    assert cfg.experiment.model.pure_continuous is True
    assert cfg.experiment.model.use_percentile_scheduling is True
    assert cfg.experiment.sampler.method == "continuous"
    assert cfg.experiment.sampler.sampling_grid == "uniform"
    assert "hybrid_exact" not in str(cfg.experiment.sampler.method)
    assert cfg.experiment.training.sample_timesteps == 256


def test_bitdiff_cifar10_uses_canonical_adm_image_bundle():
    cfg = _compose(["experiment=cifar10/bitdiff_cifar10"])

    assert cfg.experiment.model.name == "bitdiff"
    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 96
    assert cfg.experiment.model.ch_mult == [3, 4, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 4
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
    assert cfg.experiment.model.adm_num_heads == 4
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.encoding == "uint8"
    assert cfg.experiment.model.predict_target == "x0"
    assert cfg.experiment.model.self_conditioning is True
    assert cfg.experiment.model.classes == -1
    assert cfg.experiment.sampler.method == "ddim"
    assert cfg.experiment.sampler.time_difference == 0.0
    assert cfg.experiment.sampler.stochasticity == 0.0
    assert cfg.experiment.training.sample_timesteps == 256


def test_sjd_cifar10_preserves_logging_and_canonical_anchor_overrides():
    cfg = _compose(["experiment=cifar10/sjd_cifar10"])

    assert cfg.experiment.training.sample_timesteps == 256
    assert cfg.experiment.training.log_state_dependency is True
    assert cfg.experiment.training.state_dep_log_ratio_clip == cfg.experiment.sampler.log_ratio_clip
    assert cfg.experiment.sampler.categorical_sampling_policy == "legacy_low"
    assert cfg.experiment.model.anchor.family == "thermometer"
    assert cfg.experiment.model.anchor.learnable is False
    assert "outside_embed" not in cfg.experiment.model

    override_cfg = _compose(
        [
            "experiment=cifar10/sjd_cifar10",
            "model/anchor@experiment.model.anchor=ordered_normal",
        ]
    )
    assert override_cfg.experiment.model.anchor.family == "ordered_normal"
    assert override_cfg.experiment.model.anchor.learnable is True


def test_md4_cifar10_uses_md4_architecture_bundle():
    cfg = _compose(["experiment=cifar10/md4_cifar10"])

    assert cfg.experiment.model.image_backbone == "unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 128
    assert cfg.experiment.model.ch_mult == [1]
    assert cfg.experiment.sampler.method == "ancestral"
    assert cfg.experiment.sampler.categorical_sampling_policy == "legacy_low"
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.sampler.topp == 0.98
    assert cfg.experiment.training.sample_timesteps == 256


def test_mdlm_cifar10_uses_canonical_adm_image_bundle():
    cfg = _compose(["experiment=cifar10/mdlm_cifar10"])

    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 96
    assert cfg.experiment.model.ch_mult == [3, 4, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 4
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
    assert cfg.experiment.model.adm_num_heads == 4
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.name == "mdlm"
    assert cfg.experiment.model.classes == -1
    assert cfg.experiment.sampler.method == "ancestral"
    assert cfg.experiment.sampler.sampling_grid == "cosine"
    assert cfg.experiment.training.sample_timesteps == 256


def test_d3pm_cifar10_variants_use_canonical_adm_image_bundle():
    expectations = {
        "d3pm_absorb_cifar10": "absorb",
        "d3pm_uniform_cifar10": "uniform",
        "d3pm_gaussian_cifar10": "gaussian",
    }

    for experiment, transition_type in expectations.items():
        cfg = _compose([f"experiment=cifar10/{experiment}"])

        assert cfg.experiment.model.name == "d3pm"
        assert cfg.experiment.model.transition_type == transition_type
        assert cfg.experiment.model.image_backbone == "adm_unet5d"
        assert cfg.experiment.model.sequence_backbone == "auto"
        assert cfg.experiment.model.feature_dim == 96
        assert cfg.experiment.model.ch_mult == [3, 4, 4]
        assert cfg.experiment.model.adm_num_res_blocks == 4
        assert cfg.experiment.model.adm_attention_resolutions == [2, 4]
        assert cfg.experiment.model.adm_num_heads == 4
        assert cfg.experiment.model.adm_num_head_channels == 64
        assert cfg.experiment.model.adm_use_scale_shift_norm is True
        assert cfg.experiment.model.adm_use_new_attention_order is False
        assert cfg.experiment.model.classes == -1
        assert cfg.experiment.sampler.method == "ancestral"
        assert cfg.experiment.sampler.sampling_grid == "uniform"
        assert cfg.experiment.training.sample_timesteps == 256


def test_ddpm_cifar10_sampler_tracks_model_timesteps():
    cfg = _compose(["experiment=cifar10/ddpm_cifar10"])

    assert cfg.experiment.sampler.n_steps == cfg.experiment.model.timesteps
    assert cfg.experiment.training.sample_timesteps == 1000


# `cifar10_report` eval profile was removed in the config cleanup.
