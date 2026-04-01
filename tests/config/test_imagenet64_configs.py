from __future__ import annotations

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root


CONFIG_DIR = str(config_root())


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_imagenet64_experiments_compose():
    for experiment, model_name in (
        ("md4_imagenet64", "md4"),
        ("sjd_imagenet64", "sjd"),
    ):
        cfg = _compose([f"experiment=imagenet64/{experiment}", "eval=imagenet64"])

        assert cfg.experiment.task.name == experiment
        assert cfg.experiment.model.name == model_name
        assert cfg.experiment.dataset.tfds_name == "downsampled_imagenet/64x64"
        assert cfg.experiment.dataset.data_shape == [64, 64, 3]
        assert cfg.experiment.dataset.num_classes == -1
        assert cfg.experiment.dataset.eval_split == "validation"
        assert cfg.experiment.dataset.shuffle_buffer_size == 50000
        assert cfg.experiment.training.best_checkpoint_metric == "eval/fid"
        assert cfg.experiment.training.sample_timesteps == cfg.experiment.sampler.n_steps
        assert cfg.eval.fid_dataset_name == cfg.experiment.dataset.tfds_name
        assert cfg.eval.fid_split == "validation"
        assert cfg.eval.enabled is True
        assert cfg.eval.run_at_end is True


def test_imagenet64_md4_uses_64x64_adm_backbone_and_512_step_sampling():
    cfg = _compose(["experiment=imagenet64/md4_imagenet64", "eval=imagenet64"])

    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 192
    assert cfg.experiment.model.ch_mult == [1, 2, 3, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 3
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4, 8]
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.timesteps == 512
    assert cfg.experiment.sampler.n_steps == 512
    assert cfg.experiment.training.sample_timesteps == 512
    assert cfg.experiment.model.ch_mult != [3, 4, 4]
    assert cfg.experiment.model.adm_attention_resolutions != [2, 4]


def test_imagenet64_sjd_uses_64x64_adm_backbone_and_keeps_512_nfe_path():
    cfg = _compose(["experiment=imagenet64/sjd_imagenet64", "eval=imagenet64"])

    assert cfg.experiment.model.image_backbone == "adm_unet5d"
    assert cfg.experiment.model.sequence_backbone == "auto"
    assert cfg.experiment.model.feature_dim == 192
    assert cfg.experiment.model.ch_mult == [1, 2, 3, 4]
    assert cfg.experiment.model.adm_num_res_blocks == 3
    assert cfg.experiment.model.adm_attention_resolutions == [2, 4, 8]
    assert cfg.experiment.model.adm_num_head_channels == 64
    assert cfg.experiment.model.ch_mult != [3, 4, 4]
    assert cfg.experiment.model.adm_attention_resolutions != [2, 4]
    assert cfg.experiment.sampler.n_steps == 256
    assert cfg.experiment.sampler.refresh_logits_after_em_step is True
    assert cfg.experiment.training.sample_timesteps == 256


def test_imagenet64_report_profile_composes():
    for experiment in ("md4_imagenet64", "sjd_imagenet64"):
        cfg = _compose([f"experiment=imagenet64/{experiment}", "eval=imagenet64_report"])

        assert cfg.experiment.task.name == experiment
        assert cfg.eval.fid_split == "validation"
        assert cfg.eval.fid_every == 0
        assert cfg.eval.fid_num_samples == 50000
        assert cfg.eval.fid_cache_dir == "data/fid_stats"
