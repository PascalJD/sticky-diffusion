from __future__ import annotations

from hydra import compose, initialize_config_dir

from sticky.core.config_paths import config_root


CONFIG_DIR = str(config_root())


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_openwebtext_experiments_compose():
    cases = [
        ("openwebtext/sjd_openwebtext", "openwebtext_sjd", "sjd"),
        ("openwebtext/mdlm_openwebtext", "mdlm_openwebtext", "mdlm"),
        ("openwebtext/md4_openwebtext", "md4_openwebtext", "md4"),
    ]
    for exp_path, task_name, model_name in cases:
        cfg = _compose(
            [f"experiment={exp_path}"]
        )
        assert cfg.experiment.task.name == task_name
        assert cfg.experiment.model.name == model_name
        assert cfg.experiment.dataset.seq_len == 1024
        assert cfg.experiment.dataset.vocab_size == 50257
        # All three share the 12x768 backbone for fair comparison.
        assert cfg.experiment.model.sequence_backbone == "transformer"
        assert cfg.experiment.model.n_layers == 12
        assert cfg.experiment.model.feature_dim == 64
        assert cfg.experiment.model.num_heads == 12


def test_sjd_openwebtext_uses_pretrained_anchors():
    cfg = _compose(["experiment=openwebtext/sjd_openwebtext"])
    assert cfg.experiment.model.anchor.family == "pretrained"
    assert cfg.experiment.model.anchor.dim == 768
    assert cfg.experiment.model.anchor.learnable is False
    assert str(cfg.experiment.model.anchor.pretrained_path).endswith(
        "gpt2_wte.npz"
    )


def test_sjd_openwebtext_uses_antithetic_and_alpha_deriv():
    cfg = _compose(["experiment=openwebtext/sjd_openwebtext"])
    assert cfg.experiment.training.time_sampling == "antithetic"
    assert cfg.experiment.training.loss_weighting == "alpha_deriv"
