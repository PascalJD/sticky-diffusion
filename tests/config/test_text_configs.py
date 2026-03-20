from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config.yaml", overrides=overrides)


def test_openwebtext_dataset_training_and_eval_configs_compose():
    cfg = _compose(
        [
            "experiment=md4_cifar10",
            "dataset@experiment.dataset=openwebtext_gpt2_1024",
            "training@experiment.training=owt",
            "eval=text_basic",
            "experiment.task.name=openwebtext_discrete",
        ]
    )

    assert cfg.experiment.dataset.name == "openwebtext_gpt2_1024"
    assert cfg.experiment.dataset.seq_len == 1024
    assert cfg.experiment.dataset.data_shape == [1024]
    assert cfg.experiment.dataset.vocab_size == 50257
    assert cfg.experiment.dataset.tokenizer_name == "gpt2"

    assert cfg.experiment.training.name == "owt"
    assert cfg.experiment.training.log_images_every_steps == 0
    assert cfg.experiment.training.best_checkpoint_metric == "eval/loss"

    assert cfg.eval.mode == "text_basic"
    assert cfg.eval.fid_enabled is False
    assert cfg.eval.is_enabled is False
    assert cfg.eval.text_num_samples == 16
