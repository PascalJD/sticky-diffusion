from __future__ import annotations

import sys
from types import SimpleNamespace

import hydra
from omegaconf import OmegaConf

from sticky.tasks.factory import build_task


class _DummyDiscreteTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummySJDTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyOpenWebTextTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummySudokuInpaintMDMTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummySudokuMDLMTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummySudokuInpaintSJDTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_instantiate(cfg, **kwargs):
    data = OmegaConf.to_container(cfg, resolve=True)
    if kwargs:
        return {"cfg": data, **kwargs}
    return {"cfg": data}


def test_build_task_accepts_canonical_discrete_cifar_names(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_discrete",
        SimpleNamespace(CIFAR10DiscreteTask=_DummyDiscreteTask),
    )

    for task_name in (
        "bitdiff_cifar10",
        "candi_cifar10",
        "cadd_cifar10",
        "ddpm_cifar10",
        "md4_cifar10",
        "mdlm_cifar10",
        "d3pm_absorb_cifar10",
        "d3pm_uniform_cifar10",
        "d3pm_gaussian_cifar10",
    ):
        cfg = OmegaConf.create(
            {
                "task": {"name": task_name},
                "dataset": {
                    "data_dir": "/tmp/cifar10",
                    "tfds_name": "cifar10",
                    "train_split": "train",
                    "eval_split": "test",
                    "include_label": "auto",
                    "batch_size": 512,
                    "eval_batch_size": 512,
                    "data_shape": [32, 32, 3],
                    "vocab_size": 256,
                    "num_classes": -1,
                    "augment": {
                        "enabled": False,
                        "prob": 0.15,
                        "rotate": True,
                        "hflip": True,
                        "eval": False,
                    },
                },
            }
        )

        task = build_task(cfg)

        assert isinstance(task, _DummyDiscreteTask)
        assert task.kwargs["task_name"] == task_name
        assert task.kwargs["dataset_name"] == "cifar10"
        assert task.kwargs["train_split"] == "train"
        assert task.kwargs["eval_split"] == "test"
        assert task.kwargs["include_label"] == "auto"
        assert task.kwargs["batch_size"] == 512
        assert task.kwargs["data_shape"] == (32, 32, 3)
        assert task.kwargs["shuffle_buffer_size"] is None
        assert task.kwargs["augment_enabled"] is False


def test_build_task_accepts_canonical_sjd_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_sjd",
        SimpleNamespace(CIFAR10SJDTask=_DummySJDTask),
    )

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    cfg = OmegaConf.create(
        {
            "task": {"name": "sjd_cifar10"},
            "dataset": {
                "data_dir": "/tmp/cifar10",
                "tfds_name": "cifar10",
                "train_split": "train",
                "eval_split": "test",
                "include_label": "auto",
                "batch_size": 512,
                "eval_batch_size": 512,
                "data_shape": [32, 32, 3],
                "vocab_size": 256,
                "num_classes": -1,
                "augment": {
                    "enabled": False,
                    "prob": 0.15,
                    "rotate": True,
                    "hflip": True,
                    "eval": False,
                },
            },
            "forward": {
                "beta": {"name": "vp_linear"},
                "hazard": {"name": "poly_alpha", "p": 3.0},
                "jump": {"name": "vp_matched"},
            },
            "sampler": {
                "T": 1.0,
                "log_ratio_clip": 10.0,
            },
            "training": {
                "log_state_dependency": True,
                "state_dep_log_ratio_clip": 7.0,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummySJDTask)
    assert task.kwargs["task_name"] == "sjd_cifar10"
    assert task.kwargs["dataset_name"] == "cifar10"
    assert task.kwargs["train_split"] == "train"
    assert task.kwargs["eval_split"] == "test"
    assert task.kwargs["include_label"] == "auto"
    assert task.kwargs["batch_size"] == 512
    assert task.kwargs["augment_enabled"] is False
    assert task.kwargs["beta"] == {"cfg": {"name": "vp_linear"}}
    assert task.kwargs["hazard"] == {"cfg": {"name": "poly_alpha", "p": 3.0}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["jump"] == {"cfg": {"name": "vp_matched"}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["state_dep_log_ratio_clip"] == 7.0


def test_build_task_accepts_canonical_mdlm_sudoku_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.sudoku_mdlm",
        SimpleNamespace(SudokuMDLMTask=_DummySudokuMDLMTask),
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": "mdlm_sudoku"},
            "dataset": {
                "data_dir": "/tmp/sudoku",
                "train_file": "Sudoku-train-data.npy",
                "test_file": "Sudoku-test-data.npy",
                "batch_size": 128,
                "eval_batch_size": 64,
                "data_shape": [81],
                "vocab_size": 10,
                "num_classes": -1,
                "drop_remainder": True,
                "shuffle": True,
                "mmap": True,
                "max_train_examples": -1,
                "max_test_examples": -1,
                "auto_download": False,
                "download_timeout_sec": 30,
                "download_retries": 2,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummySudokuMDLMTask)
    assert task.kwargs["data_dir"] == "/tmp/sudoku"
    assert task.kwargs["train_file"] == "Sudoku-train-data.npy"
    assert task.kwargs["batch_size"] == 128
    assert task.kwargs["eval_batch_size"] == 64
    assert task.kwargs["data_shape"] == (81,)
    assert task.kwargs["vocab_size"] == 10


def test_build_task_accepts_imagenet64_discrete_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_discrete",
        SimpleNamespace(CIFAR10DiscreteTask=_DummyDiscreteTask),
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": "md4_imagenet64"},
            "dataset": {
                "data_dir": "/tmp/tfds",
                "tfds_name": "downsampled_imagenet/64x64",
                "train_split": "train",
                "eval_split": "validation",
                "include_label": False,
                "batch_size": 128,
                "eval_batch_size": 128,
                "shuffle_buffer_size": 50000,
                "data_shape": [64, 64, 3],
                "vocab_size": 256,
                "num_classes": -1,
                "augment": {
                    "enabled": False,
                    "prob": 0.15,
                    "rotate": True,
                    "hflip": True,
                    "eval": False,
                },
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummyDiscreteTask)
    assert task.kwargs["task_name"] == "md4_imagenet64"
    assert task.kwargs["dataset_name"] == "downsampled_imagenet/64x64"
    assert task.kwargs["train_split"] == "train"
    assert task.kwargs["eval_split"] == "validation"
    assert task.kwargs["include_label"] is False
    assert task.kwargs["data_shape"] == (64, 64, 3)
    assert task.kwargs["shuffle_buffer_size"] == 50000


def test_build_task_accepts_imagenet64_sjd_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_sjd",
        SimpleNamespace(CIFAR10SJDTask=_DummySJDTask),
    )

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    cfg = OmegaConf.create(
        {
            "task": {"name": "sjd_imagenet64"},
            "dataset": {
                "data_dir": "/tmp/tfds",
                "tfds_name": "downsampled_imagenet/64x64",
                "train_split": "train",
                "eval_split": "validation",
                "include_label": False,
                "batch_size": 128,
                "eval_batch_size": 64,
                "shuffle_buffer_size": 50000,
                "data_shape": [64, 64, 3],
                "vocab_size": 256,
                "num_classes": -1,
                "augment": {
                    "enabled": False,
                    "prob": 0.15,
                    "rotate": True,
                    "hflip": True,
                    "eval": False,
                },
            },
            "forward": {
                "beta": {"name": "vp_linear"},
                "hazard": {"name": "poly_alpha", "p": 3.0},
                "jump": {"name": "vp_matched"},
            },
            "sampler": {
                "T": 1.0,
                "log_ratio_clip": 10.0,
            },
            "training": {
                "log_state_dependency": True,
                "state_dep_log_ratio_clip": 6.5,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummySJDTask)
    assert task.kwargs["task_name"] == "sjd_imagenet64"
    assert task.kwargs["dataset_name"] == "downsampled_imagenet/64x64"
    assert task.kwargs["train_split"] == "train"
    assert task.kwargs["eval_split"] == "validation"
    assert task.kwargs["include_label"] is False
    assert task.kwargs["data_shape"] == (64, 64, 3)
    assert task.kwargs["shuffle_buffer_size"] == 50000
    assert task.kwargs["beta"] == {"cfg": {"name": "vp_linear"}}
    assert task.kwargs["state_dep_log_ratio_clip"] == 6.5


def test_build_task_accepts_openwebtext_discrete_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.openwebtext_discrete",
        SimpleNamespace(OpenWebTextDiscreteTask=_DummyOpenWebTextTask),
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": "openwebtext_discrete"},
            "dataset": {
                "train_tokens_path": "/tmp/owt_train.npy",
                "eval_tokens_path": "/tmp/owt_eval.npz",
                "batch_size": 32,
                "eval_batch_size": 16,
                "seq_len": 1024,
                "vocab_size": 50257,
                "tokenizer_name": "gpt2",
                "num_classes": -1,
                "drop_remainder": True,
                "shuffle": True,
                "mmap": True,
                "max_train_examples": 128,
                "max_eval_examples": 64,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummyOpenWebTextTask)
    assert task.kwargs["task_name"] == "openwebtext_discrete"
    assert task.kwargs["train_tokens_path"] == "/tmp/owt_train.npy"
    assert task.kwargs["eval_tokens_path"] == "/tmp/owt_eval.npz"
    assert task.kwargs["batch_size"] == 32
    assert task.kwargs["eval_batch_size"] == 16
    assert task.kwargs["seq_len"] == 1024
    assert task.kwargs["vocab_size"] == 50257
    assert task.kwargs["tokenizer_name"] == "gpt2"


def test_build_task_accepts_mdm_sudoku_inpaint_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.sudoku_inpaint_mdm",
        SimpleNamespace(SudokuInpaintMDMTask=_DummySudokuInpaintMDMTask),
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": "mdm_sudoku_inpaint"},
            "dataset": {
                "data_dir": "/tmp/sudoku",
                "train_file": "train.npy",
                "test_file": "test.npy",
                "batch_size": 128,
                "eval_batch_size": 64,
                "data_shape": [81],
                "vocab_size": 10,
                "num_classes": -1,
                "drop_remainder": True,
                "shuffle": True,
                "mmap": True,
                "max_train_examples": -1,
                "max_test_examples": -1,
                "auto_download": True,
                "download_timeout_sec": 120,
                "download_retries": 8,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummySudokuInpaintMDMTask)
    assert task.kwargs["batch_size"] == 128
    assert task.kwargs["eval_batch_size"] == 64
    assert task.kwargs["data_shape"] == (81,)
    assert task.kwargs["vocab_size"] == 10


def test_build_task_accepts_sjd_sudoku_inpaint_name(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.sudoku_inpaint_sjd",
        SimpleNamespace(SudokuInpaintSJDTask=_DummySudokuInpaintSJDTask),
    )
    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    cfg = OmegaConf.create(
        {
            "task": {"name": "sjd_sudoku_inpaint"},
            "dataset": {
                "data_dir": "/tmp/sudoku",
                "train_file": "train.npy",
                "test_file": "test.npy",
                "batch_size": 128,
                "eval_batch_size": 64,
                "data_shape": [81],
                "vocab_size": 10,
                "num_classes": -1,
                "drop_remainder": True,
                "shuffle": True,
                "mmap": True,
                "max_train_examples": -1,
                "max_test_examples": -1,
                "auto_download": True,
                "download_timeout_sec": 120,
                "download_retries": 8,
            },
            "forward": {
                "beta": {"name": "vp_linear"},
                "hazard": {"name": "poly_alpha", "p": 1.0},
                "jump": {"name": "vp_matched", "eta": 0.97},
            },
            "sampler": {
                "T": 1.0,
                "log_ratio_clip": 10.0,
            },
            "training": {
                "log_state_dependency": True,
                "state_dep_log_ratio_clip": 6.0,
            },
        }
    )

    task = build_task(cfg)

    assert isinstance(task, _DummySudokuInpaintSJDTask)
    assert task.kwargs["batch_size"] == 128
    assert task.kwargs["eval_batch_size"] == 64
    assert task.kwargs["data_shape"] == (81,)
    assert task.kwargs["vocab_size"] == 10
    assert task.kwargs["beta"] == {"cfg": {"name": "vp_linear"}}
    assert task.kwargs["hazard"] == {"cfg": {"name": "poly_alpha", "p": 1.0}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["jump"] == {"cfg": {"name": "vp_matched", "eta": 0.97}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["state_dep_log_ratio_clip"] == 6.0
