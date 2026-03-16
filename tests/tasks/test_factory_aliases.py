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


def test_build_task_accepts_cadd_paper_alias(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_discrete",
        SimpleNamespace(CIFAR10DiscreteTask=_DummyDiscreteTask),
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": "cadd_cifar10_paper"},
            "dataset": {
                "data_dir": "/tmp/cifar10",
                "batch_size": 512,
                "eval_batch_size": 512,
                "vocab_size": 256,
                "num_classes": -1,
                "augment": {
                    "enabled": True,
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
    assert task.kwargs["task_name"] == "cadd_cifar10_paper"
    assert task.kwargs["batch_size"] == 512
    assert task.kwargs["augment_enabled"] is True


def test_build_task_accepts_sjd_paper_alias(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sticky.tasks.cifar10_sjd",
        SimpleNamespace(CIFAR10SJDTask=_DummySJDTask),
    )

    def _fake_instantiate(cfg, **kwargs):
        data = OmegaConf.to_container(cfg, resolve=True)
        if kwargs:
            return {"cfg": data, **kwargs}
        return {"cfg": data}

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    cfg = OmegaConf.create(
        {
            "task": {"name": "sjd_cifar10_paper"},
            "dataset": {
                "data_dir": "/tmp/cifar10",
                "batch_size": 512,
                "eval_batch_size": 512,
                "data_shape": [32, 32, 3],
                "vocab_size": 256,
                "num_classes": -1,
                "augment": {
                    "enabled": True,
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
    assert task.kwargs["batch_size"] == 512
    assert task.kwargs["augment_enabled"] is True
    assert task.kwargs["beta"] == {"cfg": {"name": "vp_linear"}}
    assert task.kwargs["hazard"] == {"cfg": {"name": "poly_alpha", "p": 3.0}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["jump"] == {"cfg": {"name": "vp_matched"}, "beta": {"cfg": {"name": "vp_linear"}}}
    assert task.kwargs["state_dep_log_ratio_clip"] == 7.0
