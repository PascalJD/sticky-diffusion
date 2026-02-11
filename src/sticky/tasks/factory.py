# src/sticky/tasks/factory.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from sticky.tasks.cifar10_md4 import CIFAR10MD4Task
from sticky.tasks.cifar10_sjd import CIFAR10SJDTask


def build_task(cfg: DictConfig):
    name = cfg.task.name
    if name == "md4_cifar10":
        return CIFAR10MD4Task(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
        )

    if name == "sjd_cifar10":
        beta = hydra.utils.instantiate(cfg.forward.beta)
        T = float(cfg.sampler.get("T", getattr(beta, "T", 1.0)))

        return CIFAR10SJDTask(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            data_shape=tuple(cfg.dataset.data_shape),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
            beta=beta,
            T=T,
        )

    raise ValueError(f"Unknown task.name={name}")
