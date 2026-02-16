# src/sticky/tasks/factory.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig


def build_task(cfg: DictConfig):
    name = cfg.task.name
    if name == "md4_cifar10":
        from sticky.tasks.cifar10_md4 import CIFAR10MD4Task

        return CIFAR10MD4Task(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
        )

    if name == "sjd_cifar10":
        from sticky.tasks.cifar10_sjd import CIFAR10SJDTask

        beta = hydra.utils.instantiate(cfg.forward.beta)
        hazard_cfg = cfg.forward.get("hazard", None)
        hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None
        T = float(cfg.sampler.get("T", getattr(beta, "T", 1.0)))

        return CIFAR10SJDTask(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            data_shape=tuple(cfg.dataset.data_shape),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
            beta=beta,
            hazard=hazard,
            T=T,
        )

    raise ValueError(f"Unknown task.name={name}")
