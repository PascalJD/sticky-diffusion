# src/sticky/tasks/factory.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig


def build_task(cfg: DictConfig):
    name = cfg.task.name
    aug_cfg = cfg.dataset.get("augment", {})
    aug_enabled = bool(aug_cfg.get("enabled", True))
    aug_prob = float(aug_cfg.get("prob", 0.15))
    aug_rotate = bool(aug_cfg.get("rotate", True))
    aug_hflip = bool(aug_cfg.get("hflip", True))
    aug_eval = bool(aug_cfg.get("eval", False))

    if name == "md4_cifar10":
        from sticky.tasks.cifar10_md4 import CIFAR10MD4Task

        return CIFAR10MD4Task(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
            augment_enabled=aug_enabled,
            augment_prob=aug_prob,
            augment_rotate=aug_rotate,
            augment_hflip=aug_hflip,
            augment_eval=aug_eval,
        )

    if name == "cadd_cifar10":
        from sticky.tasks.cifar10_cadd import CIFAR10CADDTask

        return CIFAR10CADDTask(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
            augment_enabled=aug_enabled,
            augment_prob=aug_prob,
            augment_rotate=aug_rotate,
            augment_hflip=aug_hflip,
            augment_eval=aug_eval,
        )

    if name == "sjd_cifar10":
        from sticky.tasks.cifar10_sjd import CIFAR10SJDTask

        beta = hydra.utils.instantiate(cfg.forward.beta)
        hazard_cfg = cfg.forward.get("hazard", None)
        hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None
        jump_cfg = cfg.forward.get("jump", None)
        jump = hydra.utils.instantiate(jump_cfg, beta=beta) if jump_cfg is not None else None
        T = float(cfg.sampler.get("T", getattr(beta, "T", 1.0)))
        log_state_dependency = bool(cfg.training.get("log_state_dependency", True))
        state_dep_log_ratio_clip = float(
            cfg.training.get(
                "state_dep_log_ratio_clip",
                cfg.sampler.get("log_ratio_clip", 10.0),
            )
        )

        return CIFAR10SJDTask(
            data_dir=str(cfg.dataset.get("data_dir", None)),
            batch_size=int(cfg.dataset.get("batch_size")),
            eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
            data_shape=tuple(cfg.dataset.data_shape),
            vocab_size=int(cfg.dataset.get("vocab_size", 256)),
            num_classes=int(cfg.dataset.get("num_classes", -1)),
            augment_enabled=aug_enabled,
            augment_prob=aug_prob,
            augment_rotate=aug_rotate,
            augment_hflip=aug_hflip,
            augment_eval=aug_eval,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=T,
            log_state_dependency=log_state_dependency,
            state_dep_log_ratio_clip=state_dep_log_ratio_clip,
        )

    raise ValueError(f"Unknown task.name={name}")
