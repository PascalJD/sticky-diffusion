from __future__ import annotations

from typing import Optional

from sticky.tasks.cifar10_discrete import CIFAR10DiscreteTask


class CIFAR10MD4Task(CIFAR10DiscreteTask):
    """Backward-compatible wrapper around the shared discrete CIFAR-10 task."""

    def __init__(
        self,
        data_dir: Optional[str],
        batch_size: int,
        eval_batch_size: int,
        vocab_size: int = 256,
        num_classes: int = -1,
        drop_remainder: bool = True,
        augment_enabled: bool = True,
        augment_prob: float = 0.15,
        augment_rotate: bool = True,
        augment_hflip: bool = True,
        augment_eval: bool = False,
    ):
        super().__init__(
            task_name="md4_cifar10",
            data_dir=data_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            vocab_size=vocab_size,
            num_classes=num_classes,
            drop_remainder=drop_remainder,
            augment_enabled=augment_enabled,
            augment_prob=augment_prob,
            augment_rotate=augment_rotate,
            augment_hflip=augment_hflip,
            augment_eval=augment_eval,
        )
