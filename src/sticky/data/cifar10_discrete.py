from __future__ import annotations

from typing import Iterator, Mapping, Optional

from sticky.data.tfds_discrete_image import make_tfds_discrete_image_iterator


def make_cifar10_iterator(
    *,
    split: str,
    batch_size: int,
    seed: int = 0,
    data_dir: Optional[str] = None,
    shuffle: bool = True,
    repeat: bool = False,
    drop_remainder: bool = True,
    augment: bool = False,
    augment_prob: float = 0.15,
    augment_rotate: bool = True,
    augment_hflip: bool = True,
) -> Iterator[Mapping[str, object]]:
    """Compatibility wrapper for the canonical CIFAR-10 discrete image loader."""

    return make_tfds_discrete_image_iterator(
        dataset_name="cifar10",
        split=str(split),
        batch_size=int(batch_size),
        seed=int(seed),
        data_dir=data_dir,
        shuffle=bool(shuffle),
        repeat=bool(repeat),
        drop_remainder=bool(drop_remainder),
        augment=bool(augment),
        augment_prob=float(augment_prob),
        augment_rotate=bool(augment_rotate),
        augment_hflip=bool(augment_hflip),
        include_label="auto",
        dummy_label_value=-1,
    )
