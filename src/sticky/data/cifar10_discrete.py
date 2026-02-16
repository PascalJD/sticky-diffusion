from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Optional


def _resolve_data_dir(data_dir: Optional[str]) -> Optional[str]:
    if data_dir in (None, "", "null"):
        return None

    p = Path(str(data_dir))
    if p.is_absolute():
        return str(p)

    try:
        import hydra

        return str(Path(hydra.utils.get_original_cwd()) / p)
    except Exception:
        return str(p.resolve())


def _require_tf():
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError(
            "TensorFlow is required for CIFAR-10 data loading. "
            "Install dependencies from environment.yml."
        ) from e
    return tf


def _require_tfds():
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise ImportError(
            "tensorflow-datasets is required for CIFAR-10 data loading. "
            "Install dependencies from environment.yml."
        ) from e
    return tfds


def make_cifar10_iterator(
    *,
    split: str,
    batch_size: int,
    seed: int = 0,
    data_dir: Optional[str] = None,
    shuffle: bool = True,
    repeat: bool = False,
    drop_remainder: bool = True,
) -> Iterator[Mapping[str, object]]:
    """Returns an iterator over CIFAR-10 batches with integer tokens.

    Each batch is a dict:
      - `image`: int32 array [B, 32, 32, 3] in [0, 255]
      - `label`: int32 array [B]
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    tf = _require_tf()
    tfds = _require_tfds()

    resolved_data_dir = _resolve_data_dir(data_dir)
    ds = tfds.load(
        "cifar10",
        split=str(split),
        data_dir=resolved_data_dir,
        shuffle_files=bool(shuffle),
        as_supervised=False,
    )

    if shuffle:
        # CIFAR-10 train set has 50k samples; this is a reasonable default.
        ds = ds.shuffle(
            buffer_size=50_000,
            seed=int(seed),
            reshuffle_each_iteration=True,
        )

    if repeat:
        ds = ds.repeat()

    def _prep(ex):
        return {
            "image": tf.cast(ex["image"], tf.int32),
            "label": tf.cast(ex["label"], tf.int32),
        }

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(int(batch_size), drop_remainder=bool(drop_remainder))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return iter(tfds.as_numpy(ds))

