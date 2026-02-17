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


def _sample_rotation_flip_transform(
    *,
    image,
    tf,
    seed,
    use_rotate: bool,
    use_hflip: bool,
):
    """Sample a non-identity, pixel-preserving rotation/flip transform."""
    if use_rotate and use_hflip:
        # 7 non-identity D4 transforms: R90, R180, R270, F, F+R90, F+R180, F+R270
        mode = tf.random.stateless_uniform(
            shape=[],
            seed=seed,
            minval=0,
            maxval=7,
            dtype=tf.int32,
        )

        def _rot(k):
            return tf.image.rot90(image, k=k)

        branch_fns = [
            lambda: _rot(1),
            lambda: _rot(2),
            lambda: _rot(3),
            lambda: tf.image.flip_left_right(image),
            lambda: tf.image.flip_left_right(_rot(1)),
            lambda: tf.image.flip_left_right(_rot(2)),
            lambda: tf.image.flip_left_right(_rot(3)),
        ]
        return tf.switch_case(mode, branch_fns=branch_fns)

    if use_rotate:
        k = tf.random.stateless_uniform(
            shape=[],
            seed=seed,
            minval=1,
            maxval=4,
            dtype=tf.int32,
        )
        return tf.image.rot90(image, k=k)

    if use_hflip:
        return tf.image.flip_left_right(image)

    return image


def _maybe_augment_discrete_image(
    *,
    image,
    tf,
    idx,
    seed: int,
    prob: float,
    use_rotate: bool,
    use_hflip: bool,
):
    """Apply augmentation with probability `prob`, preserving exact pixel values."""
    if prob <= 0.0 or (not use_rotate and not use_hflip):
        return image

    idx32 = tf.cast(
        tf.math.floormod(
            tf.cast(idx, tf.int64),
            tf.constant(2**31 - 1, dtype=tf.int64),
        ),
        tf.int32,
    )
    seed0 = tf.constant(int(seed) % (2**31 - 1), dtype=tf.int32)
    seed_pair = tf.stack(
        [
            seed0,
            idx32,
        ],
        axis=0,
    )
    p = tf.random.stateless_uniform(shape=[], seed=seed_pair, minval=0.0, maxval=1.0)
    do_aug = p < tf.constant(float(prob), dtype=tf.float32)
    aug_seed = tf.stack([seed_pair[1], seed_pair[0] ^ tf.constant(0x9E3779B9, tf.int32)], axis=0)

    aug_image = _sample_rotation_flip_transform(
        image=image,
        tf=tf,
        seed=aug_seed,
        use_rotate=bool(use_rotate),
        use_hflip=bool(use_hflip),
    )
    return tf.cond(do_aug, lambda: aug_image, lambda: image)


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
    """Returns an iterator over CIFAR-10 batches with integer tokens.

    Each batch is a dict:
      - `image`: int32 array [B, 32, 32, 3] in [0, 255]
      - `label`: int32 array [B]

    Optional train-time augmentation is pixel-preserving: random 90-degree
    rotations and horizontal flips with per-example probability `augment_prob`.
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

    if augment_prob < 0.0 or augment_prob > 1.0:
        raise ValueError(f"augment_prob must be in [0, 1], got {augment_prob}")

    do_augment = bool(augment) and (bool(augment_rotate) or bool(augment_hflip)) and (augment_prob > 0.0)

    def _prep(ex):
        return {
            "image": tf.cast(ex["image"], tf.int32),
            "label": tf.cast(ex["label"], tf.int32),
        }

    if do_augment:
        def _prep_aug(idx, ex):
            image = tf.cast(ex["image"], tf.int32)
            image = _maybe_augment_discrete_image(
                image=image,
                tf=tf,
                idx=idx,
                seed=int(seed),
                prob=float(augment_prob),
                use_rotate=bool(augment_rotate),
                use_hflip=bool(augment_hflip),
            )
            return {
                "image": image,
                "label": tf.cast(ex["label"], tf.int32),
            }

        ds = ds.enumerate()
        ds = ds.map(_prep_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(int(batch_size), drop_remainder=bool(drop_remainder))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return iter(tfds.as_numpy(ds))
