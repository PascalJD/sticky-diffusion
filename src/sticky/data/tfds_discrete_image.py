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
            "TensorFlow is required for TFDS discrete image data loading. "
            "Install dependencies from environment.yml."
        ) from e
    return tf


def _require_tfds():
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise ImportError(
            "tensorflow-datasets is required for TFDS discrete image data loading. "
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


def _has_label_key(ds) -> bool:
    element_spec = getattr(ds, "element_spec", None)
    return isinstance(element_spec, Mapping) and ("label" in element_spec)


def _dummy_label(*, tf, dummy_label_value: int):
    return tf.cast(dummy_label_value, tf.int32)


def _shuffle_buffer_size(ds, *, tf, requested_size: int | None = None) -> int:
    if requested_size is not None:
        if int(requested_size) <= 0:
            raise ValueError(
                f"shuffle_buffer_size must be positive when provided, got {requested_size}"
            )
        return int(requested_size)

    try:
        cardinality = tf.data.experimental.cardinality(ds)
        value = int(cardinality.numpy())
    except Exception:
        return 50_000
    if value <= 0:
        return 50_000
    return min(value, 50_000)


def make_tfds_discrete_image_iterator(
    dataset_name,
    split,
    batch_size,
    seed=0,
    data_dir=None,
    shuffle=True,
    repeat=False,
    drop_remainder=True,
    augment=False,
    augment_prob=0.15,
    augment_rotate=True,
    augment_hflip=True,
    include_label="auto",
    dummy_label_value=-1,
    shuffle_buffer_size=None,
) -> Iterator[Mapping[str, object]]:
    """Return a TFDS-backed iterator with uint8 images and always-present labels."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if include_label not in ("auto", True, False):
        raise ValueError(
            f"include_label must be one of 'auto', True, or False; got {include_label!r}"
        )

    if augment_prob < 0.0 or augment_prob > 1.0:
        raise ValueError(f"augment_prob must be in [0, 1], got {augment_prob}")

    tf = _require_tf()
    tfds = _require_tfds()

    resolved_data_dir = _resolve_data_dir(data_dir)
    ds = tfds.load(
        str(dataset_name),
        split=str(split),
        data_dir=resolved_data_dir,
        shuffle_files=bool(shuffle),
        as_supervised=False,
    )

    has_label = _has_label_key(ds)
    if include_label is True and not has_label:
        raise ValueError(
            f"TFDS dataset {dataset_name!r} split {split!r} does not provide a 'label' feature."
        )

    use_dataset_label = bool(has_label) and (include_label != False)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=_shuffle_buffer_size(
                ds,
                tf=tf,
                requested_size=shuffle_buffer_size,
            ),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )

    if repeat:
        ds = ds.repeat()

    do_augment = bool(augment) and (bool(augment_rotate) or bool(augment_hflip)) and (augment_prob > 0.0)

    def _prep(ex):
        image = tf.cast(ex["image"], tf.uint8)
        label = (
            tf.cast(ex["label"], tf.int32)
            if use_dataset_label
            else _dummy_label(tf=tf, dummy_label_value=int(dummy_label_value))
        )
        return {
            "image": image,
            "label": label,
        }

    if do_augment:

        def _prep_aug(idx, ex):
            image = tf.cast(ex["image"], tf.uint8)
            image = _maybe_augment_discrete_image(
                image=image,
                tf=tf,
                idx=idx,
                seed=int(seed),
                prob=float(augment_prob),
                use_rotate=bool(augment_rotate),
                use_hflip=bool(augment_hflip),
            )
            label = (
                tf.cast(ex["label"], tf.int32)
                if use_dataset_label
                else _dummy_label(tf=tf, dummy_label_value=int(dummy_label_value))
            )
            return {
                "image": image,
                "label": label,
            }

        ds = ds.enumerate()
        ds = ds.map(_prep_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(int(batch_size), drop_remainder=bool(drop_remainder))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return iter(tfds.as_numpy(ds))
