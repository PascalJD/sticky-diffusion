# src/sticky/data/mnist_discrete.py
from __future__ import annotations
from typing import Tuple
import jax.numpy as jnp
import tensorflow as tf

def _import_tfds_cpu_only():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: F401
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    import tensorflow_datasets as tfds
    return tf, tfds

def load_mnist_split(
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
    num_workers: int = 0,
) -> Iterator:
    tf, tfds = _import_tfds_cpu_only()
    ds = tfds.load(
        "mnist",
        split=split,
        shuffle_files=shuffle,
        as_supervised=False,
    )
    def _map(ex):
        x = tf.cast(ex["image"], tf.float32) / 255.0
        x = tf.reshape(x, [28 * 28])
        return x
    num_calls = tf.data.AUTOTUNE if num_workers != 0 else 1
    ds = ds.map(_map, num_parallel_calls=num_calls)
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat() 
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)

def make_bins(L: int):
    if L == 2:
        return jnp.array([0.0, 1.0], dtype=jnp.float32)
    else:
        return jnp.linspace(0.0, 1.0, L, dtype=jnp.float32)

def dataset_stats(train_iter, n_batches: int = 200):
    import numpy as np
    xs = []
    for i, batch in enumerate(train_iter):
        xs.append(np.asarray(batch))
        if i + 1 >= n_batches:
            break
    import jax.numpy as jnp
    X = jnp.array(np.concatenate(xs, axis=0))
    mu = jnp.mean(X, axis=0)
    Xc = X - mu
    Sig = (Xc.T @ Xc) / X.shape[0]
    return mu, Sig