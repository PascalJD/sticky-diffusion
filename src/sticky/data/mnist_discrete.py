# src/sticky/data/mnist_discrete.py
from __future__ import annotations
import jax, jax.numpy as jnp
import tensorflow_datasets as tfds
from typing import Tuple

def _prep(example):
    x = example['image']  # uint8 [28,28,1]
    x = jnp.array(x, dtype=jnp.float32) / 255.0
    x = x.reshape(-1)  # vectorize to (784,)
    return x

def load_mnist_split(split='train', batch_size=128, shuffle=True, seed=0):
    ds = tfds.load('mnist', split=split, as_supervised=False)
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.map(_prep).batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)

def make_bins(L: int):
    # e.g., L=2 -> [0.0, 1.0]; L=4 -> quartiles
    if L == 2:
        return jnp.array([0.0, 1.0], dtype=jnp.float32)
    else:
        return jnp.linspace(0.0, 1.0, L)

def dataset_stats(train_iter, n_batches=200):
    # estimate mean/cov (vectorized)
    import numpy as np
    xs = []
    for i, batch in enumerate(train_iter):
        xs.append(np.array(batch))
        if i+1 >= n_batches: break
    X = jnp.array(jnp.concatenate(xs, axis=0))
    mu = jnp.mean(X, axis=0)
    Xc = X - mu
    Sig = (Xc.T @ Xc) / X.shape[0]
    return mu, Sig