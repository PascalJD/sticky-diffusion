# src/sticky/data/mnist_numpy.py
from __future__ import annotations
import os, urllib.request
import numpy as np
import jax.numpy as jnp

URL_NPZ = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def _default_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "sticky")

def _ensure_mnist_npz(data_dir: str | None = None) -> str:
    data_dir = data_dir or _default_dir()
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(path):
        print(f"[mnist_numpy] downloading to {path} ...")
        urllib.request.urlretrieve(URL_NPZ, path)
    return path

def _load_split(split: str, data_dir: str | None = None) -> np.ndarray:
    path = _ensure_mnist_npz(data_dir)
    with np.load(path, allow_pickle=False) as f:
        X = f["x_train"] if split == "train" else f["x_test"]
    X = (X.astype(np.float32) / 255.0).reshape(X.shape[0], 28 * 28)
    return X  # (N, 784) in [0,1]

def iter_mnist_split(
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
    data_dir: str | None = None,
):
    rng = np.random.default_rng(seed)
    X = _load_split(split, data_dir)
    N = X.shape[0]
    idx = np.arange(N)

    while True:
        if shuffle:
            rng.shuffle(idx)
        Xs = X[idx]
        # epoch iteration
        for start in range(0, N, batch_size):
            stop = start + batch_size
            if stop > N and drop_remainder:
                break
            yield Xs[start:stop]