"""Fréchet Inception Distance (FID) computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .inception import InceptionV3FeatureExtractor


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Matrix square root for symmetric PSD matrices via eigen-decomposition."""
    mat = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(mat)
    w = np.maximum(w, eps)
    return (v * np.sqrt(w)) @ v.T


def fid_from_stats(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute FID given two Gaussians N(mu1, sigma1) and N(mu2, sigma2)."""
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2

    sigma1_eps = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2_eps = sigma2 + np.eye(sigma2.shape[0]) * eps

    sqrt_sigma1 = _sqrtm_psd(sigma1_eps, eps=eps)
    cov_prod = sqrt_sigma1 @ sigma2_eps @ sqrt_sigma1
    covmean = _sqrtm_psd(cov_prod, eps=eps)

    fid = float(diff @ diff + np.trace(sigma1_eps) + np.trace(sigma2_eps) - 2.0 * np.trace(covmean))
    return max(0.0, fid)


@dataclass
class _OnlineMeanCov:
    """Streaming mean/covariance accumulator for features."""

    dim: int
    count: int = 0

    def __post_init__(self) -> None:
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.M2 = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x shape [B,{self.dim}], got {x.shape}")

        b = x.shape[0]
        if b == 0:
            return

        batch_mean = x.mean(axis=0)
        xc = x - batch_mean
        batch_M2 = xc.T @ xc

        if self.count == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = b
            return

        delta = batch_mean - self.mean
        n = self.count
        m = b
        total = n + m

        self.mean = self.mean + delta * (m / total)
        self.M2 = self.M2 + batch_M2 + np.outer(delta, delta) * (n * m / total)
        self.count = total

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("Need at least 2 samples to compute covariance")
        cov = self.M2 / (self.count - 1)
        return self.mean.astype(np.float64), cov.astype(np.float64)


def _require_tfds():
    try:
        import tensorflow_datasets as tfds 
    except Exception as e:
        raise ImportError(
            "tensorflow-datasets is required to compute reference FID stats. "

            "Install the project's environment (see environment.yml)."

        ) from e
    return tfds


def _extract_image_from_tfds_example(example) -> np.ndarray:
    if isinstance(example, dict):
        if "image" not in example:
            raise KeyError(f"TFDS example dict missing 'image' key: keys={list(example.keys())}")
        return example["image"]
    if isinstance(example, (tuple, list)):
        return example[0]
    raise TypeError(f"Unsupported TFDS example type: {type(example)}")


@dataclass
class FIDCalculator:
    """Compute FID for image generative models.

    Typical usage from a training loop:

    >>> fid_calc = FIDCalculator(dataset_name='cifar10', split='train')
    >>> fid_calc.ensure_reference_stats()
    >>> fid = fid_calc.compute_from_sample_fn(sample_fn, num_samples=50000)

    For CIFAR-10, the most common protocol is FID-50k using 50,000 generated
    samples. For ImageNet 64x64, it is common to use the 50k validation set
    as the reference split and generate 50k samples.
    """

    dataset_name: str
    split: str = "train"
    tfds_data_dir: Optional[str] = None
    cache_dir: str = "data/fid_stats"
    inception_batch_size: int = 128
    inception_device: Optional[str] = None

    def __post_init__(self) -> None:
        self.extractor = InceptionV3FeatureExtractor(
            batch_size=self.inception_batch_size,
            device=self.inception_device,
        )
        self._ref_mu: Optional[np.ndarray] = None
        self._ref_sigma: Optional[np.ndarray] = None

    def _stats_cache_path(self) -> Path:
        safe_name = self.dataset_name.replace("/", "_").replace(":", "_")
        return Path(self.cache_dir) / f"{safe_name}__{self.split}__inceptionv3_pool.npz"

    def ensure_reference_stats(self, force_recompute: bool = False, max_images: Optional[int] = None) -> None:
        path = self._stats_cache_path()
        if path.exists() and not force_recompute:
            data = np.load(path)
            self._ref_mu = data["mu"]
            self._ref_sigma = data["sigma"]
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        tfds = _require_tfds()
        ds = tfds.load(
            self.dataset_name,
            split=self.split,
            shuffle_files=False,
            data_dir=self.tfds_data_dir,
            as_supervised=False,
        )
        ds = tfds.as_numpy(ds)

        acc: Optional[_OnlineMeanCov] = None
        batch: list[np.ndarray] = []
        n_seen = 0

        for ex in ds:
            img = _extract_image_from_tfds_example(ex)
            batch.append(img)
            n_seen += 1
            if (max_images is not None) and (n_seen >= max_images):
                break
            if len(batch) >= self.extractor.batch_size:
                feats = self.extractor(np.stack(batch, axis=0))
                if acc is None:
                    acc = _OnlineMeanCov(dim=feats.shape[1])
                acc.update(feats)
                batch = []

        if batch:
            feats = self.extractor(np.stack(batch, axis=0))
            if acc is None:
                acc = _OnlineMeanCov(dim=feats.shape[1])
            acc.update(feats)

        if acc is None:
            raise RuntimeError("No images found in TFDS dataset; check dataset_name/split.")

        mu, sigma = acc.finalize()
        np.savez(path, mu=mu, sigma=sigma, count=acc.count)

        self._ref_mu = mu
        self._ref_sigma = sigma

    @property
    def ref_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._ref_mu is None or self._ref_sigma is None:
            raise RuntimeError("Reference stats not computed. Call ensure_reference_stats().")
        return self._ref_mu, self._ref_sigma

    def compute_from_samples(self, samples: np.ndarray, batch_size: Optional[int] = None) -> float:
        self.ensure_reference_stats()
        ref_mu, ref_sigma = self.ref_stats

        samples = np.asarray(samples)
        if samples.ndim != 4 or samples.shape[-1] != 3:
            raise ValueError(f"Expected samples shape [N,H,W,3], got {samples.shape}")

        bs = int(batch_size or self.extractor.batch_size)
        acc = _OnlineMeanCov(dim=2048)

        for i in range(0, samples.shape[0], bs):
            feats = self.extractor(samples[i : i + bs])
            acc.update(feats)

        mu, sigma = acc.finalize()
        return fid_from_stats(ref_mu, ref_sigma, mu, sigma)

    def compute_from_sample_fn(
        self,
        sample_fn: Callable[[int], np.ndarray],
        num_samples: int,
        batch_size: Optional[int] = None,
        *,
        verbose: bool = False,
        progress_every_batches: int = 20,
    ) -> float:
        self.ensure_reference_stats()
        ref_mu, ref_sigma = self.ref_stats

        bs = int(batch_size or self.extractor.batch_size)
        acc = _OnlineMeanCov(dim=2048)

        n = 0
        batch_i = 0
        t0 = time.perf_counter()
        if verbose:
            print(
                f"[fid] Start: num_samples={num_samples}, batch_size={bs}",
                flush=True,
            )
        while n < num_samples:
            cur = min(bs, num_samples - n)
            imgs = sample_fn(cur)
            feats = self.extractor(imgs)
            acc.update(feats)
            n += cur
            batch_i += 1
            if verbose and (
                (batch_i % max(1, int(progress_every_batches)) == 0) or (n >= num_samples)
            ):
                print(
                    f"[fid] Progress: {n}/{num_samples} "
                    f"({100.0 * n / max(1, num_samples):.1f}%)",
                    flush=True,
                )

        mu, sigma = acc.finalize()
        fid_val = fid_from_stats(ref_mu, ref_sigma, mu, sigma)
        if verbose:
            print(
                f"[fid] Done in {time.perf_counter() - t0:.1f}s: FID={fid_val:.3f}",
                flush=True,
            )
        return fid_val
