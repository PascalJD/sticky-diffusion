"""Inception Score (IS) computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .inception import InceptionV3ProbabilityPredictor


def _is_from_moments(sum_p: np.ndarray, sum_p_log_p: np.ndarray, count: int, eps: float) -> float:
    if count <= 0:
        raise ValueError("Inception score requires at least one sample.")
    mean_p = sum_p / float(count)
    mean_p_log_p = sum_p_log_p / float(count)
    kl = np.sum(mean_p_log_p - mean_p * np.log(np.clip(mean_p, eps, 1.0)))
    return float(np.exp(max(0.0, float(kl))))


@dataclass
class InceptionScoreCalculator:
    """Compute Inception Score on generated images.

    Returns mean/std over `num_splits` (default 10), matching common CIFAR reporting.
    """

    num_splits: int = 10
    inception_batch_size: int = 128
    inception_device: Optional[str] = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.predictor = InceptionV3ProbabilityPredictor(
            batch_size=self.inception_batch_size,
            device=self.inception_device,
        )

    def _compute_scores_from_sample_fn(
        self,
        sample_fn: Callable[[int], np.ndarray],
        num_samples: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if self.num_splits <= 0:
            raise ValueError(f"num_splits must be > 0, got {self.num_splits}")

        bs = int(batch_size or self.inception_batch_size)
        n_classes = 1000

        split_sizes = np.full((self.num_splits,), num_samples // self.num_splits, dtype=np.int32)
        split_sizes[: (num_samples % self.num_splits)] += 1
        split_ends = np.cumsum(split_sizes)

        split_sum_p = np.zeros((self.num_splits, n_classes), dtype=np.float64)
        split_sum_p_log_p = np.zeros((self.num_splits, n_classes), dtype=np.float64)
        split_counts = np.zeros((self.num_splits,), dtype=np.int32)

        n_seen = 0
        split_i = 0

        while n_seen < num_samples:
            cur = min(bs, num_samples - n_seen)
            imgs = sample_fn(cur)
            probs = np.asarray(self.predictor(imgs), dtype=np.float64)
            if probs.ndim != 2 or probs.shape[1] != n_classes:
                raise ValueError(f"Expected Inception probs [B,{n_classes}], got {probs.shape}")

            start = 0
            while start < cur:
                end_of_split = int(split_ends[split_i])
                take = min(cur - start, end_of_split - n_seen)
                if take <= 0:
                    if split_i < self.num_splits - 1:
                        split_i += 1
                    continue

                chunk = probs[start : start + take]
                chunk = np.clip(chunk, self.eps, 1.0)
                split_sum_p[split_i] += np.sum(chunk, axis=0)
                split_sum_p_log_p[split_i] += np.sum(chunk * np.log(chunk), axis=0)
                split_counts[split_i] += take
                n_seen += take
                start += take

                if (split_i < self.num_splits - 1) and (n_seen >= split_ends[split_i]):
                    split_i += 1

        split_scores = []
        for i in range(self.num_splits):
            if int(split_counts[i]) <= 0:
                continue
            score_i = _is_from_moments(
                sum_p=split_sum_p[i],
                sum_p_log_p=split_sum_p_log_p[i],
                count=int(split_counts[i]),
                eps=self.eps,
            )
            split_scores.append(score_i)

        if not split_scores:
            raise RuntimeError("No split scores computed for Inception Score.")

        scores_np = np.asarray(split_scores, dtype=np.float64)
        return float(np.mean(scores_np)), float(np.std(scores_np))

    def compute_from_sample_fn(
        self,
        sample_fn: Callable[[int], np.ndarray],
        num_samples: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        return self._compute_scores_from_sample_fn(
            sample_fn=sample_fn,
            num_samples=int(num_samples),
            batch_size=batch_size,
        )

    def compute_from_samples(
        self,
        samples: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        samples = np.asarray(samples)
        if samples.ndim != 4 or samples.shape[-1] != 3:
            raise ValueError(f"Expected samples shape [N,H,W,3], got {samples.shape}")

        ctr = {"i": 0}

        def _sample_fn(n: int) -> np.ndarray:
            i = ctr["i"]
            j = min(samples.shape[0], i + n)
            ctr["i"] = j
            return samples[i:j]

        return self._compute_scores_from_sample_fn(
            sample_fn=_sample_fn,
            num_samples=int(samples.shape[0]),
            batch_size=batch_size,
        )
