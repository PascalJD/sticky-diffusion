from __future__ import annotations

import numpy as np

from sticky.eval.fid import _normalize_tfds_data_dir, _sqrtm_psd, fid_from_stats


def test_fid_from_stats_uses_original_covariance_traces():
    mu1 = np.zeros((2,), dtype=np.float64)
    mu2 = np.zeros((2,), dtype=np.float64)
    sigma1 = np.zeros((2, 2), dtype=np.float64)
    sigma2 = np.diag([4.0, 9.0]).astype(np.float64)
    eps = 1e-3

    sqrt_sigma1 = _sqrtm_psd(sigma1 + np.eye(2) * eps, eps=eps)
    cov_prod = sqrt_sigma1 @ (sigma2 + np.eye(2) * eps) @ sqrt_sigma1
    covmean = _sqrtm_psd(cov_prod, eps=eps)
    expected = float(np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))

    got = fid_from_stats(mu1, sigma1, mu2, sigma2, eps=eps)

    assert np.isclose(got, max(0.0, expected))


def test_normalize_tfds_data_dir_accepts_dataset_leaf_paths():
    leaf = "/scratch/sticky-diffusion/tfds/cifar10/3.0.2"
    dataset_dir = "/scratch/sticky-diffusion/tfds/cifar10"
    root_dir = "/scratch/sticky-diffusion/tfds"

    assert _normalize_tfds_data_dir(leaf, "cifar10") == root_dir
    assert _normalize_tfds_data_dir(dataset_dir, "cifar10") == root_dir
    assert _normalize_tfds_data_dir(root_dir, "cifar10") == root_dir

