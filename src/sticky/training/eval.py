from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
import jax
import numpy as np
from omegaconf import DictConfig

from sticky.training.logging import numpy_available, to_numpy


def resolve_from_original_cwd(path_like: Optional[str]) -> Optional[str]:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return str(path)
    return str(Path(hydra.utils.get_original_cwd()) / path)


def sample_for_logging(
    *,
    cfg: DictConfig,
    sample_images_jit,
    params_for_sampling,
    step: int,
):
    if sample_images_jit is None:
        return None, None

    sample_rng = jax.random.fold_in(
        jax.random.PRNGKey(int(cfg.training.seed) + 999), step
    )
    out = sample_images_jit(params_for_sampling, sample_rng)

    if str(cfg.model.name) == "sjd":
        metrics = out.metrics
        samples = jax.block_until_ready(out.k_filled)
        return samples, metrics

    samples = jax.block_until_ready(out)
    return samples, None


def build_fid_logger(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    wandb_mod,
    sample_images_fid_jit,
    fid_every: int,
    fid_num_samples: int,
    fid_batch_size: int,
    fid_prefix: str,
    fid_log_at_step_zero: bool,
    fid_cache_dir: Optional[str],
    fid_tfds_data_dir: Optional[str],
):
    if (fid_every <= 0) or (sample_images_fid_jit is None):
        return None
    if not numpy_available():
        raise RuntimeError("NumPy is required for FID logging.")

    from sticky.eval.fid import FIDCalculator

    fid_calc = FIDCalculator(
        dataset_name=str(eval_cfg.get("fid_dataset_name", "cifar10")),
        split=str(eval_cfg.get("fid_split", "train")),
        tfds_data_dir=fid_tfds_data_dir,
        cache_dir=fid_cache_dir,
        inception_batch_size=int(eval_cfg.get("fid_inception_batch_size", 128)),
        inception_device=eval_cfg.get("fid_inception_device", None),
    )

    def maybe_log_fid(step_i: int, params_for_sampling):
        if wandb_mod is None:
            return
        if (step_i == 0) and (not fid_log_at_step_zero):
            return
        if (step_i > 0) and (step_i % fid_every != 0):
            return

        fid_calc.ensure_reference_stats()
        base_rng = jax.random.fold_in(
            jax.random.PRNGKey(int(cfg.training.seed) + 12345), step_i
        )
        ctr = {"i": 0}

        def sample_fn(n: int):
            i = ctr["i"]
            ctr["i"] = i + 1
            rng = jax.random.fold_in(base_rng, i)

            out = sample_images_fid_jit(params_for_sampling, rng)
            if str(cfg.model.name) == "sjd":
                imgs = out.k_filled
            else:
                imgs = out

            imgs = jax.block_until_ready(imgs)
            imgs_np = np.clip(to_numpy(imgs), 0, 255).astype(np.uint8)
            return imgs_np[:n]

        fid_val = fid_calc.compute_from_sample_fn(
            sample_fn,
            num_samples=fid_num_samples,
            batch_size=fid_batch_size,
        )
        wandb_mod.log({f"{fid_prefix}/fid": float(fid_val)}, step=step_i)

    return maybe_log_fid
