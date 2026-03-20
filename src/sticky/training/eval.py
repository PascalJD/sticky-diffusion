from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional

import hydra
import jax
import numpy as np
from omegaconf import DictConfig

from sticky.training.logging import numpy_available, to_numpy
from sticky.training.persistence import get_hydra_output_dir


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


def _should_run_eval(*, step_i: int, every: int, log_at_step_zero: bool) -> bool:
    if every <= 0:
        return False
    if step_i == 0:
        return bool(log_at_step_zero)
    return (step_i % every) == 0


def _extract_images(cfg: DictConfig, out):
    if str(cfg.model.name) == "sjd":
        imgs = out.k_filled
    else:
        imgs = out
    imgs = jax.block_until_ready(imgs)
    return np.clip(to_numpy(imgs), 0, 255).astype(np.uint8)


def _compute_joint_fid_is(
    *,
    fid_calc,
    is_calc,
    sample_fn,
    num_samples: int,
    batch_size: int,
    verbose: bool = False,
    progress_every_batches: int = 20,
):
    from sticky.eval.fid import _OnlineMeanCov, fid_from_stats
    from sticky.eval.iscore import _is_from_moments

    ref_mu, ref_sigma = fid_calc.ref_stats
    fid_acc = _OnlineMeanCov(dim=2048)

    n_classes = 1000
    num_splits = int(is_calc.num_splits)
    split_sizes = np.full((num_splits,), num_samples // num_splits, dtype=np.int32)
    split_sizes[: (num_samples % num_splits)] += 1
    split_ends = np.cumsum(split_sizes)

    split_sum_p = np.zeros((num_splits, n_classes), dtype=np.float64)
    split_sum_p_log_p = np.zeros((num_splits, n_classes), dtype=np.float64)
    split_counts = np.zeros((num_splits,), dtype=np.int32)

    n_seen = 0
    split_i = 0

    batch_i = 0
    while n_seen < num_samples:
        cur = min(batch_size, num_samples - n_seen)
        imgs = sample_fn(cur)

        feats = fid_calc.extractor(imgs)
        fid_acc.update(feats)

        probs = np.asarray(is_calc.predictor(imgs), dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != n_classes:
            raise ValueError(f"Expected Inception probs [B,{n_classes}], got {probs.shape}")

        start = 0
        while start < cur:
            end_of_split = int(split_ends[split_i])
            take = min(cur - start, end_of_split - n_seen)
            if take <= 0:
                if split_i < num_splits - 1:
                    split_i += 1
                continue

            chunk = np.clip(probs[start : start + take], is_calc.eps, 1.0)
            split_sum_p[split_i] += np.sum(chunk, axis=0)
            split_sum_p_log_p[split_i] += np.sum(chunk * np.log(chunk), axis=0)
            split_counts[split_i] += take
            n_seen += take
            start += take

            if (split_i < num_splits - 1) and (n_seen >= split_ends[split_i]):
                split_i += 1

        batch_i += 1
        if verbose and (
            (batch_i % max(1, int(progress_every_batches)) == 0) or (n_seen >= num_samples)
        ):
            print(
                f"[eval] Joint FID/IS progress: {n_seen}/{num_samples} "
                f"({100.0 * n_seen / max(1, num_samples):.1f}%)",
                flush=True,
            )

    mu, sigma = fid_acc.finalize()
    fid_val = fid_from_stats(ref_mu, ref_sigma, mu, sigma)

    split_scores = []
    for i in range(num_splits):
        if int(split_counts[i]) <= 0:
            continue
        split_scores.append(
            _is_from_moments(
                split_sum_p[i],
                split_sum_p_log_p[i],
                int(split_counts[i]),
                is_calc.eps,
            )
        )

    if not split_scores:
        raise RuntimeError("No split scores computed for Inception Score.")

    split_scores_np = np.asarray(split_scores, dtype=np.float64)
    is_mean = float(np.mean(split_scores_np))
    is_std = float(np.std(split_scores_np))
    return float(fid_val), is_mean, is_std


def build_eval_logger(
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
    task: Optional[Any] = None,
    model: Optional[Any] = None,
    eval_every: int = 0,
    sample_timesteps_override: Optional[int] = None,
):
    mode = str(eval_cfg.get("mode", "fid_is")).lower()
    if mode == "sudoku":
        from sticky.eval.sudoku import build_sudoku_eval_logger

        return build_sudoku_eval_logger(
            cfg=cfg,
            eval_cfg=eval_cfg,
            task=task,
            model=model,
            wandb_mod=wandb_mod,
            eval_every=int(eval_every),
            log_at_step_zero=bool(fid_log_at_step_zero),
            sample_timesteps_override=sample_timesteps_override,
        )

    if mode == "text_basic":
        if sample_images_fid_jit is None:
            return None

        text_every = int(eval_cfg.get("text_every", eval_every))
        text_num_samples = int(eval_cfg.get("text_num_samples", 16))
        text_batch_size = int(eval_cfg.get("text_batch_size", max(1, text_num_samples)))
        run_at_end = bool(eval_cfg.get("run_at_end", True))
        if (text_every <= 0) and (not run_at_end):
            return None

        output_dir = get_hydra_output_dir()
        output_prefix = str(eval_cfg.get("text_output_prefix", "text_samples"))

        def maybe_eval(
            step_i: int,
            params_for_sampling,
            *,
            force_fid: bool = False,
            force_is: bool = False,
        ) -> Dict[str, float]:
            del force_is
            run_text = bool(force_fid) or _should_run_eval(
                step_i=step_i,
                every=text_every,
                log_at_step_zero=fid_log_at_step_zero,
            )
            if not run_text:
                return {}

            num_samples = max(1, int(text_num_samples))
            batch_size = max(1, int(text_batch_size))
            lines: list[str] = []
            base_rng = jax.random.fold_in(
                jax.random.PRNGKey(int(cfg.training.seed) + 17_071),
                int(step_i),
            )

            batch_idx = 0
            while len(lines) < num_samples:
                sample_rng = jax.random.fold_in(base_rng, batch_idx)
                samples = sample_images_fid_jit(params_for_sampling, sample_rng)
                sample_np = np.asarray(to_numpy(jax.block_until_ready(samples)))
                remaining = num_samples - len(lines)
                sample_np = sample_np[: min(remaining, batch_size)]

                formatter = getattr(task, "format_samples_for_logging", None)
                if callable(formatter):
                    rendered = formatter(sample_np)
                else:
                    rendered = None

                if rendered is None:
                    decoded = task.decode(jnp.asarray(sample_np)) if task is not None else sample_np
                    decoded_np = np.asarray(to_numpy(decoded))
                    if decoded_np.ndim == 1:
                        decoded_np = decoded_np[None, :]
                    rendered = [
                        " ".join(str(int(tok)) for tok in row.reshape(-1))
                        for row in decoded_np
                    ]

                lines.extend(rendered)
                batch_idx += 1

            out_path = output_dir / f"{output_prefix}_step_{int(step_i):07d}.txt"
            out_path.write_text(
                "\n".join(lines[:num_samples]) + "\n",
                encoding="utf-8",
            )
            print(
                f"[eval] Wrote {num_samples} text samples to {out_path}",
                flush=True,
            )
            return {f"{fid_prefix}/text_samples_written": float(num_samples)}

        return maybe_eval

    if sample_images_fid_jit is None:
        return None
    if not numpy_available():
        raise RuntimeError("NumPy is required for FID/IS logging.")

    from sticky.eval.fid import FIDCalculator
    from sticky.eval.iscore import InceptionScoreCalculator

    fid_enabled = bool(eval_cfg.get("fid_enabled", True))
    is_enabled = bool(eval_cfg.get("is_enabled", True))
    run_at_end = bool(eval_cfg.get("run_at_end", True))
    verbose = bool(eval_cfg.get("verbose", False))
    fid_verbose = bool(eval_cfg.get("fid_verbose", verbose))
    fid_progress_every_batches = int(eval_cfg.get("fid_progress_every_batches", 20))

    is_every = int(eval_cfg.get("is_every", fid_every))
    is_num_samples = int(eval_cfg.get("is_num_samples", fid_num_samples))
    is_batch_size = int(eval_cfg.get("is_batch_size", fid_batch_size))
    is_splits = int(eval_cfg.get("is_splits", 10))

    do_fid = fid_enabled and ((fid_every > 0) or run_at_end)
    do_is = is_enabled and ((is_every > 0) or run_at_end)
    if (not do_fid) and (not do_is):
        return None

    fid_calc = None
    if do_fid:
        fid_calc = FIDCalculator(
            dataset_name=str(eval_cfg.get("fid_dataset_name", "cifar10")),
            split=str(eval_cfg.get("fid_split", "train")),
            tfds_data_dir=fid_tfds_data_dir,
            cache_dir=fid_cache_dir,
            inception_batch_size=int(eval_cfg.get("fid_inception_batch_size", 128)),
            inception_device=eval_cfg.get("fid_inception_device", None),
        )

    is_calc = None
    if do_is:
        is_calc = InceptionScoreCalculator(
            num_splits=is_splits,
            inception_batch_size=int(
                eval_cfg.get("is_inception_batch_size", eval_cfg.get("fid_inception_batch_size", 128))
            ),
            inception_device=eval_cfg.get("is_inception_device", eval_cfg.get("fid_inception_device", None)),
        )

    def maybe_eval(
        step_i: int,
        params_for_sampling,
        *,
        force_fid: bool = False,
        force_is: bool = False,
    ) -> Dict[str, float]:
        run_fid = do_fid and (
            bool(force_fid)
            or _should_run_eval(
                step_i=step_i, every=fid_every, log_at_step_zero=fid_log_at_step_zero
            )
        )
        run_is = do_is and (
            bool(force_is)
            or _should_run_eval(
                step_i=step_i, every=is_every, log_at_step_zero=fid_log_at_step_zero
            )
        )
        if (not run_fid) and (not run_is):
            return {}

        if run_fid and fid_calc is not None:
            t_ref0 = time.perf_counter()
            fid_calc.ensure_reference_stats()
            if fid_verbose:
                print(
                    f"[eval] Reference FID stats ready in {time.perf_counter() - t_ref0:.1f}s",
                    flush=True,
                )

        base_rng = jax.random.fold_in(
            jax.random.PRNGKey(int(cfg.training.seed) + 12345), step_i
        )

        def make_sample_fn(seed_offset: int):
            ctr = {"i": 0}
            rng_key = jax.random.fold_in(base_rng, seed_offset)

            def _sample_fn(n: int):
                i = ctr["i"]
                ctr["i"] = i + 1
                rng = jax.random.fold_in(rng_key, i)
                out = sample_images_fid_jit(params_for_sampling, rng)
                imgs = _extract_images(cfg, out)
                return imgs[:n]

            return _sample_fn

        metrics: Dict[str, float] = {}

        can_compute_joint = (
            run_fid
            and run_is
            and (fid_num_samples == is_num_samples)
            and (fid_batch_size == is_batch_size)
            and (fid_calc is not None)
            and (is_calc is not None)
        )

        if can_compute_joint:
            if verbose:
                print(
                    f"[eval] Running joint FID/IS at step {step_i} "
                    f"(num_samples={fid_num_samples}, batch_size={fid_batch_size})",
                    flush=True,
                )
            t0 = time.perf_counter()
            sample_fn = make_sample_fn(seed_offset=0)
            fid_val, is_mean, is_std = _compute_joint_fid_is(
                fid_calc=fid_calc,
                is_calc=is_calc,
                sample_fn=sample_fn,
                num_samples=fid_num_samples,
                batch_size=fid_batch_size,
                verbose=fid_verbose,
                progress_every_batches=fid_progress_every_batches,
            )
            metrics[f"{fid_prefix}/fid"] = float(fid_val)
            metrics[f"{fid_prefix}/is"] = float(is_mean)
            metrics[f"{fid_prefix}/is_std"] = float(is_std)
            if verbose:
                print(
                    f"[eval] Joint FID/IS done in {time.perf_counter() - t0:.1f}s: "
                    f"FID={fid_val:.3f}, IS={is_mean:.3f}±{is_std:.3f}",
                    flush=True,
                )
        else:
            if run_fid and fid_calc is not None:
                if verbose:
                    print(
                        f"[eval] Running FID at step {step_i} "
                        f"(num_samples={fid_num_samples}, batch_size={fid_batch_size})",
                        flush=True,
                    )
                fid_val = fid_calc.compute_from_sample_fn(
                    make_sample_fn(seed_offset=1),
                    num_samples=fid_num_samples,
                    batch_size=fid_batch_size,
                    verbose=fid_verbose,
                    progress_every_batches=fid_progress_every_batches,
                )
                metrics[f"{fid_prefix}/fid"] = float(fid_val)
                if verbose:
                    print(f"[eval] FID result: {fid_val:.3f}", flush=True)

            if run_is and is_calc is not None:
                t0 = time.perf_counter()
                if verbose:
                    print(
                        f"[eval] Running IS at step {step_i} "
                        f"(num_samples={is_num_samples}, batch_size={is_batch_size})",
                        flush=True,
                    )
                is_mean, is_std = is_calc.compute_from_sample_fn(
                    make_sample_fn(seed_offset=2),
                    num_samples=is_num_samples,
                    batch_size=is_batch_size,
                )
                metrics[f"{fid_prefix}/is"] = float(is_mean)
                metrics[f"{fid_prefix}/is_std"] = float(is_std)
                if verbose:
                    print(
                        f"[eval] IS done in {time.perf_counter() - t0:.1f}s: "
                        f"IS={is_mean:.3f}±{is_std:.3f}",
                        flush=True,
                    )

        if (wandb_mod is not None) and metrics:
            wandb_mod.log(metrics, step=step_i)
        return metrics

    return maybe_eval


def build_fid_logger(**kwargs):
    """Backward-compatible alias for older call sites."""
    return build_eval_logger(**kwargs)
