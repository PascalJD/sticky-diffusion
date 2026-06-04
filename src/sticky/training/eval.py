from __future__ import annotations

import time
import zlib
from typing import Any, Dict, Optional

import jax
import numpy as np
from omegaconf import DictConfig

from sticky.rng import make_rng
from sticky.core.paths import resolve_optional_path_str
from sticky.training.logging import numpy_available, to_numpy
from sticky.training.persistence import get_hydra_output_dir


def resolve_from_original_cwd(path_like: Optional[str]) -> Optional[str]:
    return resolve_optional_path_str(
        path_like,
        nullish=(None,),
        fallback_to_resolve=False,
    )


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
        make_rng(int(cfg.training.seed) + 999), step
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
    extra_fid_sampling_fns: Optional[Dict[str, Any]] = None,
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

        # --- Periodic valid-word frontier logging (CANDI calculate_word_dictionary_match)
        # at len>=5 and len>=6, computed on the SAME generated samples this hook
        # already produces and on whatever params the loop passes (EMA via
        # params_for_sampling). This surfaces the GENERATION trajectory in wandb
        # during training because the loss is a poor proxy for sample quality.
        # The 16-sample text dump is too small for a stable number, so the
        # valid-word metric runs on its own (sparser) cadence with more samples.
        vw_enabled = bool(eval_cfg.get("valid_word_enabled", False))
        vw_every = int(eval_cfg.get("valid_word_every", 25000))
        vw_num_samples = max(1, int(eval_cfg.get("valid_word_num_samples", 256)))
        vw_thresholds = [int(x) for x in eval_cfg.get("valid_word_thresholds", [5, 6])]
        vw_vocabs = None
        vw_id_to_char = None
        if vw_enabled and vw_every > 0:
            # Build the reference vocabs/char-map ONCE. Any failure here disables
            # valid-word logging with a warning; it must never break training.
            try:
                from sticky.eval.valid_word import (
                    build_threshold_vocabs,
                    load_char_map,
                    load_test_vocab,
                )

                _base_vocab = load_test_vocab(
                    str(eval_cfg.get("vocab_path", "data/text8/test_vocab_len5.json"))
                )
                vw_vocabs = build_threshold_vocabs(_base_vocab, vw_thresholds)
                vw_id_to_char = load_char_map(
                    str(eval_cfg.get("char_map_path", "data/text8/char_map.json"))
                )
                print(
                    f"[eval] valid-word logging ON: thresholds={vw_thresholds} "
                    f"every={vw_every} n_samples={vw_num_samples} "
                    f"(len5 vocab={len(_base_vocab)})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - never break training setup
                print(
                    f"[eval] WARNING: valid-word logging disabled (vocab setup "
                    f"failed): {exc}",
                    flush=True,
                )
                vw_vocabs = None

        # Optional TEMPERATURE SWEEP for the valid-word metric: build one
        # sampler per temperature so the logged number tracks the operating
        # FRONTIER (max-along-temperature), per the gate metric. Empty/absent
        # valid_word_temps => use the single configured sampler
        # (sample_images_fid_jit), i.e. prior behavior, bit-unchanged. The
        # per-temp samplers inherit cfg.sampler (incl. the fixed-sampler flags),
        # only overriding logit_temperature. Build failures fall back to the
        # single sampler and never break training.
        vw_temps = [float(t) for t in (eval_cfg.get("valid_word_temps", []) or [])]
        vw_samplers = None  # list[(temp, sample_fn)] or None => single sampler
        if (vw_vocabs is not None) and vw_temps and (model is not None):
            try:
                from omegaconf import OmegaConf

                from sticky.training.sampling import build_sampling_fns

                vw_nfe = int(
                    eval_cfg.get("valid_word_nfe", int(cfg.sampler.get("n_steps", 128)))
                )
                vw_batch = max(1, int(eval_cfg.get("valid_word_batch_size", text_batch_size)))
                _built = []
                for _t in vw_temps:
                    cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                    cfg_t.sampler.logit_temperature = float(_t)
                    _, _fn_t = build_sampling_fns(
                        cfg=cfg_t,
                        task=task,
                        model=model,
                        num_log_images=vw_batch,
                        sample_timesteps=vw_nfe,
                        fid_every=1,
                        fid_batch_size=vw_batch,
                    )
                    if _fn_t is None:
                        raise RuntimeError("build_sampling_fns returned None for valid-word temp sweep")
                    _built.append((float(_t), _fn_t))
                vw_samplers = _built
                print(
                    f"[eval] valid-word temp-sweep ON: temps={vw_temps} nfe={vw_nfe} "
                    f"(logged metric = max-along-temperature frontier)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - never break training setup
                print(
                    f"[eval] WARNING: valid-word temp-sweep disabled (build failed): "
                    f"{exc}; falling back to single configured sampler",
                    flush=True,
                )
                vw_samplers = None

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
            run_vw = (vw_vocabs is not None) and (
                bool(force_fid)
                or _should_run_eval(
                    step_i=step_i, every=vw_every, log_at_step_zero=False
                )
            )
            if not run_text and not run_vw:
                return {}

            result: Dict[str, float] = {}

            if run_text:
                num_samples = max(1, int(text_num_samples))
                batch_size = max(1, int(text_batch_size))
                lines: list[str] = []
                base_rng = jax.random.fold_in(
                    make_rng(int(cfg.training.seed) + 17_071),
                    int(step_i),
                )

                batch_idx = 0
                while len(lines) < num_samples:
                    sample_rng = jax.random.fold_in(base_rng, batch_idx)
                    samples = sample_images_fid_jit(params_for_sampling, sample_rng)
                    # SJD samplers return a ReverseSampleResult struct; the committed
                    # token grid is k_filled. Image/text baselines return the array
                    # directly. Normalize to the token array either way.
                    samples = getattr(samples, "k_filled", samples)
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
                result[f"{fid_prefix}/text_samples_written"] = float(num_samples)

            if run_vw:
                # Wrapped so any sampling/scoring failure logs a warning and lets
                # training continue (this hook must never crash a run).
                try:
                    from sticky.eval.valid_word import score_samples

                    def _collect(fn, base_rng):
                        coll: list = []
                        n = 0
                        b = 0
                        while n < vw_num_samples:
                            r = jax.random.fold_in(base_rng, b)
                            s = fn(params_for_sampling, r)
                            s = getattr(s, "k_filled", s)
                            a = np.asarray(to_numpy(jax.block_until_ready(s)))
                            if a.ndim == 1:
                                a = a[None, :]
                            coll.append(a)
                            n += int(a.shape[0])
                            b += 1
                        return np.concatenate(coll, axis=0)[:vw_num_samples]

                    base = jax.random.fold_in(
                        make_rng(int(cfg.training.seed) + 24_601), int(step_i)
                    )
                    if vw_samplers is not None:
                        # Max-along-temperature frontier: score each temp, log
                        # per-temp values plus the max as the headline metric.
                        per_thr_max = {thr: 0.0 for thr in vw_vocabs}
                        for ti, (t, fn_t) in enumerate(vw_samplers):
                            samples_t = _collect(fn_t, jax.random.fold_in(base, 1000 + ti))
                            for thr in sorted(vw_vocabs):
                                sc = float(
                                    score_samples(
                                        samples_t,
                                        id_to_char=vw_id_to_char,
                                        vocab=vw_vocabs[thr],
                                    )
                                )
                                result[f"{fid_prefix}/valid_word_len{thr}@temp{t:g}"] = sc
                                if sc > per_thr_max[thr]:
                                    per_thr_max[thr] = sc
                        for thr in sorted(vw_vocabs):
                            result[f"{fid_prefix}/valid_word_len{thr}"] = per_thr_max[thr]
                        nsamp = vw_num_samples
                    else:
                        vw_samples = _collect(sample_images_fid_jit, base)
                        for thr in sorted(vw_vocabs):
                            result[f"{fid_prefix}/valid_word_len{thr}"] = float(
                                score_samples(
                                    vw_samples,
                                    id_to_char=vw_id_to_char,
                                    vocab=vw_vocabs[thr],
                                )
                            )
                        nsamp = int(vw_samples.shape[0])
                    print(
                        f"[eval] valid_word @step {step_i} (n={nsamp}/temp, EMA, "
                        f"{'max-frontier' if vw_samplers is not None else 'single-temp'}): "
                        + ", ".join(
                            f"{k.split('/')[-1]}={v:.4f}"
                            for k, v in result.items()
                            if "valid_word_len" in k and "@" not in k
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - never crash training
                    print(
                        f"[eval] WARNING: valid_word eval failed @step {step_i} "
                        f"(training continues): {exc}",
                        flush=True,
                    )

            return result

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
            make_rng(int(cfg.training.seed) + 12345), step_i
        )

        def make_sample_fn(seed_offset: int):
            ctr = {"i": 0}
            metrics_acc: Dict[str, list] = {}
            rng_key = jax.random.fold_in(base_rng, seed_offset)

            def _sample_fn(n: int):
                i = ctr["i"]
                ctr["i"] = i + 1
                rng = jax.random.fold_in(rng_key, i)
                out = sample_images_fid_jit(params_for_sampling, rng)
                # Accumulate SJD sampling metrics across batches.
                if hasattr(out, "metrics") and out.metrics:
                    for mk, mv in out.metrics.items():
                        v = float(jax.block_until_ready(mv))
                        metrics_acc.setdefault(mk, []).append(v)
                imgs = _extract_images(cfg, out)
                return imgs[:n]

            _sample_fn.metrics_acc = metrics_acc
            return _sample_fn

        def _flush_sampling_metrics(fn, prefix: str, dst: Dict[str, float]):
            """Average accumulated SJD sampling metrics from a sample_fn and write to *dst*."""
            acc = getattr(fn, "metrics_acc", None)
            if acc:
                for mk, vals in acc.items():
                    if vals:
                        dst[f"{prefix}/{mk}"] = sum(vals) / len(vals)

        metrics: Dict[str, float] = {}

        # When extra_fid_sampling_fns is provided, each labelled run already
        # covers the default config (the "base" entry).  Skip the redundant
        # primary FID/IS computation so we don't waste ~50 % of wall time
        # on a duplicate arm.
        skip_primary = bool(extra_fid_sampling_fns)

        can_compute_joint = (
            run_fid
            and run_is
            and (fid_num_samples == is_num_samples)
            and (fid_batch_size == is_batch_size)
            and (fid_calc is not None)
            and (is_calc is not None)
        )

        if skip_primary:
            pass  # jump straight to the extra runs below
        elif can_compute_joint:
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
            _flush_sampling_metrics(sample_fn, fid_prefix, metrics)
            if verbose:
                print(
                    f"[eval] Joint FID/IS done in {time.perf_counter() - t0:.1f}s: "
                    f"FID={fid_val:.3f}, IS={is_mean:.3f}±{is_std:.3f}",
                    flush=True,
                )
        else:
            fid_flushed = False
            if run_fid and fid_calc is not None:
                if verbose:
                    print(
                        f"[eval] Running FID at step {step_i} "
                        f"(num_samples={fid_num_samples}, batch_size={fid_batch_size})",
                        flush=True,
                    )
                fid_sample = make_sample_fn(seed_offset=1)
                fid_val = fid_calc.compute_from_sample_fn(
                    fid_sample,
                    num_samples=fid_num_samples,
                    batch_size=fid_batch_size,
                    verbose=fid_verbose,
                    progress_every_batches=fid_progress_every_batches,
                )
                metrics[f"{fid_prefix}/fid"] = float(fid_val)
                _flush_sampling_metrics(fid_sample, fid_prefix, metrics)
                fid_flushed = True
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
                is_sample = make_sample_fn(seed_offset=2)
                is_mean, is_std = is_calc.compute_from_sample_fn(
                    is_sample,
                    num_samples=is_num_samples,
                    batch_size=is_batch_size,
                )
                metrics[f"{fid_prefix}/is"] = float(is_mean)
                metrics[f"{fid_prefix}/is_std"] = float(is_std)
                if not fid_flushed:
                    _flush_sampling_metrics(is_sample, fid_prefix, metrics)
                if verbose:
                    print(
                        f"[eval] IS done in {time.perf_counter() - t0:.1f}s: "
                        f"IS={is_mean:.3f}±{is_std:.3f}",
                        flush=True,
                    )

        # --- Extra labelled FID/IS runs (e.g. ablation samplers) ---
        if extra_fid_sampling_fns and (run_fid or run_is):
            for run_label, run_sample_fn in extra_fid_sampling_fns.items():
                run_prefix = f"{fid_prefix}/{run_label}"

                def make_run_sample_fn(fn, seed_offset: int):
                    ctr = {"i": 0}
                    run_metrics_acc: Dict[str, list] = {}
                    rng_key = jax.random.fold_in(base_rng, seed_offset)

                    def _sample_fn(n: int):
                        i = ctr["i"]
                        ctr["i"] = i + 1
                        rng = jax.random.fold_in(rng_key, i)
                        out = fn(params_for_sampling, rng)
                        if hasattr(out, "metrics") and out.metrics:
                            for mk, mv in out.metrics.items():
                                v = float(jax.block_until_ready(mv))
                                run_metrics_acc.setdefault(mk, []).append(v)
                        imgs = _extract_images(cfg, out)
                        return imgs[:n]

                    _sample_fn.metrics_acc = run_metrics_acc
                    return _sample_fn

                run_seed = zlib.crc32(run_label.encode("utf-8")) & 0x7FFFFFFF
                run_sample = make_run_sample_fn(run_sample_fn, seed_offset=run_seed)

                can_joint = (
                    run_fid and run_is
                    and (fid_num_samples == is_num_samples)
                    and (fid_batch_size == is_batch_size)
                    and (fid_calc is not None)
                    and (is_calc is not None)
                )
                if can_joint:
                    if verbose:
                        print(
                            f"[eval] Running joint FID/IS for '{run_label}' at step {step_i}",
                            flush=True,
                        )
                    t0_run = time.perf_counter()
                    fid_v, is_m, is_s = _compute_joint_fid_is(
                        fid_calc=fid_calc,
                        is_calc=is_calc,
                        sample_fn=run_sample,
                        num_samples=fid_num_samples,
                        batch_size=fid_batch_size,
                        verbose=fid_verbose,
                        progress_every_batches=fid_progress_every_batches,
                    )
                    metrics[f"{run_prefix}/fid"] = float(fid_v)
                    metrics[f"{run_prefix}/is"] = float(is_m)
                    metrics[f"{run_prefix}/is_std"] = float(is_s)
                    _flush_sampling_metrics(run_sample, run_prefix, metrics)
                    if verbose:
                        print(
                            f"[eval] '{run_label}' done in {time.perf_counter() - t0_run:.1f}s: "
                            f"FID={fid_v:.3f}, IS={is_m:.3f}±{is_s:.3f}",
                            flush=True,
                        )
                else:
                    run_fid_flushed = False
                    if run_fid and fid_calc is not None:
                        fid_run_sample = make_run_sample_fn(
                            run_sample_fn, seed_offset=run_seed + 1
                        )
                        fid_v = fid_calc.compute_from_sample_fn(
                            fid_run_sample,
                            num_samples=fid_num_samples,
                            batch_size=fid_batch_size,
                            verbose=fid_verbose,
                            progress_every_batches=fid_progress_every_batches,
                        )
                        metrics[f"{run_prefix}/fid"] = float(fid_v)
                        _flush_sampling_metrics(fid_run_sample, run_prefix, metrics)
                        run_fid_flushed = True
                    if run_is and is_calc is not None:
                        is_run_sample = make_run_sample_fn(
                            run_sample_fn, seed_offset=run_seed + 2
                        )
                        is_m, is_s = is_calc.compute_from_sample_fn(
                            is_run_sample,
                            num_samples=is_num_samples,
                            batch_size=is_batch_size,
                        )
                        metrics[f"{run_prefix}/is"] = float(is_m)
                        metrics[f"{run_prefix}/is_std"] = float(is_s)
                        if not run_fid_flushed:
                            _flush_sampling_metrics(is_run_sample, run_prefix, metrics)

        if (wandb_mod is not None) and metrics:
            wandb_mod.log(metrics, step=step_i)
        return metrics

    return maybe_eval


def build_fid_logger(**kwargs):
    """Backward-compatible alias for older call sites."""
    return build_eval_logger(**kwargs)
