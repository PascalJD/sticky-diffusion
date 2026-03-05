from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import jax
from flax.training import checkpoints
from omegaconf import DictConfig, OmegaConf

from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.training.eval import build_eval_logger, resolve_from_original_cwd
from sticky.training.persistence import (
    get_hydra_output_dir,
    now_utc_iso,
    resolve_run_path,
)
from sticky.training.sampling import build_sampling_fns
from sticky.training.state import init_state


def _is_nullish(value: Any) -> bool:
    return value in (None, "", "null")


def _resolve_path_from_original_cwd(path_like: Optional[str]) -> Optional[Path]:
    if _is_nullish(path_like):
        return None
    resolved = resolve_from_original_cwd(path_like)
    if resolved is None:
        return None
    return Path(resolved)


def _checkpoint_root_from_run_dir(run_dir: Path, cfg: DictConfig) -> Path:
    run_context_path = run_dir / "run_context.json"
    if run_context_path.exists():
        try:
            payload = json.loads(run_context_path.read_text(encoding="utf-8"))
            from_context = payload.get("checkpoint_dir", None)
            if isinstance(from_context, str) and from_context:
                return Path(from_context)
        except Exception:
            pass

    ckpt_cfg = Path(str(cfg.training.get("checkpoint_dir", "checkpoints")))
    if ckpt_cfg.is_absolute():
        return ckpt_cfg
    return run_dir / ckpt_cfg


def _resolve_checkpoint_root(cfg: DictConfig, offline_cfg: DictConfig) -> Path:
    ckpt_dir_cfg = offline_cfg.get("checkpoint_dir", None)
    if not _is_nullish(ckpt_dir_cfg):
        out = _resolve_path_from_original_cwd(str(ckpt_dir_cfg))
        if out is None:
            raise ValueError("offline_eval.checkpoint_dir was set but could not be resolved.")
        return out

    run_dir_cfg = offline_cfg.get("run_dir", None)
    if not _is_nullish(run_dir_cfg):
        run_dir = _resolve_path_from_original_cwd(str(run_dir_cfg))
        if run_dir is None:
            raise ValueError("offline_eval.run_dir was set but could not be resolved.")
        return _checkpoint_root_from_run_dir(run_dir, cfg)

    default_ckpt = str(cfg.training.get("checkpoint_dir", "checkpoints"))
    out = _resolve_path_from_original_cwd(default_ckpt)
    if out is None:
        raise ValueError("Could not resolve checkpoint root path.")
    return out


def _resolve_checkpoint_dir(checkpoint_root: Path, source: str) -> Path:
    key = str(source).lower()
    if key in ("periodic", "latest", "root"):
        return checkpoint_root
    if key == "best":
        return checkpoint_root / "best"
    if key == "final":
        return checkpoint_root / "final"
    raise ValueError(
        f"Unknown offline_eval.checkpoint_source={source!r}. "
        "Expected one of: periodic, latest, root, best, final."
    )


def _select_checkpoint_path(ckpt_dir: Path, step: Optional[int]) -> Path:
    if step is None:
        latest = checkpoints.latest_checkpoint(str(ckpt_dir))
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}.")
        return Path(latest)

    pattern = f"checkpoint_{int(step)}"
    exact = ckpt_dir / pattern
    if exact.exists():
        return exact

    matches = sorted(ckpt_dir.glob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for step={int(step)} under {ckpt_dir}."
        )
    return matches[0]


def _restore_checkpoint_state(state_template, ckpt_dir: Path, step: Optional[int]):
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    selected_path = _select_checkpoint_path(ckpt_dir, step)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target=state_template,
        step=step,
    )
    restored_step = int(jax.device_get(restored.step))
    if (step is not None) and (restored_step != int(step)):
        raise RuntimeError(
            f"Requested checkpoint step={int(step)}, but restored step={restored_step}."
        )
    return restored, restored_step, selected_path


def _clone_eval_cfg(eval_cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=True))


def _to_float_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _to_float_scalar(value: Any) -> Optional[float]:
    try:
        return float(jax.device_get(value))
    except Exception:
        return None


def _collect_sjd_sampler_probe_metrics(
    *,
    cfg: DictConfig,
    task,
    model,
    params_for_eval,
    sample_timesteps: int,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> Dict[str, float]:
    _, sample_images_probe_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=batch_size,
        sample_timesteps=sample_timesteps,
        fid_every=1,
        fid_batch_size=batch_size,
    )
    if sample_images_probe_jit is None:
        raise RuntimeError("Could not build SJD sampler probe function.")

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    base_rng = jax.random.PRNGKey(int(seed))
    for i in range(int(num_batches)):
        rng = jax.random.fold_in(base_rng, int(i))
        out = sample_images_probe_jit(params_for_eval, rng)
        jax.block_until_ready(out.k_filled)
        out_metrics = getattr(out, "metrics", None)
        if out_metrics is None:
            continue
        for key, value in out_metrics.items():
            scalar = _to_float_scalar(value)
            if scalar is None:
                continue
            k = str(key)
            sums[k] = sums.get(k, 0.0) + float(scalar)
            counts[k] = counts.get(k, 0) + 1

    mean_metrics: Dict[str, float] = {}
    for key, total in sums.items():
        count = counts.get(key, 0)
        if count > 0:
            mean_metrics[key] = float(total / count)
    return mean_metrics


def run_offline_checkpoint_eval(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    offline_cfg: DictConfig,
    wandb_mod=None,
) -> Dict[str, float]:
    task = build_task(cfg)
    model = build_model(
        cfg,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    rng = jax.random.PRNGKey(int(cfg.training.seed))
    state_template, _ = init_state(cfg, model, rng)

    checkpoint_root = _resolve_checkpoint_root(cfg, offline_cfg)
    checkpoint_source = str(offline_cfg.get("checkpoint_source", "best"))
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_root, checkpoint_source)
    checkpoint_step = offline_cfg.get("checkpoint_step", None)
    if _is_nullish(checkpoint_step):
        checkpoint_step = None
    else:
        checkpoint_step = int(checkpoint_step)

    state, restored_step, selected_ckpt = _restore_checkpoint_state(
        state_template=state_template,
        ckpt_dir=checkpoint_dir,
        step=checkpoint_step,
    )

    use_ema = bool(offline_cfg.get("use_ema", True))
    if use_ema and (state.ema_params is not None):
        params_for_eval = state.ema_params
        param_source = "ema_params"
    else:
        if use_ema and (state.ema_params is None):
            print(
                "[offline-eval] Requested EMA params but checkpoint has no ema_params; "
                "falling back to params.",
                flush=True,
            )
        params_for_eval = state.params
        param_source = "params"

    sample_timesteps_cfg = offline_cfg.get("sample_timesteps", None)
    if _is_nullish(sample_timesteps_cfg):
        sample_timesteps = int(cfg.training.sample_timesteps)
    else:
        sample_timesteps = int(sample_timesteps_cfg)

    eval_cfg_local = _clone_eval_cfg(eval_cfg)
    eval_cfg_local.run_at_end = True
    eval_mode = str(eval_cfg_local.get("mode", "fid_is")).lower()

    fid_every = int(eval_cfg_local.get("fid_every", 0)) if eval_mode != "sudoku" else 0
    is_every = int(eval_cfg_local.get("is_every", fid_every)) if eval_mode != "sudoku" else 0
    sudoku_every = int(eval_cfg_local.get("sudoku_every", 0)) if eval_mode == "sudoku" else 0
    fid_num_samples = int(eval_cfg_local.get("fid_num_samples", 50_000))
    fid_batch_size = int(eval_cfg_local.get("fid_batch_size", 256))
    is_batch_size = int(eval_cfg_local.get("is_batch_size", fid_batch_size))
    eval_sample_batch_size = max(fid_batch_size, is_batch_size)

    sample_images_fid_jit = None
    if eval_mode != "sudoku":
        _, sample_images_fid_jit = build_sampling_fns(
            cfg=cfg,
            task=task,
            model=model,
            num_log_images=eval_sample_batch_size,
            sample_timesteps=sample_timesteps,
            fid_every=max(1, fid_every, is_every),
            fid_batch_size=eval_sample_batch_size,
        )
        if sample_images_fid_jit is None:
            raise RuntimeError("Could not build sampling function for offline evaluation.")

    fid_prefix = str(eval_cfg_local.get("prefix", "eval"))
    fid_log_at_step_zero = bool(eval_cfg_local.get("log_at_step_zero", False))
    fid_cache_dir = resolve_from_original_cwd(
        str(eval_cfg_local.get("fid_cache_dir", "data/fid_stats"))
    )
    fid_tfds_data_dir = resolve_from_original_cwd(
        eval_cfg_local.get("fid_tfds_data_dir", None)
    )
    if fid_tfds_data_dir is None:
        fid_tfds_data_dir = resolve_from_original_cwd(
            cfg.dataset.get("data_dir", None)
        )

    maybe_log_eval = build_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg_local,
        wandb_mod=wandb_mod,
        sample_images_fid_jit=sample_images_fid_jit,
        fid_every=fid_every,
        fid_num_samples=fid_num_samples,
        fid_batch_size=fid_batch_size,
        fid_prefix=fid_prefix,
        fid_log_at_step_zero=fid_log_at_step_zero,
        fid_cache_dir=fid_cache_dir,
        fid_tfds_data_dir=fid_tfds_data_dir,
        task=task,
        model=model,
        eval_every=(sudoku_every if eval_mode == "sudoku" else max(fid_every, is_every)),
        sample_timesteps_override=(sample_timesteps if eval_mode == "sudoku" else None),
    )
    if maybe_log_eval is None:
        raise RuntimeError(
            "Offline evaluation produced no evaluator. "
            "Check eval.mode and metric-specific eval toggles."
        )

    force_fid = bool(offline_cfg.get("force_fid", True))
    force_is = bool(offline_cfg.get("force_is", True))
    metrics = _to_float_metrics(
        maybe_log_eval(
            restored_step,
            params_for_eval,
            force_fid=force_fid,
            force_is=force_is,
        )
    )
    if not metrics:
        raise RuntimeError(
            "Offline evaluation returned no metrics. "
            "Check eval settings and force flags."
        )

    probe_batches_cfg = offline_cfg.get("sampler_probe_batches", 0)
    probe_batches = 0 if _is_nullish(probe_batches_cfg) else int(probe_batches_cfg)
    sampling_probe_payload = None
    if probe_batches > 0:
        if (str(cfg.model.name) == "sjd") and (eval_mode != "sudoku"):
            probe_batch_size_cfg = offline_cfg.get("sampler_probe_batch_size", None)
            if _is_nullish(probe_batch_size_cfg):
                probe_batch_size = int(eval_sample_batch_size)
            else:
                probe_batch_size = int(probe_batch_size_cfg)

            probe_seed_offset_cfg = offline_cfg.get("sampler_probe_seed_offset", 777777)
            probe_seed_offset = (
                777777 if _is_nullish(probe_seed_offset_cfg) else int(probe_seed_offset_cfg)
            )
            probe_seed = int(cfg.training.seed) + probe_seed_offset

            probe_metrics = _collect_sjd_sampler_probe_metrics(
                cfg=cfg,
                task=task,
                model=model,
                params_for_eval=params_for_eval,
                sample_timesteps=sample_timesteps,
                batch_size=probe_batch_size,
                num_batches=probe_batches,
                seed=probe_seed,
            )
            sampling_probe_payload = {
                "enabled": True,
                "num_batches": int(probe_batches),
                "batch_size": int(probe_batch_size),
                "seed": int(probe_seed),
                "metrics": probe_metrics,
            }
        else:
            sampling_probe_payload = {
                "enabled": False,
                "requested_batches": int(probe_batches),
                "reason": (
                    "sampler probes are supported only for non-sudoku SJD runs; "
                    f"got model={cfg.model.name!r}, eval_mode={eval_mode!r}"
                ),
            }

    output_path_cfg = offline_cfg.get("output_path", "offline_eval_metrics.json")
    if _is_nullish(output_path_cfg):
        output_path_cfg = "offline_eval_metrics.json"
    output_path = resolve_run_path(
        str(output_path_cfg),
        "offline_eval_metrics.json",
        base_dir=get_hydra_output_dir(),
    )
    payload = {
        "timestamp_utc": now_utc_iso(),
        "checkpoint": {
            "root_dir": str(checkpoint_root),
            "source": checkpoint_source,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_path": str(selected_ckpt),
            "requested_step": None if checkpoint_step is None else int(checkpoint_step),
            "restored_step": int(restored_step),
            "param_source": param_source,
            "use_ema_requested": bool(use_ema),
        },
        "evaluation": {
            "mode": eval_mode,
            "sample_timesteps": int(sample_timesteps),
            "force_fid": bool(force_fid),
            "force_is": bool(force_is),
            "fid_enabled": bool(eval_cfg_local.get("fid_enabled", True)) if eval_mode != "sudoku" else False,
            "is_enabled": bool(eval_cfg_local.get("is_enabled", True)) if eval_mode != "sudoku" else False,
            "fid_num_samples": int(fid_num_samples),
            "fid_batch_size": int(fid_batch_size),
            "is_num_samples": int(eval_cfg_local.get("is_num_samples", fid_num_samples)),
            "is_batch_size": int(is_batch_size),
            "prefix": str(fid_prefix),
            "sudoku_every": int(sudoku_every),
            "sudoku_num_batches": int(eval_cfg_local.get("sudoku_num_batches", 64)),
            "sudoku_num_batches_force": int(eval_cfg_local.get("sudoku_num_batches_force", -1)),
        },
        "metrics": metrics,
    }
    if sampling_probe_payload is not None:
        payload["sampling_probe"] = sampling_probe_payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "[offline-eval] "
        f"checkpoint={selected_ckpt} step={restored_step} param_source={param_source}",
        flush=True,
    )
    print(
        f"[offline-eval] Wrote metrics report to {output_path}",
        flush=True,
    )

    return metrics
