from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import jax
from flax.training import checkpoints
from omegaconf import DictConfig, OmegaConf

from sticky.core.paths import is_nullish, resolve_optional_path
from sticky.models.factory import build_model
from sticky.rng import ensure_prng_key, legacy_prng_key_data, make_rng
from sticky.tasks.factory import build_task
from sticky.training.eval import build_eval_logger, resolve_from_original_cwd
from sticky.training.persistence import (
    get_hydra_output_dir,
    now_utc_iso,
    resolve_run_path,
)
from sticky.training.sampling import build_sampling_fns
from sticky.training.state import init_state

def _resolve_path_from_original_cwd(path_like: Optional[str]) -> Optional[Path]:
    return resolve_optional_path(
        path_like,
        nullish=(None, "", "null"),
        fallback_to_resolve=False,
    )


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _resolve_run_dir_from_offline_cfg(offline_cfg: DictConfig) -> Optional[Path]:
    run_dir_cfg = offline_cfg.get("run_dir", None)
    if is_nullish(run_dir_cfg, nullish=(None, "", "null")):
        return None
    run_dir = _resolve_path_from_original_cwd(str(run_dir_cfg))
    if run_dir is None:
        raise ValueError("offline_eval.run_dir was set but could not be resolved.")
    return run_dir


def _load_run_context_payload(run_dir: Path) -> Dict[str, Any]:
    run_context_path = run_dir / "run_context.json"
    if not run_context_path.exists():
        raise FileNotFoundError(
            f"offline_eval.use_run_config=true requires {run_context_path}."
        )
    payload = json.loads(run_context_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed run context payload: {run_context_path}")
    return payload


def _load_experiment_cfg_from_run_context(run_dir: Path) -> tuple[DictConfig, Dict[str, Any]]:
    payload = _load_run_context_payload(run_dir)
    experiment_payload = payload.get("config", {}).get("experiment", None)
    if not isinstance(experiment_payload, dict):
        raise ValueError(
            f"Run context for {run_dir} does not contain a resolved experiment config."
        )
    return OmegaConf.create(experiment_payload), payload


def _apply_environment_overrides(
    *,
    effective_cfg: DictConfig,
    fallback_cfg: DictConfig,
) -> None:
    dataset_cfg = fallback_cfg.get("dataset", None)
    if dataset_cfg is not None:
        data_dir = dataset_cfg.get("data_dir", None)
        if not is_nullish(data_dir, nullish=(None, "", "null")):
            effective_cfg.dataset.data_dir = data_dir

    runtime_cfg = fallback_cfg.get("runtime", None)
    if runtime_cfg is not None:
        effective_cfg.runtime = _clone_cfg(runtime_cfg)


def _apply_explicit_eval_overrides(
    *,
    effective_cfg: DictConfig,
    offline_cfg: DictConfig,
    sample_timesteps: int,
) -> None:
    sampler_cfg = effective_cfg.get("sampler", None)
    if sampler_cfg is not None:
        effective_cfg.sampler.n_steps = int(sample_timesteps)

    jump_eta_cfg = offline_cfg.get("jump_eta", None)
    if not is_nullish(jump_eta_cfg, nullish=(None, "", "null")):
        if effective_cfg.get("forward", None) is None:
            raise ValueError("offline_eval.jump_eta requires experiment.forward to be configured.")
        if effective_cfg.forward.get("jump", None) is None:
            raise ValueError(
                "offline_eval.jump_eta requires experiment.forward.jump to be configured."
            )
        effective_cfg.forward.jump.eta = float(jump_eta_cfg)

    tau_cfg = offline_cfg.get("logit_temperature", None)
    if not is_nullish(tau_cfg, nullish=(None, "", "null")):
        effective_cfg.sampler.logit_temperature = float(tau_cfg)


def _resolve_effective_experiment_cfg(
    *,
    cfg: DictConfig,
    offline_cfg: DictConfig,
) -> tuple[DictConfig, Optional[Path], Optional[Dict[str, Any]], str]:
    run_dir = _resolve_run_dir_from_offline_cfg(offline_cfg)
    use_run_config = bool(offline_cfg.get("use_run_config", False))
    if not use_run_config:
        return _clone_cfg(cfg), run_dir, None, "hydra_cfg"

    if run_dir is None:
        raise ValueError(
            "offline_eval.use_run_config=true requires offline_eval.run_dir to be set."
        )

    effective_cfg, run_context_payload = _load_experiment_cfg_from_run_context(run_dir)
    _apply_environment_overrides(effective_cfg=effective_cfg, fallback_cfg=cfg)
    return effective_cfg, run_dir, run_context_payload, "run_context"


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_forward_config_metadata(cfg: DictConfig) -> Dict[str, Any]:
    forward_cfg = cfg.get("forward", None)
    if forward_cfg is None:
        return {
            "forward_beta_target": None,
            "forward_hazard_target": None,
            "forward_jump_target": None,
            "jump_eta": None,
        }

    beta_cfg = forward_cfg.get("beta", None)
    hazard_cfg = forward_cfg.get("hazard", None)
    jump_cfg = forward_cfg.get("jump", None)
    return {
        "forward_beta_target": (
            str(beta_cfg.get("_target_", "")) if beta_cfg is not None else None
        ),
        "forward_hazard_target": (
            str(hazard_cfg.get("_target_", "")) if hazard_cfg is not None else None
        ),
        "forward_jump_target": (
            str(jump_cfg.get("_target_", "")) if jump_cfg is not None else None
        ),
        "jump_eta": (
            _maybe_float(jump_cfg.get("eta", None)) if jump_cfg is not None else None
        ),
    }


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
    if not is_nullish(ckpt_dir_cfg, nullish=(None, "", "null")):
        out = _resolve_path_from_original_cwd(str(ckpt_dir_cfg))
        if out is None:
            raise ValueError("offline_eval.checkpoint_dir was set but could not be resolved.")
        return out

    run_dir = _resolve_run_dir_from_offline_cfg(offline_cfg)
    if run_dir is not None:
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
    try:
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=state_template,
            step=step,
        )
    except Exception:
        legacy_target = state_template.replace(
            rng=legacy_prng_key_data(state_template.rng)
        )
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=legacy_target,
            step=step,
        )
    restored = restored.replace(rng=ensure_prng_key(restored.rng))
    restored_step = int(jax.device_get(restored.step))
    if (step is not None) and (restored_step != int(step)):
        raise RuntimeError(
            f"Requested checkpoint step={int(step)}, but restored step={restored_step}."
        )
    return restored, restored_step, selected_path


def _clone_eval_cfg(eval_cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=True))


def _is_nullish(value: Any) -> bool:
    return is_nullish(value, nullish=(None, "", "null"))


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


def _extract_sudoku_sampler_summary(
    *,
    metrics: Dict[str, float],
    eval_cfg: DictConfig,
) -> Optional[Dict[str, Any]]:
    if str(eval_cfg.get("mode", "fid_is")).lower() != "sudoku":
        return None

    prefix = str(eval_cfg.get("prefix", "eval"))
    run_all = bool(eval_cfg.get("sudoku_run_all_sampler_modes", False))
    primary_label = str(eval_cfg.get("sudoku_primary_sampler_label", "default"))

    def _summary_for(prefix_key: str, label: str) -> Dict[str, Any]:
        return {
            "label": label,
            "metrics_prefix": prefix_key,
            "solve_rate": metrics.get(f"{prefix_key}/solve_rate"),
            "board_acc_exact": metrics.get(f"{prefix_key}/board_acc_exact"),
            "board_acc": metrics.get(f"{prefix_key}/board_acc"),
            "cell_acc_unknown": metrics.get(f"{prefix_key}/cell_acc_unknown"),
            "masked_cell_acc": metrics.get(f"{prefix_key}/masked_cell_acc"),
            "full_cell_acc": metrics.get(f"{prefix_key}/full_cell_acc"),
            "acc_complete_puzzle": metrics.get(f"{prefix_key}/acc_complete_puzzle"),
            "acc_solve_strict": metrics.get(f"{prefix_key}/acc_solve_strict"),
            "avg_commits_per_step": metrics.get(f"{prefix_key}/avg_commits_per_step"),
            "mean_selected_top_probability": metrics.get(
                f"{prefix_key}/mean_selected_top_probability"
            ),
            "mean_selected_top_prob_margin": metrics.get(
                f"{prefix_key}/mean_selected_top_prob_margin"
            ),
            "final_masked_unknown_fraction": metrics.get(
                f"{prefix_key}/final_masked_unknown_fraction"
            ),
            "sampling_nfe_total": metrics.get(f"{prefix_key}/sampling/nfe_total"),
            "sampling_wallclock_sec_per_board": metrics.get(
                f"{prefix_key}/sampling/wallclock_sec_per_board"
            ),
            "sampler_steps": metrics.get(f"{prefix_key}/sampler_steps"),
            "n_steps": metrics.get(f"{prefix_key}/n_steps"),
        }

    if not run_all:
        return {
            "run_all_sampler_modes": False,
            "primary_sampler_label": primary_label,
            "samplers": [_summary_for(prefix, primary_label)],
        }

    samplers = []

    def _collect_from(container):
        out = []
        if isinstance(container, dict):
            for label in container:
                out.append(_summary_for(f"{prefix}/{label}", str(label)))
        elif isinstance(container, list):
            for entry in container:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label", "")).strip()
                if not label:
                    continue
                out.append(_summary_for(f"{prefix}/{label}", label))
        return out

    samplers_cfg = eval_cfg.get("sudoku_eval_samplers", None)
    samplers.extend(_collect_from(OmegaConf.to_container(samplers_cfg, resolve=True)))
    sjd_runs_cfg = eval_cfg.get("sudoku_eval_sjd_runs", None)
    samplers.extend(_collect_from(OmegaConf.to_container(sjd_runs_cfg, resolve=True)))

    return {
        "run_all_sampler_modes": True,
        "primary_sampler_label": primary_label,
        "samplers": samplers,
    }


def _apply_prop52_only_override(
    *,
    eval_cfg_local: DictConfig,
    offline_cfg: DictConfig,
) -> None:
    if not bool(offline_cfg.get("eval_prop52_only", False)):
        return
    if str(eval_cfg_local.get("mode", "fid_is")).lower() != "sudoku":
        raise ValueError("offline_eval.eval_prop52_only is supported only with eval.mode=sudoku.")
    eval_cfg_local.sudoku_prop52_enabled = True
    eval_cfg_local.sudoku_prop52_only = True


def _best_sampler_by_board_accuracy(
    sudoku_sampler_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not sudoku_sampler_summary:
        return None
    best = None
    best_score = None
    for sampler_info in sudoku_sampler_summary.get("samplers", []):
        score = sampler_info.get("board_acc", None)
        if score is None:
            score = sampler_info.get("board_acc_exact", None)
        score = _maybe_float(score)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = dict(sampler_info)
    return best


def _best_plugin_eta_by_board_accuracy(
    sudoku_sampler_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not sudoku_sampler_summary:
        return None
    best = None
    best_score = None
    for sampler_info in sudoku_sampler_summary.get("samplers", []):
        label = str(sampler_info.get("label", ""))
        if not label.startswith("plugin_hazard_eta_"):
            continue
        score = sampler_info.get("board_acc", None)
        if score is None:
            score = sampler_info.get("board_acc_exact", None)
        score = _maybe_float(score)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = dict(sampler_info)
    return best


def _best_pc_sampler_by_board_accuracy(
    sudoku_sampler_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not sudoku_sampler_summary:
        return None
    best = None
    best_score = None
    for sampler_info in sudoku_sampler_summary.get("samplers", []):
        label = str(sampler_info.get("label", ""))
        if not label.startswith("pc_"):
            continue
        score = sampler_info.get("board_acc", None)
        if score is None:
            score = sampler_info.get("board_acc_exact", None)
        score = _maybe_float(score)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = dict(sampler_info)
    return best


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
    base_rng = make_rng(int(seed))
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
    effective_cfg, run_dir, run_context_payload, experiment_cfg_source = (
        _resolve_effective_experiment_cfg(cfg=cfg, offline_cfg=offline_cfg)
    )

    sample_timesteps_cfg = offline_cfg.get("sample_timesteps", None)
    if _is_nullish(sample_timesteps_cfg):
        sample_timesteps = int(effective_cfg.training.sample_timesteps)
    else:
        sample_timesteps = int(sample_timesteps_cfg)
    _apply_explicit_eval_overrides(
        effective_cfg=effective_cfg,
        offline_cfg=offline_cfg,
        sample_timesteps=sample_timesteps,
    )

    task = build_task(effective_cfg)
    model = build_model(
        effective_cfg,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )

    rng = make_rng(int(effective_cfg.training.seed))
    state_template, _ = init_state(effective_cfg, model, rng)

    checkpoint_root = _resolve_checkpoint_root(effective_cfg, offline_cfg)
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
        param_source = "ema"
    else:
        if use_ema and (state.ema_params is None):
            print(
                "[offline-eval] Requested EMA params but checkpoint has no ema_params; "
                "falling back to params.",
                flush=True,
            )
        params_for_eval = state.params
        param_source = "live"

    eval_cfg_local = _clone_eval_cfg(eval_cfg)
    eval_cfg_local.run_at_end = True
    eval_cfg_local.checkpoint_source = checkpoint_source
    eval_cfg_local.param_source = param_source
    _apply_prop52_only_override(eval_cfg_local=eval_cfg_local, offline_cfg=offline_cfg)
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
            cfg=effective_cfg,
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
            effective_cfg.dataset.get("data_dir", None)
        )

    maybe_log_eval = build_eval_logger(
        cfg=effective_cfg,
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
    sudoku_sampler_summary = _extract_sudoku_sampler_summary(
        metrics=metrics,
        eval_cfg=eval_cfg_local,
    )
    sudoku_policy_table = getattr(maybe_log_eval, "sudoku_policy_table_rows", None)
    sudoku_prop52_rows_by_eta = getattr(maybe_log_eval, "sudoku_prop52_rows_by_eta", None)
    sudoku_prop52_summary_rows = getattr(maybe_log_eval, "sudoku_prop52_summary_rows", None)
    sudoku_prop52_collapse_summary = getattr(
        maybe_log_eval,
        "sudoku_prop52_collapse_summary",
        None,
    )

    probe_batches_cfg = offline_cfg.get("sampler_probe_batches", 0)
    probe_batches = 0 if _is_nullish(probe_batches_cfg) else int(probe_batches_cfg)
    sampling_probe_payload = None
    if probe_batches > 0:
        if (str(effective_cfg.model.name) == "sjd") and (eval_mode != "sudoku"):
            probe_batch_size_cfg = offline_cfg.get("sampler_probe_batch_size", None)
            if _is_nullish(probe_batch_size_cfg):
                probe_batch_size = int(eval_sample_batch_size)
            else:
                probe_batch_size = int(probe_batch_size_cfg)

            probe_seed_offset_cfg = offline_cfg.get("sampler_probe_seed_offset", 777777)
            probe_seed_offset = (
                777777 if _is_nullish(probe_seed_offset_cfg) else int(probe_seed_offset_cfg)
            )
            probe_seed = int(effective_cfg.training.seed) + probe_seed_offset

            probe_metrics = _collect_sjd_sampler_probe_metrics(
                cfg=effective_cfg,
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
                    f"got model={effective_cfg.model.name!r}, eval_mode={eval_mode!r}"
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

    effective_logit_temperature = _maybe_float(
        effective_cfg.sampler.get(
            "logit_temperature",
            effective_cfg.sampler.get("temperature", None),
        )
    )
    forward_meta = _extract_forward_config_metadata(effective_cfg)

    payload = {
        "timestamp_utc": now_utc_iso(),
        "experiment_config": {
            "source": experiment_cfg_source,
            "run_dir": None if run_dir is None else str(run_dir),
            "run_context_path": None if run_dir is None else str(run_dir / "run_context.json"),
            "task_name": str(effective_cfg.task.name),
            "model_name": str(effective_cfg.model.name),
            "forward_beta_target": forward_meta["forward_beta_target"],
            "forward_hazard_target": forward_meta["forward_hazard_target"],
            "forward_jump_target": forward_meta["forward_jump_target"],
            "jump_eta": forward_meta["jump_eta"],
            "sampler_logit_temperature": effective_logit_temperature,
            "sampler_n_steps": int(sample_timesteps),
            "training_seed": int(effective_cfg.training.seed),
        },
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
            "eval_prop52_only": bool(offline_cfg.get("eval_prop52_only", False)),
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
            "sudoku_run_all_sampler_modes": bool(
                eval_cfg_local.get("sudoku_run_all_sampler_modes", False)
            ),
            "sudoku_primary_sampler_label": str(
                eval_cfg_local.get("sudoku_primary_sampler_label", "default")
            ),
            "requested_jump_eta_override": _maybe_float(offline_cfg.get("jump_eta", None)),
            "requested_logit_temperature_override": _maybe_float(
                offline_cfg.get("logit_temperature", None)
            ),
            "use_run_config": bool(offline_cfg.get("use_run_config", False)),
        },
        "metrics": metrics,
    }
    if run_context_payload is not None:
        payload["run_context"] = {
            "checkpoint_dir": run_context_payload.get("checkpoint_dir", None),
            "metrics_dir": run_context_payload.get("metrics_dir", None),
            "timestamp_utc": run_context_payload.get("timestamp_utc", None),
        }
    if sampling_probe_payload is not None:
        payload["sampling_probe"] = sampling_probe_payload
    if sudoku_sampler_summary is not None:
        payload["sudoku_sampler_summary"] = sudoku_sampler_summary
    if sudoku_policy_table:
        payload["sudoku_policy_table"] = sudoku_policy_table
    if sudoku_prop52_rows_by_eta:
        payload["sudoku_prop52_rows_by_eta"] = sudoku_prop52_rows_by_eta
    if sudoku_prop52_summary_rows:
        payload["sudoku_prop52_summary_rows"] = sudoku_prop52_summary_rows
    if sudoku_prop52_collapse_summary:
        payload["sudoku_prop52_collapse_summary"] = sudoku_prop52_collapse_summary

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
        "[offline-eval] "
        f"config_source={experiment_cfg_source} "
        f"sample_timesteps={sample_timesteps} "
        f"jump_eta={forward_meta['jump_eta']} "
        f"logit_temperature={effective_logit_temperature}",
        flush=True,
    )
    if sudoku_sampler_summary is not None:
        print(
            "[offline-eval] "
            f"sudoku_primary_sampler={sudoku_sampler_summary['primary_sampler_label']} "
            f"run_all_sampler_modes={sudoku_sampler_summary['run_all_sampler_modes']}",
            flush=True,
        )
        for sampler_info in sudoku_sampler_summary["samplers"]:
            primary_score = (
                sampler_info["solve_rate"]
                if sampler_info["solve_rate"] is not None
                else sampler_info["acc_solve_strict"]
            )
            secondary_score = (
                sampler_info["board_acc"]
                if sampler_info["board_acc"] is not None
                else sampler_info["board_acc_exact"]
                if sampler_info["board_acc_exact"] is not None
                else sampler_info["acc_complete_puzzle"]
            )
            print(
                "[offline-eval] "
                f"{sampler_info['label']}: "
                f"primary_score={primary_score} "
                f"secondary_score={secondary_score} "
                f"avg_commits_per_step={sampler_info['avg_commits_per_step']}",
                flush=True,
            )
    best_sampler = _best_sampler_by_board_accuracy(sudoku_sampler_summary)
    best_plugin_eta = _best_plugin_eta_by_board_accuracy(sudoku_sampler_summary)
    best_pc_sampler = _best_pc_sampler_by_board_accuracy(sudoku_sampler_summary)
    if (
        best_sampler is not None
        or best_plugin_eta is not None
        or best_pc_sampler is not None
        or sudoku_prop52_collapse_summary
        or sudoku_prop52_summary_rows
    ):
        print("[offline-eval] Summary:", flush=True)
        if best_sampler is not None:
            best_sampler_score = best_sampler.get("board_acc", None)
            if best_sampler_score is None:
                best_sampler_score = best_sampler.get("board_acc_exact", None)
            print(
                f"- best Sudoku board accuracy by policy: {best_sampler.get('label')} "
                f"({best_sampler_score})",
                flush=True,
            )
        else:
            print("- best Sudoku board accuracy by policy: not run", flush=True)

        if best_plugin_eta is not None:
            best_plugin_score = best_plugin_eta.get("board_acc", None)
            if best_plugin_score is None:
                best_plugin_score = best_plugin_eta.get("board_acc_exact", None)
            print(
                f"- best eta under plug-in hazard: {best_plugin_eta.get('label')} "
                f"({best_plugin_score})",
                flush=True,
            )
        else:
            print("- best eta under plug-in hazard: not available", flush=True)

        if best_pc_sampler is not None:
            best_pc_score = best_pc_sampler.get("board_acc", None)
            if best_pc_score is None:
                best_pc_score = best_pc_sampler.get("board_acc_exact", None)
            print(
                f"- best predictor-corrector sampler: {best_pc_sampler.get('label')} "
                f"({best_pc_score})",
                flush=True,
            )
        else:
            print("- best predictor-corrector sampler: not available", flush=True)

        if sudoku_prop52_collapse_summary:
            print(
                "- whether V_state collapsed as eta approached 1: "
                f"{bool(sudoku_prop52_collapse_summary.get('v_state_collapsed', False))} "
                f"(ratio={sudoku_prop52_collapse_summary.get('eta_one_vs_max_lower_v_state_ratio')})",
                flush=True,
            )
        else:
            print("- whether V_state collapsed as eta approached 1: not evaluated", flush=True)

        caveats = []
        if bool(offline_cfg.get("eval_prop52_only", False)):
            caveats.append("board-accuracy policy sweep was skipped because eval_prop52_only=true")
        if sudoku_prop52_summary_rows:
            caveats.append("Prop 5.2 summaries are estimated on a fixed held-out subset and time bins")
        if not caveats:
            caveats.append("none")
        print(f"- caveats: {'; '.join(caveats)}", flush=True)
    print(
        f"[offline-eval] Wrote metrics report to {output_path}",
        flush=True,
    )

    return metrics
