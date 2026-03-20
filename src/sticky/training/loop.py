from __future__ import annotations

import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from omegaconf import DictConfig

from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.training.eval import (
    build_eval_logger,
    resolve_from_original_cwd,
    sample_for_logging,
)
from sticky.training.logging import (
    log_images_to_wandb,
    sanitize_metrics,
    to_py_scalar,
)
from sticky.training.persistence import (
    CheckpointWriter,
    MetricsWriter,
    get_hydra_output_dir,
    resolve_run_path,
    write_run_context,
)
from sticky.training.sampling import build_sampling_fns
from sticky.training.state import init_state, make_lr_schedule, shard_batch
from sticky.training.step import make_train_step_fn, params_for_sampling


Array = jnp.ndarray


def _is_bits_per_dim_model(cfg: DictConfig, task: Any) -> bool:
    return str(cfg.model.name) in {"md4", "cadd"} and str(task.spec.task_type) == "image"


def _with_bpd_alias(metrics: dict[str, float], *, prefix: str, enable_alias: bool) -> dict[str, float]:
    out = dict(metrics)
    if not enable_alias:
        return out

    loss_key = f"{prefix}/loss"
    if loss_key in out:
        out[f"{prefix}/bpd"] = out[loss_key]

    for suffix in ("loss_diff", "loss_prior", "loss_recon"):
        key = f"{prefix}/{suffix}"
        if key in out:
            alias = suffix.replace("loss_", "bpd_")
            out[f"{prefix}/{alias}"] = out[key]

    return out


def make_eval_step_fn(*, task, model):
    def eval_step_fn(params, rng, batch, axis_name: str | None):
        loss, metrics = task.loss_fn(rng=rng, model=model, params=params, batch=batch, train=False)
        if axis_name is not None:
            loss = jax.lax.pmean(loss, axis_name=axis_name)
            metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name=axis_name), metrics)

        metrics = dict(metrics)
        metrics["loss"] = loss
        return metrics

    return eval_step_fn


def run_likelihood_eval(
    *,
    cfg: DictConfig,
    task: Any,
    eval_step_jit,
    p_eval_step,
    params: Any,
    use_pmap: bool,
    step_i: int,
    max_batches: int,
) -> dict[str, float]:
    _, eval_iter = task.make_dataloaders(seed=int(cfg.training.seed) + 1)
    if eval_iter is None:
        return {}

    total_examples = 0
    metric_sums: dict[str, float] = {}
    local_devices = jax.local_device_count()
    base_rng = jax.random.PRNGKey(int(cfg.training.seed) + 2026)
    base_rng = jax.random.fold_in(base_rng, int(step_i))

    for batch_idx, batch in enumerate(eval_iter):
        if (max_batches > 0) and (batch_idx >= max_batches):
            break

        batch_size = int(next(iter(batch.values())).shape[0])
        if batch_size <= 0:
            continue
        if use_pmap and (batch_size % local_devices != 0):
            # Skip the final non-divisible tail batch under pmap.
            continue

        rng = jax.random.fold_in(base_rng, batch_idx)
        if use_pmap:
            per_device_rng = jax.random.split(rng, local_devices)
            metrics = p_eval_step(params, shard_batch(batch), per_device_rng)
            metrics_host = unreplicate(metrics)
        else:
            metrics_host = eval_step_jit(params, batch, rng)

        clean_metrics = sanitize_metrics(metrics_host)
        if not clean_metrics:
            continue

        total_examples += batch_size
        for key, value in clean_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + (float(value) * batch_size)

    if total_examples <= 0:
        return {}

    return {f"eval/{k}": v / total_examples for k, v in metric_sums.items()}


def main_train_loop(
    cfg: DictConfig, wandb_mod=None, eval_cfg: Optional[DictConfig] = None
):
    task = build_task(cfg)
    model = build_model(cfg, data_shape=task.spec.data_shape, vocab_size=task.spec.vocab_size)
    eval_cfg = eval_cfg or {}
    run_output_dir = get_hydra_output_dir()

    num_log_images = int(cfg.training.num_log_images)
    sample_timesteps = int(cfg.training.sample_timesteps)
    num_train_steps = int(cfg.training.num_train_steps)
    log_images_every_steps = int(cfg.training.log_images_every_steps)
    log_every_steps = int(cfg.training.log_every_steps)
    eval_every_steps = int(cfg.training.get("eval_every_steps", 0))
    timing_warn_seconds = float(cfg.training.get("timing_warn_seconds", 30.0))

    metrics_every_steps = int(cfg.training.get("metrics_every_steps", 0))
    save_final_metrics = bool(cfg.training.get("save_final_metrics", True))
    metrics_dir = resolve_run_path(
        cfg.training.get("metrics_dir", "metrics"),
        "metrics",
        base_dir=run_output_dir,
    )
    metrics_writer = None
    if (metrics_every_steps > 0) or save_final_metrics:
        metrics_writer = MetricsWriter(root_dir=metrics_dir, every_steps=metrics_every_steps)

    checkpoint_every_steps = int(cfg.training.get("checkpoint_every_steps", 0))
    checkpoint_keep = int(cfg.training.get("checkpoint_keep", 5))
    save_final_checkpoint = bool(cfg.training.get("save_final_checkpoint", True))
    checkpoint_dir = resolve_run_path(
        cfg.training.get("checkpoint_dir", "checkpoints"),
        "checkpoints",
        base_dir=run_output_dir,
    )
    checkpoint_writer = None
    if (checkpoint_every_steps > 0) or save_final_checkpoint:
        checkpoint_writer = CheckpointWriter(
            root_dir=checkpoint_dir,
            every_steps=checkpoint_every_steps,
            keep=checkpoint_keep,
            save_final=save_final_checkpoint,
            best_metric_key=str(cfg.training.get("best_checkpoint_metric", "eval/fid")),
            best_mode=str(cfg.training.get("best_checkpoint_mode", "min")),
        )

    rng = jax.random.PRNGKey(int(cfg.training.seed))
    state, tx = init_state(cfg, model, rng)
    write_run_context(
        run_dir=run_output_dir,
        experiment_cfg=cfg,
        eval_cfg=eval_cfg,
        params=state.params,
        metrics_dir=metrics_dir,
        checkpoint_dir=checkpoint_dir,
    )
    lr_schedule = make_lr_schedule(cfg)

    train_iter, _ = task.make_dataloaders(seed=int(cfg.training.seed))

    if cfg.runtime.platform == "auto":
        use_pmap = jax.device_count() > 1
    elif cfg.runtime.platform == "pmap":
        use_pmap = True
    else:
        use_pmap = False

    ema_rate = float(cfg.training.ema_rate)
    bits_per_dim_model = _is_bits_per_dim_model(cfg, task)
    likelihood_eval_every_steps = int(
        cfg.training.get(
            "likelihood_eval_every_steps",
            cfg.training.get("eval_every_steps", 0),
        )
    )
    likelihood_eval_max_batches = int(cfg.training.get("likelihood_eval_max_batches", -1))

    eval_enabled = bool(eval_cfg.get("enabled", False))
    run_eval_at_end = bool(eval_cfg.get("run_at_end", True))
    eval_mode = str(eval_cfg.get("mode", "fid_is")).lower()

    sudoku_every = 0
    text_every = 0
    fid_every = 0
    is_every = 0
    if eval_enabled and eval_mode == "sudoku":
        sudoku_every = int(eval_cfg.get("sudoku_every", eval_every_steps))
    elif eval_enabled and eval_mode == "text_basic":
        text_every = int(eval_cfg.get("text_every", eval_every_steps))
    elif eval_enabled:
        fid_every = int(eval_cfg.get("fid_every", eval_every_steps))
        is_every = int(eval_cfg.get("is_every", fid_every))

    fid_num_samples = int(eval_cfg.get("fid_num_samples", 50_000))
    fid_batch_size = int(eval_cfg.get("fid_batch_size", 256))
    is_batch_size = int(eval_cfg.get("is_batch_size", fid_batch_size))
    fid_prefix = str(eval_cfg.get("prefix", "eval"))
    fid_log_at_step_zero = bool(eval_cfg.get("log_at_step_zero", False))
    if eval_mode == "text_basic":
        eval_sample_needed = eval_enabled and ((text_every > 0) or run_eval_at_end)
        eval_sample_every = text_every if text_every > 0 else (1 if eval_sample_needed else 0)
        eval_sample_batch_size = max(1, int(eval_cfg.get("text_batch_size", num_log_images)))
    else:
        eval_sample_needed = (
            eval_enabled
            and eval_mode != "sudoku"
            and ((max(fid_every, is_every) > 0) or run_eval_at_end)
        )
        eval_sample_every = max(fid_every, is_every) if max(fid_every, is_every) > 0 else (1 if eval_sample_needed else 0)
        eval_sample_batch_size = max(fid_batch_size, is_batch_size, 1)

    fid_cache_dir = resolve_from_original_cwd(str(eval_cfg.get("fid_cache_dir", "data/fid_stats")))
    fid_tfds_data_dir = resolve_from_original_cwd(eval_cfg.get("fid_tfds_data_dir", None))
    if fid_tfds_data_dir is None:
        fid_tfds_data_dir = resolve_from_original_cwd(cfg.dataset.get("data_dir", None))

    sample_images_jit, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=num_log_images,
        sample_timesteps=sample_timesteps,
        fid_every=eval_sample_every,
        fid_batch_size=eval_sample_batch_size,
    )

    maybe_log_eval = build_eval_logger(
        cfg=cfg,
        eval_cfg=eval_cfg,
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
        eval_every=(
            sudoku_every
            if eval_mode == "sudoku"
            else (text_every if eval_mode == "text_basic" else max(fid_every, is_every))
        ),
    )

    train_step_fn = make_train_step_fn(task=task, model=model, tx=tx, ema_rate=ema_rate)
    eval_step_fn = make_eval_step_fn(task=task, model=model)
    last_train_metrics = {}
    last_eval_metrics = {}
    last_eval_step: Optional[int] = None

    if use_pmap:
        p_train_step = jax.pmap(
            lambda st, b: train_step_fn(st, b, axis_name="batch"),
            axis_name="batch",
        )
        p_eval_step = jax.pmap(
            lambda p, b, r: eval_step_fn(p, r, b, axis_name="batch"),
            axis_name="batch",
        )
        state = replicate(state)

        for step in range(num_train_steps):
            t_fetch0 = time.perf_counter()
            batch = next(train_iter)
            t_fetch = time.perf_counter() - t_fetch0
            gt_images = batch["image"][:num_log_images] if task.spec.task_type == "image" else None
            batch = shard_batch(batch)

            t_step0 = time.perf_counter()
            state, metrics = p_train_step(state, batch)
            _ = jax.block_until_ready(metrics["train/loss"])
            t_step = time.perf_counter() - t_step0

            step_i = step + 1
            if (t_fetch > timing_warn_seconds) or (t_step > timing_warn_seconds):
                print(
                    f"[step {step_i}] timing warning: "
                    f"data_fetch={t_fetch:.2f}s train_step={t_step:.2f}s",
                    flush=True,
                )
            need_train_host_metrics = (
                (step_i % log_every_steps == 0)
                or (
                    (metrics_writer is not None)
                    and metrics_writer.should_write(step_i)
                )
            )
            train_log = None
            if need_train_host_metrics:
                metrics_host = unreplicate(metrics)
                train_log = sanitize_metrics(metrics_host)
                train_log["lr"] = float(
                    to_py_scalar(lr_schedule(step)) or lr_schedule(step)
                )
                train_log["step"] = step_i
                train_log = _with_bpd_alias(train_log, prefix="train", enable_alias=bits_per_dim_model)
                last_train_metrics = dict(train_log)

            if step_i % log_every_steps == 0 and train_log is not None:
                print(
                    f"[step {step_i}] loss={float(train_log['train/loss']):.4f}",
                    flush=True,
                )
                if wandb_mod is not None:
                    wandb_mod.log(train_log, step=step_i)

            if (
                (metrics_writer is not None)
                and metrics_writer.should_write(step_i)
                and (train_log is not None)
            ):
                metrics_writer.write(step_i=step_i, metrics=train_log, tag="train")

            if (
                (task.spec.task_type == "image")
                and (wandb_mod is not None)
                and (log_images_every_steps > 0)
                and (
                step_i % log_images_every_steps == 0
                )
            ):
                state_s = unreplicate(state)
                params = params_for_sampling(state_s)
                samples, sjd_sample_metrics = sample_for_logging(
                    cfg=cfg,
                    sample_images_jit=sample_images_jit,
                    params_for_sampling=params,
                    step=step_i,
                )

                log_images_to_wandb(
                    wandb_mod=wandb_mod,
                    step_i=step_i,
                    gt_images=gt_images,
                    max_images=num_log_images,
                    samples=samples,
                )
                if sjd_sample_metrics is not None:
                    wandb_mod.log(sanitize_metrics(sjd_sample_metrics), step=step_i)

            if eval_mode == "sudoku":
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (sudoku_every > 0)
                    and (step_i % sudoku_every == 0)
                )
            elif eval_mode == "text_basic":
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (text_every > 0)
                    and (step_i % text_every == 0)
                )
            else:
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (
                        ((fid_every > 0) and (step_i % fid_every == 0))
                        or ((is_every > 0) and (step_i % is_every == 0))
                    )
                )
            checkpoint_due = (
                (checkpoint_writer is not None)
                and (checkpoint_every_steps > 0)
                and (step_i % checkpoint_every_steps == 0)
            )
            likelihood_eval_due = (
                likelihood_eval_every_steps > 0
                and (step_i % likelihood_eval_every_steps == 0)
            )

            if eval_due or checkpoint_due or likelihood_eval_due:
                state_s = unreplicate(state)

                if likelihood_eval_due:
                    likelihood_metrics = run_likelihood_eval(
                        cfg=cfg,
                        task=task,
                        eval_step_jit=None,
                        p_eval_step=p_eval_step,
                        params=params_for_sampling(state),
                        use_pmap=True,
                        step_i=step_i,
                        max_batches=likelihood_eval_max_batches,
                    )
                    if likelihood_metrics:
                        likelihood_metrics = _with_bpd_alias(
                            likelihood_metrics,
                            prefix="eval",
                            enable_alias=bits_per_dim_model,
                        )
                        last_eval_metrics.update(likelihood_metrics)
                        if metrics_writer is not None:
                            metrics_writer.write(step_i=step_i, metrics=likelihood_metrics, tag="eval_likelihood")
                        if wandb_mod is not None:
                            wandb_mod.log(likelihood_metrics, step=step_i)

                if eval_due and maybe_log_eval is not None:
                    eval_metrics = maybe_log_eval(step_i, params_for_sampling(state_s))
                    if eval_metrics:
                        last_eval_metrics.update(eval_metrics)
                        last_eval_step = step_i
                        if metrics_writer is not None:
                            metrics_writer.write(step_i=step_i, metrics=eval_metrics, tag="eval")
                        if checkpoint_writer is not None:
                            checkpoint_writer.maybe_save_best(
                                target=state_s,
                                step_i=step_i,
                                metrics=eval_metrics,
                            )

                if checkpoint_due and checkpoint_writer is not None:
                    checkpoint_writer.maybe_save_periodic(target=state_s, step_i=step_i)

    else:
        train_step_jit = jax.jit(lambda st, b: train_step_fn(st, b, axis_name=None))
        eval_step_jit = jax.jit(lambda p, b, r: eval_step_fn(p, r, b, axis_name=None))

        for step in range(num_train_steps):
            t_fetch0 = time.perf_counter()
            batch = next(train_iter)
            t_fetch = time.perf_counter() - t_fetch0
            gt_images = batch["image"][:num_log_images] if task.spec.task_type == "image" else None
            t_step0 = time.perf_counter()
            state, metrics = train_step_jit(state, batch)
            _ = jax.block_until_ready(metrics["train/loss"])
            t_step = time.perf_counter() - t_step0

            step_i = step + 1
            if (t_fetch > timing_warn_seconds) or (t_step > timing_warn_seconds):
                print(
                    f"[step {step_i}] timing warning: "
                    f"data_fetch={t_fetch:.2f}s train_step={t_step:.2f}s",
                    flush=True,
                )
            train_log = sanitize_metrics(metrics)
            train_log["lr"] = float(
                to_py_scalar(lr_schedule(step)) or lr_schedule(step)
            )
            train_log["step"] = step_i
            train_log = _with_bpd_alias(train_log, prefix="train", enable_alias=bits_per_dim_model)
            last_train_metrics = dict(train_log)

            if step_i % log_every_steps == 0:
                print(
                    f"[step {step_i}] loss={float(train_log['train/loss']):.4f}",
                    flush=True,
                )
                if wandb_mod is not None:
                    wandb_mod.log(train_log, step=step_i)

            if (metrics_writer is not None) and metrics_writer.should_write(step_i):
                metrics_writer.write(step_i=step_i, metrics=train_log, tag="train")

            if (
                (task.spec.task_type == "image")
                and (wandb_mod is not None)
                and (log_images_every_steps > 0)
                and (
                step_i % log_images_every_steps == 0
                )
            ):
                params = params_for_sampling(state)
                samples, sjd_sample_metrics = sample_for_logging(
                    cfg=cfg,
                    sample_images_jit=sample_images_jit,
                    params_for_sampling=params,
                    step=step_i,
                )

                log_images_to_wandb(
                    wandb_mod=wandb_mod,
                    step_i=step_i,
                    gt_images=gt_images,
                    max_images=num_log_images,
                    samples=samples,
                )
                if sjd_sample_metrics is not None:
                    wandb_mod.log(sanitize_metrics(sjd_sample_metrics), step=step_i)

            if eval_mode == "sudoku":
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (sudoku_every > 0)
                    and (step_i % sudoku_every == 0)
                )
            elif eval_mode == "text_basic":
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (text_every > 0)
                    and (step_i % text_every == 0)
                )
            else:
                eval_due = (
                    eval_enabled
                    and (maybe_log_eval is not None)
                    and (
                        ((fid_every > 0) and (step_i % fid_every == 0))
                        or ((is_every > 0) and (step_i % is_every == 0))
                    )
                )
            likelihood_eval_due = (
                likelihood_eval_every_steps > 0
                and (step_i % likelihood_eval_every_steps == 0)
            )
            if likelihood_eval_due:
                likelihood_metrics = run_likelihood_eval(
                    cfg=cfg,
                    task=task,
                    eval_step_jit=eval_step_jit,
                    p_eval_step=None,
                    params=params_for_sampling(state),
                    use_pmap=False,
                    step_i=step_i,
                    max_batches=likelihood_eval_max_batches,
                )
                if likelihood_metrics:
                    likelihood_metrics = _with_bpd_alias(
                        likelihood_metrics,
                        prefix="eval",
                        enable_alias=bits_per_dim_model,
                    )
                    last_eval_metrics.update(likelihood_metrics)
                    if metrics_writer is not None:
                        metrics_writer.write(step_i=step_i, metrics=likelihood_metrics, tag="eval_likelihood")
                    if wandb_mod is not None:
                        wandb_mod.log(likelihood_metrics, step=step_i)
            if eval_due and maybe_log_eval is not None:
                eval_metrics = maybe_log_eval(step_i, params_for_sampling(state))
                if eval_metrics:
                    last_eval_metrics.update(eval_metrics)
                    last_eval_step = step_i
                    if metrics_writer is not None:
                        metrics_writer.write(step_i=step_i, metrics=eval_metrics, tag="eval")
                    if checkpoint_writer is not None:
                        checkpoint_writer.maybe_save_best(
                            target=state,
                            step_i=step_i,
                            metrics=eval_metrics,
                        )

            if (checkpoint_writer is not None) and (checkpoint_every_steps > 0):
                checkpoint_writer.maybe_save_periodic(target=state, step_i=step_i)

    final_state = unreplicate(state) if use_pmap else state

    if (
        eval_enabled
        and run_eval_at_end
        and (maybe_log_eval is not None)
        and (last_eval_step != num_train_steps)
    ):
        final_eval_metrics = maybe_log_eval(
            num_train_steps,
            params_for_sampling(final_state),
            force_fid=True,
            force_is=True,
        )
        if final_eval_metrics:
            last_eval_metrics = dict(final_eval_metrics)
            if metrics_writer is not None:
                metrics_writer.write(step_i=num_train_steps, metrics=final_eval_metrics, tag="eval")
            if checkpoint_writer is not None:
                checkpoint_writer.maybe_save_best(
                    target=final_state,
                    step_i=num_train_steps,
                    metrics=final_eval_metrics,
                )

    if checkpoint_writer is not None:
        checkpoint_writer.save_final_checkpoint(target=final_state, step_i=num_train_steps)

    if metrics_writer is not None and save_final_metrics:
        final_metrics = {}
        final_metrics.update(last_train_metrics)
        final_metrics.update(last_eval_metrics)
        metrics_writer.write_final(step_i=num_train_steps, metrics=final_metrics)
