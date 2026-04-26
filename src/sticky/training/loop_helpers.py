from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import jax
import jax.numpy as jnp
from flax.jax_utils import prefetch_to_device, unreplicate
from omegaconf import DictConfig

from sticky.rng import make_rng
from sticky.training.logging import sanitize_metrics, to_py_scalar
from sticky.training.state import shard_batch


Array = jnp.ndarray


def _identity(value):
    return value


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


def _maybe_sync_training_metric(metric: Array, *, sync: bool) -> None:
    if sync:
        jax.block_until_ready(metric)


def _normalize_sudoku_eval_param_source(source: Any) -> str:
    key = str(source).strip().lower()
    if key in {"", "auto"}:
        return "auto"
    if key in {"ema", "ema_params"}:
        return "ema"
    if key in {"live", "params"}:
        return "live"
    raise ValueError(
        f"Unknown eval.param_source={source!r}. Expected one of: ema, live, auto."
    )


def _params_for_sudoku_eval(state: Any, *, eval_cfg: DictConfig):
    source = _normalize_sudoku_eval_param_source(eval_cfg.get("param_source", "ema"))
    if source == "live":
        return state.params
    if getattr(state, "ema_params", None) is not None:
        return state.ema_params
    return state.params


def _resolve_num_train_steps(cfg: DictConfig, task: Any) -> int:
    epochs_cfg = cfg.training.get("num_train_epochs", None)
    if epochs_cfg in (None, "", "null", "None"):
        return int(cfg.training.num_train_steps)

    num_train_epochs = float(epochs_cfg)
    if num_train_epochs <= 0.0:
        return int(cfg.training.num_train_steps)

    train_num_examples = None
    train_num_examples_fn = getattr(task, "train_num_examples", None)
    if callable(train_num_examples_fn):
        train_num_examples = train_num_examples_fn()
    if train_num_examples is None:
        raise ValueError(
            "training.num_train_epochs requires the active task to expose "
            "train_num_examples()."
        )

    batch_size = int(cfg.dataset.batch_size)
    if batch_size <= 0:
        raise ValueError(f"dataset.batch_size must be positive, got {batch_size}.")

    drop_remainder = bool(cfg.dataset.get("drop_remainder", True))
    if drop_remainder:
        steps_per_epoch = max(1, int(train_num_examples) // batch_size)
    else:
        steps_per_epoch = max(1, int(math.ceil(int(train_num_examples) / batch_size)))

    num_train_steps = int(steps_per_epoch * num_train_epochs)
    print(
        "[train] Derived num_train_steps from epochs: "
        f"examples={int(train_num_examples)} batch_size={batch_size} "
        f"steps_per_epoch={steps_per_epoch} epochs={num_train_epochs:g} "
        f"num_train_steps={num_train_steps}",
        flush=True,
    )
    return num_train_steps


def _make_pmap_batch_iterator(
    train_iter: Iterator[dict[str, Array]],
    *,
    prefetch_buffer_size: int,
):
    sharded_iter = (shard_batch(batch) for batch in train_iter)
    if prefetch_buffer_size > 0:
        return prefetch_to_device(sharded_iter, prefetch_buffer_size)
    return sharded_iter


def _take_gt_images(batch: dict[str, Array], *, num_log_images: int, use_pmap: bool) -> Array | None:
    images = batch.get("image")
    if images is None:
        return None
    if use_pmap:
        images = images.reshape((-1,) + images.shape[2:])
    return images[:num_log_images]


def make_eval_step_fn(*, task, model):
    def eval_step_fn(params, rng, batch, axis_name: str | None):
        loss, metrics = task.loss_fn(
            rng=rng,
            model=model,
            params=params,
            batch=batch,
            train=False,
        )
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
    base_rng = make_rng(int(cfg.training.seed) + 2026)
    base_rng = jax.random.fold_in(base_rng, int(step_i))

    for batch_idx, batch in enumerate(eval_iter):
        if (max_batches > 0) and (batch_idx >= max_batches):
            break

        batch_size = int(next(iter(batch.values())).shape[0])
        if batch_size <= 0:
            continue
        if use_pmap and (batch_size % local_devices != 0):
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

    # Whitelist (Prompt B): the only likelihood-eval metric forwarded to W&B
    # is `eval/loss`. The full per-batch metric set still flows through
    # sanitize_metrics for any internal aggregation, but only `loss` is
    # surfaced as a logged metric.
    if "loss" not in metric_sums:
        return {}
    return {"eval/loss": metric_sums["loss"] / total_examples}


@dataclass
class LoopContext:
    cfg: DictConfig
    eval_cfg: Any
    wandb_mod: Any
    task: Any
    model: Any
    state: Any
    train_iter: Iterator[dict[str, Array]]
    use_pmap: bool
    pmap_prefetch_buffer_size: int
    num_train_steps: int
    num_log_images: int
    log_images_every_steps: int
    log_every_steps: int
    timing_warn_seconds: float
    sync_train_step: bool
    save_final_metrics: bool
    checkpoint_every_steps: int
    bits_per_dim_model: bool
    likelihood_eval_every_steps: int
    likelihood_eval_max_batches: int
    eval_enabled: bool
    run_eval_at_end: bool
    eval_mode: str
    sudoku_every: int
    text_every: int
    fid_every: int
    is_every: int
    sample_images_jit: Any
    maybe_log_eval: Any
    metrics_writer: Any
    checkpoint_writer: Any
    lr_schedule: Any
    train_step_fn: Any
    eval_step_fn: Any


@dataclass
class ExecutionAdapter:
    state: Any
    train_batches: Iterator[dict[str, Array]]
    run_train_step: Any
    eval_step_jit: Any
    p_eval_step: Any
    host_state: Callable[[Any], Any]
    host_metrics: Callable[[Any], Any]
    sampling_params: Callable[[Any], Any]


def build_loop_context(
    *,
    cfg: DictConfig,
    wandb_mod: Any,
    eval_cfg: Any,
    build_task_fn,
    build_model_fn,
    init_state_fn,
    get_hydra_output_dir_fn,
    resolve_run_path_fn,
    write_run_context_fn,
    make_lr_schedule_fn,
    build_sampling_fns_fn,
    build_eval_logger_fn,
    make_train_step_fn_fn,
    build_eval_step_fn_fn,
    resolve_from_original_cwd_fn,
    metrics_writer_cls,
    checkpoint_writer_cls,
):
    task = build_task_fn(cfg)
    model = build_model_fn(cfg, data_shape=task.spec.data_shape, vocab_size=task.spec.vocab_size)
    eval_cfg = eval_cfg or {}
    run_output_dir = get_hydra_output_dir_fn()

    num_log_images = int(cfg.training.num_log_images)
    sample_timesteps = int(cfg.training.sample_timesteps)
    num_train_steps = _resolve_num_train_steps(cfg, task)
    cfg.training.num_train_steps = int(num_train_steps)
    log_images_every_steps = int(cfg.training.log_images_every_steps)
    log_every_steps = int(cfg.training.log_every_steps)
    eval_every_steps = int(cfg.training.get("eval_every_steps", 0))
    timing_warn_seconds = float(cfg.training.get("timing_warn_seconds", 30.0))
    sync_train_step = bool(cfg.runtime.get("sync_train_step", False))
    pmap_prefetch_buffer_size = int(cfg.runtime.get("pmap_prefetch_buffer_size", 2))

    metrics_every_steps = int(cfg.training.get("metrics_every_steps", 0))
    save_final_metrics = bool(cfg.training.get("save_final_metrics", True))
    metrics_dir = resolve_run_path_fn(
        cfg.training.get("metrics_dir", "metrics"),
        "metrics",
        base_dir=run_output_dir,
    )
    metrics_writer = None
    if (metrics_every_steps > 0) or save_final_metrics:
        metrics_writer = metrics_writer_cls(root_dir=metrics_dir, every_steps=metrics_every_steps)

    checkpoint_every_steps = int(cfg.training.get("checkpoint_every_steps", 0))
    checkpoint_keep = int(cfg.training.get("checkpoint_keep", 5))
    save_final_checkpoint = bool(cfg.training.get("save_final_checkpoint", True))
    checkpoint_dir = resolve_run_path_fn(
        cfg.training.get("checkpoint_dir", "checkpoints"),
        "checkpoints",
        base_dir=run_output_dir,
    )
    checkpoint_writer = None
    if (checkpoint_every_steps > 0) or save_final_checkpoint:
        checkpoint_writer = checkpoint_writer_cls(
            root_dir=checkpoint_dir,
            every_steps=checkpoint_every_steps,
            keep=checkpoint_keep,
            save_final=save_final_checkpoint,
            best_metric_key=str(cfg.training.get("best_checkpoint_metric", "eval/fid")),
            best_mode=str(cfg.training.get("best_checkpoint_mode", "min")),
            best_update_on_equal=bool(cfg.training.get("best_update_on_equal", False)),
        )

    rng = make_rng(int(cfg.training.seed))
    state, tx = init_state_fn(cfg, model, rng)
    write_run_context_fn(
        run_dir=run_output_dir,
        experiment_cfg=cfg,
        eval_cfg=eval_cfg,
        params=state.params,
        metrics_dir=metrics_dir,
        checkpoint_dir=checkpoint_dir,
    )
    lr_schedule = make_lr_schedule_fn(cfg)
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
        eval_sample_every = (
            max(fid_every, is_every) if max(fid_every, is_every) > 0 else (1 if eval_sample_needed else 0)
        )
        eval_sample_batch_size = max(fid_batch_size, is_batch_size, 1)

    fid_cache_dir = resolve_from_original_cwd_fn(str(eval_cfg.get("fid_cache_dir", "data/fid_stats")))
    fid_tfds_data_dir = resolve_from_original_cwd_fn(eval_cfg.get("fid_tfds_data_dir", None))
    if fid_tfds_data_dir is None:
        fid_tfds_data_dir = resolve_from_original_cwd_fn(cfg.dataset.get("data_dir", None))

    sample_images_jit, sample_images_fid_jit = build_sampling_fns_fn(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=num_log_images,
        sample_timesteps=sample_timesteps,
        fid_every=eval_sample_every,
        fid_batch_size=eval_sample_batch_size,
    )

    maybe_log_eval = build_eval_logger_fn(
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

    train_step_fn = make_train_step_fn_fn(task=task, model=model, tx=tx, ema_rate=ema_rate)
    eval_step_fn = build_eval_step_fn_fn(task=task, model=model)

    return LoopContext(
        cfg=cfg,
        eval_cfg=eval_cfg,
        wandb_mod=wandb_mod,
        task=task,
        model=model,
        state=state,
        train_iter=train_iter,
        use_pmap=use_pmap,
        pmap_prefetch_buffer_size=pmap_prefetch_buffer_size,
        num_train_steps=num_train_steps,
        num_log_images=num_log_images,
        log_images_every_steps=log_images_every_steps,
        log_every_steps=log_every_steps,
        timing_warn_seconds=timing_warn_seconds,
        sync_train_step=sync_train_step,
        save_final_metrics=save_final_metrics,
        checkpoint_every_steps=checkpoint_every_steps,
        bits_per_dim_model=bits_per_dim_model,
        likelihood_eval_every_steps=likelihood_eval_every_steps,
        likelihood_eval_max_batches=likelihood_eval_max_batches,
        eval_enabled=eval_enabled,
        run_eval_at_end=run_eval_at_end,
        eval_mode=eval_mode,
        sudoku_every=sudoku_every,
        text_every=text_every,
        fid_every=fid_every,
        is_every=is_every,
        sample_images_jit=sample_images_jit,
        maybe_log_eval=maybe_log_eval,
        metrics_writer=metrics_writer,
        checkpoint_writer=checkpoint_writer,
        lr_schedule=lr_schedule,
        train_step_fn=train_step_fn,
        eval_step_fn=eval_step_fn,
    )


def prepare_execution_adapter(
    ctx: LoopContext,
    *,
    make_wrapped_train_step_fn,
    make_wrapped_eval_step_fn,
    make_pmap_batch_iterator_fn,
    replicate_fn,
    unreplicate_fn,
    params_for_sampling_fn,
):
    if ctx.use_pmap:
        p_train_step = make_wrapped_train_step_fn(
            ctx.train_step_fn,
            use_pmap=True,
            axis_name="batch",
        )
        p_eval_step = make_wrapped_eval_step_fn(
            ctx.eval_step_fn,
            use_pmap=True,
            axis_name="batch",
        )
        state = replicate_fn(ctx.state)
        train_batches = make_pmap_batch_iterator_fn(
            ctx.train_iter,
            prefetch_buffer_size=ctx.pmap_prefetch_buffer_size,
        )
        return ExecutionAdapter(
            state=state,
            train_batches=train_batches,
            run_train_step=p_train_step,
            eval_step_jit=None,
            p_eval_step=p_eval_step,
            host_state=unreplicate_fn,
            host_metrics=unreplicate_fn,
            sampling_params=params_for_sampling_fn,
        )

    train_step_jit = make_wrapped_train_step_fn(ctx.train_step_fn, use_pmap=False)
    eval_step_jit = make_wrapped_eval_step_fn(ctx.eval_step_fn, use_pmap=False)
    return ExecutionAdapter(
        state=ctx.state,
        train_batches=ctx.train_iter,
        run_train_step=train_step_jit,
        eval_step_jit=eval_step_jit,
        p_eval_step=None,
        host_state=_identity,
        host_metrics=_identity,
        sampling_params=params_for_sampling_fn,
    )


def process_completed_step(
    *,
    ctx: LoopContext,
    adapter: ExecutionAdapter,
    state: Any,
    batch: dict[str, Array],
    metrics: Any,
    step: int,
    t_fetch: float,
    t_step: float,
    last_train_metrics: dict[str, float],
    last_eval_metrics: dict[str, float],
    last_eval_step: Optional[int],
    sample_for_logging_fn,
    log_images_to_wandb_fn,
    params_for_sampling_fn,
):
    step_i = step + 1

    if (t_fetch > ctx.timing_warn_seconds) or (
        ctx.sync_train_step and (t_step > ctx.timing_warn_seconds)
    ):
        print(
            f"[step {step_i}] timing warning: "
            f"data_fetch={t_fetch:.2f}s train_step={t_step:.2f}s",
            flush=True,
        )

    need_train_host_metrics = (
        ((ctx.log_every_steps > 0) and (step_i % ctx.log_every_steps == 0))
        or ((ctx.metrics_writer is not None) and ctx.metrics_writer.should_write(step_i))
        or (ctx.save_final_metrics and (step_i == ctx.num_train_steps))
    )
    train_log = None
    if need_train_host_metrics:
        metrics_host = adapter.host_metrics(metrics)
        train_log = sanitize_metrics(metrics_host)
        train_log["lr"] = float(to_py_scalar(ctx.lr_schedule(step)) or ctx.lr_schedule(step))
        train_log["step"] = step_i
        train_log = _with_bpd_alias(
            train_log,
            prefix="train",
            enable_alias=ctx.bits_per_dim_model,
        )
        last_train_metrics = dict(train_log)

    if ((ctx.log_every_steps > 0) and (step_i % ctx.log_every_steps == 0)) and (train_log is not None):
        print(
            f"[step {step_i}] loss={float(train_log['train/loss']):.4f}",
            flush=True,
        )
        if ctx.wandb_mod is not None:
            ctx.wandb_mod.log(train_log, step=step_i)

    if (
        (ctx.metrics_writer is not None)
        and ctx.metrics_writer.should_write(step_i)
        and (train_log is not None)
    ):
        ctx.metrics_writer.write(step_i=step_i, metrics=train_log, tag="train")

    if (
        (ctx.task.spec.task_type == "image")
        and (ctx.wandb_mod is not None)
        and (ctx.log_images_every_steps > 0)
        and (step_i % ctx.log_images_every_steps == 0)
    ):
        gt_images = _take_gt_images(
            batch,
            num_log_images=ctx.num_log_images,
            use_pmap=ctx.use_pmap,
        )
        state_s = adapter.host_state(state)
        params = params_for_sampling_fn(state_s)
        samples, sjd_sample_metrics = sample_for_logging_fn(
            cfg=ctx.cfg,
            sample_images_jit=ctx.sample_images_jit,
            params_for_sampling=params,
            step=step_i,
        )

        log_images_to_wandb_fn(
            wandb_mod=ctx.wandb_mod,
            step_i=step_i,
            gt_images=gt_images,
            max_images=ctx.num_log_images,
            samples=samples,
        )
        if sjd_sample_metrics is not None:
            ctx.wandb_mod.log(sanitize_metrics(sjd_sample_metrics), step=step_i)

    if ctx.eval_mode == "sudoku":
        eval_due = (
            ctx.eval_enabled
            and (ctx.maybe_log_eval is not None)
            and (ctx.sudoku_every > 0)
            and (step_i % ctx.sudoku_every == 0)
        )
    elif ctx.eval_mode == "text_basic":
        eval_due = (
            ctx.eval_enabled
            and (ctx.maybe_log_eval is not None)
            and (ctx.text_every > 0)
            and (step_i % ctx.text_every == 0)
        )
    else:
        eval_due = (
            ctx.eval_enabled
            and (ctx.maybe_log_eval is not None)
            and (
                ((ctx.fid_every > 0) and (step_i % ctx.fid_every == 0))
                or ((ctx.is_every > 0) and (step_i % ctx.is_every == 0))
            )
        )
    checkpoint_due = (
        (ctx.checkpoint_writer is not None)
        and (ctx.checkpoint_every_steps > 0)
        and (step_i % ctx.checkpoint_every_steps == 0)
    )
    likelihood_eval_due = (
        ctx.likelihood_eval_every_steps > 0
        and (step_i % ctx.likelihood_eval_every_steps == 0)
    )

    if eval_due or checkpoint_due or likelihood_eval_due:
        state_s = adapter.host_state(state)

        if likelihood_eval_due:
            likelihood_metrics = run_likelihood_eval(
                cfg=ctx.cfg,
                task=ctx.task,
                eval_step_jit=adapter.eval_step_jit,
                p_eval_step=adapter.p_eval_step,
                params=adapter.sampling_params(state),
                use_pmap=ctx.use_pmap,
                step_i=step_i,
                max_batches=ctx.likelihood_eval_max_batches,
            )
            if likelihood_metrics:
                likelihood_metrics = _with_bpd_alias(
                    likelihood_metrics,
                    prefix="eval",
                    enable_alias=ctx.bits_per_dim_model,
                )
                last_eval_metrics = dict(last_eval_metrics)
                last_eval_metrics.update(likelihood_metrics)
                if ctx.metrics_writer is not None:
                    ctx.metrics_writer.write(
                        step_i=step_i,
                        metrics=likelihood_metrics,
                        tag="eval_likelihood",
                    )
                if ctx.wandb_mod is not None:
                    ctx.wandb_mod.log(likelihood_metrics, step=step_i)
                if ctx.checkpoint_writer is not None:
                    ctx.checkpoint_writer.maybe_save_best(
                        target=state_s,
                        step_i=step_i,
                        metrics=likelihood_metrics,
                    )

        if eval_due and ctx.maybe_log_eval is not None:
            eval_params = (
                _params_for_sudoku_eval(state_s, eval_cfg=ctx.eval_cfg)
                if ctx.eval_mode == "sudoku"
                else params_for_sampling_fn(state_s)
            )
            eval_metrics = ctx.maybe_log_eval(step_i, eval_params)
            if eval_metrics:
                last_eval_metrics = dict(last_eval_metrics)
                last_eval_metrics.update(eval_metrics)
                last_eval_step = step_i
                if ctx.metrics_writer is not None:
                    ctx.metrics_writer.write(step_i=step_i, metrics=eval_metrics, tag="eval")
                if ctx.checkpoint_writer is not None:
                    ctx.checkpoint_writer.maybe_save_best(
                        target=state_s,
                        step_i=step_i,
                        metrics=eval_metrics,
                    )

        if checkpoint_due and ctx.checkpoint_writer is not None:
            ctx.checkpoint_writer.maybe_save_periodic(target=state_s, step_i=step_i)

    return last_train_metrics, last_eval_metrics, last_eval_step


def finalize_training_loop(
    *,
    ctx: LoopContext,
    adapter: ExecutionAdapter,
    state: Any,
    last_train_metrics: dict[str, float],
    last_eval_metrics: dict[str, float],
    last_eval_step: Optional[int],
    params_for_sampling_fn,
):
    final_state = adapter.host_state(state)

    if (
        ctx.eval_enabled
        and ctx.run_eval_at_end
        and (ctx.maybe_log_eval is not None)
        and (last_eval_step != ctx.num_train_steps)
    ):
        final_eval_metrics = ctx.maybe_log_eval(
            ctx.num_train_steps,
            (
                _params_for_sudoku_eval(final_state, eval_cfg=ctx.eval_cfg)
                if ctx.eval_mode == "sudoku"
                else params_for_sampling_fn(final_state)
            ),
            force_fid=True,
            force_is=True,
        )
        if final_eval_metrics:
            last_eval_metrics = dict(final_eval_metrics)
            if ctx.metrics_writer is not None:
                ctx.metrics_writer.write(
                    step_i=ctx.num_train_steps,
                    metrics=final_eval_metrics,
                    tag="eval",
                )
            if ctx.checkpoint_writer is not None:
                ctx.checkpoint_writer.maybe_save_best(
                    target=final_state,
                    step_i=ctx.num_train_steps,
                    metrics=final_eval_metrics,
                )

    if ctx.checkpoint_writer is not None:
        ctx.checkpoint_writer.save_final_checkpoint(target=final_state, step_i=ctx.num_train_steps)

    if ctx.metrics_writer is not None and ctx.save_final_metrics:
        final_metrics = {}
        final_metrics.update(last_train_metrics)
        final_metrics.update(last_eval_metrics)
        ctx.metrics_writer.write_final(step_i=ctx.num_train_steps, metrics=final_metrics)
