from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from omegaconf import DictConfig

from sticky.models.factory import build_model
from sticky.tasks.factory import build_task
from sticky.training.eval import (
    build_fid_logger,
    resolve_from_original_cwd,
    sample_for_logging,
)
from sticky.training.logging import (
    log_images_to_wandb,
    sanitize_metrics,
    to_py_scalar,
)
from sticky.training.sampling import build_sampling_fns
from sticky.training.state import init_state, make_lr_schedule, shard_batch
from sticky.training.step import make_train_step_fn, params_for_sampling


Array = jnp.ndarray


def main_train_loop(
    cfg: DictConfig, wandb_mod=None, eval_cfg: Optional[DictConfig] = None
):
    task = build_task(cfg)
    model = build_model(cfg, data_shape=task.spec.data_shape, vocab_size=task.spec.vocab_size)

    num_log_images = int(cfg.training.num_log_images)
    sample_timesteps = int(cfg.training.sample_timesteps)
    num_train_steps = int(cfg.training.num_train_steps)
    log_images_every_steps = int(cfg.training.log_images_every_steps)
    log_every_steps = int(cfg.training.log_every_steps)

    rng = jax.random.PRNGKey(int(cfg.training.seed))
    state, tx = init_state(cfg, model, rng)
    lr_schedule = make_lr_schedule(cfg)

    train_iter, _ = task.make_dataloaders(seed=int(cfg.training.seed))

    if cfg.runtime.platform == "auto":
        use_pmap = jax.device_count() > 1
    elif cfg.runtime.platform == "pmap":
        use_pmap = True
    else:
        use_pmap = False

    ema_rate = float(cfg.training.ema_rate)

    eval_cfg = eval_cfg or {}
    eval_enabled = (wandb_mod is not None) and bool(eval_cfg.get("enabled", False))

    fid_every = int(eval_cfg.get("fid_every", 0)) if eval_enabled else 0
    fid_num_samples = int(eval_cfg.get("fid_num_samples", 50_000))
    fid_batch_size = int(eval_cfg.get("fid_batch_size", 256))
    fid_prefix = str(eval_cfg.get("prefix", "eval"))
    fid_log_at_step_zero = bool(eval_cfg.get("log_at_step_zero", False))

    fid_cache_dir = resolve_from_original_cwd(str(eval_cfg.get("fid_cache_dir", "data/fid_stats")))
    fid_tfds_data_dir = resolve_from_original_cwd(eval_cfg.get("fid_tfds_data_dir", None))

    sample_images_jit, sample_images_fid_jit = build_sampling_fns(
        cfg=cfg,
        task=task,
        model=model,
        num_log_images=num_log_images,
        sample_timesteps=sample_timesteps,
        fid_every=fid_every,
        fid_batch_size=fid_batch_size,
    )

    maybe_log_fid = build_fid_logger(
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
    )

    train_step_fn = make_train_step_fn(task=task, model=model, tx=tx, ema_rate=ema_rate)

    if use_pmap:
        p_train_step = jax.pmap(
            lambda st, b: train_step_fn(st, b, axis_name="batch"),
            axis_name="batch",
        )
        state = replicate(state)

        for step in range(num_train_steps):
            if step in (4400, 4550, 4650):
                jax.profiler.save_device_memory_profile(f"/tmp/mem_{step}.prof")

            batch = next(train_iter)
            gt_images = batch["image"][:num_log_images]
            batch = shard_batch(batch)

            state, metrics = p_train_step(state, batch)
            _ = jax.block_until_ready(metrics["train/loss"])

            if (step + 1) % log_every_steps == 0:
                metrics = unreplicate(metrics)
                print(f"[step {step+1}] loss={float(metrics['train/loss']):.4f}")
                if wandb_mod is not None:
                    log_dict = sanitize_metrics(metrics)
                    log_dict["lr"] = float(
                        to_py_scalar(lr_schedule(step)) or lr_schedule(step)
                    )
                    log_dict["step"] = step + 1
                    wandb_mod.log(log_dict, step=step + 1)

            if (wandb_mod is not None) and (log_images_every_steps > 0) and (
                step % log_images_every_steps == 0
            ):
                state_s = unreplicate(state)
                params = params_for_sampling(state_s)
                samples, sjd_sample_metrics = sample_for_logging(
                    cfg=cfg,
                    sample_images_jit=sample_images_jit,
                    params_for_sampling=params,
                    step=step,
                )

                log_images_to_wandb(
                    wandb_mod=wandb_mod,
                    step_i=step,
                    gt_images=gt_images,
                    max_images=num_log_images,
                    samples=samples,
                )
                if sjd_sample_metrics is not None:
                    wandb_mod.log(sanitize_metrics(sjd_sample_metrics), step=step)

            if maybe_log_fid is not None:
                state_s = unreplicate(state)
                maybe_log_fid(step + 1, params_for_sampling(state_s))

    else:
        train_step_jit = jax.jit(lambda st, b: train_step_fn(st, b, axis_name=None))

        for step in range(num_train_steps):
            batch = next(train_iter)
            gt_images = batch["image"][:num_log_images]
            state, metrics = train_step_jit(state, batch)

            if (step + 1) % log_every_steps == 0:
                print(f"[step {step+1}] loss={float(metrics['train/loss']):.4f}")
                if wandb_mod is not None:
                    log_dict = sanitize_metrics(metrics)
                    log_dict["lr"] = float(
                        to_py_scalar(lr_schedule(step)) or lr_schedule(step)
                    )
                    log_dict["step"] = step + 1
                    wandb_mod.log(log_dict, step=step + 1)

            if (wandb_mod is not None) and (log_images_every_steps > 0) and (
                step % log_images_every_steps == 0
            ):
                params = params_for_sampling(state)
                samples, sjd_sample_metrics = sample_for_logging(
                    cfg=cfg,
                    sample_images_jit=sample_images_jit,
                    params_for_sampling=params,
                    step=step,
                )

                log_images_to_wandb(
                    wandb_mod=wandb_mod,
                    step_i=step,
                    gt_images=gt_images,
                    max_images=num_log_images,
                    samples=samples,
                )
                if sjd_sample_metrics is not None:
                    wandb_mod.log(sanitize_metrics(sjd_sample_metrics), step=step)

            if maybe_log_fid is not None:
                maybe_log_fid(step + 1, params_for_sampling(state))
