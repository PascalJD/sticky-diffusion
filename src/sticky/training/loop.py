from __future__ import annotations

import time
from typing import Optional

import jax
import jax.numpy as jnp
from flax.jax_utils import prefetch_to_device, replicate, unreplicate
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
from sticky.training.state import init_state, make_lr_schedule
from sticky.training.step import (
    make_train_step_fn,
    make_wrapped_eval_step,
    make_wrapped_train_step,
    params_for_sampling,
)
from sticky.training.loop_helpers import (
    _is_bits_per_dim_model,
    _make_pmap_batch_iterator,
    _maybe_sync_training_metric,
    _normalize_sudoku_eval_param_source,
    _params_for_sudoku_eval,
    _resolve_num_train_steps,
    _take_gt_images,
    _with_bpd_alias,
    build_loop_context,
    finalize_training_loop,
    make_eval_step_fn,
    prepare_execution_adapter,
    process_completed_step,
    run_likelihood_eval,
)


def main_train_loop(
    cfg: DictConfig, wandb_mod=None, eval_cfg: Optional[DictConfig] = None
):
    ctx = build_loop_context(
        cfg=cfg,
        wandb_mod=wandb_mod,
        eval_cfg=eval_cfg,
        build_task_fn=build_task,
        build_model_fn=build_model,
        init_state_fn=init_state,
        get_hydra_output_dir_fn=get_hydra_output_dir,
        resolve_run_path_fn=resolve_run_path,
        write_run_context_fn=write_run_context,
        make_lr_schedule_fn=make_lr_schedule,
        build_sampling_fns_fn=build_sampling_fns,
        build_eval_logger_fn=build_eval_logger,
        make_train_step_fn_fn=make_train_step_fn,
        build_eval_step_fn_fn=make_eval_step_fn,
        resolve_from_original_cwd_fn=resolve_from_original_cwd,
        metrics_writer_cls=MetricsWriter,
        checkpoint_writer_cls=CheckpointWriter,
    )
    adapter = prepare_execution_adapter(
        ctx,
        make_wrapped_train_step_fn=make_wrapped_train_step,
        make_wrapped_eval_step_fn=make_wrapped_eval_step,
        make_pmap_batch_iterator_fn=_make_pmap_batch_iterator,
        replicate_fn=replicate,
        unreplicate_fn=unreplicate,
        params_for_sampling_fn=params_for_sampling,
    )
    last_train_metrics = {}
    last_eval_metrics = {}
    last_eval_step: Optional[int] = None
    state = adapter.state
    for step in range(ctx.num_train_steps):
        t_fetch0 = time.perf_counter()
        batch = next(adapter.train_batches)
        t_fetch = time.perf_counter() - t_fetch0

        t_step0 = time.perf_counter()
        state, metrics = adapter.run_train_step(state, batch)
        _maybe_sync_training_metric(metrics["train/loss"], sync=ctx.sync_train_step)
        t_step = time.perf_counter() - t_step0 if ctx.sync_train_step else 0.0

        last_train_metrics, last_eval_metrics, last_eval_step = process_completed_step(
            ctx=ctx,
            adapter=adapter,
            state=state,
            batch=batch,
            metrics=metrics,
            step=step,
            t_fetch=t_fetch,
            t_step=t_step,
            last_train_metrics=last_train_metrics,
            last_eval_metrics=last_eval_metrics,
            last_eval_step=last_eval_step,
            sample_for_logging_fn=sample_for_logging,
            log_images_to_wandb_fn=log_images_to_wandb,
            params_for_sampling_fn=params_for_sampling,
        )

    finalize_training_loop(
        ctx=ctx,
        adapter=adapter,
        state=state,
        last_train_metrics=last_train_metrics,
        last_eval_metrics=last_eval_metrics,
        last_eval_step=last_eval_step,
        params_for_sampling_fn=params_for_sampling,
    )
