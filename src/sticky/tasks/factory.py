# src/sticky/tasks/factory.py
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import hydra
import numpy as np
from omegaconf import DictConfig

from sticky.models.sjd.blur import build_blur_kernel


def _optional_str(value):
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _image_augment_kwargs(cfg: DictConfig) -> dict[str, Any]:
    aug_cfg = cfg.dataset.get("augment", {})
    return {
        "augment_enabled": bool(aug_cfg.get("enabled", True)),
        "augment_prob": float(aug_cfg.get("prob", 0.15)),
        "augment_rotate": bool(aug_cfg.get("rotate", True)),
        "augment_hflip": bool(aug_cfg.get("hflip", True)),
        "augment_eval": bool(aug_cfg.get("eval", False)),
    }


def _tfds_include_label(cfg: DictConfig) -> bool | str:
    include_label = cfg.dataset.get("include_label", "auto")
    if isinstance(include_label, bool):
        return bool(include_label)
    return str(include_label)


def _tfds_image_dataset_kwargs(cfg: DictConfig) -> dict[str, Any]:
    shuffle_buffer_size = cfg.dataset.get("shuffle_buffer_size", None)
    return {
        "dataset_name": str(cfg.dataset.get("tfds_name", "cifar10")),
        "train_split": str(cfg.dataset.get("train_split", "train")),
        "eval_split": str(cfg.dataset.get("eval_split", "test")),
        "include_label": _tfds_include_label(cfg),
        "data_dir": _optional_str(cfg.dataset.get("data_dir", None)),
        "batch_size": int(cfg.dataset.get("batch_size")),
        "eval_batch_size": int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
        "data_shape": tuple(cfg.dataset.get("data_shape", (32, 32, 3))),
        "vocab_size": int(cfg.dataset.get("vocab_size", 256)),
        "num_classes": int(cfg.dataset.get("num_classes", -1)),
        "drop_remainder": bool(cfg.dataset.get("drop_remainder", True)),
        "shuffle_buffer_size": None if shuffle_buffer_size in (None, "", "null", "None") else int(shuffle_buffer_size),
        **_image_augment_kwargs(cfg),
    }


def _sudoku_board_dataset_kwargs(cfg: DictConfig) -> dict[str, Any]:
    return {
        "data_dir": _optional_str(cfg.dataset.get("data_dir", None)),
        "train_file": str(cfg.dataset.get("train_file", "Sudoku-train-data.npy")),
        "test_file": str(cfg.dataset.get("test_file", "Sudoku-test-data.npy")),
        "batch_size": int(cfg.dataset.get("batch_size")),
        "eval_batch_size": int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
        "data_shape": tuple(cfg.dataset.get("data_shape", (81,))),
        "vocab_size": int(cfg.dataset.get("vocab_size", 10)),
        "num_classes": int(cfg.dataset.get("num_classes", -1)),
        "drop_remainder": bool(cfg.dataset.get("drop_remainder", True)),
        "shuffle": bool(cfg.dataset.get("shuffle", True)),
        "mmap": bool(cfg.dataset.get("mmap", True)),
        "max_train_examples": int(cfg.dataset.get("max_train_examples", -1)),
        "max_test_examples": int(cfg.dataset.get("max_test_examples", -1)),
        "auto_download": bool(cfg.dataset.get("auto_download", True)),
        "download_timeout_sec": int(cfg.dataset.get("download_timeout_sec", 120)),
        "download_retries": int(cfg.dataset.get("download_retries", 8)),
    }


def _sjd_schedule_kwargs(cfg: DictConfig) -> dict[str, Any]:
    from sticky.models.sjd.freq_weighting import (
        hazard_weighting_mode,
        load_anchor_log_w,
    )

    beta = hydra.utils.instantiate(cfg.forward.beta)
    hazard_cfg = cfg.forward.get("hazard", None)
    hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None
    jump_cfg = cfg.forward.get("jump", None)
    jump = hydra.utils.instantiate(jump_cfg, beta=beta) if jump_cfg is not None else None

    # Phase 1.B: blur lives at the orthogonal `forward.blur` group. We attach
    # the kernel to the jump after instantiate so VPMatchedGaussianJump never
    # sees a `blur` kwarg.
    blur_cfg = cfg.forward.get("blur", None)
    if blur_cfg is not None and bool(blur_cfg.get("enabled", False)):
        # seq_len: prod(dataset.data_shape) for 1D tasks; the constraint kernel
        # ignores it (sudoku is hard-coded to 81). 2D image tasks are guarded
        # at the per-task builder level, so they never reach this branch.
        data_shape = tuple(cfg.dataset.get("data_shape", ()) or ())
        seq_len = int(np.prod(data_shape)) if data_shape else None
        kernel = build_blur_kernel(blur_cfg, seq_len=seq_len)
        if kernel is not None and jump is not None:
            jump = dataclasses.replace(jump, blur_kernel=kernel)

    hazard_weighting_cfg = cfg.forward.get("hazard_weighting", None)
    vocab_size = int(cfg.dataset.get("vocab_size"))
    hw_mode = hazard_weighting_mode(hazard_weighting_cfg)
    if hw_mode == "learned":
        # log_w lives in `params` under "anchors/log_w"; the task reads it from
        # there inside loss_fn so gradients can flow.
        anchor_log_w = None
        learn_log_w = True
    else:
        anchor_log_w = load_anchor_log_w(hazard_weighting_cfg, vocab_size=vocab_size)
        learn_log_w = False
    loss_weighting = str(cfg.training.get("loss_weighting", "uniform"))
    hw_loss_weighting = (
        getattr(hazard_weighting_cfg, "loss_weighting", None)
        if hazard_weighting_cfg is not None
        else None
    )
    if hw_loss_weighting not in (None, "", "null"):
        # The hazard_weighting config can pin loss_weighting (e.g. learned.yaml
        # sets hazard_deriv so log_w actually receives gradient). Field-level
        # CLI overrides on forward.hazard_weighting.loss_weighting still apply.
        loss_weighting = str(hw_loss_weighting)
    return {
        "beta": beta,
        "hazard": hazard,
        "jump": jump,
        "T": float(cfg.sampler.get("T", getattr(beta, "T", 1.0))),
        "log_state_dependency": bool(cfg.training.get("log_state_dependency", True)),
        "state_dep_log_ratio_clip": float(
            cfg.training.get(
                "state_dep_log_ratio_clip",
                cfg.sampler.get("log_ratio_clip", 10.0),
            )
        ),
        "time_sampling": str(cfg.training.get("time_sampling", "uniform")),
        "loss_weighting": loss_weighting,
        "anchor_log_w": anchor_log_w,
        "learn_log_w": learn_log_w,
        "t_floor": float(cfg.training.get("t_floor", 1e-3)),
        "log_anchor_log_w_stats": bool(
            cfg.training.get("log_anchor_log_w_stats", True)
        ),
        "pass_noisy_mask_to_model": bool(
            cfg.get("model", {}).get("use_noisy_input_bias", False)
        ),
    }


def _build_tfds_discrete_image_task(cfg: DictConfig, *, task_name: str):
    from sticky.tasks.cifar10_discrete import CIFAR10DiscreteTask

    return CIFAR10DiscreteTask(
        task_name=task_name,
        **_tfds_image_dataset_kwargs(cfg),
    )


def _build_openwebtext_discrete_task(cfg: DictConfig, *, task_name: str = "openwebtext_discrete"):
    from sticky.tasks.openwebtext_discrete import OpenWebTextDiscreteTask

    return OpenWebTextDiscreteTask(
        task_name=task_name,
        train_tokens_path=str(cfg.dataset.get("train_tokens_path")),
        eval_tokens_path=_optional_str(cfg.dataset.get("eval_tokens_path", None)),
        batch_size=int(cfg.dataset.get("batch_size")),
        eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
        seq_len=int(cfg.dataset.get("seq_len")),
        vocab_size=int(cfg.dataset.get("vocab_size")),
        tokenizer_name=_optional_str(cfg.dataset.get("tokenizer_name", None)),
        num_classes=int(cfg.dataset.get("num_classes", -1)),
        drop_remainder=bool(cfg.dataset.get("drop_remainder", True)),
        shuffle=bool(cfg.dataset.get("shuffle", True)),
        mmap=bool(cfg.dataset.get("mmap", True)),
        max_train_examples=int(cfg.dataset.get("max_train_examples", -1)),
        max_eval_examples=int(cfg.dataset.get("max_eval_examples", -1)),
    )


def _build_tfds_sjd_task(cfg: DictConfig, *, task_name: str):
    from sticky.tasks.cifar10_sjd import CIFAR10SJDTask

    # Phase 1: blur is not supported for 2D image tasks. Fail loudly if enabled.
    blur_cfg = cfg.forward.get("blur", None)
    if blur_cfg is not None and bool(blur_cfg.get("enabled", False)):
        raise ValueError(
            "Non-local blur is not supported for 2D image tasks (CIFAR10/ImageNet64) "
            "in Phase 1. Set forward.blur.enabled=false or use a sequence task."
        )

    # CIFAR10SJDTask does not accept drop_remainder / shuffle_buffer_size;
    # strip them so the shared _tfds_image_dataset_kwargs helper can evolve
    # without breaking the SJD task constructor.
    image_kw = _tfds_image_dataset_kwargs(cfg)
    image_kw.pop("drop_remainder", None)
    image_kw.pop("shuffle_buffer_size", None)

    return CIFAR10SJDTask(
        task_name=task_name,
        **image_kw,
        **_sjd_schedule_kwargs(cfg),
    )


def _build_sjd_sudoku_inpaint_task(cfg: DictConfig):
    from sticky.tasks.sudoku_inpaint_sjd import SudokuInpaintSJDTask

    return SudokuInpaintSJDTask(
        **_sudoku_board_dataset_kwargs(cfg),
        **_sjd_schedule_kwargs(cfg),
    )


def _build_openwebtext_sjd_task(cfg: DictConfig):
    from sticky.tasks.openwebtext_sjd import OpenWebTextSJDTask

    return OpenWebTextSJDTask(
        task_name="openwebtext_sjd",
        train_tokens_path=str(cfg.dataset.get("train_tokens_path")),
        eval_tokens_path=_optional_str(cfg.dataset.get("eval_tokens_path", None)),
        batch_size=int(cfg.dataset.get("batch_size")),
        eval_batch_size=int(cfg.dataset.get("eval_batch_size", cfg.dataset.batch_size)),
        seq_len=int(cfg.dataset.get("seq_len")),
        vocab_size=int(cfg.dataset.get("vocab_size")),
        tokenizer_name=_optional_str(cfg.dataset.get("tokenizer_name", None)),
        num_classes=int(cfg.dataset.get("num_classes", -1)),
        drop_remainder=bool(cfg.dataset.get("drop_remainder", True)),
        shuffle=bool(cfg.dataset.get("shuffle", True)),
        mmap=bool(cfg.dataset.get("mmap", True)),
        max_train_examples=int(cfg.dataset.get("max_train_examples", -1)),
        max_eval_examples=int(cfg.dataset.get("max_eval_examples", -1)),
        **_sjd_schedule_kwargs(cfg),
    )


TASK_BUILDERS: dict[str, Callable[[DictConfig], Any]] = {
    "md4_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="md4_cifar10"),
    "mdlm_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="mdlm_cifar10"),
    "d3pm_absorb_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="d3pm_absorb_cifar10"),
    "d3pm_uniform_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="d3pm_uniform_cifar10"),
    "d3pm_gaussian_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="d3pm_gaussian_cifar10"),
    "candi_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="candi_cifar10"),
    "cadd_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="cadd_cifar10"),
    "bitdiff_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="bitdiff_cifar10"),
    "ddpm_cifar10": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="ddpm_cifar10"),
    "md4_imagenet64": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="md4_imagenet64"),
    "mdlm_imagenet64": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="mdlm_imagenet64"),
    "bitdiff_imagenet64": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="bitdiff_imagenet64"),
    "ddpm_imagenet64": lambda cfg: _build_tfds_discrete_image_task(cfg, task_name="ddpm_imagenet64"),
    "openwebtext_discrete": _build_openwebtext_discrete_task,
    "mdlm_openwebtext": lambda cfg: _build_openwebtext_discrete_task(cfg, task_name="mdlm_openwebtext"),
    "md4_openwebtext": lambda cfg: _build_openwebtext_discrete_task(cfg, task_name="md4_openwebtext"),
    "sjd_cifar10": lambda cfg: _build_tfds_sjd_task(cfg, task_name="sjd_cifar10"),
    "sjd_imagenet64": lambda cfg: _build_tfds_sjd_task(cfg, task_name="sjd_imagenet64"),
    "sjd_sudoku_inpaint": _build_sjd_sudoku_inpaint_task,
    "openwebtext_sjd": _build_openwebtext_sjd_task,
}


def build_task(cfg: DictConfig):
    name = str(cfg.task.name)
    builder = TASK_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown task.name={name}")
    return builder(cfg)
