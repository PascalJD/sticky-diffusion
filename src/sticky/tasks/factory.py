# src/sticky/tasks/factory.py
from __future__ import annotations

from typing import Any, Callable

import hydra
from omegaconf import DictConfig


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
    beta = hydra.utils.instantiate(cfg.forward.beta)
    hazard_cfg = cfg.forward.get("hazard", None)
    hazard = hydra.utils.instantiate(hazard_cfg, beta=beta) if hazard_cfg is not None else None
    jump_cfg = cfg.forward.get("jump", None)
    jump = hydra.utils.instantiate(jump_cfg, beta=beta) if jump_cfg is not None else None
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
        "loss_weighting": str(cfg.training.get("loss_weighting", "uniform")),
    }


def _build_tfds_discrete_image_task(cfg: DictConfig, *, task_name: str):
    from sticky.tasks.cifar10_discrete import CIFAR10DiscreteTask

    return CIFAR10DiscreteTask(
        task_name=task_name,
        **_tfds_image_dataset_kwargs(cfg),
    )


def _build_openwebtext_discrete_task(cfg: DictConfig):
    from sticky.tasks.openwebtext_discrete import OpenWebTextDiscreteTask

    return OpenWebTextDiscreteTask(
        task_name="openwebtext_discrete",
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

    return CIFAR10SJDTask(
        task_name=task_name,
        **_tfds_image_dataset_kwargs(cfg),
        **_sjd_schedule_kwargs(cfg),
        **_sjd_teacher_hazard_kwargs(cfg),
    )


def _build_mdm_sudoku_inpaint_task(cfg: DictConfig):
    from sticky.tasks.sudoku_inpaint_mdm import SudokuInpaintMDMTask

    return SudokuInpaintMDMTask(**_sudoku_board_dataset_kwargs(cfg))


def _build_mdlm_sudoku_task(cfg: DictConfig):
    from sticky.tasks.sudoku_mdlm import SudokuMDLMTask

    return SudokuMDLMTask(**_sudoku_board_dataset_kwargs(cfg))


def _sjd_sudoku_order_conditioning_kwargs(cfg: DictConfig) -> dict[str, Any]:
    return {
        "order_conditioning": str(cfg.training.get("order_conditioning", "none")),
        "order_w_min": float(cfg.training.get("order_w_min", 0.5)),
        "order_w_max": float(cfg.training.get("order_w_max", 2.0)),
    }


def _sjd_teacher_hazard_kwargs(cfg: DictConfig) -> dict[str, Any]:
    mode = str(cfg.training.get("forward_site_hazard_mode", "none"))
    source = str(cfg.training.get("teacher_source", "ema"))
    if mode == "teacher_margin" and source != "ema":
        raise ValueError(
            f"training.teacher_source={source!r} not supported; only 'ema' is implemented."
        )
    return {
        "forward_site_hazard_mode": mode,
        "teacher_confidence_metric": str(cfg.training.get("teacher_confidence_metric", "margin")),
        "teacher_w_min": float(cfg.training.get("teacher_w_min", 0.5)),
        "teacher_w_max": float(cfg.training.get("teacher_w_max", 2.0)),
        "teacher_neutral_weight": float(cfg.training.get("teacher_neutral_weight", 1.0)),
        "teacher_log_metrics": bool(cfg.training.get("teacher_log_metrics", True)),
    }


def _build_sjd_sudoku_inpaint_task(cfg: DictConfig):
    from sticky.tasks.sudoku_inpaint_sjd import SudokuInpaintSJDTask

    return SudokuInpaintSJDTask(
        **_sudoku_board_dataset_kwargs(cfg),
        **_sjd_schedule_kwargs(cfg),
        **_sjd_sudoku_order_conditioning_kwargs(cfg),
        **_sjd_teacher_hazard_kwargs(cfg),
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
    "openwebtext_discrete": _build_openwebtext_discrete_task,
    "sjd_cifar10": lambda cfg: _build_tfds_sjd_task(cfg, task_name="sjd_cifar10"),
    "sjd_imagenet64": lambda cfg: _build_tfds_sjd_task(cfg, task_name="sjd_imagenet64"),
    "mdlm_sudoku": _build_mdlm_sudoku_task,
    "mdm_sudoku_inpaint": _build_mdm_sudoku_inpaint_task,
    "sjd_sudoku_inpaint": _build_sjd_sudoku_inpaint_task,
}


def build_task(cfg: DictConfig):
    name = str(cfg.task.name)
    builder = TASK_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown task.name={name}")
    return builder(cfg)
