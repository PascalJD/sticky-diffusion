from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import core as jax_core

from sticky.data.sudoku import get_sudoku_board_num_examples, make_sudoku_board_iterator
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


Array = jnp.ndarray


@dataclass
class SudokuInpaintSJDTask(Task):
    """Board-level Sudoku SJD task with hint-clamped conditional corruption."""

    data_dir: Optional[str]
    train_file: str
    test_file: str
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int
    beta: Callable[[Array], Array]
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0
    time_sampling: str = "uniform"
    loss_weighting: str = "uniform"
    order_conditioning: str = "none"
    order_w_min: float = 0.5
    order_w_max: float = 2.0
    forward_site_hazard_mode: str = "none"
    teacher_confidence_metric: str = "margin"
    teacher_w_min: float = 0.5
    teacher_w_max: float = 2.0
    teacher_neutral_weight: float = 1.0
    teacher_log_metrics: bool = True
    drop_remainder: bool = True
    shuffle: bool = True
    mmap: bool = True
    max_train_examples: int = -1
    max_test_examples: int = -1
    auto_download: bool = True
    download_timeout_sec: int = 120
    download_retries: int = 8

    def __post_init__(self):
        self.data_shape = (81,)
        self.vocab_size = 9
        self.spec = TaskSpec(
            name="sjd_sudoku_inpaint",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    @property
    def requires_teacher_params(self) -> bool:
        return str(self.forward_site_hazard_mode) == "teacher_margin"

    def _make_board_iterator(
        self,
        *,
        split: str,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        max_examples: int,
    ):
        return make_sudoku_board_iterator(
            split=str(split),
            batch_size=int(batch_size),
            seed=int(seed),
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            mmap=bool(self.mmap),
            max_examples=int(max_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_it = self._make_board_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seed=int(seed),
            shuffle=bool(self.shuffle),
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            max_examples=int(self.max_train_examples),
        )
        eval_it = self._make_board_iterator(
            split="test",
            batch_size=int(self.eval_batch_size),
            seed=int(seed) + 1,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            max_examples=int(self.max_test_examples),
        )
        return train_it, eval_it

    def train_num_examples(self) -> int | None:
        return get_sudoku_board_num_examples(
            split="train",
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            mmap=bool(self.mmap),
            max_examples=int(self.max_train_examples),
            auto_download=bool(self.auto_download),
            download_timeout_sec=int(self.download_timeout_sec),
            download_retries=int(self.download_retries),
        )

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
        teacher_params=None,
    ) -> tuple[jnp.ndarray, Metrics]:
        solution_board = jnp.asarray(batch["solution_board"], dtype=jnp.int32)
        clue_mask = jnp.asarray(batch["clue_mask"], dtype=jnp.bool_)
        solver_rank = batch.get("solver_rank", None)
        if solver_rank is not None:
            solver_rank = jnp.asarray(solver_rank, dtype=jnp.float32)
        x0_idx = solution_board - 1
        # Keep a helpful eager-mode validation without triggering Python truth-value
        # conversion on tracers inside the jitted training step.
        if not isinstance(x0_idx, jax_core.Tracer):
            if bool(jnp.any(x0_idx < 0)) or bool(jnp.any(x0_idx > 8)):
                raise ValueError(
                    "Sudoku SJD clean targets must map digits 1..9 to indices 0..8."
                )

        key_loss, key_dropout = jax.random.split(rng)
        x0_anchor = model.apply({"params": params}, x0_idx, method=model.embed)
        anchor_table = None
        if bool(self.log_state_dependency):
            try:
                anchor_table = params["anchors"]["table"]
            except Exception:
                anchor_table = model.apply({"params": params}, method=model.anchor_table)

        def apply_fn(p, xt, t_img):
            if train:
                return model.apply(
                    {"params": p},
                    xt,
                    t=t_img,
                    train=True,
                    rngs={"dropout": key_dropout},
                )
            return model.apply({"params": p}, xt, t=t_img, train=False)

        teacher_mode = str(self.forward_site_hazard_mode) == "teacher_margin"
        x0_anchor_teacher = None
        teacher_apply_fn = None
        # When teacher mode is declared but no teacher params are available
        # (eval path via loop_helpers.make_eval_step_fn), fall back to the
        # baseline scalar hazard for this call. Attribute stays unchanged.
        effective_mode = str(self.forward_site_hazard_mode)
        if teacher_mode and teacher_params is None:
            effective_mode = "none"
        if teacher_mode and teacher_params is not None:
            x0_anchor_teacher = model.apply(
                {"params": teacher_params}, x0_idx, method=model.embed
            )

            def teacher_apply_fn(p, xt, t_img):  # noqa: E306  (shadow by design)
                return model.apply({"params": p}, xt, t=t_img, train=False)

        loss, metrics = ce_allocation_loss(
            key=key_loss,
            params=params,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=self.beta,
            hazard=self.hazard,
            jump=self.jump if bool(self.log_state_dependency) else None,
            anchor_table=anchor_table,
            state_dep_log_ratio_clip=float(self.state_dep_log_ratio_clip),
            T=float(self.T),
            given_mask=clue_mask,
            time_sampling=str(self.time_sampling),
            loss_weighting=str(self.loss_weighting),
            solver_rank=solver_rank,
            order_conditioning=str(self.order_conditioning),
            order_w_min=float(self.order_w_min),
            order_w_max=float(self.order_w_max),
            forward_site_hazard_mode=effective_mode,
            x0_anchor_teacher=x0_anchor_teacher,
            teacher_params=teacher_params,
            teacher_apply_fn=teacher_apply_fn,
            teacher_confidence_metric=str(self.teacher_confidence_metric),
            teacher_w_min=float(self.teacher_w_min),
            teacher_w_max=float(self.teacher_w_max),
            teacher_neutral_weight=float(self.teacher_neutral_weight),
            teacher_log_metrics=bool(self.teacher_log_metrics),
        )

        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["clean_index_min"] = jnp.min(x0_idx).astype(jnp.float32)
        metrics["clean_index_max"] = jnp.max(x0_idx).astype(jnp.float32)
        metrics["given_fraction"] = jnp.mean(clue_mask.astype(jnp.float32))
        return loss, metrics

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(x, dtype=jnp.int32)
        if arr.size == 0:
            return arr
        if int(jnp.min(arr)) >= 0 and int(jnp.max(arr)) <= 8:
            return arr + 1
        return arr

    def format_samples_for_logging(self, x: jnp.ndarray) -> list[str] | None:
        boards = self.decode(x)
        if boards.ndim == 1:
            boards = boards[None, :]
        rendered = []
        for board in boards:
            rows = [
                "".join(str(int(token)) for token in board[row * 9 : (row + 1) * 9])
                for row in range(9)
            ]
            rendered.append(" / ".join(rows))
        return rendered
