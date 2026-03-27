from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from sticky.models.sjd.losses import ce_allocation_loss
from sticky.tasks.base import TaskSpec
from sticky.tasks.sudoku_base import SudokuTaskBase

Array = jnp.ndarray


@dataclass
class SudokuSJDTask(SudokuTaskBase):
    """Sudoku sequence task for SJD.

    Each example is a sequence of length 243 where tokens are grouped as
    `(row, col, value)` triplets for the 81 cells. The model sees tokens in
    `[0, 9]` with:
      - row/col in `[0, 8]`
      - value in `[1, 9]`

    The original Google Sudoku task is conditional sequence modeling:
      - the first `3 * start_index` tokens are the given clue prefix
      - the remaining tokens are the solution suffix
      - training loss is applied only on the suffix
    """

    # VP forward schedule
    beta: Callable[[Array], Array] = None
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0

    def __post_init__(self):
        if self.beta is None:
            raise ValueError("SudokuSJDTask requires a beta schedule.")
        self._spec = TaskSpec(
            name="sjd_sudoku",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def loss_fn(
        self,
        *,
        rng: Array,
        model,
        params: Any,
        batch: Dict[str, Array],
        train: bool,
    ):
        key_loss, key_dropout = jax.random.split(rng)

        # Discrete sequence tokens [B, 243].
        x0_idx = batch["image"].astype(jnp.int32)
        start_index = batch["start_index"].astype(jnp.int32)

        # Match the original Sudoku task definition: the clue prefix is always
        # conditioned on and never contributes to the loss.
        given_mask = self._known_prefix_mask(
            start_index=start_index,
            seq_len=x0_idx.shape[-1],
        )

        # Anchor vectors [B, 243, d_anchor].
        x0_anchor = model.apply({"params": params}, x0_idx, method=model.embed)
        anchor_table = None
        if self.log_state_dependency:
            try:
                anchor_table = params["anchors"]["table"]
            except Exception:
                anchor_table = model.apply({"params": params}, method=model.anchor_table)

        def apply_fn(p, xt, t_img):
            if train:
                return model.apply(
                    {"params": p},
                    xt,
                    t_img,
                    train=True,
                    rngs={"dropout": key_dropout},
                )
            return model.apply({"params": p}, xt, t_img, train=False)

        loss, metrics = ce_allocation_loss(
            key=key_loss,
            params=params,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=self.beta,
            hazard=self.hazard,
            jump=self.jump if self.log_state_dependency else None,
            anchor_table=anchor_table,
            state_dep_log_ratio_clip=float(self.state_dep_log_ratio_clip),
            T=float(self.T),
            given_mask=given_mask,
        )

        return loss, metrics
