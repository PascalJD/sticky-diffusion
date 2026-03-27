from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, TaskSpec
from sticky.tasks.sudoku_base import SudokuTaskBase


@dataclass
class SudokuMDLMTask(SudokuTaskBase):
    """Conditional Sudoku sequence task for vanilla MDLM.

    The first `3 * start_index` tokens are fixed clue-prefix conditioning.
    The remaining suffix tokens are the unknown solution tokens and are the
    only positions that contribute to the loss.
    """

    def __post_init__(self):
        self.spec = TaskSpec(
            name="mdlm_sudoku",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ) -> tuple[jnp.ndarray, Metrics]:
        key_sample, key_dropout = jax.random.split(rng)
        x = batch["image"].astype(jnp.int32)
        start_index = batch["start_index"].astype(jnp.int32)

        # Original Sudoku task semantics: the clue prefix is fixed conditioning
        # and only the suffix contributes to the MDLM objective.
        known_token_mask = self._known_prefix_mask(
            start_index=start_index,
            seq_len=x.shape[-1],
        )
        loss_mask = ~known_token_mask

        rngs = {"sample": key_sample}
        if train:
            rngs["dropout"] = key_dropout

        stats = model.apply(
            {"params": params},
            x,
            train=train,
            known_token_mask=known_token_mask,
            loss_mask=loss_mask,
            rngs=rngs,
        )
        return stats["loss"], stats

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)
