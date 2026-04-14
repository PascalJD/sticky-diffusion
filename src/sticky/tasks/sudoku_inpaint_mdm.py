from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.sudoku import get_sudoku_board_num_examples, make_sudoku_board_iterator
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


@dataclass
class SudokuInpaintMDMTask(Task):
    """Minimal Shah-board Sudoku inpainting task for the board MDM benchmark."""

    data_dir: Optional[str]
    train_file: str
    test_file: str
    batch_size: int
    eval_batch_size: int
    data_shape: Tuple[int, ...]
    vocab_size: int
    num_classes: int
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
        self.vocab_size = 10
        self.spec = TaskSpec(
            name="mdm_sudoku_inpaint",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

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

    def _sample_timesteps(
        self,
        *,
        rng: PRNGKey,
        batch_size: int,
        timesteps: int,
    ) -> jnp.ndarray:
        return jax.random.randint(
            rng,
            (int(batch_size),),
            minval=0,
            maxval=int(timesteps),
            dtype=jnp.int32,
        )

    def _mask_unknown_cells(
        self,
        *,
        rng: PRNGKey,
        solution_board: jnp.ndarray,
        clue_board: jnp.ndarray,
        clue_mask: jnp.ndarray,
        t: jnp.ndarray,
        timesteps: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask_prob = (t.astype(jnp.float32) + 1.0) / float(timesteps)
        draws = jax.random.uniform(rng, solution_board.shape, dtype=jnp.float32)
        masked_unknown = (~clue_mask) & (draws < mask_prob[:, None])
        x_t = jnp.where(
            clue_mask,
            clue_board,
            jnp.where(masked_unknown, 0, solution_board),
        ).astype(jnp.int32)
        return x_t, masked_unknown, mask_prob

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ) -> tuple[jnp.ndarray, Metrics]:
        solution_board = jnp.asarray(batch["solution_board"], dtype=jnp.int32)
        clue_board = jnp.asarray(batch["clue_board"], dtype=jnp.int32)
        clue_mask = jnp.asarray(batch["clue_mask"], dtype=jnp.bool_)

        timesteps = int(getattr(model, "timesteps", 0))
        if timesteps <= 0:
            raise ValueError(f"MDM-inpaint timesteps must be positive, got {timesteps}.")

        key_t, key_mask, key_dropout = jax.random.split(rng, 3)
        t = self._sample_timesteps(
            rng=key_t,
            batch_size=int(solution_board.shape[0]),
            timesteps=timesteps,
        )
        x_t, masked_unknown, mask_prob = self._mask_unknown_cells(
            rng=key_mask,
            solution_board=solution_board,
            clue_board=clue_board,
            clue_mask=clue_mask,
            t=t,
            timesteps=timesteps,
        )

        apply_kwargs = {"train": train}
        if train:
            apply_kwargs["rngs"] = {"dropout": key_dropout}
        logits = model.apply(
            {"params": params},
            x_t,
            t.astype(jnp.float32),
            method=model.predict_logits,
            **apply_kwargs,
        )

        per_token_ce = model.token_cross_entropy(logits, solution_board)
        masked_unknown_f = masked_unknown.astype(jnp.float32)
        masked_token_count = jnp.sum(masked_unknown_f)
        denom = jnp.maximum(masked_token_count, jnp.asarray(1.0, dtype=jnp.float32))
        loss = jnp.sum(per_token_ce * masked_unknown_f) / denom

        total_unknown = jnp.sum((~clue_mask).astype(jnp.float32))
        total_tokens = jnp.asarray(solution_board.size, dtype=jnp.float32)
        masked_per_example = jnp.sum(masked_unknown_f, axis=1)
        clue_board_matches = jnp.sum(
            (clue_mask & (clue_board == solution_board)).astype(jnp.float32)
        )
        clue_count = jnp.sum(clue_mask.astype(jnp.float32))

        metrics = {
            "loss": loss,
            "loss_ce": loss,
            "masked_token_count": masked_token_count,
            "masked_unknown_fraction": masked_token_count
            / jnp.maximum(total_unknown, jnp.asarray(1.0, dtype=jnp.float32)),
            "masked_token_fraction": masked_token_count
            / jnp.maximum(total_tokens, jnp.asarray(1.0, dtype=jnp.float32)),
            "masked_clue_token_count": jnp.sum(
                (masked_unknown & clue_mask).astype(jnp.float32)
            ),
            "examples_with_zero_masked_unknown_tokens": jnp.sum(
                (masked_per_example == 0).astype(jnp.float32)
            ),
            "mask_prob_mean": jnp.mean(mask_prob),
            "t_mean": jnp.mean(t.astype(jnp.float32)),
            "clue_match_fraction": clue_board_matches
            / jnp.maximum(clue_count, jnp.asarray(1.0, dtype=jnp.float32)),
        }
        return loss, metrics

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)

    def format_samples_for_logging(self, x: jnp.ndarray) -> list[str] | None:
        boards = jnp.asarray(x, dtype=jnp.int32)
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
