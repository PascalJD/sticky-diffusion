from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.data.sudoku import (
    SUDOKU_PACKED_SEQ_LEN,
    SUDOKU_VOCAB_SIZE,
    pack_sudoku_seq2seq,
)
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, TaskSpec
from sticky.tasks.sudoku_base import SudokuTaskBase


@dataclass
class SudokuMDMTask(SudokuTaskBase):
    """Ye-style packed Sudoku seq2seq task scaffold for future MDM training."""

    def __post_init__(self):
        self.data_shape = (SUDOKU_PACKED_SEQ_LEN,)
        self.vocab_size = SUDOKU_VOCAB_SIZE
        self.spec = TaskSpec(
            name="mdm_sudoku",
            task_type="text",
            data_shape=tuple(self.data_shape),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def _augment_batch(self, batch):
        packed = pack_sudoku_seq2seq(
            triplet_seq=batch["image"],
            start_index=batch["start_index"],
        )
        return {
            **dict(batch),
            "packed_seq": packed["packed_seq"],
            "prompt_mask": packed["prompt_mask"],
            "response_mask": packed["response_mask"],
            "prompt_token_count": packed["prompt_token_count"],
            "response_token_count": packed["response_token_count"],
            "sep_index": packed["sep_index"],
            "response_start_index": packed["response_start_index"],
            "eos_index": packed["eos_index"],
        }

    def _wrap_iterator(self, iterator: Iterable[Batch]):
        for batch in iterator:
            yield self._augment_batch(batch)

    def _ensure_packed_batch(self, batch: Batch):
        if all(key in batch for key in ("packed_seq", "prompt_mask", "response_mask")):
            return batch
        return self._augment_batch(batch)

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

    def _mask_response_tokens(
        self,
        *,
        rng: PRNGKey,
        x: jnp.ndarray,
        response_mask: jnp.ndarray,
        t: jnp.ndarray,
        timesteps: int,
        mask_token_id: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mask_prob = (t.astype(jnp.float32) + 1.0) / float(timesteps)
        draws = jax.random.uniform(rng, x.shape, dtype=jnp.float32)
        masked_response = jnp.asarray(response_mask, dtype=jnp.bool_) & (
            draws < mask_prob[:, None]
        )
        x_t = jnp.where(masked_response, int(mask_token_id), x).astype(jnp.int32)
        return x_t, masked_response, mask_prob

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_it, eval_it = super().make_dataloaders(seed=seed)
        train_wrapped = self._wrap_iterator(train_it)
        eval_wrapped = None if eval_it is None else self._wrap_iterator(eval_it)
        return train_wrapped, eval_wrapped

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
        teacher_params: Any = None,
    ) -> tuple[jnp.ndarray, Metrics]:
        del teacher_params
        batch = self._ensure_packed_batch(batch)

        x = jnp.asarray(batch["packed_seq"], dtype=jnp.int32)
        prompt_mask = jnp.asarray(batch["prompt_mask"], dtype=jnp.bool_)
        response_mask = jnp.asarray(batch["response_mask"], dtype=jnp.bool_)

        timesteps = int(getattr(model, "timesteps", 0))
        if timesteps <= 0:
            raise ValueError(f"MDM timesteps must be positive, got {timesteps}.")

        key_t, key_mask, key_dropout = jax.random.split(rng, 3)
        t = self._sample_timesteps(
            rng=key_t,
            batch_size=int(x.shape[0]),
            timesteps=timesteps,
        )
        x_t, masked_response, mask_prob = self._mask_response_tokens(
            rng=key_mask,
            x=x,
            response_mask=response_mask,
            t=t,
            timesteps=timesteps,
            mask_token_id=int(model.mask_token_id),
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

        per_token_ce = model.token_cross_entropy(logits, x)
        reweighted_token_ce = model.apply_token_reweighting(per_token_ce)
        time_weight = model.time_weights(t)[:, None]

        masked_response_f = masked_response.astype(jnp.float32)
        masked_token_count = jnp.sum(masked_response_f)
        denom = jnp.maximum(masked_token_count, jnp.asarray(1.0, dtype=jnp.float32))

        plain_loss = jnp.sum(per_token_ce * masked_response_f) / denom
        loss = jnp.sum(reweighted_token_ce * time_weight * masked_response_f) / denom

        total_response_tokens = jnp.sum(response_mask.astype(jnp.float32))
        total_tokens = jnp.asarray(x.size, dtype=jnp.float32)
        masked_per_example = jnp.sum(masked_response_f, axis=1)

        metrics = {
            "loss": loss,
            "loss_ce": plain_loss,
            "masked_token_count": masked_token_count,
            "masked_response_fraction": masked_token_count
            / jnp.maximum(total_response_tokens, jnp.asarray(1.0, dtype=jnp.float32)),
            "masked_token_fraction": masked_token_count
            / jnp.maximum(total_tokens, jnp.asarray(1.0, dtype=jnp.float32)),
            "masked_prompt_token_count": jnp.sum(
                (masked_response & prompt_mask).astype(jnp.float32)
            ),
            "examples_with_zero_masked_response_tokens": jnp.sum(
                (masked_per_example == 0).astype(jnp.float32)
            ),
            "mask_prob_mean": jnp.mean(mask_prob),
            "time_weight_mean": jnp.mean(model.time_weights(t)),
            "t_mean": jnp.mean(t.astype(jnp.float32)),
        }
        return loss, metrics

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)
