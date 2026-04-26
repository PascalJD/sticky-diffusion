from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sticky.data.openwebtext import make_openwebtext_iterator
from sticky.models.sjd.losses import ce_allocation_loss
from sticky.rng import PRNGKey
from sticky.tasks.base import Batch, Metrics, Task, TaskSpec


Array = jnp.ndarray


@dataclass
class OpenWebTextSJDTask(Task):
    """OpenWebText SJD task wiring GPT-2 token IDs through the SJD anchor path."""

    task_name: str
    train_tokens_path: str
    eval_tokens_path: Optional[str]
    batch_size: int
    eval_batch_size: int
    seq_len: int
    vocab_size: int
    beta: Callable[[Array], Array]
    hazard: Optional[Any] = None
    jump: Optional[Any] = None
    T: float = 1.0
    log_state_dependency: bool = True
    state_dep_log_ratio_clip: float = 10.0
    time_sampling: str = "uniform"
    loss_weighting: str = "uniform"
    tokenizer_name: Optional[str] = None
    num_classes: int = -1
    drop_remainder: bool = True
    shuffle: bool = True
    mmap: bool = True
    max_train_examples: int = -1
    max_eval_examples: int = -1
    _tokenizer: Any = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        self.spec = TaskSpec(
            name=str(self.task_name),
            task_type="text",
            data_shape=(int(self.seq_len),),
            vocab_size=int(self.vocab_size),
            num_classes=int(self.num_classes),
        )

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        train_iter = make_openwebtext_iterator(
            split="train",
            batch_size=int(self.batch_size),
            seq_len=int(self.seq_len),
            train_tokens_path=str(self.train_tokens_path),
            eval_tokens_path=self.eval_tokens_path,
            seed=int(seed),
            shuffle=bool(self.shuffle),
            repeat=True,
            drop_remainder=bool(self.drop_remainder),
            mmap=bool(self.mmap),
            max_examples=int(self.max_train_examples),
        )
        eval_iter = make_openwebtext_iterator(
            split="eval",
            batch_size=int(self.eval_batch_size),
            seq_len=int(self.seq_len),
            train_tokens_path=str(self.train_tokens_path),
            eval_tokens_path=self.eval_tokens_path,
            seed=int(seed) + 1,
            shuffle=False,
            repeat=False,
            drop_remainder=False,
            mmap=bool(self.mmap),
            max_examples=int(self.max_eval_examples),
        )
        return train_iter, eval_iter

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model,
        params,
        batch: Batch,
        train: bool,
    ) -> tuple[jnp.ndarray, Metrics]:
        x0_idx = jnp.asarray(batch["image"], dtype=jnp.int32)

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
            given_mask=None,
            time_sampling=str(self.time_sampling),
            loss_weighting=str(self.loss_weighting),
        )

        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["clean_index_min"] = jnp.min(x0_idx).astype(jnp.float32)
        metrics["clean_index_max"] = jnp.max(x0_idx).astype(jnp.float32)
        return loss, metrics

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x, dtype=jnp.int32)

    def _get_tokenizer(self):
        if self.tokenizer_name in (None, "", "null", "None"):
            return None
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except Exception as e:
                raise ImportError(
                    "transformers is required for text sample decoding when "
                    "dataset.tokenizer_name is configured."
                ) from e

            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_name))
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            self._tokenizer = tokenizer
        return self._tokenizer

    def format_samples_for_logging(self, x: jnp.ndarray) -> list[str] | None:
        samples = np.asarray(jax.device_get(x))
        if samples.ndim == 1:
            samples = samples[None, :]
        samples = samples.astype(np.int64, copy=False)

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return [" ".join(str(int(tok)) for tok in row) for row in samples]

        replacement = 0
        for candidate in (
            tokenizer.mask_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        ):
            if candidate is not None:
                replacement = int(candidate)
                break

        vocab_size = int(getattr(tokenizer, "vocab_size", self.vocab_size))
        safe_samples = np.where(
            (samples < 0) | (samples >= vocab_size),
            replacement,
            samples,
        )
        return tokenizer.batch_decode(
            safe_samples.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
