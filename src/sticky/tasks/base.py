# src/sticky/tasks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
Batch = Mapping[str, Array]
Metrics = Dict[str, Array]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str   # "image" or "text"
    data_shape: tuple[int, ...]  # e.g. (32,32,3)
    vocab_size: int  # e.g. 256 for uint8 pixels
    num_classes: int  # -1 for unconditional


class Task:
    """Lightweight interface for 'task-as-loss+data'."""

    spec: TaskSpec

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        raise NotImplementedError

    def loss_fn(
        self,
        *,
        rng: jax.random.PRNGKey,
        model: Any,
        params: Any,
        batch: Batch,
        train: bool,
    ) -> Tuple[Array, Metrics]:
        raise NotImplementedError

    def decode(self, x: Array) -> Array:
        """TODO: Convert model samples to uint8 images etc."""
        return x
