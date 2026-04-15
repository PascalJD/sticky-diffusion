# src/sticky/tasks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp

from sticky.rng import PRNGKey

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

    # True if loss_fn needs EMA teacher params threaded in from the train step.
    requires_teacher_params: ClassVar[bool] = False

    def make_dataloaders(
        self, *, seed: int
    ) -> Tuple[Iterable[Batch], Optional[Iterable[Batch]]]:
        raise NotImplementedError

    def loss_fn(
        self,
        *,
        rng: PRNGKey,
        model: Any,
        params: Any,
        batch: Batch,
        train: bool,
        teacher_params: Any = None,
    ) -> Tuple[Array, Metrics]:
        raise NotImplementedError

    def decode(self, x: Array) -> Array:
        """TODO: Convert model samples to uint8 images etc."""
        return x

    def format_samples_for_logging(self, x: Array) -> list[str] | None:
        """Optional text/task-specific rendering for generated samples."""
        return None

    def train_num_examples(self) -> int | None:
        """Optional hook for epoch-derived training schedules."""
        return None
