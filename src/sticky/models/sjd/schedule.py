"""ForwardSchedule: paper-first-class bundling of the SJD forward design axes.

The paper (Section 2) describes the SJD forward dynamics by three quantities:
  - beta(t):      the VP noise schedule (continuous-time SDE coefficient)
  - hazard:       the per-anchor unsticking hazard schedule lambda_t
  - jump:         the un-sticking kernel r_a (jump distribution at unstick time)
  - blur_kernel:  optional cross-position blending W (Section 4.4)

This dataclass keeps all four together so callers can swap them via config
without juggling four separate Optional[Any] parameters.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class ForwardSchedule:
    beta: Callable[[Array], Array]
    hazard: Any                    # HazardSchedule protocol (typed as Any for now)
    jump: Any                      # JumpDistribution protocol (typed as Any for now)
    T: float = 1.0
    blur_kernel: Optional[Any] = None

    def with_blur(self, blur_kernel: Any) -> "ForwardSchedule":
        """Return a copy with the blur kernel attached to the jump.

        Mirrors the prior factory hack of dataclasses.replace(jump, blur_kernel=kernel)
        but keeps it on the schedule type rather than inline in the factory.
        """
        new_jump = dataclasses.replace(self.jump, blur_kernel=blur_kernel)
        return dataclasses.replace(self, jump=new_jump, blur_kernel=blur_kernel)
