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

    @property
    def blur_kernel(self) -> Optional[Any]:
        """Cross-position blending W. Stored on the jump kernel; surfaced here
        so callers can read it from the schedule without reaching into jump.
        """
        return getattr(self.jump, "blur_kernel", None)

    @property
    def blur_space(self) -> str:
        """Where the blur kernel acts: "embedding" (mu = W E(X0)) or "value"
        (mu = E(B v)). Stored on the jump; "embedding" when absent. Inert
        when blur_kernel is None.
        """
        return str(getattr(self.jump, "blur_space", "embedding"))

    def with_blur(
        self, blur_kernel: Any, blur_space: str = "embedding"
    ) -> "ForwardSchedule":
        """Return a copy with the blur kernel (and the space it acts in)
        attached to the jump."""
        new_jump = dataclasses.replace(
            self.jump, blur_kernel=blur_kernel, blur_space=str(blur_space)
        )
        return dataclasses.replace(self, jump=new_jump)
