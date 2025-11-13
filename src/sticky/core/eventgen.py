# src/sticky/core/eventgen.py
from __future__ import annotations
import jax, jax.numpy as jnp
from .hazard import HazardSchedule
from .jump import GaussianJump

def sample_forward_events(
    key: jax.random.PRNGKey,
    anchors: jnp.ndarray,
    hazard: HazardSchedule,
    jump: GaussianJump,
):
    """Samples one forward unstick event per item: (t, y, has_event)."""
    B = anchors.shape[0]
    k1, k2 = jax.random.split(key)
    t = hazard.first_event_time(k1, shape=(B,))  # +inf means no event
    has = jnp.isfinite(t)
    # for CE we only care about (y,t) at actual events;
    y = jump.sample(k2, anchors)
    return t, y, has