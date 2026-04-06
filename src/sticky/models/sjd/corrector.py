from __future__ import annotations

from jax import lax
import jax.numpy as jnp


Array = jnp.ndarray

_VALID_PC_GATES = frozenset({"constant_one", "entropy", "margin"})


def normalize_pc_gate_name(gate: str | None) -> str:
    key = str("constant_one" if gate is None else gate).strip().lower()
    if key not in _VALID_PC_GATES:
        valid = ", ".join(sorted(_VALID_PC_GATES))
        raise ValueError(f"Unknown pc_gate={gate!r}. Expected one of: {valid}.")
    return key


def corrector_step_size(*, dt: Array, step_scale: float) -> Array:
    dt = jnp.asarray(dt, dtype=jnp.float32)
    return jnp.asarray(float(step_scale), dtype=jnp.float32) * jnp.maximum(dt, 1.0e-6)


def pc_gate_from_probs(
    probs: Array,
    *,
    gate: str,
    eps: float = 1.0e-12,
) -> Array:
    probs = jnp.asarray(probs, dtype=jnp.float32)
    gate_key = normalize_pc_gate_name(gate)
    if gate_key == "constant_one":
        return jnp.ones(probs.shape[:-1], dtype=jnp.float32)

    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), eps)
    if gate_key == "entropy":
        entropy = -jnp.sum(
            probs * jnp.log(jnp.maximum(probs, jnp.asarray(eps, dtype=jnp.float32))),
            axis=-1,
        )
        denom = jnp.log(jnp.asarray(probs.shape[-1], dtype=jnp.float32))
        return jnp.clip(entropy / jnp.maximum(denom, eps), 0.0, 1.0)

    if probs.shape[-1] < 2:
        return jnp.ones(probs.shape[:-1], dtype=jnp.float32)

    # We only need the two largest probabilities for the margin gate, so avoid
    # a full sort over the class axis on every corrector call.
    top2_vals, _ = lax.top_k(probs, 2)
    top1 = top2_vals[..., 0]
    top2_val = top2_vals[..., 1]
    margin = jnp.clip(top1 - top2_val, 0.0, 1.0)
    return jnp.clip(1.0 - margin, 0.0, 1.0)


def masked_sum(values: Array, mask: Array) -> Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(values * mask)
