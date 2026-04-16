from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax
from flax.training import checkpoints

from sticky.rng import ensure_prng_key, legacy_prng_key_data


def select_checkpoint_path(ckpt_dir: Path, step: Optional[int]) -> Path:
    if step is None:
        latest = checkpoints.latest_checkpoint(str(ckpt_dir))
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}.")
        return Path(latest)

    pattern = f"checkpoint_{int(step)}"
    exact = ckpt_dir / pattern
    if exact.exists():
        return exact

    matches = sorted(ckpt_dir.glob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for step={int(step)} under {ckpt_dir}."
        )
    return matches[0]


def restore_state_from_checkpoint(state_template, ckpt_dir: Path, step: Optional[int]):
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    selected_path = select_checkpoint_path(ckpt_dir, step)
    try:
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=state_template,
            step=step,
        )
    except Exception:
        legacy_target = state_template.replace(
            rng=legacy_prng_key_data(state_template.rng)
        )
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=legacy_target,
            step=step,
        )
    restored = restored.replace(rng=ensure_prng_key(restored.rng))
    restored_step = int(jax.device_get(restored.step))
    if (step is not None) and (restored_step != int(step)):
        raise RuntimeError(
            f"Requested checkpoint step={int(step)}, but restored step={restored_step}."
        )
    return restored, restored_step, selected_path
