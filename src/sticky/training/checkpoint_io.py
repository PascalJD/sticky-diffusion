from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import jax
from flax.training import checkpoints

from sticky.rng import ensure_prng_key, legacy_prng_key_data

try:
    import orbax.checkpoint as ocp
except Exception:
    ocp = None


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

    # Allow a portable pickle checkpoint artifact produced on another machine.
    if ckpt_dir.is_file() and ckpt_dir.suffix.lower() in {".pkl", ".pickle"}:
        with ckpt_dir.open("rb") as f:
            payload = pickle.load(f)

        if hasattr(payload, "replace"):
            restored = payload
        else:
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Pickle checkpoint must contain a dict or TrainState-like object; got {type(payload)!r}."
                )

            # Direct reconstruction: keep template opt_state/rng structure, swap
            # only fields required for eval-time sampling from pretrained params.
            restored = state_template.replace(
                params=payload["params"],
                ema_params=payload.get("ema_params", state_template.ema_params),
                step=int(payload.get("step", 0)),
            )

        restored = restored.replace(rng=ensure_prng_key(restored.rng))
        restored_step = int(jax.device_get(restored.step))
        if (step is not None) and (restored_step != int(step)):
            raise RuntimeError(
                f"Requested checkpoint step={int(step)}, but restored step={restored_step}."
            )
        return restored, restored_step, ckpt_dir

    selected_path = select_checkpoint_path(ckpt_dir, step)

    def _restore(restore_dir: Path, restore_step: Optional[int], target):
        # First try the default Flax/Orbax restore path.
        try:
            return checkpoints.restore_checkpoint(
                ckpt_dir=str(restore_dir),
                target=target,
                step=restore_step,
            )
        except Exception:
            # Compatibility fallback for OCDBT checkpoints that encode structure
            # in `_METADATA` rather than the legacy aggregate filename.
            if ocp is None:
                raise
            checkpointer = ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(aggregate_filename="_METADATA")
            )
            return checkpoints.restore_checkpoint(
                ckpt_dir=str(restore_dir),
                target=target,
                step=restore_step,
                orbax_checkpointer=checkpointer,
            )

    # Prefer restoring directly from the resolved checkpoint path first.
    # This is robust for Orbax single-checkpoint bundle directories.
    try:
        restored = _restore(selected_path, None, state_template)
    except Exception:
        legacy_target = state_template.replace(
            rng=legacy_prng_key_data(state_template.rng)
        )
        try:
            restored = _restore(selected_path, None, legacy_target)
        except Exception:
            try:
                restored = _restore(ckpt_dir, step, state_template)
            except Exception:
                restored = _restore(ckpt_dir, step, legacy_target)
    restored = restored.replace(rng=ensure_prng_key(restored.rng))
    restored_step = int(jax.device_get(restored.step))
    if (step is not None) and (restored_step != int(step)):
        raise RuntimeError(
            f"Requested checkpoint step={int(step)}, but restored step={restored_step}."
        )
    return restored, restored_step, selected_path
