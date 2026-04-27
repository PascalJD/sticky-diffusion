"""Boot-side helpers for the frequency-weighted SJD hazard."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from sticky.core.paths import resolve_against_original_cwd

Array = jnp.ndarray


def load_anchor_log_w(cfg_hazard_weighting: Any, vocab_size: int) -> Optional[Array]:
    """Resolve a cfg.forward.hazard_weighting block to a (vocab_size,) log_w array.

    Returns ``None`` if frequency weighting is disabled (the bit-exact baseline
    path), otherwise a jnp.float32 array of shape (vocab_size,) containing
    log w(a) ready to attach to ``AnchorTable.log_w`` and forward to
    ``ce_allocation_loss(anchor_log_w=...)``.
    """
    if cfg_hazard_weighting is None:
        return None
    enabled = bool(getattr(cfg_hazard_weighting, "enabled", False))
    if not enabled:
        return None
    log_w_path = getattr(cfg_hazard_weighting, "log_w_path", None)
    if log_w_path in (None, "", "null"):
        raise ValueError(
            "forward.hazard_weighting.enabled=true but log_w_path is empty. "
            "Run tools/compute_anchor_frequencies.py first or set "
            "log_w_path to point at the resulting .npz."
        )
    path = resolve_against_original_cwd(str(log_w_path))
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"log_w npz not found: {p}")
    with np.load(p, allow_pickle=False) as f:
        if "log_w" not in f.files:
            raise KeyError(
                f"{p}: expected key 'log_w', found {list(f.files)}. "
                "Regenerate via tools/compute_anchor_frequencies.py."
            )
        log_w_np = np.asarray(f["log_w"], dtype=np.float32)
    if log_w_np.ndim != 1:
        raise ValueError(f"log_w must be 1-D, got shape {log_w_np.shape}")
    if int(log_w_np.shape[0]) != int(vocab_size):
        raise ValueError(
            f"log_w shape {log_w_np.shape} does not match vocab_size={vocab_size}."
        )
    return jnp.asarray(log_w_np, dtype=jnp.float32)
