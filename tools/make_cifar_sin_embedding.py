"""Write the frozen CIFAR-10 sinusoidal pixel embedding to data/anchors/cifar_sin8.npz.

The table is bit-identical to analysis/field_diag/common.image_embedding('sinusoidal')
(the d=8 form characterized by the offline diagnostics): sin/cos at freqs {1,2,3,4}
of v/255, no linear term. Saved under key 'wte' with shape (256, 8) so it loads via
the repo's pretrained-anchor path (family=pretrained, learnable=false).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis" / "field_diag"))
import common as C  # noqa: E402


def main() -> None:
    E, d = C.image_embedding("sinusoidal", V=256)
    E = np.asarray(E, dtype=np.float32)
    assert E.shape == (256, 8), E.shape
    out = REPO / "data" / "anchors" / "cifar_sin8.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, wte=E)
    rn = np.linalg.norm(E, axis=1)
    print(f"saved {out}  shape={E.shape}  dtype={E.dtype}  "
          f"row_norm[min/mean/max]={rn.min():.6f}/{rn.mean():.6f}/{rn.max():.6f}")


if __name__ == "__main__":
    main()
