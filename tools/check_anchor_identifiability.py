"""Nearest-neighbor anchor identifiability under VP noise.

Compares three anchor tables used for SJD Sudoku (vocab_size=10) by their
pairwise nearest-neighbor separability under the VP corruption kernel:

    I_A(t) = median_i min_{j != i} alpha(t) * ||a_i - a_j||_2 / sigma(t)

A value near or below 1.0 means the Gaussian noise at time t is comparable
to the distance between an anchor and its nearest neighbor, so the denoising
classifier cannot reliably distinguish them. Values well above 1.0 indicate
the anchors remain separable under corruption.

VP schedule: beta_min=0.1, beta_max=20.0, T=1.0 (matches
config/forward/beta/vp_linear.yaml).
"""

from __future__ import annotations

import jax.numpy as jnp

from sticky.models.sjd.anchors import (
    AnchorTableConfig,
    AnchorTransformConfig,
    build_anchor_table,
)
from sticky.models.sjd.sdes import alpha_sigma, make_beta


VOCAB_SIZE = 10
TIMES = (0.25, 0.50, 0.75)
BETA_MIN = 0.1
BETA_MAX = 20.0
T_HORIZON = 1.0


def _identifiability(table: jnp.ndarray, alpha: float, sigma: float) -> float:
    diffs = table[:, None, :] - table[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    dists = dists + jnp.eye(dists.shape[0]) * jnp.inf
    nearest = jnp.min(dists, axis=1)
    return float(jnp.median(nearest) * alpha / sigma)


def main() -> int:
    beta = make_beta(BETA_MIN, BETA_MAX, T=T_HORIZON)
    alpha_at = {}
    sigma_at = {}
    for t in TIMES:
        a, s = alpha_sigma(beta, jnp.asarray(t))
        alpha_at[t] = float(a)
        sigma_at[t] = float(s)

    candidates = [
        (
            "ordered_scalar",
            1,
            4.0,
            "",
            AnchorTableConfig(
                family="ordered_scalar",
                vocab_size=VOCAB_SIZE,
                anchor_dim=1,
                transform=AnchorTransformConfig(scale=4.0),
            ),
        ),
        (
            "thermometer",
            64,
            1.0,
            "equalize_row_norms",
            AnchorTableConfig(
                family="thermometer",
                vocab_size=VOCAB_SIZE,
                anchor_dim=64,
                seed=0,
                projection_seed=0,
                transform=AnchorTransformConfig(
                    scale=1.0,
                    equalize_row_norms=True,
                ),
            ),
        ),
        (
            "ordered_normal",
            64,
            1.0,
            "equalize_row_norms",
            AnchorTableConfig(
                family="ordered_normal",
                vocab_size=VOCAB_SIZE,
                anchor_dim=64,
                seed=0,
                transform=AnchorTransformConfig(
                    scale=1.0,
                    equalize_row_norms=True,
                ),
            ),
        ),
    ]

    print(
        f"VP schedule: beta_min={BETA_MIN}, beta_max={BETA_MAX}, T={T_HORIZON}"
    )
    print(f"vocab_size={VOCAB_SIZE}")
    print()
    print("alpha(t), sigma(t):")
    for t in TIMES:
        print(f"  t={t:.2f}: alpha={alpha_at[t]:.4f}  sigma={sigma_at[t]:.4f}")
    print()

    header = (
        f"{'Anchor family':<16}{'dim':>5}{'scale':>8}"
        f"{'  t=0.25':>10}{'  t=0.50':>10}{'  t=0.75':>10}  notes"
    )
    print(header)
    print("-" * len(header))
    for name, dim, scale, notes, cfg in candidates:
        table = build_anchor_table(cfg)
        scores = [
            _identifiability(table, alpha_at[t], sigma_at[t]) for t in TIMES
        ]
        print(
            f"{name:<16}{dim:>5}{scale:>8.3f}"
            f"{scores[0]:>10.3f}{scores[1]:>10.3f}{scores[2]:>10.3f}  {notes}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
