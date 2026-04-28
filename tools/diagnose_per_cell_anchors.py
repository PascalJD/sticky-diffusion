"""Diagnostic for per-cell Sudoku anchors.

Loads (or freshly inits) an SJD model with a per-position anchor table of
shape (81, 9, d), then reports four diagnostics that test whether the
geometry has self-organized into Sudoku's row/column/box constraint
structure:

  1. Per-position row norms, summarized as min/median/max — a sanity check
     against under- or over-trained anchors.
  2. Cell-cell cosine matrix (81 x 81) per digit v: cos(a_{v,p}, a_{v,p'}).
     Saved as a heatmap PNG per digit.
  3. Mean intra-cell digit separation: mean of ||a_{v,p} - a_{v',p}|| over
     v != v'. Distribution across positions is reported, and any position
     with separation < 1e-3 is flagged as collapsed.
  4. Same-unit-vs-other gap: per digit, mean cosine between cells sharing
     a row/column/box vs unrelated cells. The central question of the
     experiment.

Usage:
    PYTHONPATH=src python tools/diagnose_per_cell_anchors.py \\
        --from-init experiment=sudoku/sjd_sudoku_per_cell \\
        --out diagnostics/per_cell_init/

    PYTHONPATH=src python tools/diagnose_per_cell_anchors.py \\
        --checkpoint <path/to/orbax-checkpoint-dir> \\
        --out diagnostics/per_cell_trained/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


_BOARD_SIZE = 9
_NUM_CELLS = _BOARD_SIZE * _BOARD_SIZE


def _row(p: int) -> int:
    return p // _BOARD_SIZE


def _col(p: int) -> int:
    return p % _BOARD_SIZE


def _box(p: int) -> int:
    return (_row(p) // 3) * 3 + (_col(p) // 3)


def _shares_unit(p: int, q: int) -> bool:
    return _row(p) == _row(q) or _col(p) == _col(q) or _box(p) == _box(q)


def _shares_unit_mask() -> np.ndarray:
    """Boolean (81, 81) matrix; True iff p != q and p, q share row/col/box."""
    m = np.zeros((_NUM_CELLS, _NUM_CELLS), dtype=bool)
    for p in range(_NUM_CELLS):
        for q in range(_NUM_CELLS):
            if p != q and _shares_unit(p, q):
                m[p, q] = True
    return m


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """vectors: (P, d) -> (P, P) cosine similarities."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-12)
    return unit @ unit.T


def _save_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Degrade gracefully when matplotlib isn't available.
        np.save(str(path.with_suffix(".npy")), matrix)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("cell index p'")
    ax.set_ylabel("cell index p")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _summarize(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _load_table_from_init(experiment_override: str) -> np.ndarray:
    """Compose the requested experiment, init the model, and return the
    rank-3 anchor table as a numpy array of shape (P, V, d)."""
    from hydra import compose, initialize_config_dir

    import jax
    import jax.numpy as jnp

    from sticky.core.config_paths import config_root
    from sticky.models.factory import build_model
    from sticky.tasks.factory import build_task

    overrides = [experiment_override, "eval=sudoku_sjd"]
    with initialize_config_dir(version_base=None, config_dir=str(config_root())):
        cfg = compose(config_name="config.yaml", overrides=overrides)

    task = build_task(cfg.experiment)
    model = build_model(
        cfg.experiment,
        data_shape=task.spec.data_shape,
        vocab_size=task.spec.vocab_size,
    )
    rng = jax.random.PRNGKey(0)
    P = int(task.spec.data_shape[0])
    d = int(cfg.experiment.model.anchor.dim)
    y_t = jnp.zeros((1, P, d), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    token_ids = jnp.zeros((1, P), dtype=jnp.int32)
    variables = model.init(
        {"params": rng, "dropout": rng},
        y_t,
        t,
        anchor_token_ids=token_ids,
    )
    table = np.asarray(variables["params"]["anchors"]["table"], dtype=np.float32)
    return table


def _load_table_from_checkpoint(path: Path) -> np.ndarray:
    """Load the anchor table from an Orbax checkpoint directory."""
    try:
        import orbax.checkpoint as ocp
    except ImportError as e:
        raise RuntimeError(
            "orbax.checkpoint is required to load checkpoints; "
            "install it or use --from-init for sanity checks."
        ) from e

    ckpt = ocp.PyTreeCheckpointer()
    state = ckpt.restore(str(path))
    # Walk a couple of common nesting conventions; prefer explicit failure
    # over silently returning a wrong tensor.
    candidates = []
    if isinstance(state, dict):
        candidates.append(state)
        if "params" in state and isinstance(state["params"], dict):
            candidates.append(state["params"])
        if "model" in state and isinstance(state["model"], dict):
            candidates.append(state["model"])
            inner = state["model"]
            if isinstance(inner.get("params"), dict):
                candidates.append(inner["params"])
    for candidate in candidates:
        if "anchors" in candidate and "table" in candidate["anchors"]:
            return np.asarray(candidate["anchors"]["table"], dtype=np.float32)
    raise KeyError(
        f"Could not locate params/anchors/table in checkpoint at {path}. "
        f"Available top-level keys: {list(state.keys()) if isinstance(state, dict) else type(state).__name__}"
    )


def diagnose(table: np.ndarray, out_dir: Path) -> dict:
    if table.ndim != 3:
        raise ValueError(
            f"diagnose expects a per-position table (P, V, d), got shape={table.shape}."
        )
    P, V, d = table.shape
    if P != _NUM_CELLS:
        print(
            f"[warn] P={P} != 81; constraint-structure diagnostics assume a "
            "9x9 board.",
            file=sys.stderr,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"shape": list(table.shape)}

    # 1. Per-position row norms.
    norms = np.linalg.norm(table, axis=-1)  # (P, V)
    summary["row_norms"] = _summarize(norms.flatten())

    # 2. Cell-cell cosine matrix per digit + heatmap.
    summary["cell_cell_cosine_per_digit"] = []
    for v in range(V):
        cos = _cosine_matrix(table[:, v, :])  # (P, P)
        # Off-diagonal stats only.
        off_diag = cos[~np.eye(P, dtype=bool)]
        summary["cell_cell_cosine_per_digit"].append(
            {"digit": v, **_summarize(off_diag)}
        )
        _save_heatmap(
            cos,
            out_dir / f"cell_cosine_digit_{v}.png",
            title=f"cos(a_{{v={v},p}}, a_{{v={v},p'}})",
        )

    # 3. Mean intra-cell digit separation per position.
    intra: list[float] = []
    for p in range(P):
        block = table[p]  # (V, d)
        diffs = block[:, None, :] - block[None, :, :]  # (V, V, d)
        dists = np.linalg.norm(diffs, axis=-1)  # (V, V)
        upper = dists[np.triu_indices(V, k=1)]
        intra.append(float(np.mean(upper)))
    intra_arr = np.asarray(intra)
    summary["intra_cell_digit_separation"] = _summarize(intra_arr)
    summary["collapsed_positions"] = [
        int(p) for p in np.where(intra_arr < 1e-3)[0]
    ]

    # 4. Same-unit-vs-other cosine gap per digit.
    if P == _NUM_CELLS:
        share_mask = _shares_unit_mask()
        unrelated_mask = ~share_mask
        np.fill_diagonal(unrelated_mask, False)
        per_digit_gap = []
        for v in range(V):
            cos = _cosine_matrix(table[:, v, :])
            same_mean = float(cos[share_mask].mean())
            other_mean = float(cos[unrelated_mask].mean())
            per_digit_gap.append(
                {
                    "digit": v,
                    "same_unit_mean": same_mean,
                    "unrelated_mean": other_mean,
                    "gap": same_mean - other_mean,
                }
            )
        summary["same_unit_vs_unrelated"] = per_digit_gap

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to an Orbax checkpoint directory.",
    )
    src.add_argument(
        "--from-init",
        type=str,
        metavar="EXPERIMENT_OVERRIDE",
        help=(
            "Diagnose a freshly-initialized model (sanity check). "
            "Pass a Hydra experiment override, e.g. "
            "'experiment=sudoku/sjd_sudoku_per_cell'."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("diagnostics/per_cell_anchors"),
        help="Output directory (created if missing).",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        table = _load_table_from_checkpoint(args.checkpoint)
    else:
        table = _load_table_from_init(args.from_init)

    summary = diagnose(table, args.out)

    print(f"[shape] anchor table: {tuple(summary['shape'])}")
    rn = summary["row_norms"]
    print(
        f"[norms] row norm min={rn['min']:.4f} median={rn['median']:.4f} "
        f"max={rn['max']:.4f}"
    )
    sep = summary["intra_cell_digit_separation"]
    print(
        f"[separation] intra-cell digit separation min={sep['min']:.4f} "
        f"median={sep['median']:.4f}"
    )
    collapsed = summary.get("collapsed_positions", [])
    if collapsed:
        print(
            f"[collapse] {len(collapsed)} positions show collapsed digits: "
            f"{collapsed}",
            file=sys.stderr,
        )
        return 1
    if "same_unit_vs_unrelated" in summary:
        gaps = [d["gap"] for d in summary["same_unit_vs_unrelated"]]
        print(
            f"[constraint] same-unit-minus-unrelated cosine gap per digit: "
            f"min={min(gaps):.4f} median={float(np.median(gaps)):.4f} "
            f"max={max(gaps):.4f}"
        )
    print(f"[ok] wrote {args.out}/summary.json + heatmaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
