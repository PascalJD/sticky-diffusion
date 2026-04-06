from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from sticky.training.persistence import now_utc_iso, resolve_run_path


SUDOKU_SJD_PROGRESS_COLUMNS = [
    "timestamp_utc",
    "step",
    "label",
    "kind",
    "policy",
    "eta",
    "predictor_corrector",
    "gate_type",
    "corrector_substeps",
    "corrector_strength",
    "n_steps",
    "solve_rate",
    "cell_acc_unknown",
    "board_acc_exact",
    "row_valid_fraction",
    "col_valid_fraction",
    "box_valid_fraction",
    "clue_consistency_fraction",
    "full_cell_acc",
    "avg_commits_per_step",
    "mean_masked_unknown_positions_per_step",
    "mean_selected_top_probability",
    "mean_selected_top_prob_margin",
    "final_masked_unknown_fraction",
    "sampling_nfe_total",
    "sampling_langevin_updates_total",
    "sampling_langevin_updates_per_step",
    "sampling_gate_mean_continuous",
    "sampling_gate_mean_committed",
    "sampling_frac_committed_pre_force",
    "sampling_frac_committed_final",
    "sampling_fill_frac_by_final_jump",
    "sampling_wallclock_sec_per_board",
]


def _normalize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _normalize_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        column: _normalize_csv_value(row.get(column, None))
        for column in SUDOKU_SJD_PROGRESS_COLUMNS
    }


def _read_existing_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in reader:
            key = (str(row.get("step", "")), str(row.get("label", "")))
            if key == ("", ""):
                continue
            rows[key] = {
                column: row.get(column, "")
                for column in SUDOKU_SJD_PROGRESS_COLUMNS
            }
        return rows


def _atomic_write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=SUDOKU_SJD_PROGRESS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_csv_row(row))
    os.replace(temp_path, path)


def write_sudoku_sjd_progress_artifacts(
    *,
    rows: list[Mapping[str, Any]],
    step_i: int,
    progress_path_like: str | None,
    latest_path_like: str | None,
    metrics_dir_rel: str,
    base_dir: Path,
    write_progress: bool,
    write_latest: bool,
) -> None:
    if not rows or (not write_progress and not write_latest):
        return

    progress_path = resolve_run_path(
        progress_path_like,
        f"{metrics_dir_rel}/sudoku_sjd_progress.csv",
        base_dir=base_dir,
    )
    latest_path = resolve_run_path(
        latest_path_like,
        f"{metrics_dir_rel}/sudoku_sjd_latest.csv",
        base_dir=base_dir,
    )

    timestamp = now_utc_iso()
    keyed_rows = _read_existing_rows(progress_path if write_progress else latest_path)
    for raw_row in rows:
        row = dict(raw_row)
        row["timestamp_utc"] = timestamp
        row["step"] = int(step_i)
        key = (str(int(step_i)), str(row.get("label", "")))
        keyed_rows[key] = _normalize_csv_row(row)

    ordered_rows = sorted(
        keyed_rows.values(),
        key=lambda row: (int(str(row.get("step", 0) or 0)), str(row.get("label", ""))),
    )
    if write_progress:
        _atomic_write_rows(progress_path, ordered_rows)

    if write_latest:
        latest_step = max(int(str(row.get("step", 0) or 0)) for row in ordered_rows)
        latest_rows = [
            row for row in ordered_rows if int(str(row.get("step", 0) or 0)) == latest_step
        ]
        _atomic_write_rows(latest_path, latest_rows)
