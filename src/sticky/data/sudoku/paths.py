from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sticky.core.paths import resolve_against_original_cwd


SUDOKU_DATA_URL = "https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW"
SUDOKU_DRIVE_FILE_IDS = {
    "sudoku-train-data.npy": "1msLy7AXAr4VBXXv7Xfkc_IG3E7retWza",
    "sudoku-test-data.npy": "1HlKFzrkhMUAOoDU9qGUjkMD6IjlT40v7",
}


def resolve_data_root(data_dir: Optional[str]) -> Path:
    if data_dir in (None, "", "null", "None"):
        rel = Path("data/sudoku")
    else:
        rel = Path(str(data_dir))

    if rel.is_absolute():
        return rel

    # On HPC, default Sudoku data goes to SCRATCH automatically.
    if rel.as_posix().rstrip("/") == "data/sudoku":
        scratch = os.environ.get("SCRATCH", "").strip()
        if scratch:
            return Path(scratch) / "sticky-diffusion" / "data" / "sudoku"

    return resolve_against_original_cwd(str(rel))


def resolve_data_file(
    *,
    data_dir: Optional[str],
    filename: str,
) -> Path:
    file_path = Path(str(filename))
    if file_path.is_absolute():
        return file_path

    root = resolve_data_root(data_dir)
    target = root / file_path
    if target.exists():
        return target

    # Google Drive source currently uses lowercase filenames.
    lower = target.with_name(target.name.lower())
    if lower.exists():
        return lower
    return target
