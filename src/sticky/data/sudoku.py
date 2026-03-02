from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np


_SUDOKU_DATA_URL = "https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW"
_VALID_SEQ_ORDERS = {"dataset", "fixed", "random"}
_SUDOKU_DRIVE_FILE_IDS = {
    "sudoku-train-data.npy": "1msLy7AXAr4VBXXv7Xfkc_IG3E7retWza",
    "sudoku-test-data.npy": "1HlKFzrkhMUAOoDU9qGUjkMD6IjlT40v7",
}


def _resolve_against_original_cwd(path_like: str) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path

    try:
        import hydra

        return Path(hydra.utils.get_original_cwd()) / path
    except Exception:
        return path.resolve()


def _resolve_data_root(data_dir: Optional[str]) -> Path:
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

    return _resolve_against_original_cwd(str(rel))


def _resolve_data_file(
    *,
    data_dir: Optional[str],
    filename: str,
) -> Path:
    file_path = Path(str(filename))
    if file_path.is_absolute():
        return file_path

    root = _resolve_data_root(data_dir)
    target = root / file_path
    if target.exists():
        return target

    # Google Drive source currently uses lowercase filenames.
    lower = target.with_name(target.name.lower())
    if lower.exists():
        return lower
    return target


@contextmanager
def _advisory_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _looks_like_binary_response(response) -> bool:
    ctype = str(response.headers.get("Content-Type", "")).lower()
    disp = str(response.headers.get("Content-Disposition", "")).lower()
    return (
        ("attachment" in disp)
        or ("application/octet-stream" in ctype)
        or ("application/x-npy" in ctype)
        or ("application/binary" in ctype)
    )


def _extract_confirm_token(response) -> Optional[str]:
    for k, v in response.cookies.items():
        if "download_warning" in str(k):
            return str(v)
    try:
        text = response.text
    except Exception:
        return None
    for pat in (
        r'name="confirm"\s+value="([0-9A-Za-z_]+)"',
        r"confirm=([0-9A-Za-z_]+)",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def _stream_response_to_file(response, out_path: Path, *, chunk_size_bytes: int):
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        expected_len = int(response.headers.get("Content-Length", "0") or 0)
        written = 0
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=max(1024, int(chunk_size_bytes))):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        if tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty file for {out_path.name}.")
        if expected_len > 0 and written != expected_len:
            raise RuntimeError(
                f"Incomplete download for {out_path.name}: got {written} bytes, "
                f"expected {expected_len}."
            )
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_google_drive_file(
    *,
    file_id: str,
    out_path: Path,
    timeout_sec: int,
    retries: int,
    chunk_size_bytes: int,
):
    try:
        import requests
    except Exception as e:
        raise ImportError(
            "requests is required for automatic Sudoku downloads."
        ) from e

    session = requests.Session()
    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            # Most reliable for large public files.
            with session.get(
                "https://drive.usercontent.google.com/download",
                params={"id": file_id, "export": "download", "confirm": "t"},
                stream=True,
                timeout=float(timeout_sec),
            ) as r:
                r.raise_for_status()
                if _looks_like_binary_response(r):
                    _stream_response_to_file(
                        r, out_path, chunk_size_bytes=int(chunk_size_bytes)
                    )
                    return
        except Exception as e:
            last_error = e

        try:
            with session.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id},
                stream=True,
                timeout=float(timeout_sec),
            ) as r0:
                r0.raise_for_status()
                if _looks_like_binary_response(r0):
                    _stream_response_to_file(
                        r0, out_path, chunk_size_bytes=int(chunk_size_bytes)
                    )
                    return

                token = _extract_confirm_token(r0)
                if not token:
                    raise RuntimeError(
                        f"Could not extract Google Drive confirm token for {out_path.name}."
                    )

            with session.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id, "confirm": token},
                stream=True,
                timeout=float(timeout_sec),
            ) as r1:
                r1.raise_for_status()
                if not _looks_like_binary_response(r1):
                    raise RuntimeError(
                        f"Google Drive returned non-binary response for {out_path.name}."
                    )
                _stream_response_to_file(
                    r1, out_path, chunk_size_bytes=int(chunk_size_bytes)
                )
                return
        except Exception as e:
            last_error = e

        sleep_s = min(30.0, 1.5 * (2**attempt))
        time.sleep(float(sleep_s))

    raise RuntimeError(
        f"Failed to download {out_path.name} from Google Drive after {attempts} attempts."
    ) from last_error


def ensure_sudoku_data_available(
    *,
    data_dir: Optional[str],
    train_file: str = "Sudoku-train-data.npy",
    test_file: str = "Sudoku-test-data.npy",
    timeout_sec: int = 120,
    retries: int = 8,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> tuple[Path, Path]:
    train_path = _resolve_data_file(data_dir=data_dir, filename=train_file)
    test_path = _resolve_data_file(data_dir=data_dir, filename=test_file)
    targets = (train_path, test_path)

    missing = [p for p in targets if not p.exists()]
    if not missing:
        return train_path, test_path

    print(
        "[sudoku-data] Missing files detected. "
        f"Downloading from {_SUDOKU_DATA_URL}",
        flush=True,
    )

    for path in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _advisory_lock(path.parent / ".sudoku_download.lock"):
            if path.exists():
                continue
            key = path.name.lower()
            file_id = _SUDOKU_DRIVE_FILE_IDS.get(key, None)
            if file_id is None:
                raise FileNotFoundError(
                    "Automatic download only supports standard Sudoku filenames. "
                    f"Could not map {path.name!r} to a file id."
                )
            print(f"[sudoku-data] Downloading {path.name} -> {path}", flush=True)
            _download_google_drive_file(
                file_id=file_id,
                out_path=path,
                timeout_sec=int(timeout_sec),
                retries=int(retries),
                chunk_size_bytes=int(chunk_size_bytes),
            )
            size_bytes = int(path.stat().st_size) if path.exists() else 0
            if size_bytes <= 0:
                raise RuntimeError(f"Downloaded file is empty: {path}")
            print(
                f"[sudoku-data] Ready {path.name} ({size_bytes / (1024**3):.2f} GiB).",
                flush=True,
            )

    return train_path, test_path


def _sort_row_col_val(triples: np.ndarray) -> np.ndarray:
    if triples.shape[0] <= 1:
        return triples
    keys = (triples[:, 2], triples[:, 1], triples[:, 0])
    return triples[np.lexsort(keys)]


def _to_inputs(
    *,
    triples: np.ndarray,
    start_index: np.ndarray,
    seq_order: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if seq_order == "dataset":
        return triples.reshape(triples.shape[0], -1)

    transformed = np.empty_like(triples)
    for i in range(triples.shape[0]):
        k = int(start_index[i])
        if (k < 0) or (k > 81):
            raise ValueError(f"Invalid Sudoku start_index={k}; expected 0..81.")

        prefix = triples[i, :k]
        suffix = triples[i, k:]

        if seq_order == "fixed":
            prefix = _sort_row_col_val(prefix)
            suffix = _sort_row_col_val(suffix)
        elif seq_order == "random":
            if prefix.shape[0] > 1:
                prefix = prefix[rng.permutation(prefix.shape[0])]
            if suffix.shape[0] > 1:
                suffix = suffix[rng.permutation(suffix.shape[0])]
        else:
            raise ValueError(
                f"Unknown seq_order={seq_order!r}. "
                f"Expected one of {sorted(_VALID_SEQ_ORDERS)}."
            )

        transformed[i, :k] = prefix
        transformed[i, k:] = suffix

    return transformed.reshape(transformed.shape[0], -1)


def _build_solution_board(triples: np.ndarray) -> np.ndarray:
    rows = triples[:, :, 0]
    cols = triples[:, :, 1]
    vals = triples[:, :, 2]

    if rows.min() < 0 or rows.max() > 8:
        raise ValueError("Sudoku rows must be in [0, 8].")
    if cols.min() < 0 or cols.max() > 8:
        raise ValueError("Sudoku cols must be in [0, 8].")
    if vals.min() < 1 or vals.max() > 9:
        raise ValueError("Sudoku values must be in [1, 9].")

    cell_id = rows * 9 + cols
    puzzle = np.zeros((triples.shape[0], 81), dtype=np.int32)
    puzzle[np.arange(triples.shape[0])[:, None], cell_id] = vals
    return puzzle


class _SudokuBatchIterator:
    def __init__(
        self,
        *,
        file_path: Path,
        batch_size: int,
        seed: int,
        shuffle: bool,
        repeat: bool,
        drop_remainder: bool,
        seq_order: str,
        mmap: bool,
        max_examples: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if seq_order not in _VALID_SEQ_ORDERS:
            raise ValueError(
                f"Unknown seq_order={seq_order!r}. "
                f"Expected one of {sorted(_VALID_SEQ_ORDERS)}."
            )
        if not file_path.exists():
            raise FileNotFoundError(
                f"Sudoku file not found: {file_path}. "
                "Download `Sudoku-train-data.npy` and `Sudoku-test-data.npy` from "
                f"{_SUDOKU_DATA_URL}."
            )

        mmap_mode = "r" if bool(mmap) else None
        table = np.load(str(file_path), mmap_mode=mmap_mode)
        if table.ndim != 2 or table.shape[1] != 325:
            raise ValueError(
                "Expected Sudoku array with shape (N, 325): "
                f"got {table.shape} from {file_path}."
            )

        n_total = int(table.shape[0])
        if max_examples > 0:
            n_total = min(n_total, int(max_examples))
        if n_total <= 0:
            raise ValueError("Sudoku dataset is empty after applying max_examples.")
        if bool(repeat) and bool(drop_remainder) and (n_total < int(batch_size)):
            raise ValueError(
                "No full batches available: repeat=true and drop_remainder=true "
                f"requires dataset size >= batch_size, got {n_total} < {batch_size}."
            )

        self._table = table
        self._n = n_total
        self._batch_size = int(batch_size)
        self._repeat = bool(repeat)
        self._drop_remainder = bool(drop_remainder)
        self._shuffle = bool(shuffle)
        self._seq_order = str(seq_order)
        self._rng = np.random.default_rng(int(seed))
        self._idx = np.arange(self._n, dtype=np.int64)
        self._cursor = 0
        if self._shuffle:
            self._rng.shuffle(self._idx)

    def __iter__(self):
        return self

    def _next_indices(self) -> np.ndarray:
        while True:
            if self._cursor >= self._n:
                if not self._repeat:
                    raise StopIteration
                self._cursor = 0
                if self._shuffle:
                    self._rng.shuffle(self._idx)

            remaining = self._n - self._cursor
            if remaining < self._batch_size and self._drop_remainder:
                if not self._repeat:
                    raise StopIteration
                self._cursor = self._n
                continue

            take = self._batch_size if remaining >= self._batch_size else remaining
            out = self._idx[self._cursor : self._cursor + take]
            self._cursor += take
            return out

    def __next__(self) -> Mapping[str, np.ndarray]:
        batch_idx = self._next_indices()
        rows = np.asarray(self._table[batch_idx], dtype=np.int32)

        start_index = rows[:, 0]
        triples = rows[:, 1:].reshape(-1, 81, 4)[:, :, :3]

        inputs = _to_inputs(
            triples=triples,
            start_index=start_index,
            seq_order=self._seq_order,
            rng=self._rng,
        ).astype(np.int32)
        puzzle = _build_solution_board(triples).astype(np.int32)

        return {
            "image": inputs,  # [B, 243]
            "puzzle": puzzle,  # [B, 81]
            "start_index": start_index.reshape(-1, 1).astype(np.int32),  # [B, 1]
        }


def make_sudoku_iterator(
    *,
    split: str,
    batch_size: int,
    seed: int = 0,
    data_dir: Optional[str] = None,
    train_file: str = "Sudoku-train-data.npy",
    test_file: str = "Sudoku-test-data.npy",
    shuffle: bool = True,
    repeat: bool = False,
    drop_remainder: bool = True,
    seq_order: str = "dataset",
    mmap: bool = True,
    max_examples: int = -1,
    auto_download: bool = True,
    download_timeout_sec: int = 120,
    download_retries: int = 8,
) -> Iterator[Mapping[str, np.ndarray]]:
    split_key = str(split).lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    if bool(auto_download):
        ensure_sudoku_data_available(
            data_dir=data_dir,
            train_file=train_file,
            test_file=test_file,
            timeout_sec=int(download_timeout_sec),
            retries=int(download_retries),
        )

    filename = train_file if split_key == "train" else test_file
    file_path = _resolve_data_file(data_dir=data_dir, filename=filename)

    return iter(
        _SudokuBatchIterator(
            file_path=file_path,
            batch_size=int(batch_size),
            seed=int(seed),
            shuffle=bool(shuffle),
            repeat=bool(repeat),
            drop_remainder=bool(drop_remainder),
            seq_order=str(seq_order),
            mmap=bool(mmap),
            max_examples=int(max_examples),
        )
    )
