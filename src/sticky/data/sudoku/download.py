from __future__ import annotations

import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .paths import SUDOKU_DATA_URL, SUDOKU_DRIVE_FILE_IDS, resolve_data_file


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
    train_path = resolve_data_file(data_dir=data_dir, filename=train_file)
    test_path = resolve_data_file(data_dir=data_dir, filename=test_file)
    targets = (train_path, test_path)

    missing = [p for p in targets if not p.exists()]
    if not missing:
        return train_path, test_path

    print(
        "[sudoku-data] Missing files detected. "
        f"Downloading from {SUDOKU_DATA_URL}",
        flush=True,
    )

    for path in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _advisory_lock(path.parent / ".sudoku_download.lock"):
            if path.exists():
                continue
            key = path.name.lower()
            file_id = SUDOKU_DRIVE_FILE_IDS.get(key, None)
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
