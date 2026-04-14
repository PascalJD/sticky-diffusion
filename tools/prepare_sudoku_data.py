from __future__ import annotations

import argparse

from sticky.data.sudoku import ensure_sudoku_data_available


def main():
    parser = argparse.ArgumentParser(
        description="Download Sudoku train/test .npy files if missing."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sudoku",
        help="Target data directory.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="Sudoku-train-data.npy",
        help="Train filename (or absolute path).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="Sudoku-test-data.npy",
        help="Test filename (or absolute path).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Number of download retry rounds.",
    )
    args = parser.parse_args()

    train_path, test_path = ensure_sudoku_data_available(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        timeout_sec=int(args.timeout_sec),
        retries=int(args.retries),
    )
    print(f"[sudoku-data] train: {train_path}")
    print(f"[sudoku-data] test:  {test_path}")


if __name__ == "__main__":
    main()
