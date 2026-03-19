#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _collect_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(results_root.rglob("offline_eval_metrics.json")):
        payload = _load_payload(metrics_path)
        experiment_cfg = payload.get("experiment_config", {})
        evaluation_cfg = payload.get("evaluation", {})
        checkpoint_cfg = payload.get("checkpoint", {})
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "temperature": experiment_cfg.get(
                    "sampler_logit_temperature",
                    evaluation_cfg.get("requested_logit_temperature_override"),
                ),
                "jump_eta": experiment_cfg.get("jump_eta"),
                "fid": metrics.get("eval/fid"),
                "checkpoint_step": checkpoint_cfg.get("restored_step"),
                "checkpoint_path": checkpoint_cfg.get("checkpoint_path"),
                "metrics_path": str(metrics_path),
            }
        )
    rows.sort(
        key=lambda row: (
            float("inf") if row["fid"] is None else float(row["fid"]),
            float("inf") if row["temperature"] is None else float(row["temperature"]),
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect and rank SJD temperature sweep offline-eval reports.",
    )
    parser.add_argument("results_root", type=Path, help="Sweep directory to scan.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    rows = _collect_rows(args.results_root)
    if not rows:
        print(f"No offline_eval_metrics.json files found under {args.results_root}", file=sys.stderr)
        return 1

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.csv, rows)

    print("temperature,jump_eta,fid,checkpoint_step,metrics_path")
    for row in rows:
        print(
            ",".join(
                [
                    str(row["temperature"]),
                    str(row["jump_eta"]),
                    str(row["fid"]),
                    str(row["checkpoint_step"]),
                    str(row["metrics_path"]),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
