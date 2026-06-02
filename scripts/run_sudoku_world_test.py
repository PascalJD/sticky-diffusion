"""Sudoku World-1-vs-World-2 hazard-learning test runner.

Drives FIXED / ELBO / CE training conditions on Sudoku in sequence, parses
each run's final_metrics.json, and reports solve-rate deltas + the
plan's verdict (HELPS / HURTS / NEUTRAL).

The three experiments differ only in the hazard-learning gate and the
score-term gradient gate (see plan
/Users/PascalJutras/.claude/plans/we-have-an-end-to-end-compressed-boole.md):
    FIXED : config/experiment/sudoku/sjd_sudoku_e2e_baseline.yaml
    ELBO  : config/experiment/sudoku/sjd_sudoku_e2e.yaml
    CE    : config/experiment/sudoku/sjd_sudoku_e2e_ce.yaml

USAGE (M2 smoke; ~30-60 min per condition):
    PYTHONPATH=src python scripts/run_sudoku_world_test.py

USAGE (longer M2 run, if smoke validates):
    PYTHONPATH=src python scripts/run_sudoku_world_test.py \\
        --steps 20000 --batch 64 --seeds 0,1 --eval_num_batches 8

The runner deliberately disables wandb and pins Hydra's output directory
per (condition, seed) so we can find final_metrics.json reliably.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Three experiments under config/experiment/sudoku/.
CONDITIONS = {
    "fixed": "sudoku/sjd_sudoku_e2e_baseline",
    "elbo":  "sudoku/sjd_sudoku_e2e",
    "ce":    "sudoku/sjd_sudoku_e2e_ce",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"


def run_one(*, condition: str, experiment: str, seed: int, steps: int,
             batch: int, eval_num_batches: int, predictor_nfe: int,
             out_dir: Path) -> dict:
    """Launch one (condition, seed) training run, return parsed metrics."""
    run_dir = out_dir / f"{condition}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    overrides = [
        f"experiment={experiment}",
        # Both seed slots — root is a placeholder, experiment.training.seed
        # is what the loop actually reads (per config.yaml:8 comment).
        f"seed={seed}",
        f"experiment.training.seed={seed}",
        # Compute-scale overrides for M2 smoke.
        # `training.max_steps` lives at ROOT, not under experiment.training
        # (sudoku_sjd.yaml interpolates `num_train_steps: ${training.max_steps}`).
        f"training.max_steps={steps}",
        f"experiment.dataset.batch_size={batch}",
        f"experiment.training.eval_every_steps={steps}",        # one eval at the end
        f"experiment.training.checkpoint_every_steps={steps}",
        "experiment.training.log_every_steps=100",
        # Eval: shrink the per-eval sample count so M2 finishes in minutes.
        f"eval.sudoku_num_batches={eval_num_batches}",
        # Only run the primary sampler (predictor_only) — that's what
        # best_checkpoint_metric uses. The other two samplers add cost
        # without adding signal for this test.
        "eval.sudoku_run_all_sampler_modes=false",
        # Override the predictor's NFE budget. Cheaper grids for smoke.
        f"experiment.sampler.n_steps={predictor_nfe}",
        # Pin output dir so we can find final_metrics.json.
        f"hydra.run.dir={run_dir}",
        # Disable wandb entirely.
        "wandb.enabled=false",
        "wandb.mode=disabled",
    ]
    cmd = [
        sys.executable, "-m", "sticky.cli.train",
        *overrides,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH) + os.pathsep + env.get("PYTHONPATH", "")
    # On M2 CPU, avoid GPU jax (TF GPUs are already disabled by train.py).
    env.setdefault("JAX_PLATFORM_NAME", "cpu")

    print(
        f"\n=== {condition.upper()} seed={seed}: launching "
        f"({steps} steps, batch {batch}, eval_nb={eval_num_batches}, "
        f"NFE={predictor_nfe}) ===", flush=True,
    )
    print(f"  log: {log_path}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w") as logf:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0
    print(
        f"  -> exit code {result.returncode}, {elapsed:.1f}s "
        f"({elapsed/60:.1f} min)", flush=True,
    )

    # Parse final_metrics.json. MetricsWriter places it under <run_dir>/metrics/.
    final_path = run_dir / "metrics" / "final_metrics.json"
    history_path = run_dir / "metrics" / "metrics.jsonl"
    parsed: dict = {
        "condition": condition,
        "seed": seed,
        "exit_code": int(result.returncode),
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
    }
    if not final_path.exists():
        print(f"  !! final_metrics.json missing — check {log_path}", flush=True)
        parsed["error"] = "final_metrics.json missing"
        return parsed
    with final_path.open() as f:
        final = json.load(f)
    m = final.get("metrics", {})
    # Pull the headline solve_rate and the log_w stats we care about.
    parsed["final_step"] = int(final.get("step", 0))
    parsed["solve_rate"] = m.get("eval/predictor_only/solve_rate")
    parsed["board_acc"] = m.get("eval/predictor_only/board_acc")
    parsed["full_cell_acc"] = m.get("eval/predictor_only/full_cell_acc")
    parsed["loss_ce"] = m.get("loss/ce") or m.get("train/loss/ce")
    parsed["loss_rb"] = m.get("loss/rb") or m.get("train/loss/rb")
    parsed["loss_score"] = m.get("loss/score") or m.get("train/loss/score")
    parsed["log_w_max"] = m.get("log_w/max")
    parsed["log_w_min"] = m.get("log_w/min")
    parsed["log_w_mean"] = m.get("log_w/mean")
    parsed["log_w_std"] = m.get("log_w/std")
    parsed["log_w_range"] = m.get("log_w/range")
    parsed["w_max"] = m.get("w/max")
    parsed["w_min"] = m.get("w/min")
    return parsed


def aggregate_summary(rows: list[dict], out_dir: Path) -> dict:
    """Compute per-condition means/stds and a verdict line."""
    import statistics

    def by_cond(cond: str, key: str) -> list[float]:
        vals = [r.get(key) for r in rows if r.get("condition") == cond
                 and r.get(key) is not None]
        return [float(v) for v in vals]

    def mean_std(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    sr_fixed_mu, sr_fixed_sd = mean_std(by_cond("fixed", "solve_rate"))
    sr_elbo_mu, sr_elbo_sd = mean_std(by_cond("elbo", "solve_rate"))
    sr_ce_mu, sr_ce_sd = mean_std(by_cond("ce", "solve_rate"))

    delta_ce = sr_ce_mu - sr_fixed_mu
    delta_elbo = sr_elbo_mu - sr_fixed_mu
    # 2-sigma envelope on the delta, conservative (sum, not RSS, of stds).
    delta_ce_envelope = 2.0 * (sr_fixed_sd + sr_ce_sd)
    delta_elbo_envelope = 2.0 * (sr_fixed_sd + sr_elbo_sd)

    if abs(delta_ce) <= delta_ce_envelope:
        verdict = "NEUTRAL"
    elif delta_ce > 0:
        verdict = "HELPS"
    else:
        verdict = "HURTS"

    # The training loop emits log_w/{mean, std, range} and w/{max, min}
    # but NOT log_w/{max, min}. Use w/max - w/min as a proxy for the log_w
    # range, OR the log_w/range metric directly.
    elbo_range = mean_std(by_cond("elbo", "log_w_range"))[0]
    ce_range = mean_std(by_cond("ce", "log_w_range"))[0]
    elbo_w_max = mean_std(by_cond("elbo", "w_max"))[0]
    elbo_w_min = mean_std(by_cond("elbo", "w_min"))[0]
    ce_w_max = mean_std(by_cond("ce", "w_max"))[0]
    ce_w_min = mean_std(by_cond("ce", "w_min"))[0]

    summary = {
        "verdict": verdict,
        "solve_rate": {
            "fixed": {"mean": sr_fixed_mu, "std": sr_fixed_sd},
            "elbo":  {"mean": sr_elbo_mu, "std": sr_elbo_sd},
            "ce":    {"mean": sr_ce_mu, "std": sr_ce_sd},
        },
        "delta_ce_vs_fixed": delta_ce,
        "delta_ce_envelope_2sigma": delta_ce_envelope,
        "delta_elbo_vs_fixed": delta_elbo,
        "delta_elbo_envelope_2sigma": delta_elbo_envelope,
        "log_w_range_mean": {
            "elbo": elbo_range,
            "ce":   ce_range,
        },
        "w_range_mean": {
            "elbo": {"max": elbo_w_max, "min": elbo_w_min},
            "ce":   {"max": ce_w_max,   "min": ce_w_min},
        },
        "consistency_checks": {
            "elbo_close_to_fixed_solve_rate": (
                abs(delta_elbo) <= delta_elbo_envelope
            ),
            "ce_log_w_moved": (
                ce_range is not None and ce_range > 0.1
            ),
            "elbo_log_w_stuck": (
                elbo_range is not None and elbo_range < 0.1
            ),
            "ce_moved_more_than_elbo": (
                ce_range is not None and elbo_range is not None
                and ce_range > elbo_range
            ),
        },
        "thresholds": {
            "verdict": "delta_ce > 2 * (std_fixed + std_ce)",
            "log_w_moved": "max|log_w_recentered| > 0.1",
        },
        "raw_rows": rows,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def write_csvs(rows: list[dict], out_dir: Path):
    # solve_rates.csv: condition, seed, exit_code, solve_rate, board_acc,
    # full_cell_acc, elapsed_sec, final_step
    with (out_dir / "solve_rates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "seed", "exit_code", "final_step",
            "solve_rate", "board_acc", "full_cell_acc",
            "elapsed_sec",
        ])
        for r in rows:
            w.writerow([
                r.get("condition"), r.get("seed"), r.get("exit_code"),
                r.get("final_step"), r.get("solve_rate"),
                r.get("board_acc"), r.get("full_cell_acc"),
                f"{r.get('elapsed_sec', 0.0):.1f}",
            ])
    # log_w_final.csv: condition, seed, w/log_w stats, loss components
    with (out_dir / "log_w_final.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "seed",
            "log_w_mean", "log_w_std", "log_w_max", "log_w_min", "log_w_range",
            "w_max", "w_min",
            "loss_ce", "loss_rb", "loss_score",
        ])
        for r in rows:
            w.writerow([
                r.get("condition"), r.get("seed"),
                r.get("log_w_mean"), r.get("log_w_std"),
                r.get("log_w_max"), r.get("log_w_min"), r.get("log_w_range"),
                r.get("w_max"), r.get("w_min"),
                r.get("loss_ce"), r.get("loss_rb"), r.get("loss_score"),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--seeds", type=str, default="0",
        help="comma-separated list of seeds.",
    )
    parser.add_argument(
        "--eval_num_batches", type=int, default=2,
        help="sudoku_num_batches override. Eval samples = batches * "
             "eval_batch_size (128). Smoke default = 2 (256 puzzles).",
    )
    parser.add_argument(
        "--predictor_nfe", type=int, default=64,
        help="sampler.n_steps override for the predictor_only path.",
    )
    parser.add_argument(
        "--conditions", type=str, default="fixed,elbo,ce",
        help="subset of conditions to run.",
    )
    parser.add_argument(
        "--out_root", type=str, default="runs",
        help="parent directory for the timestamped run dir.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c!r}; pick from {list(CONDITIONS)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"sudoku_world_test_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    print(
        f"conditions: {conditions}; seeds: {seeds}; "
        f"steps/run: {args.steps}; batch: {args.batch}; "
        f"eval_num_batches: {args.eval_num_batches}; "
        f"predictor_nfe: {args.predictor_nfe}",
        flush=True,
    )

    rows: list[dict] = []
    t_outer = time.perf_counter()
    for cond in conditions:
        for seed in seeds:
            row = run_one(
                condition=cond,
                experiment=CONDITIONS[cond],
                seed=seed,
                steps=args.steps,
                batch=args.batch,
                eval_num_batches=args.eval_num_batches,
                predictor_nfe=args.predictor_nfe,
                out_dir=out_dir,
            )
            rows.append(row)
            # Snapshot CSVs after each run in case the whole sweep is
            # interrupted; we keep partial results.
            write_csvs(rows, out_dir)
    wall = time.perf_counter() - t_outer

    summary = aggregate_summary(rows, out_dir)

    print("\n" + "=" * 78)
    sr = summary["solve_rate"]
    print(
        f"SUDOKU WORLD-TEST: {summary['verdict']}    "
        f"(CE {sr['ce']['mean']:.4f}±{sr['ce']['std']:.4f}  "
        f"FIXED {sr['fixed']['mean']:.4f}±{sr['fixed']['std']:.4f}  "
        f"ELBO {sr['elbo']['mean']:.4f}±{sr['elbo']['std']:.4f})"
    )
    print(
        f"  delta_CE_vs_FIXED   = {summary['delta_ce_vs_fixed']:+.4f}  "
        f"(2sigma envelope = {summary['delta_ce_envelope_2sigma']:.4f})"
    )
    print(
        f"  delta_ELBO_vs_FIXED = {summary['delta_elbo_vs_fixed']:+.4f}  "
        f"(2sigma envelope = {summary['delta_elbo_envelope_2sigma']:.4f})"
    )
    cc = summary["consistency_checks"]
    print("  consistency:")
    print(f"    ELBO ~ FIXED on solve_rate (within 2sigma): {cc['elbo_close_to_fixed_solve_rate']}")
    print(f"    CE  log_w moved (max|log_w|>0.1): {cc['ce_log_w_moved']}")
    print(f"    ELBO log_w stuck (max|log_w|<0.1): {cc['elbo_log_w_stuck']}")
    print("=" * 78)
    print(f"\nwall time: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"outputs in: {out_dir}")
    print(f"  solve_rates.csv  {out_dir / 'solve_rates.csv'}")
    print(f"  log_w_final.csv  {out_dir / 'log_w_final.csv'}")
    print(f"  summary.json     {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
