from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from sticky.data.sudoku import make_sudoku_board_iterator
from sticky.models.sjd.anchors import AnchorTable
from sticky.models.sjd.board_sampling import puzzle_digits_to_clean_indices
from sticky.models.sjd.plugin_intensity import (
    dhm_target,
    plugin_hazard_and_allocation,
)
from sticky.models.sjd.sdes import vp_perturb
from sticky.rng import make_rng


Array = jnp.ndarray


@dataclass
class Prop52DiagnosticsResult:
    metrics: Dict[str, float]
    wandb_payload: Dict[str, Any]
    rows_by_eta: Dict[str, list[dict[str, Any]]]
    eta_summary_rows: list[dict[str, Any]]
    collapse_summary: dict[str, Any]


def format_eta_label(eta: float) -> str:
    return f"{float(eta):.2f}".replace(".", "p")


def extract_prop52_eta_specs(policy_specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for spec in policy_specs:
        if str(spec.get("policy", "")).strip().lower() != "plugin_hazard":
            continue
        eta = float(spec.get("eta", 1.0))
        eta_label = format_eta_label(eta)
        if eta_label in seen:
            continue
        seen.add(eta_label)
        out.append(
            {
                "eta": eta,
                "eta_label": eta_label,
                "logit_temperature": float(spec.get("logit_temperature", 1.0)),
                "log_ratio_clip": float(spec.get("log_ratio_clip", 10.0)),
            }
        )
    return out


def _empty_result() -> Prop52DiagnosticsResult:
    return Prop52DiagnosticsResult(
        metrics={},
        wandb_payload={},
        rows_by_eta={},
        eta_summary_rows=[],
        collapse_summary={},
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size <= 1 or x.size <= 1:
        return float(0.0 if y.size == 0 else y[0] * 0.0)
    return float(np.trapezoid(y, x))


def _make_wandb_table(wandb_mod, rows: list[dict[str, Any]], columns: list[str]):
    if wandb_mod is None or not rows or not hasattr(wandb_mod, "Table"):
        return None
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb_mod.Table(columns=columns, data=data)


def _make_wandb_line_plot(wandb_mod, table, *, x: str, y: str, title: str):
    if wandb_mod is None or table is None:
        return None
    plot_mod = getattr(wandb_mod, "plot", None)
    line_fn = getattr(plot_mod, "line", None) if plot_mod is not None else None
    if not callable(line_fn):
        return None
    try:
        return line_fn(table, x, y, title=title)
    except TypeError:
        return line_fn(table, x, y, title)


def _make_wandb_scatter_plot(wandb_mod, table, *, x: str, y: str, title: str):
    if wandb_mod is None or table is None:
        return None
    plot_mod = getattr(wandb_mod, "plot", None)
    scatter_fn = getattr(plot_mod, "scatter", None) if plot_mod is not None else None
    if not callable(scatter_fn):
        return None
    try:
        return scatter_fn(table, x, y, title=title)
    except TypeError:
        return scatter_fn(table, x, y, title)


def _prepare_prop52_batch_fn(
    *,
    model,
    anchor_table: Array,
    beta,
    time_eps: float,
):
    @jax.jit
    def _prepare_batch(params, rng, solution_board, clue_board, clue_mask):
        solution_board = jnp.asarray(solution_board, dtype=jnp.int32)
        clue_board = jnp.asarray(clue_board, dtype=jnp.int32)
        clue_mask = jnp.asarray(clue_mask, dtype=jnp.bool_)

        x0_idx = solution_board - 1
        x0_anchor = anchor_table[x0_idx]

        key_t, key_vp = jax.random.split(rng)
        t_img = jax.random.uniform(
            key_t,
            shape=(solution_board.shape[0],),
            minval=jnp.asarray(float(time_eps), dtype=jnp.float32),
            maxval=jnp.asarray(float(beta.T), dtype=jnp.float32),
        )
        y_t, _ = vp_perturb(key_vp, x0_anchor, t_img, beta)

        known_idx = puzzle_digits_to_clean_indices(clue_board, known_mask=clue_mask)
        known_anchor = anchor_table[known_idx]
        y_in = jnp.where(clue_mask[..., None], known_anchor, y_t)
        logits, _ = model.apply({"params": params}, y_in, t=t_img, train=False)

        return {
            "t_img": t_img.astype(jnp.float32),
            "y_in": y_in.astype(jnp.float32),
            "logits": logits.astype(jnp.float32),
            "x0_idx": x0_idx.astype(jnp.int32),
            "unknown_mask": (~clue_mask).astype(jnp.bool_),
        }

    return _prepare_batch


def _eta_arrays_fn(
    *,
    anchors: AnchorTable,
    beta,
    hazard,
    jump,
    eta: float,
    logit_temperature: float,
    log_ratio_clip: float,
    tau_grid_size: int = 32,
):
    jump_eff = replace(jump, eta=float(eta))

    @jax.jit
    def _eta_arrays(logits, y_in, t_img, x0_idx):
        lam_hat = dhm_target(
            y=y_in,
            t_img=t_img,
            true_anchor_idx=x0_idx,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump_eff,
            log_ratio_clip=float(log_ratio_clip),
            tau_grid_size=int(tau_grid_size),
        )
        lam_plug, _ = plugin_hazard_and_allocation(
            logits=logits,
            y=y_in,
            t_img=t_img,
            anchors=anchors,
            beta=beta,
            hazard=hazard,
            jump=jump_eff,
            logit_temperature=float(logit_temperature),
            log_ratio_clip=float(log_ratio_clip),
            tau_grid_size=int(tau_grid_size),
        )
        return lam_hat.astype(jnp.float32), lam_plug.astype(jnp.float32)

    return _eta_arrays


def _empty_agg(num_bins: int) -> dict[str, np.ndarray]:
    zeros = np.zeros((int(num_bins),), dtype=np.float64)
    return {
        "count": np.zeros((int(num_bins),), dtype=np.int64),
        "sum_hat": zeros.copy(),
        "sum_hat_sq": zeros.copy(),
        "sum_plug": zeros.copy(),
        "sum_plug_sq": zeros.copy(),
        "sum_cross": zeros.copy(),
    }


def _update_agg(
    agg: dict[str, np.ndarray],
    *,
    bin_idx: np.ndarray,
    lam_hat: np.ndarray,
    lam_plug: np.ndarray,
) -> None:
    np.add.at(agg["count"], bin_idx, 1)
    np.add.at(agg["sum_hat"], bin_idx, lam_hat)
    np.add.at(agg["sum_hat_sq"], bin_idx, lam_hat * lam_hat)
    np.add.at(agg["sum_plug"], bin_idx, lam_plug)
    np.add.at(agg["sum_plug_sq"], bin_idx, lam_plug * lam_plug)
    np.add.at(agg["sum_cross"], bin_idx, lam_hat * lam_plug)


def _finalize_eta_rows(
    *,
    eta: float,
    bin_edges: np.ndarray,
    agg: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_centers: list[float] = []
    valid_counts: list[float] = []
    valid_delta: list[float] = []
    valid_v_state: list[float] = []

    for i in range(len(bin_edges) - 1):
        center = float(0.5 * (bin_edges[i] + bin_edges[i + 1]))
        count = int(agg["count"][i])
        row: dict[str, Any] = {
            "eta": float(eta),
            "t_bin_center": center,
            "count": count,
        }
        if count <= 0:
            row.update(
                {
                    "mean_lambda_hat": None,
                    "lambda_time": None,
                    "mean_lambda_plug": None,
                    "mse_time": None,
                    "mse_plug": None,
                    "delta_mse": None,
                    "V_state": None,
                }
            )
            rows.append(row)
            continue

        count_f = float(count)
        mean_hat = float(agg["sum_hat"][i] / count_f)
        mean_plug = float(agg["sum_plug"][i] / count_f)
        e_hat_sq = float(agg["sum_hat_sq"][i] / count_f)
        e_plug_sq = float(agg["sum_plug_sq"][i] / count_f)
        e_cross = float(agg["sum_cross"][i] / count_f)

        mse_time = max(0.0, e_hat_sq - (mean_hat * mean_hat))
        mse_plug = max(0.0, e_plug_sq - (2.0 * e_cross) + e_hat_sq)
        delta_mse = mse_time - mse_plug
        v_state = max(0.0, e_plug_sq - (mean_plug * mean_plug))

        row.update(
            {
                "mean_lambda_hat": mean_hat,
                "lambda_time": mean_hat,
                "mean_lambda_plug": mean_plug,
                "mse_time": float(mse_time),
                "mse_plug": float(mse_plug),
                "delta_mse": float(delta_mse),
                "V_state": float(v_state),
            }
        )
        rows.append(row)
        valid_centers.append(center)
        valid_counts.append(count_f)
        valid_delta.append(float(delta_mse))
        valid_v_state.append(float(v_state))

    if not valid_centers:
        return rows, {
            "eta": float(eta),
            "count_total": 0,
            "num_valid_bins": 0,
            "area_under_delta_mse": 0.0,
            "area_under_v_state": 0.0,
            "mean_delta_mse": 0.0,
            "mean_v_state": 0.0,
            "max_v_state": 0.0,
        }

    centers_np = np.asarray(valid_centers, dtype=np.float64)
    counts_np = np.asarray(valid_counts, dtype=np.float64)
    delta_np = np.asarray(valid_delta, dtype=np.float64)
    v_state_np = np.asarray(valid_v_state, dtype=np.float64)
    weights = counts_np / max(np.sum(counts_np), 1.0)

    summary_row = {
        "eta": float(eta),
        "count_total": int(np.sum(counts_np)),
        "num_valid_bins": int(centers_np.size),
        "area_under_delta_mse": _safe_trapz(delta_np, centers_np),
        "area_under_v_state": _safe_trapz(v_state_np, centers_np),
        "mean_delta_mse": float(np.sum(delta_np * weights)),
        "mean_v_state": float(np.sum(v_state_np * weights)),
        "max_v_state": float(np.max(v_state_np)),
    }
    return rows, summary_row


def _collapse_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        return {}
    by_eta = {float(row["eta"]): row for row in summary_rows}
    eta_one = by_eta.get(1.0, None)
    lower_etas = [row for eta, row in by_eta.items() if abs(float(eta) - 1.0) > 1.0e-6]
    if eta_one is None or not lower_etas:
        return {}

    lower_sorted = sorted(lower_etas, key=lambda row: float(row["eta"]))
    ref_row = lower_sorted[0]
    max_lower_mean_v = max(float(row["mean_v_state"]) for row in lower_sorted)
    max_lower_mean_delta = max(float(row["mean_delta_mse"]) for row in lower_sorted)

    eta_one_mean_v = float(eta_one["mean_v_state"])
    eta_one_mean_delta = float(eta_one["mean_delta_mse"])
    ref_mean_v = float(ref_row["mean_v_state"])
    ref_mean_delta = float(ref_row["mean_delta_mse"])

    return {
        "eta_reference": float(ref_row["eta"]),
        "eta_one_mean_v_state": eta_one_mean_v,
        "eta_reference_mean_v_state": ref_mean_v,
        "eta_one_mean_delta_mse": eta_one_mean_delta,
        "eta_reference_mean_delta_mse": ref_mean_delta,
        "eta_one_vs_reference_v_state_ratio": (
            float(eta_one_mean_v / ref_mean_v) if ref_mean_v > 0.0 else None
        ),
        "eta_one_vs_reference_delta_mse_ratio": (
            float(eta_one_mean_delta / ref_mean_delta) if ref_mean_delta > 0.0 else None
        ),
        "eta_one_vs_max_lower_v_state_ratio": (
            float(eta_one_mean_v / max_lower_mean_v) if max_lower_mean_v > 0.0 else None
        ),
        "eta_one_vs_max_lower_delta_mse_ratio": (
            float(eta_one_mean_delta / max_lower_mean_delta)
            if max_lower_mean_delta > 0.0
            else None
        ),
        "v_state_collapsed": bool(eta_one_mean_v <= 0.5 * max_lower_mean_v),
        "delta_mse_collapsed": bool(eta_one_mean_delta <= 0.5 * max_lower_mean_delta),
    }


def run_sudoku_prop52_diagnostics(
    *,
    cfg: DictConfig,
    eval_cfg: DictConfig,
    task,
    model,
    params,
    step_i: int,
    policy_specs: Iterable[dict[str, Any]],
    wandb_mod=None,
) -> Prop52DiagnosticsResult:
    eta_specs = extract_prop52_eta_specs(policy_specs)
    if not eta_specs:
        return _empty_result()

    num_bins = int(eval_cfg.get("sudoku_prop52_num_bins", 20))
    num_batches = int(eval_cfg.get("sudoku_prop52_num_batches", 8))
    time_eps = float(eval_cfg.get("sudoku_prop52_time_eps", 1.0e-4))
    seed_offset = int(eval_cfg.get("sudoku_prop52_seed_offset", 271828))
    fold_in_step = bool(eval_cfg.get("sudoku_prop52_fold_in_step", False))
    if num_bins <= 0 or num_batches == 0:
        return _empty_result()

    bin_edges = np.linspace(float(time_eps), float(task.beta.T), int(num_bins) + 1, dtype=np.float64)
    eta_aggs = {spec["eta_label"]: _empty_agg(num_bins) for spec in eta_specs}

    a_table = model.apply({"params": params}, method=model.anchor_table)
    anchors = AnchorTable(table_float=jnp.asarray(a_table, dtype=jnp.float32))
    prepare_batch = _prepare_prop52_batch_fn(
        model=model,
        anchor_table=anchors.table_float,
        beta=task.beta,
        time_eps=float(time_eps),
    )
    eta_fns = {
        spec["eta_label"]: _eta_arrays_fn(
            anchors=anchors,
            beta=task.beta,
            hazard=task.hazard,
            jump=task.jump,
            eta=float(spec["eta"]),
            logit_temperature=float(spec["logit_temperature"]),
            log_ratio_clip=float(spec["log_ratio_clip"]),
        )
        for spec in eta_specs
    }

    eval_iter = make_sudoku_board_iterator(
        split="test",
        batch_size=int(task.eval_batch_size),
        seed=int(cfg.training.seed) + seed_offset + 1,
        data_dir=task.data_dir,
        train_file=task.train_file,
        test_file=task.test_file,
        shuffle=False,
        repeat=False,
        drop_remainder=False,
        mmap=bool(task.mmap),
        max_examples=int(task.max_test_examples),
        include_strings=False,
        auto_download=bool(task.auto_download),
        download_timeout_sec=int(task.download_timeout_sec),
        download_retries=int(task.download_retries),
    )

    rng_seed = int(cfg.training.seed) + seed_offset
    if fold_in_step:
        rng_seed += int(step_i)
    base_rng = make_rng(rng_seed)

    consumed_batches = 0
    while consumed_batches < num_batches:
        try:
            batch = next(eval_iter)
        except StopIteration:
            break
        consumed_batches += 1
        rng_batch = jax.random.fold_in(base_rng, consumed_batches - 1)
        prepared = prepare_batch(
            params,
            rng_batch,
            jnp.asarray(batch["solution_board"], dtype=jnp.int32),
            jnp.asarray(batch["clue_board"], dtype=jnp.int32),
            jnp.asarray(batch["clue_mask"], dtype=jnp.bool_),
        )
        t_np = np.asarray(jax.device_get(prepared["t_img"]), dtype=np.float64)
        unknown_mask_np = np.asarray(jax.device_get(prepared["unknown_mask"]), dtype=np.bool_)
        if not np.any(unknown_mask_np):
            continue

        sites_per_example = int(np.prod(unknown_mask_np.shape[1:]))
        flat_mask = unknown_mask_np.reshape(-1)
        flat_t = np.repeat(t_np, sites_per_example)[flat_mask]
        bin_idx = np.clip(
            np.searchsorted(bin_edges, flat_t, side="right") - 1,
            0,
            int(num_bins) - 1,
        ).astype(np.int64)

        for spec in eta_specs:
            lam_hat, lam_plug = eta_fns[spec["eta_label"]](
                prepared["logits"],
                prepared["y_in"],
                prepared["t_img"],
                prepared["x0_idx"],
            )
            lam_hat_np = np.asarray(jax.device_get(lam_hat), dtype=np.float64).reshape(-1)[flat_mask]
            lam_plug_np = np.asarray(jax.device_get(lam_plug), dtype=np.float64).reshape(-1)[flat_mask]
            _update_agg(
                eta_aggs[spec["eta_label"]],
                bin_idx=bin_idx,
                lam_hat=lam_hat_np,
                lam_plug=lam_plug_np,
            )

    metrics: Dict[str, float] = {}
    wandb_payload: Dict[str, Any] = {}
    rows_by_eta: Dict[str, list[dict[str, Any]]] = {}
    eta_summary_rows: list[dict[str, Any]] = []

    per_eta_columns = [
        "eta",
        "t_bin_center",
        "count",
        "mean_lambda_hat",
        "lambda_time",
        "mean_lambda_plug",
        "mse_time",
        "mse_plug",
        "delta_mse",
        "V_state",
    ]

    for spec in eta_specs:
        eta = float(spec["eta"])
        eta_label = str(spec["eta_label"])
        rows, summary_row = _finalize_eta_rows(
            eta=eta,
            bin_edges=bin_edges,
            agg=eta_aggs[eta_label],
        )
        rows_by_eta[eta_label] = rows
        eta_summary_rows.append({"eta_label": eta_label, **summary_row})

        diag_prefix = f"diag/eta_{eta_label}"
        metrics[f"{diag_prefix}/area_under_delta_mse"] = float(summary_row["area_under_delta_mse"])
        metrics[f"{diag_prefix}/area_under_v_state"] = float(summary_row["area_under_v_state"])
        metrics[f"{diag_prefix}/mean_delta_mse"] = float(summary_row["mean_delta_mse"])
        metrics[f"{diag_prefix}/mean_v_state"] = float(summary_row["mean_v_state"])
        metrics[f"{diag_prefix}/max_v_state"] = float(summary_row["max_v_state"])
        metrics[f"{diag_prefix}/count_total"] = float(summary_row["count_total"])

        table = _make_wandb_table(wandb_mod, rows, per_eta_columns)
        if table is not None:
            wandb_payload[f"{diag_prefix}/prop52_table"] = table
        plot = _make_wandb_line_plot(
            wandb_mod,
            table,
            x="t_bin_center",
            y="delta_mse",
            title=f"Delta MSE vs t (eta={eta:.2f})",
        )
        if plot is not None:
            wandb_payload[f"{diag_prefix}/delta_mse_vs_t"] = plot
        plot = _make_wandb_line_plot(
            wandb_mod,
            table,
            x="t_bin_center",
            y="V_state",
            title=f"V_state vs t (eta={eta:.2f})",
        )
        if plot is not None:
            wandb_payload[f"{diag_prefix}/v_state_vs_t"] = plot
        plot = _make_wandb_line_plot(
            wandb_mod,
            table,
            x="t_bin_center",
            y="mse_time",
            title=f"Time-only MSE vs t (eta={eta:.2f})",
        )
        if plot is not None:
            wandb_payload[f"{diag_prefix}/mse_time_vs_t"] = plot
        plot = _make_wandb_line_plot(
            wandb_mod,
            table,
            x="t_bin_center",
            y="mse_plug",
            title=f"Plug-in MSE vs t (eta={eta:.2f})",
        )
        if plot is not None:
            wandb_payload[f"{diag_prefix}/mse_plug_vs_t"] = plot
        scatter = _make_wandb_scatter_plot(
            wandb_mod,
            table,
            x="V_state",
            y="delta_mse",
            title=f"Prop 5.2 scatter (eta={eta:.2f}; color=t_bin_center in table)",
        )
        if scatter is not None:
            wandb_payload[f"{diag_prefix}/delta_mse_vs_v_state"] = scatter

    summary_columns = [
        "eta_label",
        "eta",
        "count_total",
        "num_valid_bins",
        "area_under_delta_mse",
        "area_under_v_state",
        "mean_delta_mse",
        "mean_v_state",
        "max_v_state",
    ]
    summary_table = _make_wandb_table(wandb_mod, eta_summary_rows, summary_columns)
    if summary_table is not None:
        wandb_payload["diag/eta_summary_table"] = summary_table

    collapse_summary = _collapse_summary(eta_summary_rows)
    for key, value in collapse_summary.items():
        scalar = _to_float(value)
        if scalar is not None:
            metrics[f"diag/collapse/{key}"] = scalar
    if "v_state_collapsed" in collapse_summary:
        metrics["diag/collapse/v_state_collapsed"] = float(bool(collapse_summary["v_state_collapsed"]))
    if "delta_mse_collapsed" in collapse_summary:
        metrics["diag/collapse/delta_mse_collapsed"] = float(bool(collapse_summary["delta_mse_collapsed"]))

    return Prop52DiagnosticsResult(
        metrics=metrics,
        wandb_payload=wandb_payload,
        rows_by_eta=rows_by_eta,
        eta_summary_rows=eta_summary_rows,
        collapse_summary=collapse_summary,
    )
