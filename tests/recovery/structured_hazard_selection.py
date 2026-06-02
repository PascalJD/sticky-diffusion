"""Structured-data hazard-selection probe for the e2e SJD ELBO at eta=1.

The question this answers (per user framing):
  Under NON-uniform p_0 + LIMITED denoiser capacity, does jointly learning the
  per-anchor forward hazard w find a schedule whose reverse the network can
  model better than the uniform-w schedule? That is the GenMD4 mechanism;
  capacity is what makes learnable schedules useful.

This is NOT a "recover w_true" test (there is no ground-truth w; the ELBO is
tight along a manifold). The metric is finite-capacity NELBO improvement,
compared FIXED (w=1, theta-only training) vs JOINT (theta + log_w jointly),
swept across denoiser width to locate where the effect (if any) appears.

USAGE:
    PYTHONPATH=src python -m tests.recovery.structured_hazard_selection
    PYTHONPATH=src python -m tests.recovery.structured_hazard_selection \
        --steps 8000 --seeds 0,1,2 --widths 8,16,32,64,128

OUTPUTS (under runs/structured_hazard_<timestamp>/):
    metrics.csv         one row per (h, seed, condition, eval-checkpoint)
    capacity_curve.csv  per-h aggregates (mean dNELBO, std, max|log_w|)
    w_vs_p0.png         learned w(a) vs p_0(a) at h*, all seeds
    log_w_traj.png      log_w trajectories at h*, all seeds
    summary.json        verdict + all aggregates

NOTE: Phase 3 (tightness-independent sample-quality cross-check) is SKIPPED.
The repo's reverse sampler depends on AnchorTable/classifier_induced_score
plumbing that does not match a toy Flax MLP without non-trivial adapter code;
per the spec we skip rather than ship a sketchy adapter and FLAG the
bound-tightness confound in the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss

# Reuse the anchor-table helper from the previous recovery test.
from tests.recovery.e2e_hazard_recovery import make_anchor_table


# ---------------------------- config defaults -----------------------------

K = 5
D = 4
L_SEQ = 8
P_0 = jnp.array([0.5, 0.25, 0.125, 0.0625, 0.0625], dtype=jnp.float32)
LOG_W_CLIP = (-3.0, 3.0)

BATCH = 256
N_TRAIN = 50_000
N_EVAL = 4096
EVAL_NOISE_DRAWS = 64    # to reduce MC variance on the held-out NELBO

LR_THETA = 3e-4
LR_LOG_W = 3e-4 * 0.3
GRAD_CLIP = 1.0
T_FLOOR = 1e-3
BETA_MIN = 0.1
BETA_MAX = 20.0
T_TERMINAL = 1.0
HAZARD_P = 1.0

DEFAULT_WIDTHS = (8, 16, 32, 64, 128)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 8_000


# ---------------------------- helpers -------------------------------------

def fourier_t(t: jnp.ndarray, dim: int = 4) -> jnp.ndarray:
    """4-dim Fourier feature of t."""
    half = dim // 2
    freqs = 2.0 ** jnp.arange(half, dtype=jnp.float32)
    args = 2.0 * jnp.pi * t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def recenter_clamp(log_w_raw: jnp.ndarray) -> jnp.ndarray:
    """Mirror TokenAnchors.log_w_float(): recenter to mean 0, then clamp."""
    log_w = log_w_raw - jnp.mean(log_w_raw)
    return jnp.clip(log_w, LOG_W_CLIP[0], LOG_W_CLIP[1])


def sample_x0_from_p0(key, batch_size: int) -> jnp.ndarray:
    """Sample x0_idx of shape (B, L_SEQ) with sites i.i.d. from the Zipf p_0."""
    return jax.random.choice(
        key, K, shape=(batch_size, L_SEQ), p=P_0
    ).astype(jnp.int32)


# ---------------------------- denoiser (tight) ----------------------------

class Denoiser(nn.Module):
    """2-layer per-site MLP (shared across sites). No norm layers.

    Width is the swept variable. Input is (Y_site, t_fourier) -> K logits."""
    K_: int = K
    hidden: int = 32
    n_layers: int = 2
    t_dim: int = 4

    @nn.compact
    def __call__(self, y, t):
        # y: (B, L, D); t: (B,)
        t_emb = fourier_t(t, dim=self.t_dim)                          # (B, t_dim)
        B = y.shape[0]
        L = y.shape[1]
        t_per_site = jnp.broadcast_to(
            t_emb[:, None, :], (B, L, self.t_dim)
        )
        x = jnp.concatenate([y, t_per_site], axis=-1)                  # (B, L, D+t)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.gelu(x)
        return nn.Dense(self.K_)(x)                                    # (B, L, K)


# ---------------------------- train / eval --------------------------------

def make_loss_fn(denoiser, beta, hazard, jump, anchor_table_E, T_):
    def loss_pure(theta, log_w_raw, key, x0_idx, x0_anchor, fixed_log_w: bool):
        log_w_effective = recenter_clamp(log_w_raw)
        if fixed_log_w:
            log_w_effective = jax.lax.stop_gradient(log_w_effective)

        def apply_fn(p, y, t_img):
            return denoiser.apply({"params": p}, y, t_img), {}

        loss, metrics = elbo_eta1_loss(
            key=key,
            params=theta,
            apply_fn=apply_fn,
            x0_anchor=x0_anchor,
            x0_idx=x0_idx,
            beta=beta,
            hazard=hazard,
            jump=jump,
            T=T_,
            anchor_log_w=log_w_effective,
            anchor_table=anchor_table_E,
            prior_strength=0.0,
            rb_weight=1.0,
            rb_share_sample=True,
            time_sampling="uniform",
            t_floor=T_FLOOR,
        )
        return loss, metrics

    return loss_pure


def make_optimizer(fixed_log_w: bool):
    """multi_transform: log_w branch is set_to_zero() for FIXED, Adam for JOINT."""
    base_chain = [optax.clip_by_global_norm(GRAD_CLIP)]
    if fixed_log_w:
        log_w_tx = optax.set_to_zero()
    else:
        log_w_tx = optax.adam(LR_LOG_W, b1=0.9, b2=0.99)
    base_chain.append(
        optax.multi_transform(
            {
                "theta": optax.adam(LR_THETA, b1=0.9, b2=0.99),
                "log_w": log_w_tx,
            },
            param_labels={"theta": "theta_leaf", "log_w": "log_w"},
        )
    )
    # Replace label sentinel: we want every leaf of theta to label 'theta',
    # not 'theta_leaf' — but param_labels here is a tree of labels matching
    # the params tree. We'll patch this at init time per-state.
    return optax.chain(*base_chain)


def make_param_labels(theta):
    """Tree-of-labels matching {'theta': theta, 'log_w': log_w}."""
    return {
        "theta": jax.tree_util.tree_map(lambda _: "theta", theta),
        "log_w": "log_w",
    }


def init_state(denoiser, rng_init):
    dummy_y = jnp.zeros((1, L_SEQ, D), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    theta = denoiser.init(rng_init, dummy_y, dummy_t)["params"]
    log_w = jnp.zeros((K,), dtype=jnp.float32)
    return {"theta": theta, "log_w": log_w}


def make_train_step(denoiser, beta, hazard, jump, anchor_table_E):
    loss_fn = make_loss_fn(denoiser, beta, hazard, jump, anchor_table_E, T_TERMINAL)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    def train_step(state, opt_state, optimizer, key, x0_idx, x0_anchor,
                    fixed_log_w: bool):
        (loss, metrics), (g_theta, g_log_w) = grad_fn(
            state["theta"], state["log_w"], key, x0_idx, x0_anchor, fixed_log_w
        )
        merged_params = {"theta": state["theta"], "log_w": state["log_w"]}
        merged_grads = {"theta": g_theta, "log_w": g_log_w}
        updates, new_opt_state = optimizer.update(
            merged_grads, opt_state, merged_params,
        )
        new_params = optax.apply_updates(merged_params, updates)
        new_state = {"theta": new_params["theta"], "log_w": new_params["log_w"]}
        log_w_grad_norm = jnp.linalg.norm(g_log_w)
        return new_state, new_opt_state, loss, metrics, log_w_grad_norm

    return train_step


def make_eval_step(denoiser, beta, hazard, jump, anchor_table_E):
    """Returns a jitted fn that computes NELBO + components averaged over a
    held-out batch under EVAL_NOISE_DRAWS fresh corruption draws."""
    loss_fn = make_loss_fn(denoiser, beta, hazard, jump, anchor_table_E, T_TERMINAL)

    def eval_one(theta, log_w_raw, key, x0_idx, x0_anchor):
        # fixed_log_w=True only blocks gradient flow, which we don't take here.
        # Pass False so the loss path treats log_w as its sole input either way.
        return loss_fn(theta, log_w_raw, key, x0_idx, x0_anchor, False)

    def eval_avg(theta, log_w_raw, key, x0_idx, x0_anchor):
        keys = jax.random.split(key, EVAL_NOISE_DRAWS)

        def body(carry, k):
            loss, metrics = eval_one(theta, log_w_raw, k, x0_idx, x0_anchor)
            return carry, (loss, metrics["loss/ce"], metrics["loss/rb"],
                           metrics["loss/score"])
        _, (losses, ces, rbs, scores) = jax.lax.scan(body, None, keys)
        return {
            "loss": jnp.mean(losses),
            "loss/ce": jnp.mean(ces),
            "loss/rb": jnp.mean(rbs),
            "loss/score": jnp.mean(scores),
            "loss_std": jnp.std(losses),
        }

    return jax.jit(eval_avg)


# ---------------------------- one training run ----------------------------

def run_one(*, h: int, seed: int, fixed_log_w: bool, steps: int,
            beta, hazard, jump, anchor_table_E, x0_eval_idx, x0_eval_anchor,
            csv_writer, log_w_traj_log: list):
    """Train one (h, seed, condition); write per-eval rows to csv_writer.

    Returns final eval metrics + final log_w_raw + grad-norm trajectory.
    """
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    denoiser = Denoiser(K_=K, hidden=h, n_layers=2, t_dim=4)
    state = init_state(denoiser, rng_init)
    optimizer = make_optimizer(fixed_log_w=fixed_log_w)
    # Patch param_labels to per-leaf for theta.
    labels = make_param_labels(state["theta"])
    log_w_label = "log_w"
    if fixed_log_w:
        log_w_tx = optax.set_to_zero()
    else:
        log_w_tx = optax.adam(LR_LOG_W, b1=0.9, b2=0.99)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.multi_transform(
            {"theta": optax.adam(LR_THETA, b1=0.9, b2=0.99),
             "log_w": log_w_tx},
            param_labels={"theta": labels["theta"], "log_w": log_w_label},
        ),
    )
    opt_state = optimizer.init(
        {"theta": state["theta"], "log_w": state["log_w"]}
    )

    train_step_raw = make_train_step(denoiser, beta, hazard, jump, anchor_table_E)
    train_step_jit = jax.jit(
        partial(train_step_raw, optimizer=optimizer, fixed_log_w=fixed_log_w)
    )
    eval_avg = make_eval_step(denoiser, beta, hazard, jump, anchor_table_E)

    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(state["theta"]))
    condition_name = "FIXED" if fixed_log_w else "JOINT"
    print(
        f"  [{condition_name} h={h} seed={seed}] params={n_params}",
        flush=True,
    )

    grad_norms = []  # over the run

    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        rng, rng_batch, rng_step = jax.random.split(rng, 3)
        x0_idx = sample_x0_from_p0(rng_batch, BATCH)
        x0_anchor = jnp.take(anchor_table_E, x0_idx, axis=0)
        state, opt_state, loss, metrics, log_w_grad_norm = train_step_jit(
            state, opt_state, key=rng_step,
            x0_idx=x0_idx, x0_anchor=x0_anchor,
        )

        if step % 100 == 0:
            grad_norms.append(float(log_w_grad_norm))
            log_w_traj_log.append(
                (h, seed, condition_name, step,
                 np.asarray(recenter_clamp(state["log_w"])).tolist())
            )

        if step in (500, 2000, 4000, 8000) and step <= steps:
            rng, rng_eval_k = jax.random.split(rng)
            eval_metrics = eval_avg(
                state["theta"], state["log_w"], rng_eval_k,
                x0_eval_idx, x0_eval_anchor,
            )
            log_w_eff = recenter_clamp(state["log_w"])
            csv_writer.writerow([
                h, seed, condition_name, step,
                float(eval_metrics["loss"]),
                float(eval_metrics["loss/ce"]),
                float(eval_metrics["loss/rb"]),
                float(eval_metrics["loss/score"]),
                float(eval_metrics["loss_std"]),
                float(jnp.max(jnp.abs(log_w_eff))),
                float(log_w_grad_norm),
            ])
    elapsed = time.perf_counter() - t_start

    # Final eval
    rng, rng_eval_k = jax.random.split(rng)
    final_eval = eval_avg(
        state["theta"], state["log_w"], rng_eval_k,
        x0_eval_idx, x0_eval_anchor,
    )
    final_log_w_eff = recenter_clamp(state["log_w"])
    max_abs_log_w = float(jnp.max(jnp.abs(final_log_w_eff)))

    print(
        f"    -> NELBO={float(final_eval['loss']):.4f} "
        f"(ce={float(final_eval['loss/ce']):.4f}, "
        f"rb={float(final_eval['loss/rb']):.4f}, "
        f"score={float(final_eval['loss/score']):.4f}) "
        f"max|log_w_eff|={max_abs_log_w:.4f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    return {
        "nelbo": float(final_eval["loss"]),
        "nelbo_std": float(final_eval["loss_std"]),
        "ce": float(final_eval["loss/ce"]),
        "rb": float(final_eval["loss/rb"]),
        "score": float(final_eval["loss/score"]),
        "log_w_effective": np.asarray(final_log_w_eff).tolist(),
        "max_abs_log_w": max_abs_log_w,
        "grad_norm_trajectory": grad_norms,
        "elapsed_sec": elapsed,
    }


# ---------------------------- Phase 1: sweep ------------------------------

def phase1(widths, seeds, steps, beta, hazard, jump, anchor_table_E,
            x0_eval_idx, x0_eval_anchor, out_dir):
    csv_path = out_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "h", "seed", "condition", "step",
        "nelbo", "ce", "rb", "score", "nelbo_std",
        "max_abs_log_w_effective", "log_w_grad_norm",
    ])

    log_w_traj_log = []

    # Per-h aggregates.
    results = {}  # results[h] = {seed: {"FIXED": ..., "JOINT": ...}}
    for h in widths:
        print(f"\n=== width h={h} ===", flush=True)
        results[h] = {}
        for seed in seeds:
            results[h][seed] = {}
            for cond_name, fixed in (("FIXED", True), ("JOINT", False)):
                r = run_one(
                    h=h, seed=seed, fixed_log_w=fixed, steps=steps,
                    beta=beta, hazard=hazard, jump=jump,
                    anchor_table_E=anchor_table_E,
                    x0_eval_idx=x0_eval_idx, x0_eval_anchor=x0_eval_anchor,
                    csv_writer=csv_writer, log_w_traj_log=log_w_traj_log,
                )
                results[h][seed][cond_name] = r
            csv_file.flush()

    csv_file.close()

    # Capacity curve table.
    cap_rows = []
    for h in widths:
        d_nelbos = []
        max_log_ws = []
        for seed in seeds:
            r = results[h][seed]
            d_nelbo = r["FIXED"]["nelbo"] - r["JOINT"]["nelbo"]
            d_nelbos.append(d_nelbo)
            max_log_ws.append(r["JOINT"]["max_abs_log_w"])
        cap_rows.append({
            "h": h,
            "mean_dNELBO": float(np.mean(d_nelbos)),
            "std_dNELBO": float(np.std(d_nelbos, ddof=1)) if len(seeds) > 1 else 0.0,
            "max_abs_log_w_joint": float(np.mean(max_log_ws)),
            "per_seed_dNELBO": d_nelbos,
        })

    # write capacity_curve.csv
    cap_path = out_dir / "capacity_curve.csv"
    with cap_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "mean_dNELBO", "std_dNELBO", "max_abs_log_w_joint",
                    "per_seed_dNELBO"])
        for row in cap_rows:
            w.writerow([
                row["h"], row["mean_dNELBO"], row["std_dNELBO"],
                row["max_abs_log_w_joint"],
                ";".join(f"{x:.5f}" for x in row["per_seed_dNELBO"]),
            ])

    return results, cap_rows, log_w_traj_log


# ---------------------------- Phase 2: w vs p0 ----------------------------

def _spearman(x, y):
    """Spearman rank correlation between two 1-D arrays."""
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom < 1e-12:
        return 0.0
    return float((xr * yr).sum() / denom)


def phase2(results, h_star, seeds, out_dir):
    """At h_star, plot w_learned vs p_0 and report Spearman."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p0 = np.asarray(P_0)
    learned_per_seed = []
    for seed in seeds:
        log_w_eff = results[h_star][seed]["JOINT"]["log_w_effective"]
        w_learned = np.exp(np.asarray(log_w_eff))
        learned_per_seed.append(w_learned)
    learned = np.stack(learned_per_seed, axis=0)  # (S, K)
    learned_mean = learned.mean(axis=0)
    learned_std = learned.std(axis=0, ddof=1) if learned.shape[0] > 1 else np.zeros_like(learned_mean)

    spearmans = [_spearman(learned[s], p0) for s in range(learned.shape[0])]
    mean_spearman = float(np.mean(spearmans))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in range(learned.shape[0]):
        ax.scatter(p0, learned[s], label=f"seed {seeds[s]}", alpha=0.7)
    ax.errorbar(p0, learned_mean, yerr=learned_std, fmt="o-", color="k",
                label=f"mean ± std (Spearman={mean_spearman:+.3f})", capsize=4)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5,
               label="anchor-agnostic w=1")
    ax.set_xscale("log")
    ax.set_xlabel("p_0(a) (data freq, log scale)")
    ax.set_ylabel("w_learned(a) = exp(log_w_effective(a))")
    ax.set_title(f"Phase 2: learned hazard w vs anchor frequency p_0 (h*={h_star})")
    for a, (px, py) in enumerate(zip(p0, learned_mean)):
        ax.annotate(f"a={a}", (px, py), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "w_vs_p0.png", dpi=140)
    plt.close(fig)

    return {
        "h_star": h_star,
        "learned_mean": learned_mean.tolist(),
        "learned_std": learned_std.tolist(),
        "per_seed_w": learned.tolist(),
        "mean_spearman_w_p0": mean_spearman,
        "per_seed_spearman": spearmans,
    }


def plot_log_w_trajectories(log_w_traj_log, h_star, seeds, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to h_star, JOINT only.
    relevant = [r for r in log_w_traj_log
                if r[0] == h_star and r[2] == "JOINT" and r[1] in seeds]
    # Group by seed.
    by_seed = {seed: [] for seed in seeds}
    for h, seed, cond, step, lw in relevant:
        by_seed[seed].append((step, lw))

    fig, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 4),
                              sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        items = by_seed[seed]
        if not items:
            ax.set_title(f"seed {seed}: no data")
            continue
        items.sort(key=lambda r: r[0])
        steps = np.array([r[0] for r in items])
        lw_arr = np.array([r[1] for r in items])  # (T, K)
        for a in range(K):
            ax.plot(steps, lw_arr[:, a], label=f"a={a} (p0={float(P_0[a]):.3g})")
        ax.set_xlabel("step")
        ax.set_title(f"seed {seed}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("log_w_effective (recentered, clamped)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"log_w trajectories during JOINT training (h*={h_star})")
    fig.tight_layout()
    fig.savefig(out_dir / "log_w_traj.png", dpi=140)
    plt.close(fig)


# ---------------------------- main ----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--widths", type=str, default=",".join(str(w) for w in DEFAULT_WIDTHS),
    )
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    widths = tuple(int(w) for w in args.widths.split(","))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"structured_hazard_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    print(f"widths: {widths}; seeds: {seeds}; steps/run: {args.steps}",
          flush=True)
    print(f"p_0 = {np.asarray(P_0).tolist()}", flush=True)

    beta = make_beta(BETA_MIN, BETA_MAX, T=T_TERMINAL)
    hazard = make_hazard_poly_alpha(beta, p=HAZARD_P)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    anchor_table_E = make_anchor_table()

    # Held-out eval pool fixed across all (h, seed, condition).
    rng_eval = jax.random.PRNGKey(99_999)
    x0_eval_idx = sample_x0_from_p0(rng_eval, N_EVAL)
    x0_eval_anchor = jnp.take(anchor_table_E, x0_eval_idx, axis=0)

    t_start = time.perf_counter()
    results, cap_rows, log_w_traj_log = phase1(
        widths, seeds, args.steps, beta, hazard, jump, anchor_table_E,
        x0_eval_idx, x0_eval_anchor, out_dir,
    )
    wall_p1 = time.perf_counter() - t_start
    print(f"\nphase 1 wall time: {wall_p1:.1f}s ({wall_p1/60:.1f} min)",
          flush=True)

    # ----- pick h* -----
    # Best mean dNELBO that is also robust (> 2 * seed-std).
    h_star = None
    best_score = -np.inf
    for row in cap_rows:
        mu = row["mean_dNELBO"]
        sd = row["std_dNELBO"]
        # Robustness margin: mu - 2*sd.
        margin = mu - 2.0 * sd
        if mu > best_score:
            best_score = mu
            h_star_max = row["h"]
        if mu > 0 and margin > 0 and mu > best_score - 1e-9:
            h_star = row["h"]
    if h_star is None:
        # Fall back to argmax for plotting purposes; verdict still uses the
        # strict robustness check.
        h_star = h_star_max

    print(f"\nh* (best robust width, or argmax fallback): {h_star}",
          flush=True)
    phase2_out = phase2(results, h_star, seeds, out_dir)
    plot_log_w_trajectories(log_w_traj_log, h_star, seeds, out_dir)

    # ----- verdict -----
    h_star_row = next(r for r in cap_rows if r["h"] == h_star)
    h_star_mu = h_star_row["mean_dNELBO"]
    h_star_sd = h_star_row["std_dNELBO"]
    h_star_margin = h_star_mu - 2.0 * h_star_sd
    h_star_max_log_w = h_star_row["max_abs_log_w_joint"]
    spearman = phase2_out["mean_spearman_w_p0"]

    nelbo_robust = (h_star_mu > 0) and (h_star_margin > 0)
    log_w_moved = h_star_max_log_w > 0.1

    if nelbo_robust and log_w_moved:
        verdict = "HELPS"
    elif nelbo_robust and not log_w_moved:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO EFFECT"

    # Phase 3 was skipped — add caveat string.
    phase3_caveat = (
        "Phase 3 (sample-quality cross-check) was NOT run; bound-tightness "
        "confound is uncontrolled."
    )

    # ----- write summary.json -----
    summary = {
        "verdict": verdict,
        "h_star": h_star,
        "h_star_mean_dNELBO": h_star_mu,
        "h_star_std_dNELBO": h_star_sd,
        "h_star_robust_margin": h_star_margin,
        "h_star_max_abs_log_w_joint": h_star_max_log_w,
        "h_star_mean_spearman_w_p0": spearman,
        "capacity_curve": cap_rows,
        "phase2": phase2_out,
        "phase3_skipped_caveat": phase3_caveat,
        "raw_results_per_h_per_seed": {
            str(h): {
                str(seed): {
                    "FIXED": results[h][seed]["FIXED"],
                    "JOINT": results[h][seed]["JOINT"],
                } for seed in seeds
            } for h in widths
        },
        "wall_time_sec": wall_p1,
        "config": {
            "K": K, "D": D, "L_SEQ": L_SEQ,
            "p_0": np.asarray(P_0).tolist(),
            "log_w_clip": list(LOG_W_CLIP),
            "batch": BATCH, "n_train": N_TRAIN, "n_eval": N_EVAL,
            "eval_noise_draws": EVAL_NOISE_DRAWS,
            "lr_theta": LR_THETA, "lr_log_w": LR_LOG_W,
            "steps": args.steps,
            "seeds": list(seeds), "widths": list(widths),
        },
        "thresholds": {
            "nelbo_margin_factor": 2.0,
            "log_w_moved_floor": 0.1,
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ----- print verdict + capacity table -----
    print("\n" + "=" * 78)
    if verdict == "HELPS":
        print(
            f"STRUCTURED HAZARD SELECTION: HELPS    "
            f"(h*={h_star}, dNELBO={h_star_mu:+.4f}±{h_star_sd:.4f}, "
            f"spearman={spearman:+.3f}, TV J/F=skipped)"
        )
    elif verdict == "NO EFFECT":
        best_row = max(cap_rows, key=lambda r: r["mean_dNELBO"])
        print(
            f"STRUCTURED HAZARD SELECTION: NO EFFECT "
            f"(best dNELBO={best_row['mean_dNELBO']:+.4f}±"
            f"{best_row['std_dNELBO']:.4f} at h={best_row['h']}; "
            f"log_w_moved at h*={h_star} was max|log_w|="
            f"{h_star_max_log_w:.3f})"
        )
    else:
        print(
            f"STRUCTURED HAZARD SELECTION: INCONCLUSIVE "
            f"(NELBO helps at h*={h_star} but log_w didn't move; "
            f"max|log_w|={h_star_max_log_w:.3f})"
        )
    print("=" * 78)

    # capacity table
    print("\nCapacity curve:")
    print(f"  {'h':>4}  {'mean_dNELBO':>14}  {'std':>8}  "
          f"{'max|log_w| (JOINT)':>18}")
    for row in cap_rows:
        marker = " *" if row["h"] == h_star else ""
        print(
            f"  {row['h']:>4}  {row['mean_dNELBO']:>+14.5f}  "
            f"{row['std_dNELBO']:>8.5f}  {row['max_abs_log_w_joint']:>18.4f}"
            f"{marker}"
        )

    print(f"\n{phase3_caveat}")
    print(f"\noutputs in: {out_dir}")
    print(f"  metrics.csv         {out_dir / 'metrics.csv'}")
    print(f"  capacity_curve.csv  {out_dir / 'capacity_curve.csv'}")
    print(f"  summary.json        {out_dir / 'summary.json'}")
    print(f"  w_vs_p0.png         {out_dir / 'w_vs_p0.png'}")
    print(f"  log_w_traj.png      {out_dir / 'log_w_traj.png'}")
    print(f"\nwall time: {wall_p1:.1f}s ({wall_p1/60:.1f} min)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
