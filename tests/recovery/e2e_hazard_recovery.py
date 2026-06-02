"""End-to-end hazard recovery test for the e2e SJD ELBO objective at eta=1.

Question: does the joint (theta, log_w) optimizer recover a known w_true on
synthetic data under the CURRENT loss implementation (L_CE + L_RB + L_score)?

If yes (L2 < 0.15 AND cos > 0.95 on recentered log_w), the implementation is
good enough to launch Sudoku. If no, the L_score derivation likely needs more
work.

USAGE:
    PYTHONPATH=src python -m tests.recovery.e2e_hazard_recovery

OUTPUTS (under runs/recovery_eta1_<timestamp>/):
    metrics.csv     -- per-eval log_w_hat, errors, losses, grad norm
    log_w_trajectory.png  -- 5 anchor curves vs log_w_true_recentered dashed
    summary.json    -- final L2, cos, per-anchor err, PASSED flag

This file does NOT modify src/sticky/ — it consumes the loss as-is.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from sticky.models.sjd.sdes import make_beta
from sticky.models.sjd.hazard import make_hazard_poly_alpha
from sticky.models.sjd.jump import VPMatchedGaussianJump
from sticky.models.sjd.sjd_elbo_loss import elbo_eta1_loss


# ---------------------------- config defaults -----------------------------

K = 5
D = 4
L_SEQ = 8
HIDDEN = 128
N_TRAIN = 50_000
N_EVAL = 2048
BATCH = 256
STEPS_DEFAULT = 8_000
SEED_DEFAULT = 0
LR_THETA = 3e-4
LR_LOG_W = 1e-4  # ~ LR_THETA * 0.3
T_FLOOR = 1e-3
BETA_MIN = 0.1
BETA_MAX = 20.0
T_TERMINAL = 1.0
HAZARD_P = 1.0
W_TRUE = jnp.array([0.5, 0.7, 1.0, 1.5, 2.0], dtype=jnp.float32)


# ---------------------------- anchor table --------------------------------

def make_anchor_table(K_=K, d=D, scale=3.0, seed=42):
    """Frozen anchor embedding table E with rows roughly orthonormal scaled.

    With K=5 in R^4 we can't be exactly orthogonal; we orthonormalize the
    first d via QR and append the (K-d) remaining as normalized random.
    """
    rng = jax.random.PRNGKey(seed)
    raw = jax.random.normal(rng, shape=(K_, d), dtype=jnp.float32)
    rows = []
    # Greedy Gram–Schmidt: each row orthogonalized vs previous, then
    # normalized. For K > d the later rows can only be a unit vector in the
    # span — they just won't all be mutually orthogonal.
    for i in range(K_):
        v = raw[i]
        for j in range(len(rows)):
            v = v - jnp.dot(v, rows[j]) * rows[j]
        norm = jnp.linalg.norm(v)
        if norm < 1e-6:
            # backup: just use the raw row (K>d case after rank exhausted)
            v = raw[i]
            v = v / jnp.linalg.norm(v)
        else:
            v = v / norm
        rows.append(v)
    E = jnp.stack(rows, axis=0)
    return scale * E


# ---------------------------- denoiser ------------------------------------

def _sinusoidal_t_emb(t: jnp.ndarray, dim: int = 32) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class Denoiser(nn.Module):
    """Per-site MLP shared across sites; takes (y, t) -> logits over K.

    Inputs:
        y: (B, L_seq, D)
        t: (B,)
    Output:
        logits: (B, L_seq, K)
    """
    K_: int = K
    hidden: int = HIDDEN
    n_layers: int = 3
    t_emb_dim: int = 32

    @nn.compact
    def __call__(self, y, t):
        # (B, t_emb_dim) -> broadcast across sites to (B, L_seq, t_emb_dim)
        t_emb = _sinusoidal_t_emb(t, dim=self.t_emb_dim)
        B_ = y.shape[0]
        L = y.shape[1]
        t_emb_per_site = jnp.broadcast_to(
            t_emb[:, None, :], (B_, L, self.t_emb_dim)
        )
        x = jnp.concatenate([y, t_emb_per_site], axis=-1)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.gelu(x)
        return nn.Dense(self.K_)(x)


# ---------------------------- train state ---------------------------------

@dataclass
class TrainState:
    theta: dict        # denoiser params
    log_w: jnp.ndarray  # (K,)
    opt_state: optax.OptState


def init_optimizer(state):
    """Multi-transform optimizer: separate Adam for theta vs log_w."""
    # Label leaves: every theta leaf -> 'theta'; log_w -> 'log_w'.
    label_tree = {
        "theta": jax.tree_util.tree_map(lambda _: "theta", state["theta"]),
        "log_w": "log_w",
    }
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "theta": optax.adam(LR_THETA, b1=0.9, b2=0.99),
                "log_w": optax.adam(LR_LOG_W, b1=0.9, b2=0.99),
            },
            param_labels=label_tree,
        ),
    )
    return opt


# ---------------------------- train step ----------------------------------

def make_train_step(denoiser, beta, hazard, jump, anchor_table_E, T_):
    apply_fn_inner = denoiser.apply

    def _loss_pure(params_log_w_tuple, key, x0_idx, x0_anchor):
        theta, log_w = params_log_w_tuple
        def apply_fn(p, y, t_img):
            return apply_fn_inner({"params": p}, y, t_img), {}
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
            anchor_log_w=log_w,
            anchor_table=anchor_table_E,
            prior_strength=0.0,
            rb_weight=1.0,
            rb_share_sample=True,
            time_sampling="uniform",
            t_floor=T_FLOOR,
        )
        return loss, metrics

    grad_fn = jax.value_and_grad(_loss_pure, argnums=0, has_aux=True)

    def train_step(state, key, x0_idx, x0_anchor, optimizer):
        (loss, metrics), grads = grad_fn(
            (state["theta"], state["log_w"]), key, x0_idx, x0_anchor
        )
        grad_theta, grad_log_w = grads
        merged_grads = {"theta": grad_theta, "log_w": grad_log_w}
        merged_params = {"theta": state["theta"], "log_w": state["log_w"]}
        updates, new_opt_state = optimizer.update(
            merged_grads, state["opt_state"], merged_params
        )
        new_params = optax.apply_updates(merged_params, updates)
        new_state = {
            "theta": new_params["theta"],
            "log_w": new_params["log_w"],
            "opt_state": new_opt_state,
        }
        # Grad norm on log_w (the optimizer-visible signal)
        log_w_grad_norm = jnp.linalg.norm(grad_log_w)
        return new_state, loss, metrics, log_w_grad_norm

    return train_step


# ---------------------------- main ----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--eval-every", type=int, default=500,
        help="Eval cadence in steps (also CSV-row cadence).",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"recovery_eta1_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    # --- schedules ---
    beta = make_beta(BETA_MIN, BETA_MAX, T=T_TERMINAL)
    hazard = make_hazard_poly_alpha(beta, p=HAZARD_P)
    jump = VPMatchedGaussianJump(beta=beta, eta=1.0)
    anchor_table_E = make_anchor_table()  # (K, D), frozen

    log_w_true = jnp.log(W_TRUE)  # (K,)
    log_w_true_centered = log_w_true - jnp.mean(log_w_true)
    log_w_true_norm = float(jnp.linalg.norm(log_w_true_centered))
    print(f"w_true = {np.asarray(W_TRUE).tolist()}", flush=True)
    print(f"log_w_true_centered = {np.asarray(log_w_true_centered).tolist()}",
          flush=True)
    print(f"||log_w_true_centered|| = {log_w_true_norm:.4f}", flush=True)

    # --- denoiser + log_w init ---
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)
    denoiser = Denoiser(K_=K, hidden=HIDDEN, n_layers=3)
    dummy_y = jnp.zeros((1, L_SEQ, D), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    theta_init = denoiser.init(rng_init, dummy_y, dummy_t)["params"]
    log_w_init = jnp.zeros((K,), dtype=jnp.float32)

    state_pre = {"theta": theta_init, "log_w": log_w_init}
    optimizer = init_optimizer(state_pre)
    opt_state = optimizer.init({"theta": theta_init, "log_w": log_w_init})
    state = {"theta": theta_init, "log_w": log_w_init, "opt_state": opt_state}

    n_params = sum(
        int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(theta_init)
    )
    print(f"denoiser params: {n_params}", flush=True)

    # --- fixed training pool ---
    rng, rng_pool = jax.random.split(rng)
    x0_pool = jax.random.randint(
        rng_pool, shape=(N_TRAIN, L_SEQ), minval=0, maxval=K, dtype=jnp.int32
    )
    rng, rng_eval = jax.random.split(rng)
    x0_eval_idx = jax.random.randint(
        rng_eval, shape=(N_EVAL, L_SEQ), minval=0, maxval=K, dtype=jnp.int32
    )
    x0_eval_anchor = jnp.take(anchor_table_E, x0_eval_idx, axis=0)

    # --- compile train step + eval-loss fn ---
    train_step_raw = make_train_step(
        denoiser, beta, hazard, jump, anchor_table_E, T_TERMINAL
    )
    train_step = jax.jit(
        lambda s, k, idx, anc: train_step_raw(s, k, idx, anc, optimizer)
    )

    def _eval_loss(theta, log_w, key, x0_idx, x0_anchor):
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
            T=T_TERMINAL,
            anchor_log_w=log_w,
            anchor_table=anchor_table_E,
            prior_strength=0.0,
            rb_weight=1.0,
            rb_share_sample=True,
            time_sampling="uniform",
            t_floor=T_FLOOR,
        )
        return loss, metrics

    eval_loss = jax.jit(_eval_loss)

    # --- training loop ---
    csv_path = out_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step",
        "log_w_hat_0", "log_w_hat_1", "log_w_hat_2", "log_w_hat_3", "log_w_hat_4",
        "log_w_hat_c0", "log_w_hat_c1", "log_w_hat_c2", "log_w_hat_c3",
        "log_w_hat_c4",
        "L2_err", "cosine", "per_anchor_max_err",
        "loss_ce", "loss_rb", "loss_score", "loss_total",
        "log_w_grad_norm",
    ])
    csv_file.flush()

    log_w_history = [np.asarray(log_w_init).tolist()]
    log_w_history_steps = [0]

    print(f"\nstarting training: {args.steps} steps, batch={args.batch}",
          flush=True)
    t0 = time.perf_counter()
    t_first100 = None

    for step in range(1, args.steps + 1):
        rng, rng_batch, rng_step = jax.random.split(rng, 3)
        batch_idx = jax.random.randint(
            rng_batch, shape=(args.batch,), minval=0, maxval=N_TRAIN,
        )
        x0_idx = x0_pool[batch_idx]  # (B, L_seq)
        x0_anchor = jnp.take(anchor_table_E, x0_idx, axis=0)

        state, loss, metrics, log_w_grad_norm = train_step(
            state, rng_step, x0_idx, x0_anchor
        )

        if step == 100:
            t_first100 = time.perf_counter() - t0
            sps = 100 / t_first100
            eta = (args.steps - 100) / sps
            print(
                f"  step {step}: loss={float(loss):.4f}, "
                f"{sps:.1f} steps/sec, ETA {eta/60:.1f} min", flush=True
            )

        if step % 100 == 0:
            log_w_hat = np.asarray(state["log_w"])
            log_w_history.append(log_w_hat.tolist())
            log_w_history_steps.append(step)

        if step % args.eval_every == 0 or step == args.steps:
            log_w_hat = state["log_w"]
            log_w_hat_centered = log_w_hat - jnp.mean(log_w_hat)
            l2_err = float(jnp.linalg.norm(
                log_w_hat_centered - log_w_true_centered
            ))
            cos = float(
                jnp.dot(log_w_hat_centered, log_w_true_centered)
                / (jnp.linalg.norm(log_w_hat_centered)
                   * jnp.linalg.norm(log_w_true_centered) + 1e-12)
            )
            per_anchor_err = jnp.abs(log_w_hat_centered - log_w_true_centered)
            max_err = float(jnp.max(per_anchor_err))

            # eval losses on the held-out eval batch
            rng, rng_eval_key = jax.random.split(rng)
            eval_total, eval_m = eval_loss(
                state["theta"], state["log_w"], rng_eval_key,
                x0_eval_idx, x0_eval_anchor,
            )

            row = [
                step,
                *np.asarray(log_w_hat).tolist(),
                *np.asarray(log_w_hat_centered).tolist(),
                l2_err, cos, max_err,
                float(eval_m["loss/ce"]),
                float(eval_m["loss/rb"]),
                float(eval_m["loss/score"]),
                float(eval_total),
                float(log_w_grad_norm),
            ]
            csv_writer.writerow(row)
            csv_file.flush()

            print(
                f"  step {step}: loss={float(eval_total):.4f} "
                f"(ce={float(eval_m['loss/ce']):.4f}, "
                f"rb={float(eval_m['loss/rb']):.4f}, "
                f"score={float(eval_m['loss/score']):.4f}) | "
                f"L2={l2_err:.4f}, cos={cos:.4f}, "
                f"max_err={max_err:.4f}, "
                f"||grad_log_w||={float(log_w_grad_norm):.4g}",
                flush=True,
            )

            if not bool(jnp.isfinite(eval_total)):
                print(f"  !! NaN/Inf in eval loss at step {step}; stopping early",
                      flush=True)
                break

    wall_time = time.perf_counter() - t0
    final_steps = step
    sps_overall = final_steps / wall_time
    print(f"\nwall time: {wall_time:.1f}s ({sps_overall:.1f} steps/sec)",
          flush=True)

    csv_file.close()

    # --- final metrics ---
    log_w_hat = state["log_w"]
    log_w_hat_centered = log_w_hat - jnp.mean(log_w_hat)
    l2_err = float(jnp.linalg.norm(log_w_hat_centered - log_w_true_centered))
    cos = float(
        jnp.dot(log_w_hat_centered, log_w_true_centered)
        / (jnp.linalg.norm(log_w_hat_centered)
           * jnp.linalg.norm(log_w_true_centered) + 1e-12)
    )
    per_anchor_err = np.asarray(
        jnp.abs(log_w_hat_centered - log_w_true_centered)
    )
    max_err = float(np.max(per_anchor_err))

    # Baseline (no learning): log_w_hat stayed at zero
    baseline_centered = jnp.zeros_like(log_w_true_centered)
    base_l2 = float(jnp.linalg.norm(baseline_centered - log_w_true_centered))
    base_cos = 0.0  # zero vector → cosine undefined; report 0

    recovered = (l2_err < 0.15) and (cos > 0.95)

    summary = {
        "recovered": bool(recovered),
        "l2_err": l2_err,
        "cosine": cos,
        "per_anchor_err": per_anchor_err.tolist(),
        "per_anchor_max_err": max_err,
        "log_w_hat": np.asarray(log_w_hat).tolist(),
        "log_w_hat_centered": np.asarray(log_w_hat_centered).tolist(),
        "log_w_true_centered": np.asarray(log_w_true_centered).tolist(),
        "baseline_l2_err": base_l2,
        "baseline_cosine": base_cos,
        "wall_time_sec": wall_time,
        "steps": int(final_steps),
        "steps_per_sec": sps_overall,
        "thresholds": {"l2": 0.15, "cosine": 0.95},
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- plot log_w trajectory ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = np.array(log_w_history)  # (n_evals, K)
        steps_arr = np.array(log_w_history_steps)
        # Recenter at each step for comparability with the dashed true lines.
        hist_centered = hist - hist.mean(axis=1, keepdims=True)
        log_w_true_centered_np = np.asarray(log_w_true_centered)

        fig, ax = plt.subplots(figsize=(8, 5))
        for a in range(K):
            color = f"C{a}"
            ax.plot(steps_arr, hist_centered[:, a], color=color,
                    label=f"log_w_hat[{a}] (w_true={float(W_TRUE[a]):.2f})")
            ax.axhline(log_w_true_centered_np[a], color=color, linestyle="--",
                       alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("log_w (recentered to mean 0)")
        ax.set_title(
            f"e2e ELBO hazard recovery (RECOVERED={recovered}, "
            f"L2={l2_err:.3f}, cos={cos:.3f})"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "log_w_trajectory.png", dpi=140)
        plt.close(fig)
        png_msg = str(out_dir / "log_w_trajectory.png")
    except Exception as e:  # noqa: BLE001
        png_msg = f"(plot skipped: {e!r})"

    # --- report ---
    print("\n" + "=" * 72)
    status = "PASSED" if recovered else "FAILED"
    print(
        f"RECOVERY TEST: {status}   "
        f"(L2 = {l2_err:.4f}, cos = {cos:.4f}, "
        f"per-anchor max err = {max_err:.4f})"
    )
    print(
        f"Baseline (no learning):  L2 = {base_l2:.4f}, cos = {base_cos:.4f}"
    )
    print("=" * 72)
    print(f"  csv: {csv_path}")
    print(f"  json: {out_dir / 'summary.json'}")
    print(f"  png: {png_msg}")

    return 0 if recovered else 1


if __name__ == "__main__":
    sys.exit(main())
