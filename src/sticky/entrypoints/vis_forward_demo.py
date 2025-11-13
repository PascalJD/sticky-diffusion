from __future__ import annotations
from typing import Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sticky.core.sde_vp import make_beta, B_of_t
from sticky.core.hazard import make_hazard_early
from sticky.core.jump import GaussianJump
from sticky.core.eventgen import sample_forward_events

Array = jnp.ndarray


def vp_step_between(
    key: jax.random.PRNGKey,
    x_start: Array,
    t_start: Array,
    t_end: Array,
    beta, 
) -> Array:
    while t_start.ndim < x_start.ndim:
        t_start = t_start[..., None]
    while t_end.ndim < x_start.ndim:
        t_end = t_end[..., None]
    B_end = B_of_t(beta, t_end)
    B_start = B_of_t(beta, t_start)
    dB = jnp.maximum(0.0, B_end - B_start)
    rho = jnp.exp(-0.5 * dB)
    sig2 = jnp.clip(1.0 - jnp.exp(-dB), 1e-12)
    eps = jax.random.normal(key, shape=x_start.shape)
    return rho * x_start + jnp.sqrt(sig2) * eps


def simulate_paths(
    key: jax.random.PRNGKey,
    anchors_support: Array,
    anchors_probs: Array,
    B: int = 20,
    T: float = 1.0,
    n_steps: int = 200,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    kappa: float = 6.0,
    jump_std: float = 0.05,
):
    # 1) Sample anchors from histogram
    kA, krest = jax.random.split(key)
    idx = jax.random.categorical(kA, jnp.log(anchors_probs), shape=(B,))
    anchors = anchors_support[idx].astype(jnp.float32)
    anchors = anchors[:, None]

    # 2) Make schedules & kernels
    beta = make_beta(beta_min=beta_min, beta_max=beta_max, T=T)
    hazard = make_hazard_early(beta, kappa=kappa)
    jump = GaussianJump(std=jump_std)

    # 3) Sample first-event times & jump draws using your eventgen
    k_ev, k_rest = jax.random.split(krest)
    t_event, y, has_event = sample_forward_events(k_ev, anchors, hazard, jump)

    # 4) Time grid and scan carry
    tgrid = jnp.linspace(0.0, T, n_steps + 1)
    x0 = anchors
    has0 = jnp.zeros((B,), dtype=bool)
    carry0 = (x0, jnp.array(0.0, dtype=jnp.float32), has0, t_event, y, k_rest)

    def evolve_one(carry, tn):
        x_prev, t_prev, has_jumped, t_e, y, key = carry
        k0, k2, key_next = jax.random.split(key, 3)

        # Will the first event occur in (t_prev, tn] and not yet jumped?
        will_jump = (~has_jumped) & (t_e > t_prev) & (t_e <= tn)

        # Paths that already jumped: diffuse (t_prev -> tn); else stay stuck
        x_cont = vp_step_between(k0, x_prev, t_prev, tn, beta)
        x_nojump = jnp.where(has_jumped[:, None], x_cont, x_prev)

        # For those that jump in this interval: jump at t_e, then diffuse (t_e -> tn)
        te_clamped = jnp.minimum(t_e, tn)
        x_after_jump = vp_step_between(k2, y, te_clamped, tn, beta)

        # Select per-path
        x_next = jnp.where(will_jump[:, None], x_after_jump, x_nojump)
        has_next = has_jumped | will_jump

        return (x_next, tn, has_next, t_e, y, key_next), x_next

    # 5) Scan forward in time
    carry_final, xs_tail = jax.lax.scan(evolve_one, carry0, tgrid[1:])
    xs = jnp.concatenate([x0[None, ...], xs_tail], axis=0)

    # Pack outputs
    sim = {
        "tgrid": tgrid,
        "X": xs,
        "anchors": anchors.squeeze(-1),
        "t_event": t_event,
        "has_event": has_event,
        "T": T,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "kappa": kappa,
        "jump_std": jump_std,
    }
    return sim


def plot_trajectories(sim, n_show: int = 24):
    t = np.asarray(sim["tgrid"])
    X = np.asarray(sim["X"])
    anchors = np.asarray(sim["anchors"])
    t_evt = np.asarray(sim["t_event"])
    B = X.shape[1]

    show_idx = np.random.choice(B, size=min(n_show, B), replace=False)

    plt.figure(figsize=(10, 5))
    for i in show_idx:
        plt.plot(t, X[:, i, 0], lw=1.0, alpha=0.8)
        if np.isfinite(t_evt[i]):
            plt.axvline(t_evt[i], color="k", lw=0.5, ls="--", alpha=0.35)
        plt.scatter([0.0], [anchors[i]], s=10, c="k", alpha=0.7)

    plt.title("Forward Sticky Jump Diffusion (vertical = unstick time)")
    plt.xlabel("time t")
    plt.ylabel("state X_t")
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_histograms(sim, times=(0.0, 0.1, 0.25, 0.5, 1.0)):
    tgrid = np.asarray(sim["tgrid"])
    X = np.asarray(sim["X"])

    idxs = [int(np.argmin(np.abs(tgrid - ts))) for ts in times]
    ncols = len(idxs)
    plt.figure(figsize=(2.8 * ncols, 3.0))

    for j, (ts, ix) in enumerate(zip(times, idxs), start=1):
        plt.subplot(1, ncols, j)
        plt.hist(X[ix, :, 0], bins=40, density=True, alpha=0.85)
        plt.title(f"t = {ts:.2f}")
        plt.xlabel("X_t")
        if j == 1:
            plt.ylabel("density")

    plt.suptitle("Snapshots of X_t under early-hazard SJD + VP")
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    # Histogram over anchors {-1, 0, 1, 2}
    anchors_support = jnp.array([-1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    anchors_probs = jnp.array([0.20, 0.30, 0.35, 0.15], dtype=jnp.float32)

    key = jax.random.PRNGKey(1)

    sim = simulate_paths(
        key=key,
        anchors_support=anchors_support,
        anchors_probs=anchors_probs,
        B=6,
        T=1.0,
        n_steps=512,
        beta_min=0.1,
        beta_max=20.0,
        kappa=2.0,  # larger => earlier unsticks
        jump_std=0.05,
    )

    SAVE_DIR = "content"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Plot 1: trajectories
    plot_trajectories(sim, n_show=20)
    traj_path = os.path.join(SAVE_DIR, "sjd_trajectories.png")
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {traj_path}")

    # # Plot 2: histograms
    # plot_histograms(sim, times=(0.0, 0.05, 0.15, 0.4, 1.0))
    # hist_path = os.path.join(SAVE_DIR, f"sjd_histograms.png")
    # plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"[saved] {hist_path}")


if __name__ == "__main__":
    main()