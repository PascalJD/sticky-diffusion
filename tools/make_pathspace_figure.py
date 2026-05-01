"""Create a 3-panel path-space figure.

Panels:
  (1) Purely discrete (masked) model on a finite state space (cadlag path).
  (2) Continuous diffusion path that requires a rounding / decoding step.
  (3) Sticky Jump Diffusion (SJD): continuous evolution + late sticking event.

The continuous and SJD trajectories are built from forward-time VP corruptions,
then displayed on a reverse-time axis to match the paper narrative.

Example:
  python tools/make_pathspace_figure.py --out outputs/pathspaces_toy1d.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


_PANEL_FACECOLOR = "#eef2f7"
_PATHSPACE_DARK = "#1f4e79"
_PATHSPACE_CMAP = LinearSegmentedColormap.from_list(
    "pathspace_blue", [_PANEL_FACECOLOR, _PATHSPACE_DARK]
)
_TRAJ_COLOR = "#1f4e79"


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",

    # Bigger fonts for NeurIPS readability
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    # Slightly thicker lines (survive downscaling)
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.0,

    # Improve PDF output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass(frozen=True)
class Toy1DDiscreteSpec:
    anchors: Tuple[float, ...]
    probs: Tuple[float, ...]


def default_toy1d_spec() -> Toy1DDiscreteSpec:
    return Toy1DDiscreteSpec(
        anchors=(-2.0, -1.0, 0.0, 1.0, 2.0),
        probs=(0.08, 0.16, 0.52, 0.16, 0.08),
    )


@dataclass(frozen=True)
class VPSchedule:
    """Simple VP schedule with linear beta(t) on [0, T]."""

    T: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 12.0

    def beta(self, t: np.ndarray) -> np.ndarray:
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T)

    def _int_beta(self, t: np.ndarray) -> np.ndarray:
        b0, b1 = self.beta_min, self.beta_max
        return b0 * t + (b1 - b0) * (t * t) / (2.0 * self.T)

    def alpha_sigma(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ib = self._int_beta(t)
        alpha = np.exp(-0.5 * ib)
        sigma2 = np.clip(1.0 - alpha * alpha, 1e-12, None)
        sigma = np.sqrt(sigma2)
        return alpha, sigma

    def step_alpha_sigma(self, t0: float, t1: float) -> Tuple[float, float]:
        ib = float(
            self._int_beta(np.asarray([t1], dtype=np.float64))[0]
            - self._int_beta(np.asarray([t0], dtype=np.float64))[0]
        )
        alpha = float(np.exp(-0.5 * ib))
        sigma2 = float(np.clip(1.0 - alpha * alpha, 1e-12, None))
        return alpha, float(np.sqrt(sigma2))


def simulate_forward_vp_path(
    *,
    rng: np.random.Generator,
    vp: VPSchedule,
    t_grid: np.ndarray,
    init: float,
) -> np.ndarray:
    """Exact Gaussian transitions for the forward-time VP corruption."""
    y = np.zeros_like(t_grid)
    y[0] = init
    for i in range(len(t_grid) - 1):
        alpha_step, sigma_step = vp.step_alpha_sigma(float(t_grid[i]), float(t_grid[i + 1]))
        y[i + 1] = alpha_step * y[i] + sigma_step * rng.normal()
    return y


def reverse_for_plot(
    t_forward: np.ndarray,
    y_forward: np.ndarray,
    *,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a forward-time path onto the paper's reverse-time axis."""
    tau = T - t_forward[::-1]
    return tau, y_forward[::-1]


def simulate_discrete_cadlag_path(
    *,
    T: float,
    mask_y: float,
    target_y: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A cadlag path on a purely discrete (masked) state space with *one* unmask event.

    This avoids token-to-token transitions and avoids remasking, matching the
    most common "monotone unmasking" depiction.
    """
    rng = np.random.default_rng(seed)

    # One unmask time (slight jitter for aesthetics).
    tau_unmask = float(0.65 * T + rng.normal(scale=0.03 * T))
    tau_unmask = float(np.clip(tau_unmask, 0.15 * T, 0.95 * T))

    tau = np.asarray([0.0, tau_unmask, T], dtype=np.float64)
    y = np.asarray([mask_y, target_y, target_y], dtype=np.float64)
    return tau, y


def render_vp_pathspace_background(
    ax: plt.Axes,
    *,
    spec: Toy1DDiscreteSpec,
    vp: VPSchedule,
    n_steps: int,
    y_min: float,
    y_max: float,
    n_y: int = 360,
    t_floor: float = 0.01,
    alpha: float = 0.55,
    temperature: float = 2.0,
    gamma: float = 1.6,
):
    """Render a light VP path space background as a time-space density map."""

    T = vp.T
    tau = np.linspace(0.0, T, n_steps + 1)
    t = np.clip(T - tau, t_floor, T)
    alpha_t, sigma_t = vp.alpha_sigma(t)

    anchors = np.asarray(spec.anchors, dtype=np.float64)
    p0 = np.asarray(spec.probs, dtype=np.float64)

    # Grid in y for the heatmap.
    y_grid = np.linspace(y_min, y_max, n_y, dtype=np.float64)

    # Compute p_t(y) = sum_a p0(a) N(y; alpha(t)a, sigma^2(t)).
    y = y_grid[:, None, None]
    mean = alpha_t[None, :, None] * anchors[None, None, :]
    sigma2 = (sigma_t[None, :, None] ** 2)
    log_pdf = -0.5 * (np.log(2.0 * np.pi * sigma2) + (y - mean) ** 2 / sigma2)
    pdf = np.exp(log_pdf)
    dens = np.sum(p0[None, None, :] * pdf, axis=2)  # (n_y, n_tau)

    logd = np.log(dens + 1e-30)
    logd = logd - np.max(logd, axis=0, keepdims=True)

    img = np.exp(logd / max(float(temperature), 1e-6))
    img = np.clip(img, 0.0, 1.0) ** max(float(gamma), 1e-6)

    ax.imshow(
        img,
        extent=[0.0, T, y_min, y_max],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap=_PATHSPACE_CMAP,
        alpha=alpha,
        zorder=0,
    )


def render_masked_pathspace_background(
    ax: plt.Axes,
    *,
    spec: Toy1DDiscreteSpec,
    T: float,
    mask_y: float,
    n_steps: int,
    y_min: float,
    y_max: float,
    n_y: int = 360,
    band_half_width: float = 0.18,
    alpha: float = 0.65,
):
    """Discrete masked-diffusion path-space background.

    At reverse time tau, P(MASK) = 1 - tau/T and P(a_k) = (tau/T) * p_data(a_k).
    Renders thin horizontal bands centered on each anchor row and on the MASK
    row, intensity proportional to the marginal mass at that tau.
    """
    anchors = np.asarray(spec.anchors, dtype=np.float64)
    p0 = np.asarray(spec.probs, dtype=np.float64)
    p0 = p0 / p0.sum()

    tau = np.linspace(0.0, T, n_steps + 1)
    p_unmask = tau / T
    p_mask = 1.0 - p_unmask

    rows_y = np.concatenate([anchors, [mask_y]])  # shape (K+1,)
    rows_p = np.concatenate(
        [p_unmask[None, :] * p0[:, None], p_mask[None, :]], axis=0
    )  # (K+1, n_tau)

    y_grid = np.linspace(y_min, y_max, n_y)
    img = np.zeros((n_y, len(tau)), dtype=np.float64)
    for ry, rp in zip(rows_y, rows_p):
        mask_band = np.abs(y_grid - ry) <= band_half_width
        if not mask_band.any():
            continue
        img[mask_band, :] = np.maximum(img[mask_band, :], rp[None, :])

    ax.imshow(
        img,
        extent=[0.0, T, y_min, y_max],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=_PATHSPACE_CMAP,
        alpha=alpha,
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )


_OK_COLOR = "#2ca02c"   # Matplotlib default green
_BAD_COLOR = "#d62728"  # Matplotlib default red


def _add_checkcross_row(
    ax: plt.Axes,
    *,
    x_icon: float,
    y: float,
    ok: bool,
    label: str,
    icon_size: float = 120.0,   # points^2 (scatter size)
    text_dx: float = 0.07,      # in Axes fraction units
    fontsize: float = 10.5,
    label_box: bool = False,
):
    """Draw one row: colored circle icon + '✓/✗' + label, in axes coordinates."""
    color = _OK_COLOR if ok else _BAD_COLOR
    mark = "✓" if ok else "✗"

    ax.scatter(
        [x_icon],
        [y],
        s=icon_size,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
        zorder=50,
    )

    # White mark on top of the circle.
    ax.text(
        x_icon,
        y,
        mark,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        clip_on=False,
        zorder=51,
    )

    bbox = None
    if label_box:
        bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", lw=0.8, alpha=0.95)

    ax.text(
        x_icon + text_dx,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="black",
        fontsize=fontsize,
        bbox=bbox,
        clip_on=False,
        zorder=51,
    )


def add_method_property_flags(
    ax: plt.Axes,
    *,
    uses_continuous_gradients: bool,
    outputs_discrete_tokens: bool,
    x_icon: float = 0.06,
    y0: float = -0.28,
    dy: float = -0.13,
    icon_size: float = 120.0,
    fontsize: float = 10.5,
):
    """Add the 2-row property block under a panel."""
    _add_checkcross_row(
        ax,
        x_icon=x_icon,
        y=y0,
        ok=uses_continuous_gradients,
        label="Uses Continuous Gradients",
        icon_size=icon_size,
        fontsize=fontsize,
    )
    _add_checkcross_row(
        ax,
        x_icon=x_icon,
        y=y0 + dy,
        ok=outputs_discrete_tokens,
        label="Outputs Discrete Tokens",
        icon_size=icon_size,
        fontsize=fontsize,
        label_box=False,
    )


def plot_data_pmf(
    ax: plt.Axes,
    *,
    anchors: np.ndarray,
    probs: np.ndarray,
    y_min: float,
    y_max: float,
):
    """Shared marginal axis: a clear PMF (mass function) over discrete tokens."""
    anchors = np.asarray(anchors, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.sum(probs)

    ax.barh(
        anchors,
        probs,
        height=0.32,
        color=_TRAJ_COLOR,
        alpha=0.9,
        edgecolor="none",
        zorder=5,
    )

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0.0, float(np.max(probs) * 1.15))

    ax.set_title("Data PMF")
    # ax.set_xlabel("PMF")

    ax.set_yticks(list(anchors))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)

    ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    # Make it feel like a marginal plot.
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def make_figure(
    *,
    out_path: str,
    seed: int = 0,
    target_k: int = 3,
    n_steps: int = 128,
    T: float = 1.0,
    cont_terminal_frac: float = 0.58,
    sjd_hold_frac: float = 0.16,
    sjd_jump_frac: float = 0.22,
    draw_flags: bool = True,
):
    spec = default_toy1d_spec()
    vp = VPSchedule(T=T)
    anchors = np.asarray(spec.anchors)

    target_k = int(np.clip(target_k, 0, len(anchors) - 1))
    target_y = float(anchors[target_k])
    mask_y = float(np.max(anchors) + 1.2)

    neighbor_k = target_k - 1 if target_k > 0 else min(target_k + 1, len(anchors) - 1)
    neighbor_y = float(anchors[neighbor_k])
    lo_y = min(target_y, neighbor_y)
    hi_y = max(target_y, neighbor_y)

    t_forward = np.linspace(0.0, T, n_steps + 1)

    # ------------------------------------------------------------------
    # Continuous panel: choose an ambiguous off-token endpoint, then
    # simulate its forward VP corruption and plot it in reverse time.
    # ------------------------------------------------------------------
    cont_fraction = cont_terminal_frac if target_y >= neighbor_y else 1.0 - cont_terminal_frac
    cont_terminal_y = float(lo_y + cont_fraction * (hi_y - lo_y))
    rng_cont = np.random.default_rng(seed * 10_000 + 17)
    y_cont_forward = simulate_forward_vp_path(
        rng=rng_cont,
        vp=vp,
        t_grid=t_forward,
        init=cont_terminal_y,
    )
    tau_cont, y_cont = reverse_for_plot(t_forward, y_cont_forward, T=T)
    y_end = cont_terminal_y
    y_round = target_y

    # ------------------------------------------------------------------
    # SJD panel: hold on-anchor in forward time, jump off-anchor once,
    # then continue with forward VP corruption and plot in reverse time.
    # ------------------------------------------------------------------
    jump_idx = int(np.clip(round(sjd_hold_frac * n_steps), 1, n_steps - 2))
    jump_time = float(t_forward[jump_idx])
    sjd_fraction = sjd_jump_frac if target_y >= neighbor_y else 1.0 - sjd_jump_frac
    sjd_y_prejump = float(lo_y + sjd_fraction * (hi_y - lo_y))

    rng_sjd = np.random.default_rng(seed * 20_000 + 19)
    y_sjd_forward = simulate_forward_vp_path(
        rng=rng_sjd,
        vp=vp,
        t_grid=t_forward[jump_idx:],
        init=sjd_y_prejump,
    )
    tau_sjd_diff, y_sjd_diff = reverse_for_plot(t_forward[jump_idx:], y_sjd_forward, T=T)
    sjd_commit_tau = float(T - jump_time)
    sjd_k = target_k
    tau_sjd = np.concatenate([tau_sjd_diff[:-1], [sjd_commit_tau, T]])
    y_sjd = np.concatenate([y_sjd_diff[:-1], [target_y, target_y]])

    # ------------------------------------------------------------------
    # Discrete masked panel: one MASK -> token unmask event.
    # ------------------------------------------------------------------
    tau_mask, y_mask = simulate_discrete_cadlag_path(
        T=T,
        mask_y=mask_y,
        target_y=target_y,
        seed=seed + 123,
    )

    # ------------------------------------------------------------------
    # Layout: 3 panels + 1 shared narrow PMF marginal axis on the right.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 4,
        figsize=(9.6, 3.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.27]},
    )

    ax0, ax1, ax2, ax_pmf = axes

    y_min = float(np.min(anchors) - 1.0)
    y_max = mask_y + 0.8

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0.0, T)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("reverse time  $\\tau$")
        ax.title.set_fontweight("bold")
        ax.set_facecolor(_PANEL_FACECOLOR)

    # Discrete (token + MASK) path-space background for panel 1.
    render_masked_pathspace_background(
        ax0,
        spec=spec,
        T=T,
        mask_y=mask_y,
        n_steps=n_steps,
        y_min=y_min,
        y_max=y_max,
    )

    # Continuous VP path-space background for panels (2) and (3).
    render_vp_pathspace_background(
        ax1, spec=spec, vp=vp, n_steps=n_steps, y_min=y_min, y_max=y_max
    )
    render_vp_pathspace_background(
        ax2, spec=spec, vp=vp, n_steps=n_steps, y_min=y_min, y_max=y_max
    )

    # Panel-specific y ticks.
    ax0.set_yticks(list(anchors) + [mask_y])
    ax0.set_yticklabels([str(int(a)) for a in anchors] + ["MASK"])
    for ax in (ax1, ax2):
        ax.set_yticks(list(anchors))
        ax.set_yticklabels([str(int(a)) for a in anchors])

    # ------------------------------------------------------------------
    # Panel 1: discrete + mask path space (single unmask).
    # ------------------------------------------------------------------
    for a in anchors:
        ax0.axhline(a, linewidth=0.8, alpha=0.25)
    ax0.axhline(mask_y, linewidth=1.0, alpha=0.35)
    ax0.step(tau_mask, y_mask, where="post", linewidth=2.3, color=_TRAJ_COLOR)
    ax0.set_title("Masked diffusion")

    # ------------------------------------------------------------------
    # Panel 2: continuous path space + end rounding.
    # ------------------------------------------------------------------
    for a in anchors:
        ax1.axhline(a, linewidth=0.8, alpha=0.2, linestyle="--")
    ax1.plot(tau_cont, y_cont, linewidth=2.3, color=_TRAJ_COLOR)
    ax1.plot([T, T], [y_end, y_round], linestyle="-", linewidth=4, color="red")
    ax1.annotate(
        "projection required",
        xy=(T, 0.5 * (y_end + y_round)),
        xycoords="data",
        xytext=(-18, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9),
        zorder=11,
        clip_on=False,
    )
    ax1.set_title("Continuous diffusion")

    # ------------------------------------------------------------------
    # Panel 3: SJD path space + explicit jump (vertical line).
    # ------------------------------------------------------------------
    for a in anchors:
        ax2.axhline(a, linewidth=0.7, alpha=0.15, linestyle="--")

    if np.isfinite(sjd_commit_tau) and np.isfinite(sjd_y_prejump) and (sjd_k >= 0):
        y_anchor = float(anchors[sjd_k])
        commit_idx = int(np.argmin(np.abs(tau_sjd - sjd_commit_tau)))

        # Pre-commit segment + point at commit time with the pre-jump value.
        tau_pre = np.concatenate([tau_sjd[:commit_idx], [sjd_commit_tau]])
        y_pre = np.concatenate([y_sjd[:commit_idx], [sjd_y_prejump]])
        ax2.plot(tau_pre, y_pre, linewidth=2.3, color=_TRAJ_COLOR)

        # Jump (vertical line at commit time).
        ax2.plot(
            [sjd_commit_tau, sjd_commit_tau],
            [sjd_y_prejump, y_anchor],
            linewidth=2,
            color=_TRAJ_COLOR,
            zorder=5,
        )
        y_mid = 0.5 * (sjd_y_prejump + y_anchor)
        ax2.annotate(
            "jump",
            xy=(sjd_commit_tau, y_mid),          # point you are labeling (on the jump)
            xycoords="data",
            xytext=(-18, 12),                    # center-left offset in *points*
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", lw=0.8, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9),  # small leader line
            zorder=11,
            clip_on=False,
        )

        # Post-commit sticky segment.
        ax2.plot(tau_sjd[commit_idx:], y_sjd[commit_idx:], linewidth=2.3, color=_TRAJ_COLOR)
    else:
        ax2.plot(tau_sjd, y_sjd, linewidth=2.3, color=_TRAJ_COLOR)

    ax2.set_title("Sticky Jump Diffusion")

    # ------------------------------------------------------------------
    # Shared marginal axis: data PMF over tokens.
    # ------------------------------------------------------------------
    plot_data_pmf(ax_pmf, anchors=anchors, probs=np.asarray(spec.probs), y_min=y_min, y_max=y_max)

    # ------------------------------------------------------------------
    # Property flags.
    # ------------------------------------------------------------------
    if draw_flags:
        add_method_property_flags(
            ax0,
            uses_continuous_gradients=False,
            outputs_discrete_tokens=True,
        )
        add_method_property_flags(
            ax1,
            uses_continuous_gradients=True,
            outputs_discrete_tokens=False,
        )
        add_method_property_flags(
            ax2,
            uses_continuous_gradients=True,
            outputs_discrete_tokens=True,
        )

    out_file = Path(out_path).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/pathspaces_toy1d.pdf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_k", type=int, default=3)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--no_flags", action="store_true", help="Disable ✓/✗ property flags.")
    args = p.parse_args()

    make_figure(
        out_path=args.out,
        seed=args.seed,
        target_k=args.target_k,
        n_steps=args.steps,
        T=args.T,
        draw_flags=(not args.no_flags),
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
