"""Create a 3-panel path-space figure.

Panels:
  (1) Purely discrete (masked) model on a finite state space (cadlag path).
  (2) Continuous diffusion path that requires a rounding / decoding step.
  (3) Sticky Jump Diffusion (SJD): continuous evolution + late sticking event.

This script is intentionally illustrative (toy dynamics) and is meant for
papers / slides.

Example:
  python scripts/make_pathspace_figure.py --out outputs/pathspaces_toy1d.png
"""

from __future__ import annotations

# Allow running directly via `python scripts/...` without installing the package.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sticky.data.toy1d_discrete import default_toy1d_spec, Toy1DDiscreteSpec


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",

    # Bigger fonts for NeurIPS readability
    "font.size": 11,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Slightly thicker lines (survive downscaling)
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.0,

    # Improve PDF output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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


def _log_normal_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


def posterior_pa_given_y(
    y: float,
    t: float,
    *,
    spec: Toy1DDiscreteSpec,
    vp: VPSchedule,
) -> np.ndarray:
    """Compute p(a | y, t) under the VP corruption: y ~ N(alpha(t) a, sigma^2(t))."""
    a = np.asarray(spec.anchors)
    p0 = np.asarray(spec.probs)
    alpha, sigma = vp.alpha_sigma(np.asarray([t], dtype=np.float64))
    alpha = float(alpha[0])
    sigma2 = float(sigma[0] ** 2)

    loglik = _log_normal_pdf(y, mean=alpha * a, var=sigma2)
    logp = np.log(p0 + 1e-30) + loglik
    logp = logp - np.max(logp)
    p = np.exp(logp)
    p = p / np.sum(p)
    return p


def score_mixture(y: float, t: float, *, spec: Toy1DDiscreteSpec, vp: VPSchedule) -> float:
    """Classifier-induced score for a mixture of Gaussians (1D)."""
    p = posterior_pa_given_y(y, t, spec=spec, vp=vp)
    a = np.asarray(spec.anchors)
    mu = float(np.sum(p * a))
    alpha, sigma = vp.alpha_sigma(np.asarray([t], dtype=np.float64))
    alpha = float(alpha[0])
    sigma2 = float(sigma[0] ** 2)
    return -(y - alpha * mu) / sigma2


def simulate_continuous_reverse_path(
    *,
    rng: np.random.Generator,
    spec: Toy1DDiscreteSpec,
    vp: VPSchedule,
    n_steps: int,
    init: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse-time VP SDE path on the continuous state space."""
    T = vp.T
    dt = T / n_steps
    tau = np.linspace(0.0, T, n_steps + 1)
    y = np.zeros_like(tau)
    y[0] = init
    for i in range(n_steps):
        t = T - tau[i]  # forward-time parameter
        bt = float(vp.beta(np.asarray([t]))[0])
        s = score_mixture(float(y[i]), t, spec=spec, vp=vp)
        drift = 0.5 * bt * y[i] + bt * s
        y[i + 1] = y[i] + drift * dt + np.sqrt(bt * dt) * rng.normal()
    return tau, y


def simulate_sjd_reverse_path(
    *,
    rng: np.random.Generator,
    spec: Toy1DDiscreteSpec,
    vp: VPSchedule,
    n_steps: int,
    init: float,
    hazard_scale: float = 4.0,
    eta: float = 0.55,
    commit_min_tau: float = 0.0,
    hazard_ramp_gamma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """Reverse-time toy SJD sampler for a *single* 1D site.

    We simulate a VP reverse SDE until a Poisson jump event occurs.
    When it jumps, we commit to an anchor and stay there.

    Returns:
        (tau, y_path, committed_idx, commit_tau, y_prejump)
        where committed_idx == -1 if no jump occurred.
    """
    T = vp.T
    dt = T / n_steps
    tau = np.linspace(0.0, T, n_steps + 1)
    y = np.zeros_like(tau)
    y[0] = init
    committed = False
    k_star = -1

    # For plotting a *vertical* jump line: store the pre-jump continuous value
    # at the commit time.
    commit_tau = float("nan")
    y_prejump = float("nan")

    anchors = np.asarray(spec.anchors)

    for i in range(n_steps):
        if committed:
            y[i + 1] = y[i]
            continue

        t = T - tau[i]

        # 1) Diffuse (same as continuous panel)
        bt = float(vp.beta(np.asarray([t]))[0])
        s = score_mixture(float(y[i]), t, spec=spec, vp=vp)
        drift = 0.5 * bt * y[i] + bt * s
        y_prop = y[i] + drift * dt + np.sqrt(bt * dt) * rng.normal()

        # 2) Jump (toy plug-in hazard)
        p = posterior_pa_given_y(float(y_prop), t, spec=spec, vp=vp)
        alpha, sigma = vp.alpha_sigma(np.asarray([t], dtype=np.float64))
        alpha = float(alpha[0])
        sigma2 = float(sigma[0] ** 2)

        resid2 = (y_prop - alpha * anchors) ** 2
        ratio = (1.0 / eta) * np.exp(-0.5 * ((1.0 / (eta * eta)) - 1.0) * resid2 / sigma2)
        per_anchor = ratio * p

        # Ramp the hazard so commits happen later in reverse time.
        ramp = float(np.clip(tau[i] / max(T, 1e-12), 0.0, 1.0) ** hazard_ramp_gamma)
        lam_total = hazard_scale * ramp * float(np.sum(per_anchor))
        p_jump = 1.0 - np.exp(-lam_total * dt)

        # Optional hard delay for commits (aesthetic control for the figure).
        allow_jump = tau[i] >= commit_min_tau

        if allow_jump and rng.uniform() < p_jump and lam_total > 0.0:
            per_anchor = per_anchor / np.sum(per_anchor)
            k_star = int(rng.choice(len(anchors), p=per_anchor))

            # Jump happens after the diffusion proposal (operator splitting):
            commit_tau = float(tau[i + 1])
            y_prejump = float(y_prop)

            y[i + 1] = float(anchors[k_star])
            committed = True
        else:
            y[i + 1] = y_prop

    return tau, y, k_star, commit_tau, y_prejump


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
    alpha: float = 0.18,
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
        cmap="Blues",
        alpha=alpha,
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
    icon_size: float = 95.0,    # points^2 (scatter size)
    text_dx: float = 0.07,      # in Axes fraction units
    fontsize: float = 8.5,
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
    decoding_model_implied: bool,
    x_icon: float = 0.06,
    y0: float = -0.34,
    dy: float = -0.115,
    icon_size: float = 95.0,
    fontsize: float = 8.5,
):
    """Add the 3-row property block under a panel."""
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
    )
    _add_checkcross_row(
        ax,
        x_icon=x_icon,
        y=y0 + 2 * dy,
        ok=decoding_model_implied,
        label="Token decoding is model implied",
        icon_size=icon_size,
        fontsize=fontsize,
        label_box=True,
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
        color="C0",
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
    cont_min_end_gap: float = 0.25,
    sjd_commit_min_frac: float = 0.75,
    draw_flags: bool = True,
):
    spec = default_toy1d_spec()
    vp = VPSchedule(T=T)

    rng = np.random.default_rng(seed)
    anchors = np.asarray(spec.anchors)

    target_k = int(np.clip(target_k, 0, len(anchors) - 1))
    target_y = float(anchors[target_k])

    # ------------------------------------------------------------------
    # Continuous panel: find a path that ends clearly between tokens.
    # ------------------------------------------------------------------
    mids = (anchors[:-1] + anchors[1:]) / 2.0
    lower = float(mids[target_k - 1]) if target_k > 0 else -np.inf
    upper = float(mids[target_k]) if target_k < len(anchors) - 1 else np.inf

    tau_cont = y_cont = None
    y_end = np.nan
    y_round = target_y
    for attempt in range(240):
        init_cand = float(rng.normal())
        rng_cont = np.random.default_rng(seed * 10_000 + attempt)
        tau_c, y_c = simulate_continuous_reverse_path(
            rng=rng_cont, spec=spec, vp=vp, n_steps=n_steps, init=init_cand
        )
        y_end_cand = float(y_c[-1])
        k_round = int(np.argmin(np.abs(anchors - y_end_cand)))
        if k_round != target_k:
            continue
        if not (lower < y_end_cand < upper):
            continue
        if abs(y_end_cand - target_y) < cont_min_end_gap:
            continue
        tau_cont, y_cont = tau_c, y_c
        y_end = y_end_cand
        y_round = float(anchors[k_round])
        break

    if tau_cont is None:
        rng_cont = np.random.default_rng(seed * 10_000 + 1)
        tau_cont, y_cont = simulate_continuous_reverse_path(
            rng=rng_cont, spec=spec, vp=vp, n_steps=n_steps, init=float(rng.normal())
        )
        y_end = float(y_cont[-1])
        y_round = float(anchors[int(np.argmin(np.abs(anchors - y_end)))])

    # If we still don't visually end "between tokens", nudge the endpoint.
    k_round = int(np.argmin(np.abs(anchors - y_end)))
    cont_ok = (k_round == target_k) and (lower < y_end < upper) and (abs(y_end - target_y) >= cont_min_end_gap)
    if not cont_ok:
        if np.isfinite(lower):
            desired_end = lower + 0.2 * (target_y - lower)
        elif np.isfinite(upper):
            desired_end = upper - 0.2 * (upper - target_y)
        else:
            desired_end = target_y + 0.5

        delta = float(desired_end - y_cont[-1])
        ramp = (tau_cont / max(T, 1e-12)) ** 2
        y_cont = y_cont + delta * ramp
        y_end = float(y_cont[-1])
        y_round = target_y

    # ------------------------------------------------------------------
    # SJD panel: prefer a late commit in reverse time.
    # ------------------------------------------------------------------
    tau_sjd = y_sjd = None  # type: ignore
    sjd_k = -1
    sjd_commit_tau = float("nan")
    sjd_y_prejump = float("nan")

    min_jump_height = 0.55  # 0.4–0.8 works well for unit-spaced tokens

    best = None
    best_jump = -np.inf

    for attempt in range(240):
        init_cand = float(rng.normal())
        rng_sjd = np.random.default_rng(seed * 20_000 + attempt)
        tau_j, y_j, k_j, commit_tau, y_prejump = simulate_sjd_reverse_path(
            rng=rng_sjd,
            spec=spec,
            vp=vp,
            n_steps=n_steps,
            init=init_cand,
            hazard_scale=8.0,
            eta=0.55,
            commit_min_tau=sjd_commit_min_frac * T,
            hazard_ramp_gamma=3.0,
        )
        if k_j != target_k:
            continue
        if not np.isfinite(commit_tau):
            continue

        jump_h = abs(float(y_prejump) - float(anchors[k_j]))
        if jump_h > best_jump:
            best_jump = jump_h
            best = (tau_j, y_j, k_j, commit_tau, y_prejump)

        if jump_h >= min_jump_height:
            tau_sjd, y_sjd, sjd_k, sjd_commit_tau, sjd_y_prejump = tau_j, y_j, k_j, commit_tau, y_prejump
            break

    # Fallback: take the “best available jump” if none cleared threshold
    if tau_sjd is None and best is not None:
        tau_sjd, y_sjd, sjd_k, sjd_commit_tau, sjd_y_prejump = best

    # ------------------------------------------------------------------
    # Discrete masked panel: one MASK -> token unmask event.
    # ------------------------------------------------------------------
    mask_y = float(np.max(anchors) + 1.2)
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
        figsize=(9.0, 2.65),
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

    # Light VP background for the continuous panels (2) and (3).
    render_vp_pathspace_background(
        ax1, spec=spec, vp=vp, n_steps=n_steps, y_min=y_min, y_max=y_max, alpha=0.3
    )
    render_vp_pathspace_background(
        ax2, spec=spec, vp=vp, n_steps=n_steps, y_min=y_min, y_max=y_max, alpha=0.3
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
    ax0.step(tau_mask, y_mask, where="post", linewidth=2.3)
    ax0.set_title("Masked diffusion")

    # ------------------------------------------------------------------
    # Panel 2: continuous path space + end rounding.
    # ------------------------------------------------------------------
    for a in anchors:
        ax1.axhline(a, linewidth=0.8, alpha=0.2, linestyle="--")
    ax1.plot(tau_cont, y_cont, linewidth=2.3)
    ax1.plot([T, T], [y_end, y_round], linestyle="-", linewidth=4, color="red")
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
        ax2.plot(tau_pre, y_pre, linewidth=2.3, color="C0")

        # Jump (vertical line at commit time).
        ax2.plot(
            [sjd_commit_tau, sjd_commit_tau],
            [sjd_y_prejump, y_anchor],
            linewidth=2,
            color="C0",
            zorder=5,
        )
        y_mid = 0.5 * (sjd_y_prejump + y_anchor)
        ax2.annotate(
            "jump",
            xy=(sjd_commit_tau, y_mid),          # point you are labeling (on the jump)
            xycoords="data",
            xytext=(18, 10),                    # (right, up) offset in *points*
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", lw=0.8, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9),  # small leader line
            zorder=11,
            clip_on=False,
        )

        # Post-commit sticky segment.
        ax2.plot(tau_sjd[commit_idx:], y_sjd[commit_idx:], linewidth=2.3)
    else:
        ax2.plot(tau_sjd, y_sjd, linewidth=2.3)

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
            decoding_model_implied=False,
        )
        add_method_property_flags(
            ax1,
            uses_continuous_gradients=True,
            outputs_discrete_tokens=False,
            decoding_model_implied=False,
        )
        add_method_property_flags(
            ax2,
            uses_continuous_gradients=True,
            outputs_discrete_tokens=True,
            decoding_model_implied=True,
        )

    fig.savefig(out_path, bbox_inches="tight", dpi=300)


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
