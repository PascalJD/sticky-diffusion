"""Forward/backward state-dependence figure for SJD (tight two-panel).

Correctness checklist:
  1. At η=1, γ=0 the heatmap column is uniformly white (ratio ≡ 1).
  2. Moving away from trivial knob, heatmap develops bilaterally structured
     red/blue bands aligned with modes of CIFAR-10's p0.
  3. Risk marginal is exactly zero at trivial knob and grows as knob moves away.
  4. NO annotation labels the trivial knob as "optimal" — it is simply a
     degenerate regime, not a virtue of the time-only schedule.
  5. Colorbar is shared across both panels and centered at 1.0.
  6. Two panels visually parallel (same heights, spine/tick style, marginal).
  7. Reads as "two expressions of the same principle" without verbal guidance.
"""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.special import logsumexp as _logsumexp
except ImportError:
    def _logsumexp(a, axis=None):
        a = np.asarray(a)
        amax = np.max(a, axis=axis, keepdims=True)
        amax_f = np.where(np.isfinite(amax), amax, 0.0)
        out = np.log(np.sum(np.exp(a - amax_f), axis=axis))
        if axis is None:
            return amax_f.squeeze() + out
        return np.squeeze(amax_f, axis=axis) + out


HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "_cifar10_pixel_p0.npy"
OUT_PDF = HERE / "forward_backward_state_dependence.pdf"
OUT_PNG = HERE / "forward_backward_state_dependence.png"
DATA_DIR = HERE.parent / "data"

ANCHORS = np.linspace(-1.0, 1.0, 256)
BETA_MIN = 0.1
BETA_MAX = 20.0
N_SAMPLES = 200_000

ETA_GRID = np.linspace(0.15, 1.0, 40)
GAMMA_GRID = np.linspace(-1.0, 4.0, 40)
Y_GRID = np.linspace(-2.8, 2.8, 280)

SLATE = "#4a5568"
CORAL = "#e07a5f"
DIV = "#333333"
VMAX_CAP = 5.0  # readability cap; literal 99%ile printed to stdout


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _read_cifar10_train(data_dir):
    bins = sorted(Path(data_dir).rglob("data_batch_[1-5].bin"))
    if len(bins) >= 5:
        imgs = []
        for p in bins[:5]:
            raw = np.fromfile(str(p), dtype=np.uint8).reshape(-1, 3073)
            imgs.append(raw[:, 1:])
        return np.concatenate(imgs, axis=0)
    pys = [p for p in sorted(Path(data_dir).rglob("data_batch_[1-5]")) if p.is_file()]
    if len(pys) >= 5:
        imgs = []
        for p in pys[:5]:
            with open(p, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            imgs.append(d[b"data"])
        return np.concatenate(imgs, axis=0)
    raise FileNotFoundError(f"No CIFAR-10 training batches found under {data_dir}")


def load_cifar10_prior(data_dir=None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if CACHE_PATH.exists():
        counts = np.load(CACHE_PATH).astype(np.float64)
    else:
        pixels = _read_cifar10_train(data_dir)
        counts = np.bincount(pixels.reshape(-1), minlength=256).astype(np.float64)
        np.save(CACHE_PATH, counts)
    counts = np.maximum(counts, 1e-12)
    p0 = counts / counts.sum()
    return p0, np.log(p0)


def vp_coefficients(t):
    t = np.asarray(t, dtype=np.float64)
    alpha = np.exp(-0.5 * (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t * t))
    sigma = np.sqrt(np.maximum(1.0 - alpha * alpha, 0.0))
    return alpha, sigma


def find_t_ref(tol=1e-10):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        a, s = vp_coefficients(mid)
        if a > s:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def log_lambda_of_a(log_p0, gamma):
    return -gamma * log_p0 - _logsumexp((1.0 - gamma) * log_p0)


def time_only_hazard(gamma, t, log_p0):
    log_lam = log_lambda_of_a(log_p0, gamma)
    lam = np.exp(log_lam)
    p0 = np.exp(log_p0)
    return float(np.sum(lam * p0 * np.exp(-lam * t)))


def hazard_ratio_grid(y_grid, eta, gamma, alpha, sigma, t, anchors, log_p0):
    """λ*_t(y) / λ*_time evaluated over y_grid. Fully vectorized (y × anchors)."""
    log_lam = log_lambda_of_a(log_p0, gamma)
    lam = np.exp(log_lam)
    s2 = sigma * sigma
    es2 = (eta * sigma) ** 2
    y = np.asarray(y_grid)
    sq = (y[:, None] - alpha * anchors[None, :]) ** 2
    log_q = -0.5 * np.log(2.0 * np.pi * s2) - sq / (2.0 * s2)
    log_r = -0.5 * np.log(2.0 * np.pi * es2) - sq / (2.0 * es2)
    log_num = _logsumexp(log_lam[None, :] + log_p0[None, :] - lam[None, :] * t + log_r, axis=1)
    log_den = _logsumexp(log_p0[None, :] + log_q, axis=1)
    return np.exp(log_num - log_den) / time_only_hazard(gamma, t, log_p0)


def sample_marginal_y(n, alpha, sigma, anchors, p0, rng):
    a_idx = rng.choice(len(anchors), size=n, p=p0)
    eps = rng.standard_normal(n)
    return alpha * anchors[a_idx] + sigma * eps


def build_heatmap(knob_grid, knob_name, y_grid, alpha, sigma, t, anchors, log_p0):
    out = np.empty((len(y_grid), len(knob_grid)), dtype=np.float64)
    for i, k in enumerate(knob_grid):
        eta, gamma = (float(k), 0.0) if knob_name == "eta" else (1.0, float(k))
        out[:, i] = hazard_ratio_grid(y_grid, eta, gamma, alpha, sigma, t, anchors, log_p0)
    return out


def build_variance_curve(y_samples, knob_grid, knob_name, alpha, sigma, t, anchors,
                         log_p0, chunk=20_000):
    out = np.empty(len(knob_grid))
    n = len(y_samples)
    for i, k in enumerate(knob_grid):
        eta, gamma = (float(k), 0.0) if knob_name == "eta" else (1.0, float(k))
        ratios = np.empty(n)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            ratios[s:e] = hazard_ratio_grid(y_samples[s:e], eta, gamma, alpha, sigma, t,
                                            anchors, log_p0)
        out[i] = float(np.var(ratios))
    return out


def _compute_colornorm(heat_eta, heat_gamma, cap=VMAX_CAP):
    both = np.concatenate([heat_eta.ravel(), heat_gamma.ravel()])
    both_c = np.clip(both, 1e-8, 1e8)
    sym = np.maximum(both_c, 1.0 / both_c)
    vmax_lit = float(np.quantile(sym, 0.99))
    vmax = min(vmax_lit, cap)
    vmin = 1.0 / vmax
    return vmin, vmax, vmax_lit


def _plot_marginal(ax, knob_grid, var_curve, trivial_x, arrow_xy, arrow_text_xy):
    pts = np.stack([knob_grid, var_curve], axis=-1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors=[SLATE] * len(segs), linewidths=1.6))
    ax.fill_between(knob_grid, 0, var_curve, color=SLATE, alpha=0.12, linewidth=0)
    ax.axvline(trivial_x, color=DIV, lw=0.8, alpha=0.7, ls="--")
    ax.set_xlim(knob_grid[0], knob_grid[-1])

    nz = var_curve[var_curve > 1e-10]
    if len(nz) > 0 and (nz.max() / max(nz.min(), 1e-12)) > 10 ** 1.5:
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, var_curve.max() * 1.15)
    else:
        ax.set_ylim(0, max(var_curve.max() * 1.15, 1e-6))

    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel(
        r"Var$_Y[\lambda^{\star}_t\,/\,\lambda^{\star}_{\mathrm{time}}]$",
        labelpad=2, fontsize=8,
    )
    ax.annotate(
        r"$\uparrow$ risk of time-only reversal",
        xy=arrow_xy, xytext=arrow_text_xy,
        fontsize=7, style="italic", color=CORAL,
        arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=0.6,
                        connectionstyle="arc3,rad=0.15"),
    )


def make_figure(heat_eta, heat_gamma, var_eta, var_gamma, vmin, vmax):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    heat_eta_c = np.clip(heat_eta, vmin, vmax)
    heat_gamma_c = np.clip(heat_gamma, vmin, vmax)

    fig = plt.figure(figsize=(11.6, 4.8), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 1))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12, width_ratios=[1, 1])
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=[0.22, 1.0], hspace=0.04)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[0.22, 1.0], hspace=0.04)
    ax_marg_L = fig.add_subplot(left_gs[0])
    ax_heat_L = fig.add_subplot(left_gs[1], sharex=ax_marg_L)
    ax_marg_R = fig.add_subplot(right_gs[0])
    ax_heat_R = fig.add_subplot(right_gs[1], sharex=ax_marg_R)

    im = ax_heat_L.imshow(
        heat_eta_c, origin="lower", cmap="RdBu_r", norm=norm,
        extent=[ETA_GRID[0], ETA_GRID[-1], Y_GRID[0], Y_GRID[-1]],
        aspect="auto", interpolation="bilinear",
    )
    ax_heat_R.imshow(
        heat_gamma_c, origin="lower", cmap="RdBu_r", norm=norm,
        extent=[GAMMA_GRID[0], GAMMA_GRID[-1], Y_GRID[0], Y_GRID[-1]],
        aspect="auto", interpolation="bilinear",
    )
    ax_heat_L.axvline(1.0, color=DIV, lw=0.8, alpha=0.7, ls="--")
    ax_heat_R.axvline(0.0, color=DIV, lw=0.8, alpha=0.7, ls="--")
    ax_heat_L.set_xlabel(r"kernel mismatch  $\eta$")
    ax_heat_L.set_ylabel(r"noisy value  $y$")
    ax_heat_R.set_xlabel(r"anchor-weighted hazard  $\gamma$")
    ax_heat_L.set_xlim(ETA_GRID[0], ETA_GRID[-1])
    ax_heat_R.set_xlim(GAMMA_GRID[0], GAMMA_GRID[-1])
    ax_heat_L.set_ylim(Y_GRID[0], Y_GRID[-1])
    ax_heat_R.set_ylim(Y_GRID[0], Y_GRID[-1])

    _plot_marginal(
        ax_marg_L, ETA_GRID, var_eta, trivial_x=1.0,
        arrow_xy=(ETA_GRID[2], var_eta[2]),
        arrow_text_xy=(0.45, var_eta.max() * 0.55),
    )
    ax_marg_L.set_title(r"(a)  Kernel mismatch  $\eta$",
                        loc="left", fontweight="semibold", fontsize=10, pad=4)
    _plot_marginal(
        ax_marg_R, GAMMA_GRID, var_gamma, trivial_x=0.0,
        arrow_xy=(GAMMA_GRID[-3], var_gamma[-3]),
        arrow_text_xy=(1.0, var_gamma.max() * 0.55),
    )
    ax_marg_R.set_title(r"(b)  Anchor-weighted hazard  $\gamma$",
                        loc="left", fontweight="semibold", fontsize=10, pad=4)

    ticks = ([vmin, 0.5, 1.0, 2.0, vmax] if (vmin <= 0.5 and 2.0 <= vmax)
             else [vmin, 1.0, vmax])
    cbar = fig.colorbar(
        im, ax=[ax_heat_L, ax_heat_R], orientation="horizontal",
        shrink=0.4, aspect=40, pad=0.08, ticks=ticks,
    )
    cbar.ax.set_xticklabels([f"{t:.2g}" for t in ticks])
    cbar.set_label(
        r"$\lambda^{\star}_t(y)\,/\,\lambda^{\star}_{\mathrm{time}}(t)$",
        fontsize=8,
    )
    cbar.outline.set_linewidth(0.6)

    fig.text(
        0.5, 0.018,
        r"As forward state-dependence is introduced — via kernel mismatch "
        r"$\eta$ (left) or anchor-weighted hazard $\gamma$ (right) — SJD's "
        r"reverse hazard automatically develops the matching structure "
        r"(heatmaps). Using a time-only reverse schedule discards this "
        r"structure, incurring the Prop. 4.2 excess risk plotted above.",
        ha="center", fontsize=8, style="italic", color="#444",
    )
    return fig


def main():
    t_start = time.perf_counter()
    setup_style()
    p0, log_p0 = load_cifar10_prior()
    t_ref = float(find_t_ref())
    alpha, sigma = vp_coefficients(t_ref)
    alpha, sigma = float(alpha), float(sigma)

    gammas_check = np.linspace(-2.0, 6.0, 21)
    err_max = max(
        abs(float(np.sum(p0 * np.exp(log_lambda_of_a(log_p0, float(g))))) - 1.0)
        for g in gammas_check
    )

    rng = np.random.default_rng(0)
    y_s = sample_marginal_y(N_SAMPLES, alpha, sigma, ANCHORS, p0, rng)

    heat_eta = build_heatmap(ETA_GRID, "eta", Y_GRID, alpha, sigma, t_ref, ANCHORS, log_p0)
    heat_gamma = build_heatmap(GAMMA_GRID, "gamma", Y_GRID, alpha, sigma, t_ref, ANCHORS, log_p0)
    var_eta = build_variance_curve(y_s, ETA_GRID, "eta", alpha, sigma, t_ref, ANCHORS, log_p0)
    var_gamma = build_variance_curve(y_s, GAMMA_GRID, "gamma", alpha, sigma, t_ref, ANCHORS, log_p0)

    vmin, vmax, vmax_lit = _compute_colornorm(heat_eta, heat_gamma)

    def _var_at(eta, gamma):
        r = np.empty(N_SAMPLES)
        for s in range(0, N_SAMPLES, 20_000):
            e = min(s + 20_000, N_SAMPLES)
            r[s:e] = hazard_ratio_grid(y_s[s:e], eta, gamma, alpha, sigma, t_ref,
                                       ANCHORS, log_p0)
        return float(np.var(r))

    v00 = _var_at(1.0, 0.0)
    v_eta_lo = _var_at(0.15, 0.0)
    v_g_hi = _var_at(1.0, 4.0)

    print(f"t* = {t_ref:.4f}, alpha(t*) = {alpha:.4f}, sigma(t*) = {sigma:.4f}")
    print(f"Color range: vmin = {vmin:.3f}, vmax = {vmax:.3f}  "
          f"(literal 99%ile of symmetric ratio = {vmax_lit:.3g}, cap = {VMAX_CAP})")
    print(f"Var @ (η=1,    γ=0) = {v00:.3e}")
    print(f"Var @ (η=0.15, γ=0) = {v_eta_lo:.3e}")
    print(f"Var @ (η=1,    γ=4) = {v_g_hi:.3e}")
    print(f"Max |Σ p0·λ(γ) − 1| over γ ∈ [-2, 6]: {err_max:.3e}")

    fig = make_figure(heat_eta, heat_gamma, var_eta, var_gamma, vmin, vmax)
    with warnings.catch_warnings():
        # Spurious with nested gridspec + shared horizontal colorbar; layout is fine.
        warnings.filterwarnings("ignore", message="constrained_layout not applied")
        fig.savefig(OUT_PDF)
        fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Figure saved. Total elapsed: {time.perf_counter() - t_start:.1f} s.")


if __name__ == "__main__":
    main()
