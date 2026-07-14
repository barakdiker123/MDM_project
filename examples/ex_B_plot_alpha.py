#!/usr/bin/env python
# examples/ex_B_plot_alpha.py
r"""
Violin + box plot of the ordinary-MDM alpha estimates on the Example-B system.

Each panel shows, for one MC run:
  - violin  : full kernel-density estimate of the estimate distribution
  - box     : median, IQR, and 1.5*IQR whiskers (Tukey style)
  - dots    : individual outlier estimates beyond the whiskers
  - red line: true alpha_i value

IMPORTANT -- this system is open-loop unstable (F has eigenvalue -1.01):
  state grows as 1.01^tau.  Float64 has ~15 significant digits, so once
  1.01^tau > 1e14 (tau > ~3200), the state cancellation inside the MDM
  residue is dominated by roundoff and estimates deteriorate.

  DO NOT simply increase --tau to get better estimates on this system.
  Instead, use --n-seg > 1: this runs n_seg independent short trajectories
  (each of length --tau) and averages their sufficient statistics, which is
  mathematically equivalent to more data but numerically safe.

Run:
    python examples/ex_B_plot_alpha.py --mc 200 --tau 1000 --n-seg 1
    python examples/ex_B_plot_alpha.py --mc 200 --tau 1000 --n-seg 10   # 10x more data, same tau
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # -> mdm package
sys.path.insert(0, _HERE)  # -> examples/ as flat modules

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ex_B_unobservable_unknown_input import (
    F,
    E,
    H,
    D,
    G_fn,
    u_fn,
    BQ,
    BR,
    alpha_true,
    L,
)

# Spectral radius of F for the instability warning
_SPEC_RAD = max(abs(np.linalg.eigvals(F)))
_SAFE_TAU = int(14 / np.log10(_SPEC_RAD)) if _SPEC_RAD > 1 else int(1e9)


def _warn_if_unstable(tau):
    growth = _SPEC_RAD**tau
    if growth > 1e14:
        print(
            f"\n  WARNING: F is unstable (spectral radius {_SPEC_RAD:.4f}).\n"
            f"  At tau={tau}, state grows by {growth:.1e} -- this exceeds float64\n"
            f"  precision (~1e15) and will corrupt the MDM residue cancellation.\n"
            f"  Estimates will be WORSE than at smaller tau.\n"
            f"  Use --tau <= {_SAFE_TAU} for a single trajectory, or increase\n"
            f"  --n-seg to aggregate multiple short segments instead.\n"
        )


# ------------------------------------------------------------------ simulate
def _compute_metrics(estimates, basis):
    """Per-MC Stein and squared-Frobenius distances to ground-truth (Q, R),
    plus the rate of positive-definite reconstructions."""
    from mdm.neural.losses import stein_loss
    from mdm.linalg import psd_project, is_psd

    Q_true, R_true = basis.to_QR(alpha_true)
    steins, frobs, pd_flags = [], [], []
    for a in estimates:
        Qh, Rh = basis.to_QR(a)
        pd = is_psd(Qh) and is_psd(Rh)
        pd_flags.append(float(pd))
        steins.append(
            stein_loss(psd_project(Qh), Q_true) + stein_loss(psd_project(Rh), R_true)
        )
        frobs.append(
            np.linalg.norm(Qh - Q_true, "fro") ** 2
            + np.linalg.norm(Rh - R_true, "fro") ** 2
        )
    return np.array(steins), np.array(frobs), np.array(pd_flags)


def run(MC=200, TAU=1000, n_seg=1, seed=0):
    from mdm import StateSpaceModel, NoiseBasis, MDM

    _warn_if_unstable(TAU)
    model = StateSpaceModel.from_constant(F, E, H, D, TAU, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(MC):
        s = np.mean(
            [
                est.sufficient_stat(est.residue_cov(model.simulate(Q, R, rng)))
                for _ in range(n_seg)
            ],
            axis=0,
        )
        estimates.append(np.linalg.solve(est.S, s))
    estimates = np.array(estimates)
    steins, frobs, pd_flags = _compute_metrics(estimates, basis)
    return estimates, basis, steins, frobs, pd_flags


# ------------------------------------------------------------------ shared drawing helpers
def _violin_box(ax, col, c):
    """Draw violin + Tukey box on ax. Returns outlier array.
    Falls back to a thin horizontal bar if data is near-constant
    (std < 1e-6) which would crash matplotlib's KDE-based violinplot."""
    ax.set_xticks([])
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    if col.std() < 1e-6:
        # degenerate: all values essentially identical — draw a thick line
        med = float(np.median(col))
        ax.plot([-0.3, 0.3], [med, med], color=c, lw=4, zorder=4)
        ax.set_ylim(med - 0.1, med + 0.1)
        return np.array([])

    vp = ax.violinplot(
        col, positions=[0], widths=0.8, showmedians=False, showextrema=False
    )
    for body in vp["bodies"]:
        body.set_facecolor(c)
        body.set_alpha(0.35)
        body.set_edgecolor("none")
    q1, med, q3 = np.percentile(col, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    wlo = col[col >= lo_f].min()
    whi = col[col <= hi_f].max()
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (-0.22, q1),
            0.44,
            max(iqr, 1e-9),
            boxstyle="square,pad=0",
            lw=1.2,
            edgecolor=c,
            facecolor=c,
            alpha=0.6,
            zorder=3,
        )
    )
    ax.plot([-0.22, 0.22], [med, med], color="white", lw=2, zorder=4)
    for y0, y1 in [(wlo, q1), (q3, whi)]:
        ax.plot([0, 0], [y0, y1], color=c, lw=1.4, zorder=3)
    for yy in [wlo, whi]:
        ax.plot([-0.1, 0.1], [yy, yy], color=c, lw=1.4)
    out = col[(col < lo_f) | (col > hi_f)]
    if out.size:
        xj = np.random.default_rng(int(abs(col[0]) * 1e6) % 2**31).uniform(
            -0.15, 0.15, size=out.size
        )
        ax.scatter(
            xj,
            out,
            color=c,
            s=16,
            alpha=0.7,
            edgecolors="k",
            lw=0.4,
            zorder=5,
            label=f"{out.size} outliers",
        )
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.5)
    return out


def _metric_panel(ax, vals, c, name):
    """Single bar (mean ± std) for one scalar metric across MC runs."""
    ax.bar(
        [0],
        [vals.mean()],
        yerr=[vals.std()],
        color=c,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.8,
        width=0.5,
        capsize=6,
        error_kw=dict(lw=1.5, ecolor="black"),
    )
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([0])
    ax.set_xticklabels([name], fontsize=9)
    ax.set_title(
        f"mean={vals.mean():.3f}\nstd={vals.std():.3f}  S.cov={vals.var():.3f}",
        fontsize=8.5,
    )
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=8)


# ------------------------------------------------------------------ plot
def plot(
    estimates, basis, steins, frobs, pd_flags, out="alpha_violin.png", tau=1000, n_seg=1
):
    MC, na = estimates.shape
    seg_str = f", {n_seg} segs" if n_seg > 1 else ""
    colors = plt.cm.tab10.colors

    # layout: 2 rows — alpha panels (top), metric panels (bottom)
    # bottom row has 2 panels (Stein, Frobenius), centred under the alpha panels
    fig = plt.figure(figsize=(3 * na, 10))
    fig.suptitle(
        f"Ordinary MDM  —  Example B  (MC={MC}, $\\tau$={tau}{seg_str})\n"
        r"Distribution of $\hat{\alpha}_i$ estimates and distance metrics to GT",
        fontsize=11,
    )
    gs_top = fig.add_gridspec(1, na, top=0.88, bottom=0.52, hspace=0.05)
    gs_bot = fig.add_gridspec(1, 2, top=0.44, bottom=0.08, left=0.15, right=0.85)

    # ---- alpha panels (top row) ----
    for i in range(na):
        ax = fig.add_subplot(gs_top[0, i])
        col = estimates[:, i]
        true = alpha_true[i]
        c = colors[i % len(colors)]
        _violin_box(ax, col, c)
        ax.axhline(
            true, color="crimson", lw=1.8, ls="--", zorder=6, label=f"true={true:.2g}"
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax.set_xlabel(f"$\\hat{{\\alpha}}_{i+1}$", fontsize=11)
        ax.set_title(
            f"a{i+1}\nstd={col.std():.3f}  S.cov={col.var():.3f}\n"
            f"bias={abs(col.mean()-true):.3f}",
            fontsize=8.5,
        )
        ax.yaxis.set_tick_params(labelsize=8)

    # ---- metric panels (bottom row) ----
    ax_s = fig.add_subplot(gs_bot[0, 0])
    ax_f = fig.add_subplot(gs_bot[0, 1])

    _metric_panel(ax_s, steins, c="steelblue", name="Stein$(Q+R)$")
    _metric_panel(ax_f, frobs, c="darkorange", name=r"$\|Q-Q^*\|_F^2+\|R-R^*\|_F^2$")

    # print summary
    print(
        f"\n  Stein:     mean={steins.mean():.4f}  S.cov={steins.var():.4f}"
        f"  std={steins.std():.4f}"
    )
    print(
        f"  Frobenius: mean={frobs.mean():.4f}  S.cov={frobs.var():.4f}"
        f"  std={frobs.std():.4f}"
    )
    print(
        f"  PD rate:   {100*pd_flags.mean():.1f}%  "
        f"({int(pd_flags.sum())}/{len(pd_flags)} estimates have PSD Q and R)"
    )

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=200, help="Monte-Carlo repeats")
    p.add_argument(
        "--tau",
        type=int,
        default=1000,
        help=f"trajectory length (safe limit ~{_SAFE_TAU})",
    )
    p.add_argument(
        "--n-seg", type=int, default=1, help="segments averaged per estimate"
    )
    p.add_argument("--out", type=str, default="alpha_violin.png")
    args = p.parse_args()
    estimates, basis, steins, frobs, pd_flags = run(
        MC=args.mc, TAU=args.tau, n_seg=args.n_seg
    )
    plot(
        estimates,
        basis,
        steins,
        frobs,
        pd_flags,
        out=args.out,
        tau=args.tau,
        n_seg=args.n_seg,
    )
