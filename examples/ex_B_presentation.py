#!/usr/bin/env python
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from examples.ex_B_unobservable_unknown_input import (
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
        # average sufficient statistics across n_seg independent segments
        s = np.mean(
            [
                est.sufficient_stat(est.residue_cov(model.simulate(Q, R, rng)))
                for _ in range(n_seg)
            ],
            axis=0,
        )
        estimates.append(np.linalg.solve(est.S, s))
    return np.array(estimates), basis  # (MC, n_alpha)


# ------------------------------------------------------------------ plot
def plot(estimates, basis, out="alpha_violin.png", tau=1000, n_seg=1):
    MC, na = estimates.shape
    total = MC * tau * n_seg
    seg_str = f", {n_seg} seg{'s' if n_seg > 1 else ''}/estimate" if n_seg > 1 else ""
    fig, axes = plt.subplots(1, na, figsize=(3 * na, 5), sharey=False)
    fig.suptitle(
        f"Ordinary MDM  —  Example B  "
        f"(MC={MC}, $\\tau$={tau}{seg_str})\n"
        r"Distribution of $\hat{\alpha}_i$ estimates across independent trajectories",
        fontsize=11,
    )

    colors = plt.cm.tab10.colors

    for i, ax in enumerate(axes):
        col = estimates[:, i]
        true = alpha_true[i]
        c = colors[i % len(colors)]

        # violin
        vp = ax.violinplot(
            col, positions=[0], widths=0.8, showmedians=False, showextrema=False
        )
        for body in vp["bodies"]:
            body.set_facecolor(c)
            body.set_alpha(0.35)
            body.set_edgecolor("none")

        # box
        q1, median, q3 = np.percentile(col, [25, 50, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        whisker_lo = col[col >= lo_fence].min()
        whisker_hi = col[col <= hi_fence].max()
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (-0.22, q1),
                0.44,
                iqr,
                boxstyle="square,pad=0",
                linewidth=1.2,
                edgecolor=c,
                facecolor=c,
                alpha=0.6,
                zorder=3,
            )
        )
        ax.plot([-0.22, 0.22], [median, median], color="white", lw=2, zorder=4)
        for y0, y1 in [(whisker_lo, q1), (q3, whisker_hi)]:
            ax.plot([0, 0], [y0, y1], color=c, lw=1.4, zorder=3)
        for yy in [whisker_lo, whisker_hi]:
            ax.plot([-0.1, 0.1], [yy, yy], color=c, lw=1.4)

        # outliers
        outliers = col[(col < lo_fence) | (col > hi_fence)]
        if outliers.size:
            xjit = np.random.default_rng(i).uniform(-0.15, 0.15, size=outliers.size)
            ax.scatter(
                xjit,
                outliers,
                color=c,
                s=18,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label=f"{outliers.size} outliers",
            )
            ax.legend(fontsize=7, loc="upper right", framealpha=0.5)

        # true value
        ax.axhline(
            true,
            color="crimson",
            lw=1.8,
            linestyle="--",
            zorder=6,
            label=f"true={true:.2g}",
        )
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.6)

        ax.set_xticks([])
        ax.set_xlabel(f"$\\hat{{\\alpha}}_{i+1}$", fontsize=12)
        ax.set_title(
            f"a{i+1}\nstd={col.std():.3f}  S.cov={col.var():.3f}\nbias={abs(col.mean()-true):.3f}",
            fontsize=9,
        )
        ax.yaxis.set_tick_params(labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    fig.tight_layout()
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
        help=f"trajectory length per segment (safe limit ~{_SAFE_TAU} for this system)",
    )
    p.add_argument(
        "--n-seg",
        type=int,
        default=1,
        help="segments averaged per estimate (use this instead of larger --tau)",
    )
    p.add_argument("--out", type=str, default="alpha_violin.png")
    args = p.parse_args()
    estimates, basis = run(MC=args.mc, TAU=args.tau, n_seg=args.n_seg)
    plot(estimates, basis, out=args.out, tau=args.tau, n_seg=args.n_seg)
