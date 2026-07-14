#!/usr/bin/env python
# examples/ex_B_plot_alpha_nn_v2.py
r"""
MDM (segment-averaged) vs NN (single-trajectory) — Version 2.

Violin rows are IDENTICAL to v1: each shows the single-trajectory distribution
of estimates across all MC runs, so you can see the raw variance of both.

What changes is the BOTTOM section:
  - MDM segmented:  split the MC trajectories into n_seg groups of
    (MC // n_seg) trajectories each.  Within each group, average the
    individual alpha_hat estimates -> one high-quality alpha per group.
  - NN single:      keep every single-trajectory NN estimate as-is (no averaging).
  - Bottom panels compare PD rate and Stein distance between these two.

The key question answered:  at what segment size does segment-averaged MDM
match or beat single-trajectory NN on Stein and PD rate?

Flags
-----
    --mc         total MC trajectories, default 200
    --n-seg      number of segments to average into (default 10),
                 so each segment averages MC // n_seg individual estimates
    --tau        trajectory length (default 1000)
    --ckpt       path to nn_best.pt (auto-trains if missing)
    --n-train    training samples (default 20000)
    --epochs     training epochs (default 150)
    --out        output PNG (default alpha_nn_vs_mdm_v2.png)

Run:
    python examples/ex_B_plot_alpha_nn_v2.py --mc 200 --n-seg 10
    python examples/ex_B_plot_alpha_nn_v2.py --mc 200 --n-seg 1   # no averaging
    python examples/ex_B_plot_alpha_nn_v2.py --mc 200 --n-seg 40  # fine averaging
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse everything from v1 unchanged
from ex_B_plot_alpha_nn import _build, _train, _load_net, _warn_if_unstable, _SAFE_TAU
from ex_B_unobservable_unknown_input import alpha_true
from ex_B_plot_alpha import _compute_metrics, _violin_box

_c_mdm = "steelblue"
_c_nn = "darkorange"


# ── run ─────────────────────────────────────────────────────────────
def run(
    MC=200, TAU=1000, n_seg=10, ckpt="nn_best.pt", n_train=20000, epochs=150, seed=0
):
    """
    Returns
    -------
    mdm_single   : (MC, na)   single-trajectory MDM estimates  (violin row)
    nn_single    : (MC, na)   single-trajectory NN  estimates  (violin row)
    mdm_seg      : (n_seg, na) segment-averaged MDM estimates  (bottom panels)
    nn_single    : also used for bottom NN Stein (no averaging)
    basis, metrics ...
    """
    import torch

    assert MC % n_seg == 0, f"MC={MC} must be divisible by n_seg={n_seg}"
    seg_size = MC // n_seg  # trajectories per segment

    _warn_if_unstable(TAU)
    model, basis, est = _build(TAU)

    if not os.path.exists(ckpt):
        print(f"No checkpoint at '{ckpt}', training...")
        _train(basis, est, model, n_train=n_train, epochs=epochs, ckpt=ckpt)
    net = _load_net(basis, ckpt)

    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(seed)

    mdm_single, nn_single = [], []
    print(f"Simulating {MC} trajectories ({n_seg} segments x {seg_size} each)...")
    for i in range(MC):
        z = model.simulate(Q, R, rng)
        s = est.sufficient_stat(est.residue_cov(z))
        mdm_single.append(np.linalg.solve(est.S, s))
        with torch.no_grad():
            nn_single.append(net(torch.tensor(s[None], dtype=torch.float32)).numpy()[0])
        if (i + 1) % max(1, MC // 5) == 0:
            print(f"  {i+1}/{MC}")

    mdm_single = np.array(mdm_single)  # (MC, na)
    nn_single = np.array(nn_single)  # (MC, na)

    # segment-averaged MDM: reshape into (n_seg, seg_size, na), then mean over axis 1
    mdm_seg = mdm_single.reshape(n_seg, seg_size, -1).mean(axis=1)  # (n_seg, na)

    # metrics for violin rows (single-trajectory)
    ms_s, ms_f, ms_pd = _compute_metrics(mdm_single, basis)
    ns_s, ns_f, ns_pd = _compute_metrics(nn_single, basis)

    # metrics for bottom rows
    mseg_s, mseg_f, mseg_pd = _compute_metrics(mdm_seg, basis)
    # NN stays single-trajectory for bottom comparison
    nn_bot_s, nn_bot_f, nn_bot_pd = ns_s, ns_f, ns_pd

    return (
        mdm_single,
        nn_single,
        mdm_seg,
        basis,
        ms_s,
        ms_f,
        ms_pd,  # MDM single violin metrics
        ns_s,
        ns_f,
        ns_pd,  # NN  single violin metrics
        mseg_s,
        mseg_f,
        mseg_pd,  # MDM segment bottom metrics
        nn_bot_s,
        nn_bot_f,
        nn_bot_pd,  # NN  single  bottom metrics
        n_seg,
        seg_size,
    )


# ── plot ─────────────────────────────────────────────────────────────
def plot(
    mdm_single,
    nn_single,
    mdm_seg,
    basis,
    ms_s,
    ms_f,
    ms_pd,
    ns_s,
    ns_f,
    ns_pd,
    mseg_s,
    mseg_f,
    mseg_pd,
    nn_bot_s,
    nn_bot_f,
    nn_bot_pd,
    n_seg,
    seg_size,
    out="alpha_nn_vs_mdm_v2.png",
    tau=1000,
):

    MC, na = mdm_single.shape
    colors_mdm = plt.cm.tab10.colors
    colors_nn = plt.cm.Set2.colors

    fig = plt.figure(figsize=(3 * na, 18))
    fig.suptitle(
        f"MDM (single-traj) & NN (single-traj) — violin rows\n"
        f"MDM segment-averaged ({n_seg} seg × {seg_size} traj) vs NN single — bottom bars\n"
        f"MC={MC}, $\\tau$={tau}",
        fontsize=10,
    )

    gs_top = fig.add_gridspec(1, na, top=0.93, bottom=0.72)
    gs_mid = fig.add_gridspec(1, na, top=0.67, bottom=0.46)
    gs_bot = fig.add_gridspec(
        1, 2, top=0.40, bottom=0.24, left=0.08, right=0.96, wspace=0.30
    )
    gs_bot2 = fig.add_gridspec(
        1, 2, top=0.18, bottom=0.02, left=0.08, right=0.96, wspace=0.30
    )

    # ── violin rows (identical to v1) ────────────────────────────────
    row_specs = [
        (mdm_single, colors_mdm, "MDM  (single trajectory)", gs_top, ms_pd),
        (nn_single, colors_nn, "NN   (single trajectory)", gs_mid, ns_pd),
    ]
    for ests, colors, rlabel, gs, pd_flags in row_specs:
        pd_rate = 100 * pd_flags.mean()
        for i in range(na):
            ax = fig.add_subplot(gs[0, i])
            col = ests[:, i]
            true = alpha_true[i]
            c = colors[i % len(colors)]
            _violin_box(ax, col, c)
            ax.axhline(
                true,
                color="crimson",
                lw=1.8,
                ls="--",
                zorder=6,
                label=f"true={true:.2g}",
            )
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.6)
            ax.set_xlabel(f"$\\hat{{\\alpha}}_{i+1}$", fontsize=10)
            ax.set_title(
                f"a{i+1}\nstd={col.std():.3f}  S.cov={col.var():.3f}\n"
                f"bias={abs(col.mean()-true):.3f}",
                fontsize=8,
            )
            ax.yaxis.set_tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel(rlabel, fontsize=9, labelpad=6)
            if i == na - 1:
                ax.annotate(
                    f"PD rate\n(Q & R):\n{pd_rate:.1f}%",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    fontsize=9,
                    color="darkgreen" if pd_rate >= 50 else "crimson",
                    fontweight="bold",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                    ),
                )

    # ── grouped bar helper ────────────────────────────────────────────
    def _gbar(ax, v1, v2, lbl1, lbl2, title, c1=_c_mdm, c2=_c_nn):
        means = [v1.mean(), v2.mean()]
        stds = [v1.std(), v2.std()]
        ax.bar(
            [0, 1],
            means,
            yerr=stds,
            width=0.5,
            color=[c1, c2],
            alpha=0.82,
            edgecolor="k",
            lw=0.8,
            capsize=8,
            error_kw=dict(lw=1.8, ecolor="black"),
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([lbl1, lbl2], fontsize=10)
        ax.set_title(
            f"{title}\n"
            f"{lbl1} mean={means[0]:.3f} std={stds[0]:.3f}   "
            f"{lbl2} mean={means[1]:.3f} std={stds[1]:.3f}",
            fontsize=8.5,
        )
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_tick_params(labelsize=8)

    # ── row 2: Stein and PD rate (segmented MDM vs single NN) ────────
    lbl_mdm = f"MDM\nseg-avg\n({seg_size}/seg)"
    lbl_nn = f"NN\nsingle\ntraj"

    _gbar(
        fig.add_subplot(gs_bot[0, 0]),
        mseg_s,
        nn_bot_s,
        lbl_mdm,
        lbl_nn,
        f"Stein$(Q,Q^*)+$Stein$(R,R^*)$\n"
        f"MDM: {n_seg} segment averages   NN: {MC} single trajectories",
    )

    _gbar(
        fig.add_subplot(gs_bot[0, 1]),
        mseg_f,
        nn_bot_f,
        lbl_mdm,
        lbl_nn,
        r"$\|Q-Q^*\|_F^2+\|R-R^*\|_F^2$"
        f"\nMDM: {n_seg} segment averages   NN: {MC} single trajectories",
    )

    # ── row 3: PD rate  |  Stein PD-only ─────────────────────────────
    ax_pd = fig.add_subplot(gs_bot2[0, 0])
    ax_spd = fig.add_subplot(gs_bot2[0, 1])

    # PD rate: segmented MDM vs single NN
    pd_means = [100 * mseg_pd.mean(), 100 * nn_bot_pd.mean()]
    n_labels = [f"{n_seg} segs", f"{MC} runs"]
    ax_pd.bar(
        [0, 1],
        pd_means,
        width=0.5,
        color=[_c_mdm, _c_nn],
        alpha=0.82,
        edgecolor="k",
        lw=0.8,
    )
    ax_pd.set_xticks([0, 1])
    ax_pd.set_xticklabels(
        [f"MDM seg-avg\n({seg_size}/seg)", "NN single\ntraj"], fontsize=9
    )
    ax_pd.set_ylim(0, 110)
    ax_pd.set_ylabel("PD rate  Q & R (%)", fontsize=9)
    ax_pd.set_title(
        f"PD rate: Q and R both positive definite\n"
        f"MDM seg-avg={pd_means[0]:.1f}%  ({n_seg} segments)   "
        f"NN single={pd_means[1]:.1f}%  ({MC} runs)",
        fontsize=8.5,
    )
    for xi, (val, n) in enumerate(zip(pd_means, n_labels)):
        ax_pd.text(
            xi, val + 1.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold"
        )
    ax_pd.grid(axis="y", ls=":", alpha=0.5)
    ax_pd.spines[["top", "right"]].set_visible(False)

    # Stein PD-only: runs where BOTH segmented MDM and single NN are PD
    # Note: sizes differ (n_seg vs MC), so we compare distributions directly
    mdm_pd_runs = mseg_pd == 1
    nn_pd_runs = nn_bot_pd == 1
    n_mdm_pd = int(mdm_pd_runs.sum())
    n_nn_pd = int(nn_pd_runs.sum())
    if n_mdm_pd >= 2 and n_nn_pd >= 2:
        means = [mseg_s[mdm_pd_runs].mean(), nn_bot_s[nn_pd_runs].mean()]
        stds = [mseg_s[mdm_pd_runs].std(), nn_bot_s[nn_pd_runs].std()]
        ax_spd.bar(
            [0, 1],
            means,
            yerr=stds,
            width=0.5,
            color=[_c_mdm, _c_nn],
            alpha=0.82,
            edgecolor="k",
            lw=0.8,
            capsize=8,
            error_kw=dict(lw=1.8, ecolor="black"),
        )
        ax_spd.set_xticks([0, 1])
        ax_spd.set_xticklabels(
            [
                f"MDM seg-avg\n(PD only n={n_mdm_pd})",
                f"NN single\n(PD only n={n_nn_pd})",
            ],
            fontsize=9,
        )
        ax_spd.set_title(
            f"Stein  [PD-only subset]\n"
            f"MDM seg mean={means[0]:.3f} std={stds[0]:.3f}   "
            f"NN mean={means[1]:.3f} std={stds[1]:.3f}",
            fontsize=8.5,
        )
        ax_spd.grid(axis="y", ls=":", alpha=0.5)
        ax_spd.spines[["top", "right"]].set_visible(False)
        ax_spd.yaxis.set_tick_params(labelsize=8)
    else:
        ax_spd.text(
            0.5,
            0.5,
            f"MDM PD: {n_mdm_pd}/{n_seg}   NN PD: {n_nn_pd}/{MC}\n"
            "Not enough PD runs for comparison\n"
            "(increase n-seg or MC)",
            ha="center",
            va="center",
            transform=ax_spd.transAxes,
            fontsize=10,
        )
        ax_spd.set_title("Stein  [PD-only]", fontsize=8.5)
        ax_spd.axis("off")

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # console summary
    print(f"\n  Segment size={seg_size}  n_seg={n_seg}  MC={MC}  tau={tau}")
    print(f"  {'metric':<22s}  {'MDM seg-avg':>12s}  {'NN single':>12s}")
    for name, mv, nv in [
        ("Stein mean", mseg_s.mean(), nn_bot_s.mean()),
        ("Stein std", mseg_s.std(), nn_bot_s.std()),
        ("Frobenius mean", mseg_f.mean(), nn_bot_f.mean()),
        ("PD rate %", 100 * mseg_pd.mean(), 100 * nn_bot_pd.mean()),
    ]:
        print(f"  {name:<22s}  {mv:12.4f}  {nv:12.4f}")
    if n_mdm_pd >= 2 and n_nn_pd >= 2:
        print(
            f"  {'Stein PD-only mean':<22s}  "
            f"{mseg_s[mdm_pd_runs].mean():12.4f}  "
            f"{nn_bot_s[nn_pd_runs].mean():12.4f}"
        )


# ── entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mc",
        type=int,
        default=200,
        help="total MC trajectories (must be divisible by n-seg)",
    )
    p.add_argument(
        "--n-seg",
        type=int,
        default=10,
        help="number of segments (MC // n-seg trajectories per segment)",
    )
    p.add_argument(
        "--tau",
        type=int,
        default=1000,
        help=f"trajectory length (safe limit ~{_SAFE_TAU})",
    )
    p.add_argument("--ckpt", type=str, default="nn_best.pt")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--out", type=str, default="alpha_nn_vs_mdm_v2.png")
    args = p.parse_args()

    result = run(
        MC=args.mc,
        TAU=args.tau,
        n_seg=args.n_seg,
        ckpt=args.ckpt,
        n_train=args.n_train,
        epochs=args.epochs,
    )
    (
        mdm_single,
        nn_single,
        mdm_seg,
        basis,
        ms_s,
        ms_f,
        ms_pd,
        ns_s,
        ns_f,
        ns_pd,
        mseg_s,
        mseg_f,
        mseg_pd,
        nn_bot_s,
        nn_bot_f,
        nn_bot_pd,
        n_seg,
        seg_size,
    ) = result

    plot(
        mdm_single,
        nn_single,
        mdm_seg,
        basis,
        ms_s,
        ms_f,
        ms_pd,
        ns_s,
        ns_f,
        ns_pd,
        mseg_s,
        mseg_f,
        mseg_pd,
        nn_bot_s,
        nn_bot_f,
        nn_bot_pd,
        n_seg=n_seg,
        seg_size=seg_size,
        out=args.out,
        tau=args.tau,
    )
