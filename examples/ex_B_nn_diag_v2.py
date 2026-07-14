#!/usr/bin/env python
# examples/ex_B_nn_diag_v2.py
r"""
Example B: window-based diagonal noise covariance estimator (NN v2).

Trains a 1D-CNN that takes a window of [z_k | imu_k] measurements and
directly predicts diag(Q) and diag(R), bypassing the MDM sufficient
statistic entirely.  Compares against the MDM ordinary estimate
(which uses the full alpha parameterisation and recovers the full Q, R).

Architecture (DiagCovEstimator):
    Input  : (batch, nz + n_imu, W) = (batch, 9, 100) for Example B
    CNN    : Conv1d 9->32->64->128, BatchNorm + ReLU, AdaptiveAvgPool
    MLP    : FC 128->64->6  (log-space diagonal outputs)
    Output : exp( net(x) ) = [diag(Q_hat), diag(R_hat)]

Key difference from NN v1:
    v1: input = MDM sufficient statistic s (6-dim, post-processed)
    v2: input = raw measurement + IMU window (W * 9-dim, time-series)
    v1: output = full alpha vector (6 weights for structured Q, R)
    v2: output = diag(Q) and diag(R) directly (6 positive scalars)

Training:
    Diagonal Q = diag(q1, q2, q3), diagonal R = diag(r1, r2, r3)
    drawn log-uniformly from [lo_q, hi_q] and [lo_r, hi_r].
    Loss = MSE on log-space outputs (scale-invariant).

Run:
    python examples/ex_B_nn_diag_v2.py --n-train 20000 --window 100
    python examples/ex_B_nn_diag_v2.py --ckpt nn_v2_best.pt   # skip training
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdm import StateSpaceModel, NoiseBasis, MDM
from mdm.neural_ver2.dataset import generate_dataset_v2
from mdm.neural_ver2.train import train_v2, load_v2
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

# ── true Q, R diagonal (from Example B alpha_true) ────────────────────
_basis_full = NoiseBasis(BQ, BR)
_Q_true, _R_true = _basis_full.to_QR(alpha_true)
Q_DIAG_TRUE = np.diag(_Q_true)  # [Q11, Q22, Q33]
R_DIAG_TRUE = np.diag(_R_true)  # [R11, R22, R33]

# Sampling box (log-uniform): bounds on each diagonal element
LO_Q = np.array([0.1, 0.1, 0.1])
HI_Q = np.array([5.0, 5.0, 5.0])
LO_R = np.array([0.2, 0.2, 0.2])
HI_R = np.array([8.0, 8.0, 8.0])

_SPEC_RAD = max(abs(np.linalg.eigvals(F)))
_SAFE_TAU = int(14 / np.log10(_SPEC_RAD)) if _SPEC_RAD > 1 else int(1e9)


def _build(tau):
    model = StateSpaceModel.from_constant(F, E, H, D, tau, G_fn, u_fn)
    return model


def _build_mdm(tau):
    model = _build(tau)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


# ── build one input window from a simulated trajectory ────────────────
def _make_window(model, Q, R, rng, window, n_imu=6):
    """Return one (n_ch, window) float32 input array."""
    nw = model.nw
    nz = model.nz
    tau = model.tau

    # need w_k for the IMU proxy; regenerate noise arrays
    cQ = np.linalg.cholesky(Q)
    cR = np.linalg.cholesky(R)
    w_seq = (cQ @ rng.standard_normal((nw, tau))).T  # (tau, nw)
    _ = (cR @ rng.standard_normal((model.nv, tau))).T
    z_list = model.simulate(Q, R, rng)

    start = rng.integers(0, max(tau - window, 1))
    Z = np.stack([z_list[start + t] for t in range(window)])  # (W, nz)
    Ew = w_seq[start : start + window] @ model.E[start].T  # (W, nx)
    if Ew.shape[1] >= n_imu:
        imu = Ew[:, :n_imu]
    else:
        imu = np.hstack([Ew, np.zeros((window, n_imu - Ew.shape[1]))])
    return np.hstack([Z, imu]).T.astype(np.float32)  # (n_ch, W)


# ── run comparison: NN v2 vs MDM ──────────────────────────────────────
def run_comparison(
    model, mdm_basis, est, net, MC=100, window=100, n_imu=6, tau=1000, seed=0
):
    import torch

    rng = np.random.default_rng(seed)
    Q_use = np.diag(Q_DIAG_TRUE)
    R_use = np.diag(R_DIAG_TRUE)
    nw = model.nw
    nv = model.nv

    nn_q, nn_r, mdm_q, mdm_r = [], [], [], []
    for _ in range(MC):
        # NN v2: from raw window
        x = _make_window(model, Q_use, R_use, rng, window, n_imu)
        with torch.no_grad():
            pred = net.predict(torch.tensor(x[None], dtype=torch.float32))[0].numpy()
        nn_q.append(pred[:nw])
        nn_r.append(pred[nw:])

        # MDM: from full trajectory, recover full Q/R then extract diagonal
        z = model.simulate(Q_use, R_use, rng)
        a_hat = est.fit_ordinary(z)
        Q_hat, R_hat = mdm_basis.to_QR(a_hat)
        mdm_q.append(np.diag(Q_hat))
        mdm_r.append(np.diag(R_hat))

    return (np.array(nn_q), np.array(nn_r), np.array(mdm_q), np.array(mdm_r))


# ── plot ──────────────────────────────────────────────────────────────
def plot_comparison(nn_q, nn_r, mdm_q, mdm_r, out, tau, window, MC):
    from mdm.neural.losses import stein_loss
    from mdm.linalg import psd_project, is_psd
    import matplotlib.patches as mpatches

    nw, nv = nn_q.shape[1], nn_r.shape[1]
    na = nw + nv  # total diagonal elements = 6
    c_nn = "darkorange"
    c_mdm = "steelblue"
    colors_mdm = plt.cm.tab10.colors
    colors_nn = plt.cm.Set2.colors

    # ── per-MC Stein and Frobenius ────────────────────────────────────
    Q_true = np.diag(Q_DIAG_TRUE)
    R_true = np.diag(R_DIAG_TRUE)

    def _metrics(q_ests, r_ests):
        steins, frobs, pds = [], [], []
        for q, r in zip(q_ests, r_ests):
            Qh = np.diag(q)
            Rh = np.diag(r)
            pd = is_psd(Qh) and is_psd(Rh)
            pds.append(float(pd))
            steins.append(
                stein_loss(psd_project(Qh), Q_true)
                + stein_loss(psd_project(Rh), R_true)
            )
            frobs.append(
                np.linalg.norm(Qh - Q_true, "fro") ** 2
                + np.linalg.norm(Rh - R_true, "fro") ** 2
            )
        return np.array(steins), np.array(frobs), np.array(pds)

    mdm_s, mdm_f, mdm_pd = _metrics(mdm_q, mdm_r)
    nn_s, nn_f, nn_pd = _metrics(nn_q, nn_r)

    # ── figure layout: 3 rows ─────────────────────────────────────────
    #   row 0: MDM diagonal violin  (na panels, Q then R)
    #   row 1: NN  diagonal violin  (na panels)
    #   row 2: Stein + Frobenius + PD rate bars
    fig = plt.figure(figsize=(3 * na, 14))
    fig.suptitle(
        f"NN v2 (1D-CNN window, diag output)  vs  MDM (ordinary)\n"
        f"MC={MC}  $\\tau$={tau}  window={window}  "
        f"|  NN input: [{nw+6}-ch raw window]  MDM input: full trajectory",
        fontsize=10,
    )
    gs_top = fig.add_gridspec(1, na, top=0.90, bottom=0.65)
    gs_mid = fig.add_gridspec(1, na, top=0.60, bottom=0.35)
    gs_bot = fig.add_gridspec(
        1, 3, top=0.28, bottom=0.04, left=0.06, right=0.97, wspace=0.30
    )

    # true values and labels for each panel
    true_vals = list(Q_DIAG_TRUE) + list(R_DIAG_TRUE)
    pan_labels = [f"Q[{i+1},{i+1}]" for i in range(nw)] + [
        f"R[{i+1},{i+1}]" for i in range(nv)
    ]

    def _vp_row(gs, ests_q, ests_r, colors, row_label, pd_flags):
        all_ests = np.hstack([ests_q, ests_r])  # (MC, na)
        pd_rate = 100 * pd_flags.mean()
        for i in range(na):
            ax = fig.add_subplot(gs[0, i])
            col = all_ests[:, i]
            true = true_vals[i]
            c = colors[i % len(colors)]
            # violin
            vp = ax.violinplot(
                col, positions=[0], widths=0.8, showmedians=False, showextrema=False
            )
            for b in vp["bodies"]:
                b.set_facecolor(c)
                b.set_alpha(0.35)
                b.set_edgecolor("none")
            # box
            q1, med, q3 = np.percentile(col, [25, 50, 75])
            iqr = q3 - q1
            lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            wlo = col[col >= lo_f].min()
            whi = col[col <= hi_f].max()
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (-0.22, q1),
                    0.44,
                    iqr,
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
            # outliers
            out_pts = col[(col < lo_f) | (col > hi_f)]
            if out_pts.size:
                xj = np.random.default_rng(i).uniform(-0.15, 0.15, out_pts.size)
                ax.scatter(
                    xj,
                    out_pts,
                    color=c,
                    s=16,
                    alpha=0.7,
                    edgecolors="k",
                    lw=0.4,
                    zorder=5,
                    label=f"{out_pts.size} outliers",
                )
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.5)
            # true line
            ax.axhline(
                true,
                color="crimson",
                lw=1.8,
                ls="--",
                zorder=6,
                label=f"true={true:.3f}",
            )
            ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
            ax.set_xlabel(pan_labels[i], fontsize=10)
            ax.set_title(
                f"std={col.std():.3f}  S.cov={col.var():.3f}\n"
                f"bias={abs(col.mean()-true):.3f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.grid(axis="y", ls=":", alpha=0.5)
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.yaxis.set_tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel(row_label, fontsize=9, labelpad=6)
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

    _vp_row(gs_top, mdm_q, mdm_r, colors_mdm, "MDM  (ordinary, diag extracted)", mdm_pd)
    _vp_row(gs_mid, nn_q, nn_r, colors_nn, "NN v2  (1D-CNN window input)", nn_pd)

    # ── bottom bars ───────────────────────────────────────────────────
    def _gbar(ax, mv, nv_, title):
        means = [mv.mean(), nv_.mean()]
        stds = [mv.std(), nv_.std()]
        ax.bar(
            [0, 1],
            means,
            yerr=stds,
            width=0.5,
            color=[c_mdm, c_nn],
            alpha=0.82,
            edgecolor="k",
            lw=0.8,
            capsize=8,
            error_kw=dict(lw=1.8, ecolor="black"),
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MDM", "NN v2"], fontsize=10)
        ax.set_title(
            f"{title}\nMDM mean={means[0]:.3f} std={stds[0]:.3f}   "
            f"NN mean={means[1]:.3f} std={stds[1]:.3f}",
            fontsize=8.5,
        )
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_tick_params(labelsize=8)

    _gbar(
        fig.add_subplot(gs_bot[0, 0]),
        mdm_s,
        nn_s,
        "Stein$(\\hat{Q},Q^*)+$Stein$(\\hat{R},R^*)$",
    )
    _gbar(
        fig.add_subplot(gs_bot[0, 1]),
        mdm_f,
        nn_f,
        r"$\|\hat{Q}-Q^*\|_F^2+\|\hat{R}-R^*\|_F^2$",
    )

    # PD rate bar
    ax_pd = fig.add_subplot(gs_bot[0, 2])
    pd_means = [100 * mdm_pd.mean(), 100 * nn_pd.mean()]
    ax_pd.bar(
        [0, 1],
        pd_means,
        width=0.5,
        color=[c_mdm, c_nn],
        alpha=0.82,
        edgecolor="k",
        lw=0.8,
    )
    ax_pd.set_xticks([0, 1])
    ax_pd.set_xticklabels(["MDM", "NN v2"], fontsize=10)
    ax_pd.set_ylim(0, 110)
    ax_pd.set_ylabel("PD rate (%)", fontsize=9)
    ax_pd.set_title(
        f"Rate: diag(Q̂) and diag(R̂) both PD\n"
        f"MDM={pd_means[0]:.1f}%   NN v2={pd_means[1]:.1f}%",
        fontsize=8.5,
    )
    for xi, val in enumerate(pd_means):
        ax_pd.text(
            xi, val + 1.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold"
        )
    ax_pd.grid(axis="y", ls=":", alpha=0.5)
    ax_pd.spines[["top", "right"]].set_visible(False)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # console summary
    print(
        f"\n{'':10s}  {'NN v2 mean':>12s}  {'NN v2 std':>10s}"
        f"  {'MDM mean':>10s}  {'MDM std':>10s}  {'true':>8s}"
    )
    for i in range(nw):
        t = Q_DIAG_TRUE[i]
        print(
            f"  Q[{i+1},{i+1}]    {nn_q[:,i].mean():12.4f}  {nn_q[:,i].std():10.4f}"
            f"  {mdm_q[:,i].mean():10.4f}  {mdm_q[:,i].std():10.4f}  {t:8.4f}"
        )
    for i in range(nv):
        t = R_DIAG_TRUE[i]
        print(
            f"  R[{i+1},{i+1}]    {nn_r[:,i].mean():12.4f}  {nn_r[:,i].std():10.4f}"
            f"  {mdm_r[:,i].mean():10.4f}  {mdm_r[:,i].std():10.4f}  {t:8.4f}"
        )
    print(f"\n  Stein MDM={mdm_s.mean():.3f}  NN v2={nn_s.mean():.3f}")
    print(f"  Frob  MDM={mdm_f.mean():.3f}  NN v2={nn_f.mean():.3f}")
    print(f"  PD%   MDM={pd_means[0]:.1f}%  NN v2={pd_means[1]:.1f}%")


# ── entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tau",
        type=int,
        default=1000,
        help=f"trajectory length (single-traj safe limit ~{_SAFE_TAU})",
    )
    p.add_argument("--window", type=int, default=100, help="CNN input window length W")
    p.add_argument(
        "--n-imu", type=int, default=6, help="IMU proxy channels appended to z_k"
    )
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--mc", type=int, default=100, help="Monte Carlo runs for evaluation"
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="nn_v2_best.pt",
        help="checkpoint path (skip training if exists)",
    )
    p.add_argument("--out", type=str, default="diag_nn_v2_vs_mdm.png")
    args = p.parse_args()

    model = _build(args.tau)
    mdm_model, mdm_basis, est = _build_mdm(args.tau)
    n_ch = model.nz + args.n_imu

    if not os.path.exists(args.ckpt):
        print(f"No checkpoint at '{args.ckpt}', training NN v2...")
        # derive prefix by stripping _best.pt or .pt so train_v2 saves to exactly args.ckpt
        stem = os.path.splitext(args.ckpt)[0]  # e.g. "nn_v2_best"
        prefix = stem[:-5] if stem.endswith("_best") else stem  # e.g. "nn_v2"
        train_v2(
            model,
            model.nw,
            model.nv,
            LO_Q,
            HI_Q,
            LO_R,
            HI_R,
            n_imu=args.n_imu,
            window=args.window,
            n_train=args.n_train,
            n_val=args.n_val,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            out_prefix=prefix,
        )
    else:
        print(f"Loading checkpoint from '{args.ckpt}'")

    net, ckpt_info = load_v2(args.ckpt)
    print(
        f"Loaded NN v2: "
        f"channels={ckpt_info['n_channels']}  "
        f"window={ckpt_info['window']}  "
        f"nw={ckpt_info['nw']}  nv={ckpt_info['nv']}"
    )

    print(f"\nRunning comparison (MC={args.mc})...")
    nn_q, nn_r, mdm_q, mdm_r = run_comparison(
        mdm_model,
        mdm_basis,
        est,
        net,
        MC=args.mc,
        window=args.window,
        n_imu=args.n_imu,
        tau=args.tau,
    )

    plot_comparison(
        nn_q,
        nn_r,
        mdm_q,
        mdm_r,
        out=args.out,
        tau=args.tau,
        window=args.window,
        MC=args.mc,
    )
