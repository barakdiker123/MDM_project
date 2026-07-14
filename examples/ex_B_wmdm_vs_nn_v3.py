#!/usr/bin/env python
# examples/ex_B_wmdm_vs_nn_v3.py
r"""
Weighted MDM (segment-averaged) vs NN v3 — Example B.

Identical plot layout to ex_B_nn_v3_vs_mdm.py but using the
WEIGHTED MDM estimator (eq. 23) instead of ordinary MDM (eq. 21).

Weighted MDM procedure per segment:
  1. Collect covRes from seg_size trajectories
  2. Average covRes across trajectories (pool data)
  3. Call fit_weighted(covRes=avg_covRes):
       a. Ordinary MDM alpha_init = S^{-1} s
       b. Build weight matrix P(alpha_init)  (eq. 22)
       c. Solve weighted LS -> alpha_w       (eq. 23)

Flags:
    --mc, --seg-size, --tau, --window, --ckpt, --out   (same as ordinary script)

Run:
    python examples/ex_B_wmdm_vs_nn_v3.py --ckpt nn_v3_best.pt --mc 200 --seg-size 10 --window 200
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
from mdm.neural_ver3.train import train_v3, load_v3
from mdm.neural.losses import kl_loss
from mdm.linalg import psd_project, is_psd
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
from ex_B_plot_alpha import _violin_box

_SPEC_RAD = max(abs(np.linalg.eigvals(F)))
_SAFE_TAU = int(14 / np.log10(_SPEC_RAD)) if _SPEC_RAD > 1 else int(1e9)
_c_mdm = "steelblue"
_c_nn = "darkorange"


def _build(tau):
    model = StateSpaceModel.from_constant(F, E, H, D, tau, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


def _make_window(model, Q, R, rng, window, n_imu):
    nw = model.nw
    nz = model.nz
    tau = model.tau
    cQ = np.linalg.cholesky(Q)
    w_seq = (cQ @ rng.standard_normal((nw, tau))).T
    z_list = model.simulate(Q, R, rng)
    start = rng.integers(0, max(tau - window, 1))
    Z = np.stack([z_list[start + t] for t in range(window)])
    E_k = model.E[start]
    imu_raw = w_seq[start : start + window] @ E_k.T
    if imu_raw.shape[1] >= n_imu:
        imu = imu_raw[:, :n_imu]
    else:
        imu = np.hstack([imu_raw, np.zeros((window, n_imu - imu_raw.shape[1]))])
    return np.hstack([Z, imu]).T.astype(np.float32), z_list


def _metrics(Q_ests, R_ests, Q_true, R_true):
    """Full-matrix Stein (PD-only), Frobenius (all), PD rate."""
    steins_Q, steins_R = [], []  # only PD runs
    frobs_Q, frobs_R = [], []  # all runs
    pd_flags = []
    pd_mask = []
    for Qh, Rh in zip(Q_ests, R_ests):
        pd = is_psd(Qh) and is_psd(Rh)
        pd_flags.append(float(pd))
        pd_mask.append(pd)
        frobs_Q.append(np.linalg.norm(Qh - Q_true, "fro") ** 2)
        frobs_R.append(np.linalg.norm(Rh - R_true, "fro") ** 2)
        if pd:
            steins_Q.append(kl_loss(Qh, Q_true))
            steins_R.append(kl_loss(Rh, R_true))
    return (
        np.array(steins_Q),
        np.array(steins_R),  # PD-only Stein
        np.array(frobs_Q),
        np.array(frobs_R),  # all-runs Frobenius
        np.array(pd_flags),
        np.array(pd_mask),
    )


def run(
    MC=200,
    TAU=1000,
    window=100,
    seg_size=10,
    ckpt="nn_v3_best.pt",
    n_imu=6,
    n_train=20000,
    epochs=200,
    seed=0,
):
    import torch

    assert MC % seg_size == 0, f"MC={MC} must be divisible by seg_size={seg_size}"
    n_segs = MC // seg_size

    if _SPEC_RAD**TAU > 1e14:
        print(
            f"\n  WARNING: tau={TAU} may cause numerical issues. "
            f"Use --tau <= {_SAFE_TAU}.\n"
        )

    model, basis, est = _build(TAU)
    Q_true, R_true = basis.to_QR(alpha_true)

    if not os.path.exists(ckpt):
        print(f"No checkpoint at '{ckpt}', training NN v3 ...")
        stem = os.path.splitext(ckpt)[0]
        prefix = stem[:-5] if stem.endswith("_best") else stem
        train_v3(
            model,
            model.nw,
            model.nv,
            n_imu=n_imu,
            window=window,
            n_train=n_train,
            n_val=n_train // 5,
            epochs=epochs,
            out_prefix=prefix,
        )
    else:
        print(f"Loading '{ckpt}'")

    net, info = load_v3(ckpt)
    W = info.get("window", window)
    print(
        f"Loaded NN v3: window={W}  nw={info['nw']}  nv={info['nv']}"
        f"  n_channels={info['n_channels']}"
    )
    print(
        f"Weighted MDM (eq.23): {n_segs} segments x {seg_size} traj  "
        f"-> avg covRes per segment, fit_weighted\n"
        f"NN v3: {MC} single windows"
    )

    rng = np.random.default_rng(seed)

    # collect per-trajectory covRes lists (preserve window structure for P)
    all_covres = []
    nn_Q, nn_R = [], []

    print(f"Simulating {MC} trajectories ...")
    for i in range(MC):
        x_win, z_list = _make_window(model, Q_true, R_true, rng, W, n_imu)

        # NN v3: one window → (Q_hat, R_hat)
        with torch.no_grad():
            Qh_t, Rh_t = net.predict_QR(torch.tensor(x_win[None], dtype=torch.float32))
        nn_Q.append(Qh_t[0].numpy())
        nn_R.append(Rh_t[0].numpy())

        # weighted MDM: keep covRes list (window structure needed for P)
        all_covres.append(est.residue_cov(z_list))

        if (i + 1) % max(1, MC // 5) == 0:
            print(f"  {i+1}/{MC}")

    # ── for each segment: average covRes then call fit_weighted ────────
    n_windows = len(all_covres[0])
    mdm_Q, mdm_R, mdm_alpha = [], [], []
    for seg in range(n_segs):
        idx = range(seg * seg_size, (seg + 1) * seg_size)
        avg_covres = [
            np.mean([all_covres[i][k] for i in idx], axis=0) for k in range(n_windows)
        ]
        a_hat = est.fit_weighted(covRes=avg_covres)
        Qh, Rh = basis.to_QR(a_hat)
        mdm_Q.append(Qh)
        mdm_R.append(Rh)
        mdm_alpha.append(a_hat)

    mdm_Q = np.array(mdm_Q)  # (n_segs, nw, nw)
    mdm_R = np.array(mdm_R)  # (n_segs, nv, nv)
    mdm_alpha = np.array(mdm_alpha)  # (n_segs, n_alpha)
    nn_Q = np.array(nn_Q)
    nn_R = np.array(nn_R)  # (MC, nw, nw)
    mdm_sQ, mdm_sR, mdm_fQ, mdm_fR, mdm_pd, mdm_pd_mask = _metrics(
        mdm_Q, mdm_R, Q_true, R_true
    )
    nn_sQ, nn_sR, nn_fQ, nn_fR, nn_pd, nn_pd_mask = _metrics(nn_Q, nn_R, Q_true, R_true)
    return (
        mdm_Q,
        mdm_R,
        mdm_alpha,
        nn_Q,
        nn_R,
        mdm_sQ,
        mdm_sR,
        mdm_fQ,
        mdm_fR,
        mdm_pd,
        mdm_pd_mask,
        nn_sQ,
        nn_sR,
        nn_fQ,
        nn_fR,
        nn_pd,
        nn_pd_mask,
        Q_true,
        R_true,
        W,
        n_segs,
        seg_size,
    )


def _upper_entries(M):
    """Return upper-triangle entries (including diagonal) of a symmetric matrix."""
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx]


def plot(
    mdm_Q,
    mdm_R,
    mdm_alpha,
    nn_Q,
    nn_R,
    mdm_sQ,
    mdm_sR,
    mdm_fQ,
    mdm_fR,
    mdm_pd,
    mdm_pd_mask,
    nn_sQ,
    nn_sR,
    nn_fQ,
    nn_fR,
    nn_pd,
    nn_pd_mask,
    Q_true,
    R_true,
    window,
    n_segs,
    seg_size,
    out="wmdm_vs_nn_v3.png",
    tau=1000,
):
    MC = nn_Q.shape[0]
    na = mdm_alpha.shape[1]  # n_alpha = 6 for Example B
    nw = Q_true.shape[0]
    nv = R_true.shape[0]
    n_Q_entries = nw * (nw + 1) // 2
    n_R_entries = nv * (nv + 1) // 2
    colors_mdm = plt.cm.tab10.colors

    fig = plt.figure(figsize=(3.5 * na, 14))
    fig.suptitle(
        f"Weighted MDM segment-averaged vs NN v3 — Example B\n"
        f"Weighted MDM: {n_segs} segments × {seg_size} traj (avg covRes, fit_weighted)  |  "
        f"NN v3: {MC} single windows  |  τ={tau}",
        fontsize=10,
        y=0.99,
    )

    # 2 violin rows (alpha_1..alpha_6), then 2 bar rows
    gs_alpha = fig.add_gridspec(1, na, top=0.93, bottom=0.60)
    gs_b1 = fig.add_gridspec(
        1, 3, top=0.54, bottom=0.28, left=0.06, right=0.97, wspace=0.35
    )

    lbl_mdm = f"MDM\n({seg_size}/seg)"
    lbl_nn = "NN v3"

    # Stein: PD-only (passed in from run()).  Frobenius: all runs.
    n_mdm_pd = int(mdm_pd.sum())
    n_mdm = len(mdm_pd)
    n_nn_pd = int(nn_pd.sum())
    n_nn = len(nn_pd)

    def _split_bar(
        ax,
        mdm_Q_vals,
        nn_Q_vals,
        mdm_R_vals,
        nn_R_vals,
        title,
        ylabel,
        n_mdm_shown,
        n_mdm_tot,
        n_nn_shown,
        n_nn_tot,
    ):
        x = np.array([0, 1, 3, 4])
        means = np.array(
            [mdm_Q_vals.mean(), nn_Q_vals.mean(), mdm_R_vals.mean(), nn_R_vals.mean()]
        )
        stds = np.array(
            [mdm_Q_vals.std(), nn_Q_vals.std(), mdm_R_vals.std(), nn_R_vals.std()]
        )
        err_lo = np.minimum(stds, means)  # clip at 0 (metrics always >= 0)
        ax.bar(
            x,
            means,
            yerr=[err_lo, stds],
            width=0.7,
            color=[_c_mdm, _c_nn, _c_mdm, _c_nn],
            alpha=0.82,
            edgecolor="k",
            lw=0.8,
            capsize=6,
            error_kw=dict(lw=1.5, ecolor="black"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"wMDM Q\n{n_mdm_shown}/{n_mdm_tot}",
                f"NN v3 Q\n{n_nn_shown}/{n_nn_tot}",
                f"wMDM R\n{n_mdm_shown}/{n_mdm_tot}",
                f"NN v3 R\n{n_nn_shown}/{n_nn_tot}",
            ],
            fontsize=8,
        )
        ax.axvline(2, color="gray", lw=0.8, ls=":")
        ax.set_title(
            f"{title}\n"
            f"Q: wMDM={means[0]:.3f} NN={means[1]:.3f}  |  "
            f"R: wMDM={means[2]:.3f} NN={means[3]:.3f}",
            fontsize=8,
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_tick_params(labelsize=8)

    # Stein bar: PD-only estimates, label shows n_PD / n_total
    _split_bar(
        fig.add_subplot(gs_b1[0, 0]),
        mdm_sQ,
        nn_sQ,
        mdm_sR,
        nn_sR,
        "KL divergence  [PD-only estimates]",
        "KL divergence",
        n_mdm_pd,
        n_mdm,
        n_nn_pd,
        n_nn,
    )

    # Frobenius bar: all estimates, label shows n_total / n_total
    _split_bar(
        fig.add_subplot(gs_b1[0, 1]),
        mdm_fQ,
        nn_fQ,
        mdm_fR,
        nn_fR,
        "Frobenius distance  [all estimates]",
        r"$\|\hat{M} - M^*\|_F^2$",
        n_mdm,
        n_mdm,
        n_nn,
        n_nn,
    )

    # PD rate
    ax_pd = fig.add_subplot(gs_b1[0, 2])
    pd_means = [100 * mdm_pd.mean(), 100 * nn_pd.mean()]
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
    ax_pd.set_xticklabels([f"MDM\n({seg_size}/seg)", "NN v3"], fontsize=9)
    ax_pd.set_ylim(0, 110)
    ax_pd.set_ylabel("PD rate %", fontsize=9)
    ax_pd.set_title(
        f"Q and R both positive definite\n"
        f"MDM={pd_means[0]:.1f}%  NN v3={pd_means[1]:.1f}%",
        fontsize=8.5,
    )
    for xi, val in enumerate(pd_means):
        ax_pd.text(
            xi, val + 1.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold"
        )
    ax_pd.grid(axis="y", ls=":", alpha=0.5)
    ax_pd.spines[["top", "right"]].set_visible(False)

    # ── alpha violin row ──────────────────────────────────────────────
    pd_rate = 100 * mdm_pd.mean()
    for i in range(na):
        ax = fig.add_subplot(gs_alpha[0, i])
        col = mdm_alpha[:, i]
        true = alpha_true[i]
        c = colors_mdm[i % len(colors_mdm)]
        _violin_box(ax, col, c)
        ax.axhline(
            true, color="crimson", lw=1.8, ls="--", zorder=6, label=f"true={true:.2g}"
        )
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.6)
        ax.set_xlabel(f"$\\hat{{\\alpha}}_{i+1}$", fontsize=11)
        ax.set_title(
            f"a{i+1}\nstd={col.std():.3f}  S.cov={col.var():.3f}\n"
            f"bias={abs(col.mean()-true):.3f}",
            fontsize=8.5,
        )
        ax.yaxis.set_tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel(f"wMDM ({seg_size}/seg)", fontsize=9, labelpad=6)
        if i == na - 1:
            ax.annotate(
                f"PD:\n{pd_rate:.1f}%",
                xy=(1.05, 0.5),
                xycoords="axes fraction",
                fontsize=9,
                color="darkgreen" if pd_rate >= 50 else "crimson",
                fontweight="bold",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
            )

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    print(f"\n  {'metric':<26s}  {'wMDM':>10s}  {'NN v3':>10s}")
    for name, mv, nv_ in [
        (
            "KL Q mean (PD-only)",
            mdm_sQ.mean() if len(mdm_sQ) else float("nan"),
            nn_sQ.mean() if len(nn_sQ) else float("nan"),
        ),
        (
            "KL R mean (PD-only)",
            mdm_sR.mean() if len(mdm_sR) else float("nan"),
            nn_sR.mean() if len(nn_sR) else float("nan"),
        ),
        ("Frobenius Q mean", mdm_fQ.mean(), nn_fQ.mean()),
        ("Frobenius R mean", mdm_fR.mean(), nn_fR.mean()),
        ("PD rate %", 100 * mdm_pd.mean(), 100 * nn_pd.mean()),
    ]:
        print(f"  {name:<26s}  {mv:10.4f}  {nv_:10.4f}")
    print(f"\n  Q_true:\n{Q_true.round(4)}")
    print(f"  wMDM mean Q_hat:\n{mdm_Q.mean(0).round(4)}")
    print(f"  NN v3 mean Q_hat:\n{nn_Q.mean(0).round(4)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=200)
    p.add_argument(
        "--seg-size",
        type=int,
        default=10,
        help="MDM trajectories averaged per segment (MC must be divisible)",
    )
    p.add_argument("--tau", type=int, default=1000)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--n-imu", type=int, default=6)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--ckpt", type=str, default="nn_v3_best.pt")
    p.add_argument("--out", type=str, default="wmdm_vs_nn_v3.png")
    args = p.parse_args()

    result = run(
        MC=args.mc,
        TAU=args.tau,
        window=args.window,
        seg_size=args.seg_size,
        ckpt=args.ckpt,
        n_imu=args.n_imu,
        n_train=args.n_train,
        epochs=args.epochs,
    )
    (
        mdm_Q,
        mdm_R,
        mdm_alpha,
        nn_Q,
        nn_R,
        mdm_sQ,
        mdm_sR,
        mdm_fQ,
        mdm_fR,
        mdm_pd,
        mdm_pd_mask,
        nn_sQ,
        nn_sR,
        nn_fQ,
        nn_fR,
        nn_pd,
        nn_pd_mask,
        Q_true,
        R_true,
        W,
        n_segs,
        seg_size,
    ) = result

    plot(
        mdm_Q,
        mdm_R,
        mdm_alpha,
        nn_Q,
        nn_R,
        mdm_sQ,
        mdm_sR,
        mdm_fQ,
        mdm_fR,
        mdm_pd,
        mdm_pd_mask,
        nn_sQ,
        nn_sR,
        nn_fQ,
        nn_fR,
        nn_pd,
        nn_pd_mask,
        Q_true,
        R_true,
        window=W,
        n_segs=n_segs,
        seg_size=seg_size,
        out=args.out,
        tau=args.tau,
    )

import argparse
