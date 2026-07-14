#!/usr/bin/env python
# examples/ex_B_nn_v2_vs_mdm.py
r"""
Comparison: NN v2 (window-based CNN) vs MDM ordinary estimator.

Plots the SAME layout as ex_B_plot_alpha_nn.py but with the v2 NN:
  Row 1: violin plots of diag(Q) estimates  -- MDM (extract diag) vs NN v2 (direct output)
  Row 2: violin plots of diag(R) estimates  -- same
  Row 3: Stein + Frobenius bar charts (MDM vs NN v2)
  Row 4: PD rate bar + Stein PD-only bar

Ground truth: diag(Q_true) and diag(R_true).
Both estimators are compared on the same diagonal targets because NN v2 is
designed for diagonal covariance estimation only.

MDM uses the full trajectory (tau steps).
NN v2 uses a single window of W steps extracted from the same trajectory.

Run:
    python examples/ex_B_nn_v2_vs_mdm.py --ckpt nn_v2_best.pt --mc 200 --window 100
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mdm import StateSpaceModel, NoiseBasis, MDM
from mdm.neural_ver2.train import load_v2
from mdm.neural.losses import stein_loss
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


# ── ground truth diagonals ────────────────────────────────────────────
def _build(tau):
    model = StateSpaceModel.from_constant(F, E, H, D, tau, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


def _gt_diags(basis):
    Q_true, R_true = basis.to_QR(alpha_true)
    return np.diag(Q_true), np.diag(R_true)  # diag(Q), diag(R)


# ── build one raw input window ────────────────────────────────────────
def _make_window(model, Q, R, rng, window, n_imu):
    """Same logic as dataset_v2.generate_dataset_v2, one sample."""
    nw = model.nw
    nz = model.nz
    tau = model.tau
    cQ = np.linalg.cholesky(Q)
    w_seq = (cQ @ rng.standard_normal((nw, tau))).T  # (tau, nw)
    z_list = model.simulate(Q, R, rng)
    start = rng.integers(0, max(tau - window, 1))
    Z = np.stack([z_list[start + t] for t in range(window)])  # (W, nz)
    E_k = model.E[start]
    imu_raw = w_seq[start : start + window] @ E_k.T  # (W, nx)
    if imu_raw.shape[1] >= n_imu:
        imu = imu_raw[:, :n_imu]
    else:
        pad = np.zeros((window, n_imu - imu_raw.shape[1]))
        imu = np.hstack([imu_raw, pad])
    return np.hstack([Z, imu]).T.astype(np.float32), z_list  # (n_ch, W), z_list


# ── per-run metrics (diagonal-only ground truth) ──────────────────────
def _diag_metrics(q_ests, r_ests, q_true, r_true):
    """
    Stein and Frobenius vs diagonal ground truth, plus PD flag.
    q_ests, r_ests : (MC, nw) and (MC, nv)
    """
    steins, frobs, pd_flags = [], [], []
    for q, r in zip(q_ests, r_ests):
        Qh = np.diag(q)
        Rh = np.diag(r)
        Qt = np.diag(q_true)
        Rt = np.diag(r_true)
        pd = is_psd(Qh) and is_psd(Rh)
        pd_flags.append(float(pd))
        steins.append(stein_loss(psd_project(Qh), Qt) + stein_loss(psd_project(Rh), Rt))
        frobs.append(
            np.linalg.norm(Qh - Qt, "fro") ** 2 + np.linalg.norm(Rh - Rt, "fro") ** 2
        )
    return np.array(steins), np.array(frobs), np.array(pd_flags)


# ── MC run ────────────────────────────────────────────────────────────
def run(MC=200, TAU=1000, window=100, ckpt="nn_v2_best.pt", n_imu=6, seed=0):
    import torch

    if TAU**0.01 > 1 and _SPEC_RAD**TAU > 1e14:
        print(
            f"\n  WARNING: tau={TAU} may cause numerical issues "
            f"(1.01^{TAU}={_SPEC_RAD**TAU:.1e}). Use --tau <= {_SAFE_TAU}.\n"
        )

    model, basis, est = _build(TAU)
    q_true, r_true = _gt_diags(basis)
    Q_use = np.diag(q_true)
    R_use = np.diag(r_true)

    net, ckpt_info = load_v2(ckpt)
    n_imu = ckpt_info.get("n_imu", n_imu)
    W = ckpt_info.get("window", window)
    print(
        f"Loaded {ckpt}: window={W}  n_imu={n_imu}  nw={ckpt_info['nw']}  nv={ckpt_info['nv']}"
    )

    rng = np.random.default_rng(seed)
    mdm_q, mdm_r, nn_q, nn_r = [], [], [], []

    for i in range(MC):
        # build window and z_list together (same rng state)
        x_win, z_list = _make_window(model, Q_use, R_use, rng, W, n_imu)

        # NN v2: predict diag(Q), diag(R) from window
        with torch.no_grad():
            pred = net.predict(torch.tensor(x_win[None], dtype=torch.float32))[
                0
            ].numpy()
        nn_q.append(pred[: model.nw])
        nn_r.append(pred[model.nw :])

        # MDM: use the SAME z_list, extract diag from full alpha estimate
        a_hat = est.fit_ordinary(z_list)
        Q_hat, R_hat = basis.to_QR(a_hat)
        mdm_q.append(np.diag(Q_hat))
        mdm_r.append(np.diag(R_hat))

        if (i + 1) % max(1, MC // 5) == 0:
            print(f"  {i+1}/{MC}")

    mdm_q = np.array(mdm_q)
    mdm_r = np.array(mdm_r)
    nn_q = np.array(nn_q)
    nn_r = np.array(nn_r)

    mdm_s, mdm_f, mdm_pd = _diag_metrics(mdm_q, mdm_r, q_true, r_true)
    nn_s, nn_f, nn_pd = _diag_metrics(nn_q, nn_r, q_true, r_true)

    return (
        mdm_q,
        mdm_r,
        nn_q,
        nn_r,
        mdm_s,
        mdm_f,
        mdm_pd,
        nn_s,
        nn_f,
        nn_pd,
        q_true,
        r_true,
        W,
    )


# ── plotting ──────────────────────────────────────────────────────────
def plot(
    mdm_q,
    mdm_r,
    nn_q,
    nn_r,
    mdm_s,
    mdm_f,
    mdm_pd,
    nn_s,
    nn_f,
    nn_pd,
    q_true,
    r_true,
    window,
    out="nn_v2_vs_mdm.png",
    tau=1000,
):

    MC = mdm_q.shape[0]
    nw = mdm_q.shape[1]
    nv = mdm_r.shape[1]
    na = max(nw, nv)  # columns
    colors_mdm = plt.cm.tab10.colors
    colors_nn = plt.cm.Set2.colors

    fig = plt.figure(figsize=(3.5 * na, 18))
    fig.suptitle(
        f"NN v2 (window CNN, W={window}) vs MDM ordinary — Example B\n"
        f"MC={MC}  τ={tau}  |  diag(Q) and diag(R) estimates vs diag(Q_true), diag(R_true)",
        fontsize=10,
    )

    gs_q = fig.add_gridspec(1, nw, top=0.93, bottom=0.72)
    gs_r = fig.add_gridspec(1, nv, top=0.67, bottom=0.46)
    gs_b1 = fig.add_gridspec(
        1, 2, top=0.40, bottom=0.24, left=0.08, right=0.96, wspace=0.30
    )
    gs_b2 = fig.add_gridspec(
        1, 2, top=0.18, bottom=0.02, left=0.08, right=0.96, wspace=0.30
    )

    def _vp(ax, col, true, c, xlabel, pd_rate=None):
        _violin_box(ax, col, c)
        ax.axhline(
            true, color="crimson", lw=1.8, ls="--", zorder=6, label=f"true={true:.3f}"
        )
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(
            f"std={col.std():.3f}  S.cov={col.var():.3f}\n"
            f"bias={abs(col.mean()-true):.3f}",
            fontsize=8,
        )
        ax.yaxis.set_tick_params(labelsize=7)
        if pd_rate is not None:
            ax.annotate(
                f"PD: {pd_rate:.1f}%",
                xy=(1.05, 0.5),
                xycoords="axes fraction",
                fontsize=8,
                color="darkgreen" if pd_rate >= 50 else "crimson",
                fontweight="bold",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
            )

    # ── Row 1: diag(Q) violins (MDM left, NN v2 right per element) ───
    for i in range(nw):
        ax_m = fig.add_subplot(gs_q[0, i])
        _vp(
            ax_m,
            mdm_q[:, i],
            q_true[i],
            colors_mdm[i % len(colors_mdm)],
            f"MDM  $Q_{{{i+1}{i+1}}}$",
            pd_rate=100 * mdm_pd.mean() if i == nw - 1 else None,
        )
        if i == 0:
            ax_m.set_ylabel("MDM  diag(Q) estimates", fontsize=9, labelpad=6)

    # NN v2 diag(Q) — overlay on same column positions using twin axes
    for i in range(nw):
        ax_n = fig.add_subplot(gs_q[0, i], label=f"nn_q_{i}")
        ax_n.patch.set_visible(False)
        _vp(
            ax_n,
            nn_q[:, i],
            q_true[i],
            colors_nn[i % len(colors_nn)],
            f"NN v2  $Q_{{{i+1}{i+1}}}$",
            pd_rate=100 * nn_pd.mean() if i == nw - 1 else None,
        )

    # ── Actually use separate subplots for MDM and NN rows ───────────
    # (simpler than twin axes — redo with proper separate gridspecs)
    fig.clear()
    gs_mdm_q = fig.add_gridspec(1, nw, top=0.93, bottom=0.78)
    gs_nn_q = fig.add_gridspec(1, nw, top=0.74, bottom=0.59)
    gs_mdm_r = fig.add_gridspec(1, nv, top=0.55, bottom=0.40)
    gs_nn_r = fig.add_gridspec(1, nv, top=0.36, bottom=0.21)
    gs_b1 = fig.add_gridspec(
        1, 2, top=0.17, bottom=0.10, left=0.08, right=0.96, wspace=0.30
    )
    gs_b2 = fig.add_gridspec(
        1, 2, top=0.06, bottom=0.00, left=0.08, right=0.96, wspace=0.30
    )
    fig.suptitle(
        f"NN v2 (window CNN, W={window}) vs MDM ordinary — Example B\n"
        f"MC={MC}  τ={tau}  |  diag(Q) and diag(R) estimates vs true diagonals",
        fontsize=10,
        y=0.99,
    )

    row_specs = [
        (mdm_q, q_true, colors_mdm, "MDM   diag(Q)", gs_mdm_q, mdm_pd),
        (nn_q, q_true, colors_nn, "NN v2 diag(Q)", gs_nn_q, nn_pd),
        (mdm_r, r_true, colors_mdm, "MDM   diag(R)", gs_mdm_r, mdm_pd),
        (nn_r, r_true, colors_nn, "NN v2 diag(R)", gs_nn_r, nn_pd),
    ]
    for ests, true_diag, colors, ylabel, gs, pd_flags in row_specs:
        n_cols = ests.shape[1]
        pd_rate = 100 * pd_flags.mean()
        for i in range(n_cols):
            ax = fig.add_subplot(gs[0, i])
            _violin_box(ax, ests[:, i], colors[i % len(colors)])
            ax.axhline(
                true_diag[i],
                color="crimson",
                lw=1.8,
                ls="--",
                zorder=6,
                label=f"true={true_diag[i]:.3f}",
            )
            ax.legend(fontsize=6, loc="upper right", framealpha=0.6)
            ax.set_xlabel(f"$[{i+1},{i+1}]$", fontsize=10)
            ax.set_title(
                f"std={ests[:,i].std():.3f}  S.cov={ests[:,i].var():.3f}\n"
                f"bias={abs(ests[:,i].mean()-true_diag[i]):.3f}",
                fontsize=7.5,
            )
            ax.yaxis.set_tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, labelpad=6)
            if i == n_cols - 1:
                ax.annotate(
                    f"PD:\n{pd_rate:.1f}%",
                    xy=(1.05, 0.5),
                    xycoords="axes fraction",
                    fontsize=8,
                    color="darkgreen" if pd_rate >= 50 else "crimson",
                    fontweight="bold",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8
                    ),
                )

    # ── Row 5-6: metric bar charts ────────────────────────────────────
    def _gbar(ax, v1, v2, lbl1, lbl2, title):
        means = [v1.mean(), v2.mean()]
        stds = [v1.std(), v2.std()]
        ax.bar(
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
        ax.set_xticks([0, 1])
        ax.set_xticklabels([lbl1, lbl2], fontsize=10)
        ax.set_title(
            f"{title}\n{lbl1} mean={means[0]:.3f} std={stds[0]:.3f}   "
            f"{lbl2} mean={means[1]:.3f} std={stds[1]:.3f}",
            fontsize=8,
        )
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_tick_params(labelsize=8)

    _gbar(
        fig.add_subplot(gs_b1[0, 0]),
        mdm_s,
        nn_s,
        "MDM",
        "NN v2",
        "Stein(diag Q + diag R vs true diagonals)",
    )
    _gbar(
        fig.add_subplot(gs_b1[0, 1]),
        mdm_f,
        nn_f,
        "MDM",
        "NN v2",
        r"$\|$diag(Q)$-$diag($Q^*$)$\|^2+\|$diag(R)$-$diag($R^*$)$\|^2$",
    )

    # PD rate bar
    ax_pd = fig.add_subplot(gs_b2[0, 0])
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
    ax_pd.set_xticklabels(["MDM", "NN v2"], fontsize=10)
    ax_pd.set_ylim(0, 110)
    ax_pd.set_ylabel("PD rate %", fontsize=9)
    ax_pd.set_title(
        f"Q and R both positive definite\n"
        f"MDM={pd_means[0]:.1f}%  NN v2={pd_means[1]:.1f}%",
        fontsize=8.5,
    )
    for xi, val in enumerate(pd_means):
        ax_pd.text(
            xi, val + 1.5, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold"
        )
    ax_pd.grid(axis="y", ls=":", alpha=0.5)
    ax_pd.spines[["top", "right"]].set_visible(False)

    # Stein PD-only
    ax_spd = fig.add_subplot(gs_b2[0, 1])
    mdm_pd_mask = mdm_pd == 1
    nn_pd_mask = nn_pd == 1
    n_m = int(mdm_pd_mask.sum())
    n_n = int(nn_pd_mask.sum())
    if n_m >= 2 and n_n >= 2:
        _gbar(
            ax_spd,
            mdm_s[mdm_pd_mask],
            nn_s[nn_pd_mask],
            f"MDM\n(PD n={n_m})",
            f"NN v2\n(PD n={n_n})",
            "Stein  [PD-only subset]",
        )
    else:
        ax_spd.text(
            0.5,
            0.5,
            f"MDM PD: {n_m}  NN PD: {n_n}\nNot enough for comparison",
            ha="center",
            va="center",
            transform=ax_spd.transAxes,
        )
        ax_spd.axis("off")

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # console summary
    print(f"\n  {'metric':<22s}  {'MDM':>10s}  {'NN v2':>10s}")
    for name, mv, nv_ in [
        ("Stein mean", mdm_s.mean(), nn_s.mean()),
        ("Stein std", mdm_s.std(), nn_s.std()),
        ("Frobenius mean", mdm_f.mean(), nn_f.mean()),
        ("PD rate %", 100 * mdm_pd.mean(), 100 * nn_pd.mean()),
    ]:
        print(f"  {name:<22s}  {mv:10.4f}  {nv_:10.4f}")
    print(f"\n  diag(Q_true) = {q_true.round(4)}")
    print(f"  diag(R_true) = {r_true.round(4)}")
    print(f"\n  MDM diag(Q) mean  = {mdm_q.mean(0).round(4)}")
    print(f"  NN v2 diag(Q) mean = {nn_q.mean(0).round(4)}")
    print(f"  MDM diag(R) mean  = {mdm_r.mean(0).round(4)}")
    print(f"  NN v2 diag(R) mean = {nn_r.mean(0).round(4)}")


# ── entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=200)
    p.add_argument("--tau", type=int, default=1000, help=f"safe limit ~{_SAFE_TAU}")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--n-imu", type=int, default=6)
    p.add_argument("--ckpt", type=str, default="nn_v2_best.pt")
    p.add_argument("--out", type=str, default="nn_v2_vs_mdm.png")
    args = p.parse_args()

    result = run(
        MC=args.mc, TAU=args.tau, window=args.window, ckpt=args.ckpt, n_imu=args.n_imu
    )
    (
        mdm_q,
        mdm_r,
        nn_q,
        nn_r,
        mdm_s,
        mdm_f,
        mdm_pd,
        nn_s,
        nn_f,
        nn_pd,
        q_true,
        r_true,
        W,
    ) = result

    plot(
        mdm_q,
        mdm_r,
        nn_q,
        nn_r,
        mdm_s,
        mdm_f,
        mdm_pd,
        nn_s,
        nn_f,
        nn_pd,
        q_true,
        r_true,
        window=W,
        out=args.out,
        tau=args.tau,
    )


import argparse
