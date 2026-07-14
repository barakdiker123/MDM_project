#!/usr/bin/env python
# examples/ex_B_plot_alpha_nn.py
r"""
Side-by-side violin + box plots comparing the ordinary-MDM estimator and the
neural-network estimator on the Example-B system, using the SAME trajectories.

For each MC repeat:
  1. Simulate one trajectory (same Q, R, system matrices for both)
  2. Compute sufficient statistic s  (identical input for both estimators)
  3. MDM:  solve  S alpha = s  (linear, unbiased)
  4. NN:   forward pass  s -> alpha_hat  (learned, biased, lower variance)

If no checkpoint (nn_best.pt) is found in the working directory, the script
trains the network automatically before plotting.

Run:
    python examples/ex_B_plot_alpha_nn.py --mc 200 --tau 1000
    python examples/ex_B_plot_alpha_nn.py --mc 200 --tau 1000 --ckpt path/to/nn_best.pt

Flags:
    --mc        Monte-Carlo repeats (default 200)
    --tau       trajectory length   (default 1000, safe limit ~3200 for this system)
    --n-seg     segments averaged per estimate (default 1)
    --ckpt      path to nn_best.pt  (default: nn_best.pt in cwd)
    --n-train   training samples if network must be trained (default 20000)
    --epochs    training epochs     (default 150)
    --out       output PNG filename (default alpha_nn_vs_mdm.png)
"""

import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # -> mdm package
sys.path.insert(0, _HERE)  # -> examples/ as flat modules

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mdm import StateSpaceModel, NoiseBasis, MDM
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
from ex_B_plot_alpha import _compute_metrics, _violin_box, _metric_panel

# ------------------------------------------------------------------ helpers
_SPEC_RAD = max(abs(np.linalg.eigvals(F)))
_SAFE_TAU = int(14 / np.log10(_SPEC_RAD)) if _SPEC_RAD > 1 else int(1e9)


def _warn_if_unstable(tau):
    g = _SPEC_RAD**tau
    if g > 1e14:
        print(
            f"\n  WARNING: F is unstable (spectral radius {_SPEC_RAD:.4f}).\n"
            f"  At tau={tau}, state grows by {g:.1e} -- exceeds float64 precision.\n"
            f"  Use --tau <= {_SAFE_TAU} or increase --n-seg instead.\n"
        )


def _build(tau):
    model = StateSpaceModel.from_constant(F, E, H, D, tau, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


# ------------------------------------------------------------------ training
def _train(basis, est, model, n_train=20000, epochs=150, ckpt="nn_best.pt"):
    import torch
    from mdm.neural.dataset import make_feasible_sampler, generate_dataset
    from mdm.neural.network import CovarianceEstimator
    from mdm.neural.losses import frobenius_loss_torch

    lo = np.array([0.2, 0.2, -2.0, 0.5, 0.5, -1.0])
    hi = np.array([3.0, 3.0, 0.5, 4.0, 4.0, 2.5])
    sampler = make_feasible_sampler(basis, lo, hi)

    print(f"Training NN on {n_train} samples for {epochs} epochs ...")
    Xtr, Ytr = generate_dataset(model, basis, est, sampler, n_train, seed=0)
    Xva, Yva = generate_dataset(model, basis, est, sampler, n_train // 5, seed=1)

    s_mean = torch.tensor(Xtr.mean(0))
    s_std = torch.tensor(Xtr.std(0))
    net = CovarianceEstimator(s_mean, s_std, n_alpha=basis.n_alpha)
    BQt = torch.tensor(np.stack(basis.BQ), dtype=torch.float32)
    BRt = torch.tensor(np.stack(basis.BR), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
        batch_size=256,
        shuffle=True,
    )
    Xva_t = torch.tensor(Xva)
    Yva_t = torch.tensor(Yva)

    best = np.inf
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            frobenius_loss_torch(net(xb), yb, BQt, BRt).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = frobenius_loss_torch(net(Xva_t), Yva_t, BQt, BRt).item()
        if vl < best:
            best = vl
            torch.save(net.state_dict(), ckpt)
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  val={vl:.4f}  best={best:.4f}")
    print(f"Saved checkpoint -> {ckpt}")
    return ckpt


def _load_net(basis, ckpt):
    import torch
    from mdm.neural.network import CovarianceEstimator

    sd = torch.load(ckpt, map_location="cpu")
    net = CovarianceEstimator(sd["s_mean"], sd["s_std"], n_alpha=basis.n_alpha)
    net.load_state_dict(sd)
    net.eval()
    return net


# ------------------------------------------------------------------ MC run
def run(
    MC=200, TAU=1000, n_seg=1, ckpt="nn_best.pt", n_train=20000, epochs=150, seed=0
):
    import torch

    _warn_if_unstable(TAU)
    model, basis, est = _build(TAU)

    if not os.path.exists(ckpt):
        print(f"No checkpoint found at '{ckpt}'.")
        _train(basis, est, model, n_train=n_train, epochs=epochs, ckpt=ckpt)
    net = _load_net(basis, ckpt)

    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(seed)
    mdm_ests, nn_ests = [], []

    for i in range(MC):
        # aggregate n_seg segments
        ss = [
            est.sufficient_stat(est.residue_cov(model.simulate(Q, R, rng)))
            for _ in range(n_seg)
        ]
        s = np.mean(ss, axis=0)

        # MDM (same s)
        mdm_ests.append(np.linalg.solve(est.S, s))

        # NN (same s)
        with torch.no_grad():
            a_nn = net(torch.tensor(s[None, :], dtype=torch.float32)).numpy()[0]
        nn_ests.append(a_nn)

        if (i + 1) % 50 == 0:
            print(f"  MC {i+1}/{MC}")

    mdm_ests = np.array(mdm_ests)
    nn_ests = np.array(nn_ests)
    mdm_steins, mdm_frobs, mdm_pd = _compute_metrics(mdm_ests, basis)
    nn_steins, nn_frobs, nn_pd = _compute_metrics(nn_ests, basis)
    return (
        mdm_ests,
        nn_ests,
        basis,
        mdm_steins,
        mdm_frobs,
        mdm_pd,
        nn_steins,
        nn_frobs,
        nn_pd,
    )


# ------------------------------------------------------------------ plot
def plot(
    mdm_ests,
    nn_ests,
    basis,
    mdm_steins,
    mdm_frobs,
    mdm_pd,
    nn_steins,
    nn_frobs,
    nn_pd,
    out="alpha_nn_vs_mdm.png",
    tau=1000,
    n_seg=1,
):
    MC, na = mdm_ests.shape
    seg_str = f", {n_seg} segs" if n_seg > 1 else ""
    colors_mdm = plt.cm.tab10.colors
    colors_nn = plt.cm.Set2.colors

    # layout: 4 rows
    #   row 0: MDM alpha violins  (na panels)
    #   row 1: NN  alpha violins  (na panels)  + PD rate annotation
    #   row 2: Stein MDM vs NN  |  Frobenius MDM vs NN  (all MC runs)
    #   row 3: PD rate MDM vs NN  |  Stein MDM vs NN (PD-only runs)
    fig = plt.figure(figsize=(3 * na, 18))
    fig.suptitle(
        f"Ordinary MDM  vs  Neural Network  —  Example B\n"
        f"MC={MC}, $\\tau$={tau}{seg_str}   |   same trajectories, same $s$",
        fontsize=11,
    )
    gs_top = fig.add_gridspec(1, na, top=0.93, bottom=0.72)
    gs_mid = fig.add_gridspec(1, na, top=0.67, bottom=0.46)
    gs_bot = fig.add_gridspec(
        1, 2, top=0.40, bottom=0.24, left=0.08, right=0.96, wspace=0.30
    )
    gs_bot2 = fig.add_gridspec(
        1, 2, top=0.18, bottom=0.02, left=0.08, right=0.96, wspace=0.30
    )

    row_specs = [
        (mdm_ests, colors_mdm, "MDM (ordinary, eq. 21)", gs_top, mdm_pd),
        (nn_ests, colors_nn, "NN  (learned)", gs_mid, nn_pd),
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
            # annotate PD rate on last panel of each row
            if i == na - 1:
                ax.annotate(
                    f"PD rate\n(Q & R): {pd_rate:.1f}%",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    fontsize=9,
                    color="crimson" if pd_rate < 50 else "darkgreen",
                    fontweight="bold",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                    ),
                )

    # ---- row 2: Stein and Frobenius (all MC runs) ----
    c_mdm = "steelblue"
    c_nn = "darkorange"

    def _grouped_bar(ax, mdm_vals, nn_vals, title, ylabel=""):
        x = np.array([0, 1])
        means = [mdm_vals.mean(), nn_vals.mean()]
        stds = [mdm_vals.std(), nn_vals.std()]
        ax.bar(
            x,
            means,
            yerr=stds,
            width=0.5,
            color=[c_mdm, c_nn],
            alpha=0.82,
            edgecolor="k",
            linewidth=0.8,
            capsize=8,
            error_kw=dict(lw=1.8, ecolor="black"),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["MDM", "NN"], fontsize=10)
        ax.set_title(
            f"{title}\n"
            f"MDM mean={means[0]:.3f} std={stds[0]:.3f}   "
            f"NN mean={means[1]:.3f} std={stds[1]:.3f}",
            fontsize=8.5,
        )
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_tick_params(labelsize=8)

    ax_s = fig.add_subplot(gs_bot[0, 0])
    ax_f = fig.add_subplot(gs_bot[0, 1])
    _grouped_bar(
        ax_s, mdm_steins, nn_steins, "Stein$(Q,Q^*)+$Stein$(R,R^*)$  [all runs]"
    )
    _grouped_bar(
        ax_f, mdm_frobs, nn_frobs, r"$\|Q-Q^*\|_F^2+\|R-R^*\|_F^2$  [all runs]"
    )

    # ---- row 3: PD rate bar  |  Stein (PD-only runs) ----
    ax_pd = fig.add_subplot(gs_bot2[0, 0])
    ax_spd = fig.add_subplot(gs_bot2[0, 1])

    # PD rate grouped bar
    pd_means = [100 * mdm_pd.mean(), 100 * nn_pd.mean()]
    ax_pd.bar(
        [0, 1],
        pd_means,
        width=0.5,
        color=[c_mdm, c_nn],
        alpha=0.82,
        edgecolor="k",
        linewidth=0.8,
    )
    ax_pd.set_xticks([0, 1])
    ax_pd.set_xticklabels(["MDM", "NN"], fontsize=10)
    ax_pd.set_ylim(0, 105)
    ax_pd.set_ylabel("PD rate (%)", fontsize=9)
    ax_pd.set_title(
        f"Rate: Q and R both positive definite\n"
        f"MDM = {pd_means[0]:.1f}%   NN = {pd_means[1]:.1f}%",
        fontsize=8.5,
    )
    for xi, val in enumerate(pd_means):
        ax_pd.text(
            xi,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
    ax_pd.grid(axis="y", ls=":", alpha=0.5)
    ax_pd.spines[["top", "right"]].set_visible(False)
    ax_pd.yaxis.set_tick_params(labelsize=8)

    # Stein filtered to PD-only runs (runs where BOTH mdm and nn are PD)
    pd_both = (mdm_pd == 1) & (nn_pd == 1)
    n_pd_both = pd_both.sum()
    if n_pd_both >= 2:
        mdm_s_pd = mdm_steins[pd_both]
        nn_s_pd = nn_steins[pd_both]
        _grouped_bar(
            ax_spd,
            mdm_s_pd,
            nn_s_pd,
            f"Stein  [PD-only runs,  n={n_pd_both}]\n"
            r"(runs where $\hat{Q},\hat{R}$ both PD for both estimators)",
        )
    else:
        ax_spd.text(
            0.5,
            0.5,
            f"Only {n_pd_both} runs have both estimators PD\n"
            "(increase MC or n_seg for more)",
            ha="center",
            va="center",
            transform=ax_spd.transAxes,
            fontsize=10,
        )
        ax_spd.set_title("Stein  [PD-only runs]", fontsize=8.5)
        ax_spd.axis("off")

    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # print summary table
    print(
        f"\n{'':6s} {'MDM Stein':>12s} {'MDM Frob':>12s} "
        f"{'NN Stein':>12s} {'NN Frob':>12s}"
    )
    for label, vals in [
        (
            "mean",
            [mdm_steins.mean(), mdm_frobs.mean(), nn_steins.mean(), nn_frobs.mean()],
        ),
        ("S.cov", [mdm_steins.var(), mdm_frobs.var(), nn_steins.var(), nn_frobs.var()]),
    ]:
        print(
            f"  {label:<6s} {vals[0]:12.4f} {vals[1]:12.4f} "
            f"{vals[2]:12.4f} {vals[3]:12.4f}"
        )
    print(
        f"  {'PD %':<6s} {100*mdm_pd.mean():12.1f} {'--':>12s} "
        f"{100*nn_pd.mean():12.1f} {'--':>12s}"
    )

    print(
        f"\n{'':6s} {'MDM std':>10s} {'MDM S.cov':>12s} {'MDM bias':>10s}"
        f" {'NN std':>10s} {'NN S.cov':>12s} {'NN bias':>10s}"
    )
    for i in range(na):
        ms, mv = mdm_ests[:, i].std(), mdm_ests[:, i].var()
        mb = abs(mdm_ests[:, i].mean() - alpha_true[i])
        ns, nv = nn_ests[:, i].std(), nn_ests[:, i].var()
        nb = abs(nn_ests[:, i].mean() - alpha_true[i])
        print(
            f"  a{i+1}  true={alpha_true[i]:5.2f}  {ms:10.4f} {mv:12.4f} {mb:10.4f}"
            f"  {ns:10.4f} {nv:12.4f} {nb:10.4f}"
        )


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=200)
    p.add_argument(
        "--tau", type=int, default=1000, help=f"safe single-traj limit ~{_SAFE_TAU}"
    )
    p.add_argument(
        "--n-seg", type=int, default=1, help="segments averaged per estimate"
    )
    p.add_argument("--ckpt", type=str, default="nn_best.pt")
    p.add_argument(
        "--n-train",
        type=int,
        default=20000,
        help="training samples if network must be trained",
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--out", type=str, default="alpha_nn_vs_mdm.png")
    args = p.parse_args()

    (
        mdm_ests,
        nn_ests,
        basis,
        mdm_steins,
        mdm_frobs,
        mdm_pd,
        nn_steins,
        nn_frobs,
        nn_pd,
    ) = run(
        MC=args.mc,
        TAU=args.tau,
        n_seg=args.n_seg,
        ckpt=args.ckpt,
        n_train=args.n_train,
        epochs=args.epochs,
    )
    plot(
        mdm_ests,
        nn_ests,
        basis,
        mdm_steins,
        mdm_frobs,
        mdm_pd,
        nn_steins,
        nn_frobs,
        nn_pd,
        out=args.out,
        tau=args.tau,
        n_seg=args.n_seg,
    )
