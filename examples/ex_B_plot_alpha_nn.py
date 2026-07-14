#!/usr/bin/env python
#!/usr/bin/env python
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mdm import StateSpaceModel, NoiseBasis, MDM
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

    return np.array(mdm_ests), np.array(nn_ests), basis


# ------------------------------------------------------------------ plot
def _panel(ax, col, true, c, label):
    """Draw one violin+box panel and return (std, scov, bias)."""
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

    out = col[(col < lo_f) | (col > hi_f)]
    if out.size:
        xj = np.random.default_rng(42).uniform(-0.15, 0.15, size=out.size)
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

    ax.axhline(
        true, color="crimson", lw=1.8, ls="--", zorder=6, label=f"true={true:.2g}"
    )
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax.set_xlabel(f"$\\hat{{\\alpha}}_{{{label}}}$", fontsize=11)
    ax.set_xticks([])
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    return col.std(), col.var(), abs(col.mean() - true)


def plot(mdm_ests, nn_ests, basis, out="alpha_nn_vs_mdm.png", tau=1000, n_seg=1):
    MC, na = mdm_ests.shape
    seg_str = f", {n_seg} segs" if n_seg > 1 else ""

    # 2 rows (MDM top, NN bottom), na columns
    fig, axes = plt.subplots(2, na, figsize=(3 * na, 8), sharey=False, sharex=False)
    fig.suptitle(
        f"Ordinary MDM  vs  Neural Network  —  Example B\n"
        f"MC={MC}, $\\tau$={tau}{seg_str}   |   same trajectories, same sufficient statistic $s$",
        fontsize=11,
        y=1.01,
    )

    row_labels = ["MDM (ordinary, eq. 21)", "NN (learned, eq. 21 input)"]
    colors_mdm = plt.cm.tab10.colors
    colors_nn = plt.cm.Set2.colors

    for row, (ests, colors, rlabel) in enumerate(
        [(mdm_ests, colors_mdm, row_labels[0]), (nn_ests, colors_nn, row_labels[1])]
    ):
        axes[row, 0].set_ylabel(rlabel, fontsize=10, labelpad=8)
        for i in range(na):
            ax = axes[row, i]
            std, scov, bias = _panel(
                ax, ests[:, i], alpha_true[i], colors[i % len(colors)], str(i + 1)
            )
            ax.set_title(
                f"a{i+1}\nstd={std:.3f}  S.cov={scov:.3f}\nbias={bias:.3f}",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # print summary table
    print(
        f"\n{'':6s} {'':>6s} {'MDM std':>10s} {'MDM S.cov':>12s} {'MDM bias':>10s}"
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

    mdm_ests, nn_ests, basis = run(
        MC=args.mc,
        TAU=args.tau,
        n_seg=args.n_seg,
        ckpt=args.ckpt,
        n_train=args.n_train,
        epochs=args.epochs,
    )
    plot(mdm_ests, nn_ests, basis, out=args.out, tau=args.tau, n_seg=args.n_seg)
