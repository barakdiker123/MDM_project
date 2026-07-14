r"""
Training entry point for the neural noise-covariance estimator.

Run as a script with the example-B system (default), or import ``train`` and
pass any (model, basis, MDM) triple. Trains sim2real on feasible alpha and
exports both a PyTorch checkpoint and a flat ``nn_weights.mat`` (so the MATLAB
comparison scripts can load the network without the Deep Learning Toolbox).

Usage
-----
    python -m mdm.neural.train --n-train 40000 --n-val 4000 --epochs 200
"""
from __future__ import annotations
import argparse
import numpy as np


def train(model, basis, est, lo, hi,
          n_train=40000, n_val=4000, epochs=200, batch=256, lr=1e-3,
          seed=0, out_prefix="nn", device=None):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from .network import CovarianceEstimator
    from .dataset import make_feasible_sampler, generate_dataset
    from .losses import frobenius_loss_torch

    device = device or ("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    sampler = make_feasible_sampler(basis, lo, hi)

    print(f"Generating {n_train} train / {n_val} val samples ...")
    Xtr, Ytr = generate_dataset(model, basis, est, sampler, n_train, seed=seed)
    Xva, Yva = generate_dataset(model, basis, est, sampler, n_val, seed=seed + 1)

    s_mean = torch.tensor(Xtr.mean(0)); s_std = torch.tensor(Xtr.std(0))
    net = CovarianceEstimator(s_mean, s_std, n_alpha=basis.n_alpha).to(device)
    BQ = torch.tensor(np.stack(basis.BQ), dtype=torch.float32, device=device)
    BR = torch.tensor(np.stack(basis.BR), dtype=torch.float32, device=device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=batch, shuffle=True)
    Xva_t = torch.tensor(Xva, device=device); Yva_t = torch.tensor(Yva, device=device)

    best = np.inf
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = frobenius_loss_torch(net(xb), yb, BQ, BR)
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vloss = frobenius_loss_torch(net(Xva_t), Yva_t, BQ, BR).item()
        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), f"{out_prefix}_best.pt")
            _export_mat(net, f"{out_prefix}_weights.mat")
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  val Frobenius {vloss:.5f}  (best {best:.5f})")
    print(f"Done. Best val {best:.5f}. Saved {out_prefix}_best.pt and {out_prefix}_weights.mat")
    return net


def _export_mat(net, path):
    """Flatten the MLP to a .mat (W1,b1,...,s_mean,s_std) for MATLAB inference."""
    from scipy.io import savemat
    sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    lin = [k for k in sd if k.endswith(".weight")]
    out = {"s_mean": sd["s_mean"], "s_std": sd["s_std"]}
    for i, wk in enumerate(lin, start=1):
        bk = wk.replace(".weight", ".bias")
        out[f"W{i}"] = sd[wk]; out[f"b{i}"] = sd[bk]
    savemat(path, out)


def _build_example_B():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    from ex_B_unobservable_unknown_input import build
    model, basis, est = build()
    lo = np.array([0.2, 0.2, -2.0, 0.5, 0.5, -1.0])
    hi = np.array([3.0, 3.0, 0.5, 4.0, 4.0, 2.5])
    return model, basis, est, lo, hi


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-val", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    model, basis, est, lo, hi = _build_example_B()
    train(model, basis, est, lo, hi,
          n_train=args.n_train, n_val=args.n_val,
          epochs=args.epochs, batch=args.batch, lr=args.lr)
