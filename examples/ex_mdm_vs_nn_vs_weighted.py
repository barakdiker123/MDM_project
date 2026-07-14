#!/usr/bin/env python
r"""
MDM (ordinary AND weighted) vs. neural estimator -- a fair-data comparison on
the Example-B system, extending ``ex_mdm_vs_nn.py`` with the weighted MDM
(eq. 23) to see whether weighting reduces the single-trajectory non-PSD rate
that plagues ordinary MDM on this system.

Rows
----
1. MDM-ordinary  (1 segment)    eq. 21 on one tau=1000 trajectory
2. MDM-weighted  (1 segment)    eq. 23 on the SAME trajectory
3. MDM-ordinary  (40 segments)  eq. 21, sufficient stat averaged over 40 trajectories
4. MDM-weighted  (40 segments)  eq. 23, covRes averaged over 40 trajectories
5. NN            (1 segment)    learned estimator on one trajectory

Why averaging covRes (not re-running fit_weighted per segment) is valid for the
40-segment weighted row
--------------------------------------------------------------------------
The weighted-LS point estimate is invariant to a constant rescaling of the
weight: alpha_w = (A' (cP)^{-1} A)^{-1} A' (cP)^{-1} r = (A' P^{-1} A)^{-1} A' P^{-1} r
for any c > 0. Since segments are i.i.d., the residue covariance estimated from
ONE segment's plug-in is, up to that irrelevant constant, the same weighting
direction as the (true) covariance of the 40-segment average. So we can simply
average the per-window covRes vectors across the 40 segments and call
``fit_weighted`` ONCE on the averaged covRes -- no need to refit 40 times. This
also means the weighted-MDM point estimate is unbiased here regardless of the
scale mismatch; only a *reported* covariance of the estimate would need an
explicit 1/40 correction, which we don't use below (we use the empirical MC
spread instead, which is exact).
"""

from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ex_B_unobservable_unknown_input import build, alpha_true  # noqa: E402
from mdm.linalg import is_psd  # noqa: E402
from mdm.neural.losses import stein_loss  # noqa: E402


def _try_load_net(basis, ckpt="nn_best.pt"):
    if not os.path.exists(ckpt):
        return None
    try:
        import torch
        from mdm.neural.network import CovarianceEstimator

        sd = torch.load(ckpt, map_location="cpu")
        net = CovarianceEstimator(sd["s_mean"], sd["s_std"], n_alpha=basis.n_alpha)
        net.load_state_dict(sd)
        net.eval()
        return net
    except Exception as e:  # pragma: no cover
        print(f"(could not load network: {e})")
        return None


def _avg_covRes(list_of_covRes):
    r"""Elementwise mean of covRes across segments, window by window."""
    n_windows = len(list_of_covRes[0])
    return [np.mean([cr[k] for cr in list_of_covRes], axis=0) for k in range(n_windows)]


def _metrics(name, est_arr, basis):
    e = est_arr - alpha_true
    rmse = np.sqrt((e**2).mean())
    bias = np.linalg.norm(est_arr.mean(0) - alpha_true)
    npsd, dS = 0, 0.0
    Qt, Rt = basis.to_QR(alpha_true)
    for a in est_arr:
        Q, R = basis.to_QR(a)
        if not (is_psd(Q) and is_psd(R)):
            npsd += 1
        dS += stein_loss(Q, Qt) + stein_loss(R, Rt)
    print(
        f"  {name:<26s} RMSE={rmse:6.3f}  ||bias||={bias:6.3f}  "
        f"non-PSD={100*npsd/len(est_arr):5.1f}%  Stein={dS/len(est_arr):7.3f}"
    )


# def main(MC=100, n_seg=40, seed=0):
def main(MC=5000, n_seg=40, seed=0):
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    net = _try_load_net(basis)
    rng = np.random.default_rng(seed)

    mdm_o1, mdm_w1, mdm_o40, mdm_w40, nn1 = [], [], [], [], []
    t0 = time.time()
    for i in range(MC):
        cr_list = []
        for k in range(n_seg):
            z = model.simulate(Q, R, rng)
            cr_list.append(est.residue_cov(z))

        # --- 1 segment (the last one drawn) ---
        cr1 = cr_list[-1]
        s1 = est.sufficient_stat(cr1)
        mdm_o1.append(np.linalg.solve(est.S, s1))  # eq. 21
        mdm_w1.append(est.fit_weighted(covRes=cr1))  # eq. 23

        # --- 40 segments, aggregated ---
        s40 = np.mean([est.sufficient_stat(cr) for cr in cr_list], axis=0)
        mdm_o40.append(np.linalg.solve(est.S, s40))  # eq. 21, averaged stat
        cr40 = _avg_covRes(cr_list)
        mdm_w40.append(est.fit_weighted(covRes=cr40))  # eq. 23, averaged covRes

        if net is not None:
            import torch

            with torch.no_grad():
                nn1.append(
                    net(torch.tensor(s1[None, :], dtype=torch.float32)).numpy()[0]
                )

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{MC}  ({time.time()-t0:.0f}s)")

    print(
        f"\nMDM ordinary vs. weighted vs. NN on the Example-B system "
        f"(MC={MC}, segments={n_seg}):"
    )
    _metrics("MDM-ordinary (1 segment)", np.array(mdm_o1), basis)
    _metrics("MDM-weighted (1 segment)", np.array(mdm_w1), basis)
    _metrics(f"MDM-ordinary ({n_seg} segments)", np.array(mdm_o40), basis)
    _metrics(f"MDM-weighted ({n_seg} segments)", np.array(mdm_w40), basis)
    if nn1:
        _metrics("NN  (1 segment)", np.array(nn1), basis)
    else:
        print(
            "  NN  (1 segment)             -- no checkpoint found, skipped "
            "(train with: python -m mdm.neural.train)"
        )


if __name__ == "__main__":
    main()
