r"""
MDM vs. neural estimator -- a *fair-data* comparison (Example-B system).

Key methodological points (the same ones that matter on real data):

1. The Example-B system is open-loop unstable (F eigenvalues {1, -1.01, 1}), so a
   single very long trajectory diverges and the state cancellation collapses to
   roundoff. "More data" must be realised as many independent fixed-length
   SEGMENTS, with the MDM sufficient statistic aggregated across them.

2. The network is trained sim2real and evaluated on a single segment (its design
   regime). To compare fairly, the MDM is also given a single segment AND a
   larger multi-segment budget; only then is the comparison about estimator
   quality rather than data volume.

This script needs a trained network (``nn_best.pt``); if absent it runs the MDM
rows only and explains how to train. PyTorch is optional.
"""

from __future__ import annotations
import os, sys
import numpy as np

# sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from ex_B_unobservable_unknown_input import build, alpha_true  # noqa: E402
from mdm.linalg import is_psd, psd_project  # noqa: E402
from mdm.neural.losses import stein_loss  # noqa: E402
from mdm import StateSpaceModel, NoiseBasis, MDM
from tqdm import tqdm

# This simulation is from example B !
TAU = 1000
# BIG PROBLEM FOR IMPLEMENTATION !!!!!
# L = 2 # About 85 percent of the time we get non-PSD Matrix !!
L = 2

F = np.array([[1, 2, 1], [0, -1.01, 2], [0, 0, 1]], float)
E = np.array([[-3, 2, 0], [2, 2, 2], [5, 0, 1]], float)
H = np.array([[0, 1, 0], [0, 0, 2], [0, 1, 1]], float)
D = np.array([[1, 1, 0], [0, 2, 1], [1, 0, -1]], float)

G_fn = lambda k: np.array([[0.0], [np.sin(10 * k / TAU)], [1.0]])  # eq. (37a)
u_fn = lambda k: np.sin(k / TAU)  # Sec. VII preamble

BQ = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], float),
]
BR = [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
# weights 4,5,6 act on R (and not on Q):
BQ += [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
BR += [
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], float),
    np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], float),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], float),
]

alpha_true = np.array([1, 1, -1, 2, 2, 1], float)


def build():
    model = StateSpaceModel.from_constant(F, E, H, D, TAU, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


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
        f"  {name:<22s} RMSE={rmse:6.3f}  ||bias||={bias:6.3f}  "
        f"non-PSD={100*npsd/len(est_arr):4.1f}%  Stein={dS/len(est_arr):6.3f}"
    )


def main(MC=100, n_seg=40, seed=0):
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    net = _try_load_net(basis)
    rng = np.random.default_rng(seed)

    mdm1, mdm40, nn1 = [], [], []
    for _ in tqdm(range(MC)):
        s_list = []
        for k in range(n_seg):
            z = model.simulate(Q, R, rng)
            s_list.append(est.sufficient_stat(est.residue_cov(z)))
        s_list = np.array(s_list)
        mdm1.append(np.linalg.solve(est.S, s_list[-1]))  # MDM, 1 segment
        mdm40.append(np.linalg.solve(est.S, s_list.mean(0)))  # MDM, n_seg segments
        if net is not None:
            import torch

            with torch.no_grad():
                nn1.append(
                    net(torch.tensor(s_list[-1:], dtype=torch.float32)).numpy()[0]
                )

    print(f"\nFair comparison on the Example-B system  (MC={MC}, segments={n_seg}):")
    _metrics("MDM (1 segment)", np.array(mdm1), basis)
    _metrics(f"MDM ({n_seg} segments)", np.array(mdm40), basis)
    if nn1:
        _metrics("NN  (1 segment)", np.array(nn1), basis)
    else:
        print("  NN  (1 segment)       -- train first: python -m mdm.neural.train")


if __name__ == "__main__":
    main()
