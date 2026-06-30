r"""
Example B (paper Section VII-B): unobservable LTV system with UNKNOWN input.

Reproduces the model of eqs. (37)-(40) with true weights
    alpha = [1, 1, -1, 2, 2, 1].
The state is unobservable over an L=2 window (rank(O_k) = 2 < nx = 3) and the
input u_k is treated as unknown, so the annihilation matrix of [O_k, Gamma_k G_k]
is used (eq. 33).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mdm import StateSpaceModel, NoiseBasis, MDM

TAU = 1000
L = 2

F = np.array([[1, 2, 1], [0, -1.01, 2], [0, 0, 1]], float)
E = np.array([[-3, 2, 0], [2, 2, 2], [5, 0, 1]], float)
H = np.array([[0, 1, 0], [0, 0, 2], [0, 1, 1]], float)
D = np.array([[1, 1, 0], [0, 2, 1], [1, 0, -1]], float)

G_fn = lambda k: np.array([[0.0], [np.cos(10 * (k + 1) / TAU)], [1.0]])
u_fn = lambda k: np.sin((k + 1) / TAU)

BQ = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
      np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], float),
      np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]], float)]
BR = [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
# weights 4,5,6 act on R (and not on Q):
BQ += [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
BR += [np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], float),
       np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], float),
       np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], float)]

alpha_true = np.array([1, 1, -1, 2, 2, 1], float)


def build():
    model = StateSpaceModel.from_constant(F, E, H, D, TAU, G_fn, u_fn)
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=True)
    return model, basis, est


def main(MC=200, seed=0):
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    from mdm import observability_gamma
    O, _ = observability_gamma(model.F, model.H, est.nz_list, L, 0)
    print(f"rank(O) over L={L} window = {np.linalg.matrix_rank(O)}  (nx={model.nx}, unobservable)")
    print(f"identifiable weights = rank(script_A) = {est.n_identifiable()} of {basis.n_alpha}")

    rng = np.random.default_rng(seed)
    est_o = np.array([est.fit_ordinary(model.simulate(Q, R, rng)) for _ in range(MC)])
    print("\n  alpha   true     MDM mean     MDM std")
    for i in range(basis.n_alpha):
        print(f"  a{i+1:<5d}{alpha_true[i]:8.3f}{est_o[:, i].mean():12.3f}{est_o[:, i].std():12.3f}")
    print(f"\n  ||bias|| = {np.linalg.norm(est_o.mean(0) - alpha_true):.4f}")


if __name__ == "__main__":
    main()
