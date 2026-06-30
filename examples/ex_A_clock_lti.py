r"""
Example A (paper Section VII-A): unobservable LTI clock ensemble.

Three two-state clocks (time/frequency deviation) observed only through phase
*differences*, so the common time mode is unobservable. Known input (G=0).
Eqs. (34)-(36); structure-defining matrices (35); true weights (36).
The window length L=10 satisfies n_{z,k,L} > rank(O_k).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mdm import StateSpaceModel, NoiseBasis, MDM

TAU = 1000
L = 10
Ts = 10.0

Fs = np.array([[1, Ts], [0, 1]], float)
F = np.kron(np.eye(3), Fs)                       # 6x6
E = np.eye(6)
H = np.array([[1, 0, -1, 0, 0, 0],
              [1, 0, 0, 0, -1, 0]], float)        # phase differences only
D = np.eye(2)

# structure-defining matrices (eq. 35): 6 for Q, 2 for R, total n_alpha = 8
diffu = np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])
white = np.array([[Ts, 0], [0, 0]])
sel = [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])]
BQ, BR = [], []
for s in sel:                                     # per clock: diffusion + white
    BQ += [np.kron(s, diffu), np.kron(s, white)]
    BR += [np.zeros((2, 2)), np.zeros((2, 2))]
BQ += [np.zeros((6, 6)), np.zeros((6, 6))]        # the two R weights
BR += [np.array([[1, 0], [0, 0]], float), np.array([[0, 0], [0, 1]], float)]

alpha_true = 1e-19 * np.array([6, 0.05, 20, 0.3, 7, 0.04, 80, 100], float)


def build():
    model = StateSpaceModel.from_constant(F, E, H, D, TAU)   # G=0, no input
    basis = NoiseBasis(BQ, BR)
    est = MDM(model, basis, L=L, unknown_input=False)
    return model, basis, est


def main(MC=200, seed=0):
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    print(f"identifiable weights = {est.n_identifiable()} of {basis.n_alpha}")
    rng = np.random.default_rng(seed)
    e = np.array([est.fit_ordinary(model.simulate(Q, R, rng)) for _ in range(MC)])
    print("\n  alpha          true            MDM mean")
    for i in range(basis.n_alpha):
        print(f"  a{i+1:<5d}{alpha_true[i]:14.3e}{e[:, i].mean():16.3e}")


if __name__ == "__main__":
    main()
