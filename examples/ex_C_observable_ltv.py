r"""
Example C (paper Section VII-C): observable scalar LTV system.

Used in the paper to confront ordinary vs weighted MDM (Table III, Fig. 3):
the weighted MDM attains significantly lower estimate covariance. Eqs. (41)-(42),
true (Q, R) = (2, 1), L = 2.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mdm import StateSpaceModel, NoiseBasis, MDM

TAU = 1000
L = 2


def build():
    F = [np.array([[0.8 - 0.1 * np.sin(7 * np.pi * k / TAU)]]) for k in range(TAU)]
    E = [np.array([[1.0]]) for _ in range(TAU)]
    H = [np.array([[1 + 0.99 * np.sin(100 * np.pi * k / TAU)]]) for k in range(TAU)]
    D = [np.array([[1.0]]) for _ in range(TAU)]
    G = [np.zeros((1, 1)) for _ in range(TAU)]
    model = StateSpaceModel(F, E, H, D, G, u=None)
    basis = NoiseBasis(BQ=[np.array([[1.0]]), np.array([[0.0]])],
                       BR=[np.array([[0.0]]), np.array([[1.0]])])
    est = MDM(model, basis, L=L, unknown_input=False)
    return model, basis, est


alpha_true = np.array([2.0, 1.0])   # (Q, R)


def main(MC=300, seed=0):
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(seed)
    eo, ew = [], []
    for _ in range(MC):
        z = model.simulate(Q, R, rng)
        cr = est.residue_cov(z)
        eo.append(est.fit_ordinary(covRes=cr))
        ew.append(est.fit_weighted(covRes=cr))
    eo, ew = np.array(eo), np.array(ew)
    print(f"{'':6s}{'true':>8s}{'OLS mean':>11s}{'OLS std':>10s}{'WLS mean':>11s}{'WLS std':>10s}")
    for i, nm in enumerate(["Q", "R"]):
        print(f"{nm:>6s}{alpha_true[i]:8.3f}{eo[:, i].mean():11.3f}{eo[:, i].std():10.3f}"
              f"{ew[:, i].mean():11.3f}{ew[:, i].std():10.3f}")
    print("\nWeighted MDM attains lower std (paper Table III / Fig. 3).")


if __name__ == "__main__":
    main()
