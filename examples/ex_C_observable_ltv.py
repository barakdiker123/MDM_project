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


def main(MC=1000, seed=0):
    # Paper uses MC = 1e4; the weighted fit factorises a banded P each call, so
    # 1000 is a practical default (S.cov stable to a few %). Bump MC to match.
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(seed)
    eo = np.empty((MC, 2)); ew = np.empty((MC, 2)); est_cov = np.zeros(2)
    for j in range(MC):
        z = model.simulate(Q, R, rng); cr = est.residue_cov(z)
        eo[j] = est.fit_ordinary(covRes=cr)
        aw, cov = est.fit_weighted(covRes=cr, return_cov=True)
        ew[j] = aw; est_cov += np.diag(cov)
    est_cov /= MC

    # Table III: S. mean, S. cov, and (weighted only) Est. cov from eq. 24.
    print(f"Example C, L={L}, tau={TAU}, MC={MC}   (paper Table III)")
    hdr = f"{'method':<10}{'S.mean Q':>9}{'R':>7}{'S.cov Q':>10}{'R':>9}{'Est.cov Q':>12}{'R':>9}"
    print(hdr)
    print(f"{'ordinary':<10}{eo[:,0].mean():9.3f}{eo[:,1].mean():7.3f}"
          f"{eo[:,0].var():10.4f}{eo[:,1].var():9.4f}{'--':>12}{'--':>9}")
    print(f"{'weighted':<10}{ew[:,0].mean():9.3f}{ew[:,1].mean():7.3f}"
          f"{ew[:,0].var():10.4f}{ew[:,1].var():9.4f}{est_cov[0]:12.4f}{est_cov[1]:9.4f}")
    print("\n  Weighted MDM: lower S. cov than ordinary, and S. cov ~ Est. cov")
    print("  (eq. 24 gives a realistic uncertainty estimate) -- paper Fig. 3.")


if __name__ == "__main__":
    main()
