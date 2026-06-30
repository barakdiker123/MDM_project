r"""
Tests that the package reproduces the paper's identifiability and recovers the
true weights on the three numerical illustrations (Sections VII-A/B/C).

Run:  pytest -q   (from the package root)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_example_B_recovers_alpha():
    from ex_B_unobservable_unknown_input import build, alpha_true
    model, basis, est = build()
    assert est.n_identifiable() == 6                     # fully identifiable
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(0)
    e = np.array([est.fit_ordinary(model.simulate(Q, R, rng)) for _ in range(150)])
    assert np.linalg.norm(e.mean(0) - alpha_true) < 0.2  # unbiased (MC noise)


def test_example_C_weighted_beats_ordinary():
    from ex_C_observable_ltv import build, alpha_true
    model, basis, est = build()
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(0)
    eo, ew = [], []
    for _ in range(200):
        z = model.simulate(Q, R, rng); cr = est.residue_cov(z)
        eo.append(est.fit_ordinary(covRes=cr)); ew.append(est.fit_weighted(covRes=cr))
    eo, ew = np.array(eo), np.array(ew)
    # both recover (Q, R) = (2, 1); weighted has lower std (paper Table III)
    assert np.allclose(eo.mean(0), alpha_true, atol=0.1)
    assert ew.std(0).mean() < eo.std(0).mean()


def test_example_A_wellconditioned_weights():
    from ex_A_clock_lti import build, alpha_true
    model, basis, est = build()
    assert est.n_identifiable() == 8
    Q, R = basis.to_QR(alpha_true)
    rng = np.random.default_rng(0)
    e = np.array([est.fit_ordinary(model.simulate(Q, R, rng)) for _ in range(200)])
    # the three well-conditioned diffusion weights (1,3,5) recover to ~1%
    for i in (0, 2, 4):
        assert abs(e[:, i].mean() / alpha_true[i] - 1.0) < 0.1


if __name__ == "__main__":
    test_example_B_recovers_alpha()
    test_example_C_weighted_beats_ordinary()
    test_example_A_wellconditioned_weights()
    print("all example tests passed")
