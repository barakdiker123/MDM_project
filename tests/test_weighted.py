r"""
Unit tests for the linear-algebra conventions and the Gaussian weighting blocks.
These guard the index-level fidelity to the paper (column-major vec, commutation
matrix, and the closed-form residue fourth moment used by the weighted MDM).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdm.linalg import vec, unvec, commutation_matrix, kron_power, upper_tri_selector
from mdm.moments import unique_cov_block


def test_vec_roundtrip_and_kron_identity():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3))
    assert np.allclose(unvec(vec(A), (4, 3)), A)
    a = rng.standard_normal(5)
    # for a vector, a kron a == vec(a a^T)  (used to equate eq. 8 forms)
    assert np.allclose(kron_power(a, 2), vec(np.outer(a, a)))


def test_commutation_matrix():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 4))
    K = commutation_matrix(3, 4)
    assert np.allclose(K @ vec(X), vec(X.T))


def test_weighting_block_matches_montecarlo():
    # Cov of the unique entries of (Z Z^T) for Gaussian Z must match the
    # closed-form  Xi (I+K)(Sigma kron Sigma) Xi^T.
    rng = np.random.default_rng(2)
    n = 3
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T + np.eye(n)
    Xi = upper_tri_selector(n)
    closed = unique_cov_block(Sigma, Xi)
    L = np.linalg.cholesky(Sigma)
    N = 400000
    Z = (L @ rng.standard_normal((n, N)))
    u = np.array([np.outer(Z[:, i], Z[:, i]).ravel(order="F")[Xi] for i in range(N)])
    emp = np.cov(u.T)
    assert np.allclose(closed, emp, atol=0.05 * np.abs(closed).max())


if __name__ == "__main__":
    test_vec_roundtrip_and_kron_identity()
    test_commutation_matrix()
    test_weighting_block_matches_montecarlo()
    print("all internal tests passed")
