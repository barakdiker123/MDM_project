r"""
Linear-algebra primitives used throughout the MDM, with the *column-major*
(Fortran) vectorisation convention so that results match the reference MATLAB
implementation of Kost et al. exactly.

Notation (paper: Kost, Duník, Puncochář, Straka, IEEE TAC 2026)
---------------------------------------------------------------
* Kronecker power           A^{\otimes 2} = A \otimes A                  -> kron_power
* Vectorisation             (\cdot)_V  = column-major stacking of a matrix
* Inverse vectorisation     (\cdot)_M
* Commutation matrix        K_{m,n} with  K vec(X) = vec(X^T)
* Unification matrix        \Xi   selects the unique (upper-triangular)
                            entries of a symmetric vec(\cdot)            (eq. 14)
* Replication matrix        \Psi  = \Xi^{+}, duplicates them back        (eq. 30)

For a column vector a, the Kronecker power a \otimes a equals vec(a a^T)
because a_i a_j = a_j a_i; this is why the residue second moment can be written
either as a Kronecker power (paper) or as vec of an outer product (code).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag as _block_diag


def kron_power(A: np.ndarray, n: int = 2) -> np.ndarray:
    r"""Kronecker power A^{\otimes n} = A \otimes A \otimes ... (n factors)."""
    out = A
    for _ in range(n - 1):
        out = np.kron(out, A)
    return out


def vec(M: np.ndarray) -> np.ndarray:
    r"""Column-major vectorisation (\cdot)_V."""
    return M.ravel(order="F")


def unvec(v: np.ndarray, shape) -> np.ndarray:
    r"""Inverse vectorisation (\cdot)_M."""
    return v.reshape(shape, order="F")


def blkdiag(*mats: np.ndarray) -> np.ndarray:
    """Block-diagonal assembly (handles the empty case)."""
    mats = [m for m in mats if m is not None and np.size(m) > 0]
    if not mats:
        return np.zeros((0, 0))
    return _block_diag(*mats)


def commutation_matrix(m: int, n: int) -> np.ndarray:
    r"""K_{m,n} such that K vec(X) = vec(X^T) for X an m-by-n matrix."""
    K = np.zeros((m * n, m * n))
    for i in range(m):
        for j in range(n):
            K[j + i * n, i + j * m] = 1.0
    return K


def upper_tri_selector(n: int) -> np.ndarray:
    r"""
    Unification index set \Xi (eq. 14): column-major positions of the upper
    triangle of an n-by-n matrix, matching MATLAB ``triu(ones(n))`` applied to
    a column-major ``reshape(M, n^2, 1)``.

    Returns integer indices into a length-n^2 column-major vector.
    """
    idx = []
    for col in range(n):
        for row in range(n):
            if row <= col:
                idx.append(row + col * n)
    return np.asarray(idx, dtype=int)


def selector_matrix(n: int) -> np.ndarray:
    r"""Dense form of the unification matrix \Xi  (shape  n(n+1)/2  x  n^2)."""
    idx = upper_tri_selector(n)
    Xi = np.zeros((idx.size, n * n))
    for r, c in enumerate(idx):
        Xi[r, c] = 1.0
    return Xi


def is_psd(M: np.ndarray, tol: float = 1e-9) -> bool:
    """True if M is (numerically) positive semidefinite."""
    M = 0.5 * (M + M.T)
    return bool(np.all(np.linalg.eigvalsh(M) >= -tol))


def psd_project(M: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clamping eigenvalues."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    return (V * np.clip(w, floor, None)) @ V.T
