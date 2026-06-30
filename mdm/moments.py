r"""
Fourth-moment weighting for the *weighted* MDM (paper Section IV-B, eqs. 18-24).

The weighted least squares in eq. (23) needs the covariance P of the residual
process  L eta  in eq. (19). For Gaussian noises this is fully determined by the
noise covariances (footnote 4): the fourth moments factor through second moments
by Isserlis' theorem. Rather than forming E[script_E^{\otimes 4}] symbolically
(as ``EwvLS4_compute.m`` does), we use the equivalent closed form in residue
space:

    Cov( Z_k^{\otimes 2}, Z_j^{\otimes 2} )
        = (I + K)( Sigma_{Z_k,Z_j} \otimes Sigma_{Z_k,Z_j} )

with the residue cross-covariance  Sigma_{Z_k,Z_j} = (A_k C_k) Sigma_{E_k,E_j}
(A_j C_j)^T, where  Sigma_{E_k,E_j}  is the cross-covariance of the augmented
noise vectors of the two windows (nonzero only when they overlap, |k-j| < L).
Selecting the unique elements with the unification matrix Xi gives the blocks of
P. This produces the same weighting as the symbolic route but numerically and
without the Symbolic Math Toolbox.
"""
from __future__ import annotations
import numpy as np
from .linalg import commutation_matrix, blkdiag


def augmented_cross_cov(Q, R, L, nw, nv, s):
    r"""
    Cross-covariance  Sigma_{E_k, E_{k+s}} = E[ script_E_k script_E_{k+s}^T ]
    of the augmented noise vectors  script_E = [W; V] of two windows separated
    by lag ``s`` (>= 0). Depends only on the lag because the noise is stationary.

    Layout of script_E (length (L-1)*nw + L*nv):
        w_k, ..., w_{k+L-2},  v_k, ..., v_{k+L-1}.
    """
    Lbar = L - 1
    nE = Lbar * nw + L * nv
    S = np.zeros((nE, nE))
    # process-noise blocks: w_{k+b1} vs w_{k+s+b2} coincide iff b1 = s + b2
    for b1 in range(Lbar):
        b2 = b1 - s
        if 0 <= b2 < Lbar:
            S[b1 * nw:(b1 + 1) * nw, b2 * nw:(b2 + 1) * nw] = Q
    # measurement-noise blocks: v_{k+l1} vs v_{k+s+l2} coincide iff l1 = s + l2
    off = Lbar * nw
    for l1 in range(L):
        l2 = l1 - s
        if 0 <= l2 < L:
            S[off + l1 * nv: off + (l1 + 1) * nv,
              off + l2 * nv: off + (l2 + 1) * nv] = R
    return S


def residue_cross_cov(AC_k, AC_j, Sigma_E):
    r"""Sigma_{Z_k,Z_j} = (A_k C_k) Sigma_{E_k,E_j} (A_j C_j)^T."""
    return AC_k @ Sigma_E @ AC_j.T


def unique_cov_block(Sigma_Z, Xi_idx):
    r"""
    Unique-element covariance block
        Xi (I + K)(Sigma_Z \otimes Sigma_Z) Xi^T
    for a residue (cross-)covariance ``Sigma_Z`` of size naO x naO. ``Xi_idx``
    are the column-major upper-triangle indices (the unification matrix Xi).
    """
    n = Sigma_Z.shape[0]
    K = commutation_matrix(n, n)
    full = (np.eye(n * n) + K) @ np.kron(Sigma_Z, Sigma_Z)   # Cov(vec(ZZ'))
    return full[np.ix_(Xi_idx, Xi_idx)]
