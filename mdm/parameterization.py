r"""
Noise-covariance parameterisation via *structure-defining matrices*
(paper Section III-D, eqs. 9-12).

The state- and measurement-noise covariance matrices are written as a weighted
sum of known matrices:

    Q = sum_i  alpha_i  B_Q^{(i)}              (eq. 9)
    R = sum_i  alpha_i  B_R^{(i)}

The unknown weights are gathered in  alpha = [alpha_1, ..., alpha_{n_alpha}]^T.
A single weight may drive entries in *both* Q and R, so each basis index i
carries a pair (B_Q^{(i)}, B_R^{(i)}); unused halves are zero matrices.

The defining-replication matrix  Upsilon_{E2}  (eq. 12) maps alpha to the
column-major vec of the augmented-noise covariance

    R_{E2} = vec( blkdiag( I_{Lbar} kron Q,  I_L kron R ) ) = Upsilon_{E2} alpha

with Lbar = L - 1.
"""
from __future__ import annotations
import numpy as np
from .linalg import vec, blkdiag


class NoiseBasis:
    r"""
    Container for the structure-defining matrices  {B_Q^{(i)}, B_R^{(i)}}.

    Parameters
    ----------
    BQ, BR : list[np.ndarray]
        Equal-length lists of n_w x n_w and n_v x n_v matrices. Entry i is the
        pair driven by weight alpha_i. Use a zero matrix where a weight does not
        act on Q (or on R).
    """

    def __init__(self, BQ, BR):
        if len(BQ) != len(BR):
            raise ValueError("BQ and BR must have the same length (one pair per alpha_i).")
        self.BQ = [np.asarray(b, float) for b in BQ]
        self.BR = [np.asarray(b, float) for b in BR]
        self.n_alpha = len(BQ)
        self.nw = self.BQ[0].shape[0]
        self.nv = self.BR[0].shape[0]

    # ---- alpha  <->  (Q, R) -------------------------------------------------
    def to_QR(self, alpha):
        r"""Reconstruct (Q, R) from the weight vector alpha  (eq. 9)."""
        alpha = np.asarray(alpha, float)
        Q = sum(alpha[i] * self.BQ[i] for i in range(self.n_alpha))
        R = sum(alpha[i] * self.BR[i] for i in range(self.n_alpha))
        return Q, R

    # ---- Upsilon_{E2}  (defining-replication matrix, eq. 12) ----------------
    def Upsilon_E2(self, L: int) -> np.ndarray:
        r"""
        Build  Upsilon_{E2}  of shape  ((Lbar n_w + L n_v)^2)  x  n_alpha,
        mapping alpha to vec of the augmented-noise covariance over a window.
        """
        Lbar = L - 1
        dim = Lbar * self.nw + L * self.nv
        Ups = np.zeros((dim * dim, self.n_alpha))
        for i in range(self.n_alpha):
            blockQ = np.kron(np.eye(Lbar), self.BQ[i]) if Lbar > 0 else np.zeros((0, 0))
            blockR = np.kron(np.eye(L), self.BR[i])
            Ups[:, i] = vec(blkdiag(blockQ, blockR))
        return Ups
