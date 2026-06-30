r"""
Core measurement-difference-method estimator (paper Sections III-V).

Pipeline per window k = 0, ..., tau - Lbar  (Lbar = L - 1):

  1. (O_k, Gamma_k)                                            (eqs. 3a, 3e)
  2. annihilation matrix  am(.)                                (eq. 4 / 33)
        known input  :  am(O_k)                  -> cancels x_k
        unknown input:  am([O_k, Gamma_k script_G_k])  -> cancels x_k AND u_k
  3. A_k = am(.) [Gamma_k, I],   C_k = blkdiag(script_E_k, script_D_k)  (eq. 7)
  4. residue            Z_k = A_k C_k [W_k; V_k]               (eq. 6b)
  5. regression block   script_A_k = Xi_k (A_k C_k)^{\otimes 2} Upsilon_{E2}  (eq. 16)
  6. sample residue cov  R^U_{Z2,k} = Xi_k (Z_k \otimes Z_k)   (eq. 17, lhs)

Stacking all windows gives the linear system  R_{Z2} = script_A alpha + L Y
(eq. 19). The noise weights alpha are estimated by ordinary LS (eq. 21) or
weighted LS (eq. 23).

The matrices in steps 1-5 depend only on the (known) system, so they are
precomputed once in ``__init__`` and reused for every data realisation.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, factorized

from .augmented import observability_gamma
from .parameterization import NoiseBasis
from .linalg import kron_power, blkdiag, upper_tri_selector
from .moments import augmented_cross_cov, residue_cross_cov, unique_cov_block


def annihilation_matrix(M, tol=None):
    r"""
    Row basis of the left null space of ``M``: am(M) with am(M) M = 0.
    Returned with shape (n_null, rows(M)). Uses the SVD (numerically robust).
    """
    U, sv, _ = np.linalg.svd(M, full_matrices=True)
    r = M.shape[0]
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps * (sv[0] if sv.size else 1.0)
    rank = int(np.sum(sv > tol))
    return U[:, rank:].conj().T          # (r - rank) x r


class MDM:
    r"""
    Measurement difference method on a :class:`StateSpaceModel`.

    Parameters
    ----------
    model        : StateSpaceModel
    basis        : NoiseBasis (structure-defining matrices)
    L            : window length (>= 1). Must satisfy n_{z,k,L} > rank(O_k).
    unknown_input: if True, also annihilate the (unknown) input via eq. (33).
    """

    def __init__(self, model, basis: NoiseBasis, L: int, unknown_input: bool = False):
        self.model = model
        self.basis = basis
        self.L = L
        self.Lbar = L - 1
        self.unknown_input = unknown_input
        m = model
        self.nz_list = [m.H[k].shape[0] for k in range(m.tau)]
        self.n_windows = m.tau - self.Lbar
        self.Upsilon = basis.Upsilon_E2(L)               # (nE^2) x n_alpha

        self.am = [None] * self.n_windows                # annihilation matrices
        self.AC = [None] * self.n_windows                # A_k C_k  (naO x nE)
        self.A_k = [None] * self.n_windows               # A_k
        self.Gamma = [None] * self.n_windows
        self.Xi_idx = [None] * self.n_windows
        self.script_A = [None] * self.n_windows          # eq. 16 block  (nq x n_alpha)
        self._precompute()

        # ordinary-LS normal matrix  S = sum_k script_A_k^T script_A_k
        self.S = sum(a.T @ a for a in self.script_A)

    # ------------------------------------------------------------------ build
    def _precompute(self):
        m = self.model
        L, Lbar = self.L, self.Lbar
        for k in range(self.n_windows):
            O, Gamma = observability_gamma(m.F, m.H, self.nz_list, L, k)
            self.Gamma[k] = Gamma
            if self.unknown_input:
                # [O_k, Gamma_k script_G_k]   (script_G_k = blkdiag of inputs)
                Gcols = np.hstack([
                    Gamma[:, j * m.nx:(j + 1) * m.nx] @ m.G[k + j]
                    for j in range(Lbar)
                ]) if Lbar > 0 else np.zeros((O.shape[0], 0))
                aug = np.hstack([O, Gcols])
            else:
                aug = O
            am = annihilation_matrix(aug)
            if am.shape[0] == 0:
                raise RuntimeError(
                    f"Empty annihilation matrix at window {k}: need larger L "
                    f"(require n_z,k,L > rank(O_k)).")
            self.am[k] = am
            naO = am.shape[0]

            # A_k = am [Gamma_k, I];   C_k = blkdiag(script_E_k, script_D_k)
            A_k = am @ np.hstack([Gamma, np.eye(O.shape[0])])
            E_bd = blkdiag(*[m.E[k + j] for j in range(Lbar)]) if Lbar > 0 else np.zeros((0, 0))
            D_bd = blkdiag(*[m.D[k + j] for j in range(L)])
            C_k = blkdiag(E_bd, D_bd)
            self.A_k[k] = A_k
            self.AC[k] = A_k @ C_k                        # naO x nE

            Xi_idx = upper_tri_selector(naO)
            self.Xi_idx[k] = Xi_idx
            # script_A_k = Xi_k (A_k C_k)^{\otimes 2} Upsilon       (eq. 16)
            AC2 = kron_power(self.AC[k], 2)              # naO^2 x nE^2
            self.script_A[k] = AC2[Xi_idx, :] @ self.Upsilon

    # ---------------------------------------------------- regression assembly
    @property
    def script_A_stacked(self) -> np.ndarray:
        r"""Paper matrix  script_A  (eq. 20a), stacked over all windows."""
        return np.vstack(self.script_A)

    def n_identifiable(self) -> int:
        r"""Number of identifiable weights = rank(script_A)  (Section V-C)."""
        return int(np.linalg.matrix_rank(self.script_A_stacked))

    # ----------------------------------------------------- residue covariances
    def residue_cov(self, z, u=None):
        r"""
        Sample residue covariances  R^U_{Z2,k} = Xi_k (Z_k \otimes Z_k) for one
        data realisation. ``z`` is a length-tau list of measurement vectors.
        Returns a list of length ``n_windows`` (each of length naO(naO+1)/2).
        """
        m = self.model
        L, Lbar = self.L, self.Lbar
        out = []
        for k in range(self.n_windows):
            Z = np.concatenate([z[k + j] for j in range(L)])      # augmented meas.
            res = self.am[k] @ Z
            u_seq = u if u is not None else m.u
            if (not self.unknown_input) and (u_seq is not None) and Lbar > 0:
                # subtract the known control term  am(O_k) Gamma_k script_G_k U_k
                U = np.concatenate([np.atleast_1d(u_seq[k + j]) for j in range(Lbar)])
                Gbd = blkdiag(*[m.G[k + j] for j in range(Lbar)])
                res = res - self.am[k] @ self.Gamma[k] @ Gbd @ U
            outer = np.outer(res, res).ravel(order="F")
            out.append(outer[self.Xi_idx[k]])
        return out

    def sufficient_stat(self, covRes):
        r"""s = sum_k script_A_k^T R^U_{Z2,k}  (the 6-vector both MDM and the NN use)."""
        return sum(self.script_A[k].T @ covRes[k] for k in range(self.n_windows))

    # ---------------------------------------------------------- ordinary MDM
    def fit_ordinary(self, z=None, u=None, covRes=None):
        r"""
        Ordinary-LS estimate  alpha_o = (script_A^T script_A)^{-1} script_A^T R_{Z2}
        (eq. 21). Unbiased and consistent (Section V-D).
        """
        if covRes is None:
            covRes = self.residue_cov(z, u)
        s = self.sufficient_stat(covRes)
        return np.linalg.solve(self.S, s)

    # ----------------------------------------------------------- weighted MDM
    def _weighting_blocks(self, alpha):
        r"""
        Build the diagonal and lag-s unique-element covariance blocks of P
        (eq. 22) from a plug-in ``alpha`` (Gaussian closed form). Returns a list
        ``blocks[s]`` of per-window (n_windows-s) lists of nq x nq matrices for
        lags s = 0, ..., Lbar.
        """
        Q, R = self.basis.to_QR(alpha)
        m = self.model
        SigmaE = {s: augmented_cross_cov(Q, R, self.L, m.nw, m.nv, s)
                  for s in range(self.L)}                      # lags 0..L-1
        blocks = {}
        for s in range(self.L):                                # lag s
            row = []
            for k in range(self.n_windows - s):
                Sz = residue_cross_cov(self.AC[k], self.AC[k + s], SigmaE[s])
                row.append(unique_cov_block(Sz, self.Xi_idx[k]))
            blocks[s] = row
        return blocks

    def _assemble_P(self, alpha):
        r"""Sparse banded covariance  P  (eq. 22) in unique-element space."""
        blocks = self._weighting_blocks(alpha)
        nq = [self.script_A[k].shape[0] for k in range(self.n_windows)]
        offs = np.concatenate([[0], np.cumsum(nq)])
        N = offs[-1]
        P = sp.lil_matrix((N, N))
        for s in range(self.L):                                # lag s (and -s)
            for k, B in enumerate(blocks[s]):
                ri, rj = offs[k], offs[k + s]
                P[ri:ri + B.shape[0], rj:rj + B.shape[1]] = B
                if s > 0:
                    P[rj:rj + B.shape[1], ri:ri + B.shape[0]] = B.T
        return P.tocsc()

    def fit_weighted(self, z=None, u=None, covRes=None, alpha_init=None,
                     return_cov=False):
        r"""
        Weighted-LS estimate (eq. 23):
            alpha_w = (script_A^T P^{-1} script_A)^{-1} script_A^T P^{-1} R_{Z2}
        with P estimated from a plug-in ordinary estimate (feasible GLS). Lower
        variance than ordinary MDM; mildly biased in finite samples (Sec. V-D).

        If ``return_cov`` also returns the approximate COV(alpha_w) (eq. 24).
        """
        if covRes is None:
            covRes = self.residue_cov(z, u)
        if alpha_init is None:
            alpha_init = self.fit_ordinary(covRes=covRes)

        A = self.script_A_stacked                              # (N x n_alpha)
        r = np.concatenate(covRes)                             # R_{Z2}  (N,)
        P = self._assemble_P(alpha_init)
        P = P + 1e-12 * sp.eye(P.shape[0])
        solve = factorized(P)                                  # P^{-1} (.)  fast
        PiA = np.column_stack([solve(A[:, j]) for j in range(A.shape[1])])  # P^{-1} A
        Pir = solve(r)                                         # P^{-1} R_{Z2}
        AtPiA = A.T @ PiA
        alpha_w = np.linalg.solve(AtPiA, A.T @ Pir)
        if return_cov:
            return alpha_w, np.linalg.inv(AtPiA)
        return alpha_w
