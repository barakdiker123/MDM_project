r"""
Observability matrix O_k and propagation matrix Gamma_k for one window
(paper eqs. 3a, 3e). Port of ``O_Gamma.m`` (author: Oliver Kost).

The augmented measurement vector over a window of L measurements starting at k
(eq. 2) is

    Z_k = O_k x_k + Gamma_k ( script_G_k U_k + script_E_k W_k ) + script_D_k V_k

where (with Lbar = L - 1, F^{j}_k = F_{k+j-1} ... F_{k+1} F_k):

    O_k     = [ H_k ; H_{k+1}F_k ; H_{k+2}F_{k+1}F_k ; ... ]          (eq. 3a)
    Gamma_k = block-lower-triangular map carrying each state increment
              forward into later measurements                         (eq. 3e)
"""
from __future__ import annotations
import numpy as np


def observability_gamma(F, H, nz_list, L, k):
    r"""
    Compute (O_k, Gamma_k) for the window starting at 0-indexed time ``k``.

    Parameters
    ----------
    F, H      : list of system matrices.
    nz_list   : list of measurement dimensions n_{z, k}.
    L         : window length.
    k         : 0-indexed window start.

    Returns
    -------
    O     : (sum nz over window) x nx
    Gamma : (sum nz over window) x ((L-1) nx)
    """
    nx = F[0].shape[0]
    nz_total = int(sum(nz_list[k:k + L]))
    O = np.zeros((nz_total, nx))
    Gamma = np.zeros((nz_total, (L - 1) * nx))

    # j indexes the "source" column block (j=1 -> O_k, j>=2 -> Gamma columns);
    # i indexes the measurement row block. Mirrors the 1-indexed MATLAB loops.
    for j in range(1, L + 1):
        part = np.zeros((nz_total, nx))
        partF = np.eye(nx)
        row = 0
        for i in range(1, L + 1):
            nz_i = int(nz_list[k + i - 1])
            if i >= j:
                if i == j:
                    part[row:row + nz_i, :] = H[k + i - 1]
                else:
                    partF = F[k + i - 2] @ partF      # F_{k+i-2} ... (accumulated)
                    part[row:row + nz_i, :] = H[k + i - 1] @ partF
            row += nz_i
        if j == 1:
            O = part.copy()
        else:
            Gamma[:, nx * (j - 2): nx * (j - 1)] = part
    return O, Gamma
