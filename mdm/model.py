r"""
Linear time-varying state-space model (paper eqs. 1a-1b):

    x_{k+1} = F_k x_k + G_k u_k + E_k w_k
    z_k     = H_k x_k + D_k v_k

with w_k ~ (0, Q),  v_k ~ (0, R), independent and zero-mean. The model stores
the (possibly time-varying) system matrices as length-tau lists. Convenience
constructors are provided for the common case where F, E, H, D are constant and
only G_k, u_k vary with time.
"""
from __future__ import annotations
import numpy as np


def _safe_cholesky(M, n):
    """Cholesky with a *relative* jitter, so it is correct for any matrix scale
    (a fixed absolute jitter would swamp e.g. the 1e-19-scale clock covariances)."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        scale = np.trace(M) / max(n, 1)
        return np.linalg.cholesky(M + 1e-10 * max(scale, 1e-300) * np.eye(n))


class StateSpaceModel:
    r"""
    Parameters
    ----------
    F, E, H, D : list[np.ndarray]
        Length-tau lists of system matrices. F_k: nx x nx, E_k: nx x nw,
        H_k: nz x nx, D_k: nz x nv.
    G : list[np.ndarray]
        Length-tau list of input matrices nx x nu (use zeros if no input).
    u : list[np.ndarray] or None
        Length-tau list of input vectors (nu,). None if there is no input.
    """

    def __init__(self, F, E, H, D, G, u=None):
        self.F = [np.asarray(m, float) for m in F]
        self.E = [np.asarray(m, float) for m in E]
        self.H = [np.asarray(m, float) for m in H]
        self.D = [np.asarray(m, float) for m in D]
        self.G = [np.atleast_2d(np.asarray(m, float)) for m in G]
        self.u = None if u is None else [np.atleast_1d(np.asarray(ui, float)) for ui in u]
        self.tau = len(self.F)
        self.nx = self.F[0].shape[0]
        self.nw = self.E[0].shape[1]
        self.nv = self.D[0].shape[1]
        self.nz = self.H[0].shape[0]

    # ---- convenience constructor -------------------------------------------
    @classmethod
    def from_constant(cls, F, E, H, D, tau, G_fn=None, u_fn=None):
        r"""
        Build a model with constant F, E, H, D and optional time-varying input.

        G_fn(k) -> (nx, nu) array,  u_fn(k) -> scalar/array. k is 0-indexed.
        """
        F = np.asarray(F, float)
        nx = F.shape[0]
        Fl = [F] * tau
        El = [np.asarray(E, float)] * tau
        Hl = [np.asarray(H, float)] * tau
        Dl = [np.asarray(D, float)] * tau
        if G_fn is None:
            Gl = [np.zeros((nx, 1))] * tau
            ul = None
        else:
            Gl = [np.atleast_2d(G_fn(k)) for k in range(tau)]
            ul = None if u_fn is None else [np.atleast_1d(u_fn(k)) for k in range(tau)]
        return cls(Fl, El, Hl, Dl, Gl, ul)

    # ---- simulation ---------------------------------------------------------
    def simulate(self, Q, R, rng=None, x0_mean=None, x0_cov=None):
        r"""
        Draw one trajectory and return the measurement list ``z`` (length tau).

        Initial state defaults to the paper's convention
        E[x0] = 1_{nx}, VAR[x0] = I_{nx}.
        """
        rng = np.random.default_rng() if rng is None else rng
        Q = np.asarray(Q, float); R = np.asarray(R, float)
        cQ = _safe_cholesky(Q, self.nw)
        cR = _safe_cholesky(R, self.nv)
        x0_mean = np.ones(self.nx) if x0_mean is None else np.asarray(x0_mean, float)
        L0 = np.eye(self.nx) if x0_cov is None else np.linalg.cholesky(x0_cov)

        w = cQ @ rng.standard_normal((self.nw, self.tau))
        v = cR @ rng.standard_normal((self.nv, self.tau))
        x = np.empty((self.nx, self.tau))
        x[:, 0] = x0_mean + L0 @ rng.standard_normal(self.nx)
        z = [None] * self.tau
        for k in range(self.tau):
            z[k] = self.H[k] @ x[:, k] + self.D[k] @ v[:, k]
            if k < self.tau - 1:
                xin = self.G[k] @ self.u[k] if self.u is not None else 0.0
                x[:, k + 1] = self.F[k] @ x[:, k] + xin + self.E[k] @ w[:, k]
        return z
