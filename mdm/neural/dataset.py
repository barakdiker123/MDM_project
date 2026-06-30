r"""
Sim2real dataset generator for the neural noise-covariance estimator.

For each sample: draw a PSD-feasible weight vector alpha, simulate one
trajectory from the given :class:`StateSpaceModel`, and compute the MDM
sufficient statistic ``s``. The network learns the map  s -> alpha  across the
distribution of feasible alpha (this is what makes it a learned prior).

The generator is model-agnostic: pass any (model, basis, MDM) triple, e.g. the
one from ``examples/ex_B_unobservable_unknown_input.py``.
"""
from __future__ import annotations
import numpy as np
from ..linalg import is_psd


def make_feasible_sampler(basis, lo, hi, n_tries=500):
    r"""
    Return a function ``sample(rng) -> alpha`` drawing uniformly from the box
    [lo, hi] subject to Q(alpha), R(alpha) both PSD (rejection sampling).
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)

    def sample(rng):
        for _ in range(n_tries):
            alpha = rng.uniform(lo, hi)
            Q, R = basis.to_QR(alpha)
            if is_psd(Q) and is_psd(R):
                return alpha
        raise RuntimeError("PSD rejection rate too high; tighten the box [lo, hi].")

    return sample


def generate_dataset(model, basis, est, sampler, n_samples, seed=0, verbose=True):
    r"""
    Returns
    -------
    S_arr : (n_samples, n_alpha) float32  -- sufficient statistics s
    a_arr : (n_samples, n_alpha) float32  -- true weight vectors alpha
    """
    rng = np.random.default_rng(seed)
    na = basis.n_alpha
    S_arr = np.empty((n_samples, na), np.float32)
    a_arr = np.empty((n_samples, na), np.float32)
    for i in range(n_samples):
        if verbose and i % 2000 == 0:
            print(f"  sample {i}/{n_samples}", flush=True)
        alpha = sampler(rng)
        Q, R = basis.to_QR(alpha)
        z = model.simulate(Q, R, rng)
        s = est.sufficient_stat(est.residue_cov(z))
        S_arr[i] = s.astype(np.float32)
        a_arr[i] = alpha.astype(np.float32)
    return S_arr, a_arr
