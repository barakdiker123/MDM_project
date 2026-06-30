r"""
mdm -- Measurement Difference Method for noise covariance identification.

A Python re-implementation, in the notation of

    O. Kost, J. Duník, I. Puncochář, O. Straka,
    "Unobservable Systems: No Problem for Noise Identification,"
    IEEE Transactions on Automatic Control, 71(2):1223-1230, Feb. 2026.

of the publicly available MATLAB toolbox. Supports linear time-varying systems,
unobservable states (via an annihilation matrix instead of a pseudoinverse),
unknown inputs, and both ordinary and weighted least-squares estimation.

Quick start
-----------
>>> from mdm import StateSpaceModel, NoiseBasis, MDM
>>> model = StateSpaceModel.from_constant(F, E, H, D, tau, G_fn, u_fn)
>>> basis = NoiseBasis(BQ, BR)
>>> est   = MDM(model, basis, L=2, unknown_input=True)
>>> z     = model.simulate(Q_true, R_true, rng)
>>> alpha_hat = est.fit_ordinary(z)            # eq. 21
>>> alpha_w   = est.fit_weighted(z)            # eq. 23
"""
from .model import StateSpaceModel
from .parameterization import NoiseBasis
from .mdm import MDM, annihilation_matrix
from .augmented import observability_gamma

__all__ = [
    "StateSpaceModel",
    "NoiseBasis",
    "MDM",
    "annihilation_matrix",
    "observability_gamma",
]
__version__ = "1.0.0"
