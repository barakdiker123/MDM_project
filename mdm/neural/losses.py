r"""
Losses for the neural noise-covariance estimator.

Training loss (Frobenius)
-------------------------
Compare predicted and true noise covariances directly in matrix space:

    L = || Q(alpha_hat) - Q(alpha) ||_F^2 + || R(alpha_hat) - R(alpha) ||_F^2

This is well defined for every prediction (no PSD requirement) and weights the
unique CM entries by their multiplicity, which is the natural metric for the
covariance-identification task.

Evaluation metric (Stein loss)
------------------------------
Scale-aware discrepancy between two SPD matrices,

    d_Stein(A, B) = tr(A B^{-1}) - log det(A B^{-1}) - n  >= 0,

zero iff A = B. Used only for *evaluation*: it needs both matrices SPD, which an
unconstrained network does not guarantee early in training, so predictions are
PSD-projected first. (This is why Stein is unsuitable as the training loss.)
"""

from __future__ import annotations
import numpy as np


def frobenius_loss_torch(alpha_hat, alpha_true, BQ, BR):
    r"""
    Torch Frobenius loss. ``BQ``/``BR`` are (n_alpha, n, n) tensors of
    structure-defining matrices; ``alpha_*`` are (batch, n_alpha) tensors.
    """
    import torch

    Qh = torch.einsum("bi,ijk->bjk", alpha_hat, BQ)
    Rh = torch.einsum("bi,ijk->bjk", alpha_hat, BR)
    Qt = torch.einsum("bi,ijk->bjk", alpha_true, BQ)
    Rt = torch.einsum("bi,ijk->bjk", alpha_true, BR)
    return ((Qh - Qt) ** 2).sum((-1, -2)).mean() + ((Rh - Rt) ** 2).sum((-1, -2)).mean()


def stein_loss(A, B):
    r"""Numpy Stein loss between SPD matrices A and B (PSD-projected if needed)."""
    from ..linalg import psd_project

    A = psd_project(A)
    B = psd_project(B)
    M = A @ np.linalg.inv(B)
    sign, logdet = np.linalg.slogdet(M)
    return float(np.trace(M) - logdet - A.shape[0])


def kl_loss(A, B):
    r"""
    KL divergence KL( N(0,A) || N(0,B) ) between zero-mean Gaussians.

        KL = 0.5 * [ tr(B^{-1} A) - n + log(det(B)/det(A)) ]
           = 0.5 * stein_loss(A, B)

    Requires both A and B to be positive definite.
    """
    return 0.5 * stein_loss(A, B)
