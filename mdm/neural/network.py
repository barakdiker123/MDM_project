r"""
MLP mapping the MDM sufficient statistic  s  to the noise-weight vector alpha.

Architecture (matches the reference port):
    Normalise -> Linear(n,128) -> ELU -> Linear(128,128) -> ELU
              -> Linear(128,64) -> ELU -> Linear(64,n)

Input standardisation statistics are stored as non-trainable buffers so the
normalisation is baked into any export (ONNX / weights).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CovarianceEstimator(nn.Module):
    def __init__(self, s_mean: torch.Tensor, s_std: torch.Tensor, n_alpha: int = 6):
        super().__init__()
        self.register_buffer("s_mean", s_mean.float())
        self.register_buffer("s_std", s_std.float().clamp(min=1e-6))
        self.net = nn.Sequential(
            nn.Linear(n_alpha, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, n_alpha),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net((s - self.s_mean) / self.s_std)
