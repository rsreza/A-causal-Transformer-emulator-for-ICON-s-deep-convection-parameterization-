"""Input/output normalization for the convection emulator.

Physics data has wildly different scales across variables:
  T ~ 300 K, p ~ 100000 Pa, q ~ 0.02 kg/kg, Q_rad ~ 1e-5 K/s

Without normalization, neural networks struggle to learn. We compute
per-variable mean and std from the training set and apply:

    x_norm = (x - mean) / (std + eps)

Both inputs and targets are normalized. The model sees normalized inputs
and produces normalized outputs. The training loop then denormalizes
the predictions before applying the physics-constrained loss, so the
mass/energy/positivity terms work in physical units while the data term
works in normalized space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


EPS = 1e-6


@dataclass
class Normalizer:
    """Per-variable normalization statistics.

    state_mean, state_std       : (N_LEVELS, N_INPUT_VARS)
    tendency_mean, tendency_std : (N_LEVELS, N_OUTPUT_VARS)
    precip_mean, precip_std     : (N_PRECIP_VARS,)
    """
    state_mean: np.ndarray
    state_std: np.ndarray
    tendency_mean: np.ndarray
    tendency_std: np.ndarray
    precip_mean: np.ndarray
    precip_std: np.ndarray

    # ----- Save / load -----
    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            state_mean=self.state_mean, state_std=self.state_std,
            tendency_mean=self.tendency_mean, tendency_std=self.tendency_std,
            precip_mean=self.precip_mean, precip_std=self.precip_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = np.load(path, allow_pickle=False)
        return cls(
            state_mean=d["state_mean"], state_std=d["state_std"],
            tendency_mean=d["tendency_mean"], tendency_std=d["tendency_std"],
            precip_mean=d["precip_mean"], precip_std=d["precip_std"],
        )

    # ----- Compute from data -----
    @classmethod
    def compute(cls,
                state: np.ndarray,
                tendency: np.ndarray,
                precip: np.ndarray,
                ) -> "Normalizer":
        """Compute stats over all samples and timesteps.

        state    : (N, T, L, V_in)
        tendency : (N, T, L, V_out)
        precip   : (N, T, V_precip)
        """
        N, T, L, V = state.shape
        state_flat = state.reshape(-1, L, V)
        tendency_flat = tendency.reshape(-1, L, tendency.shape[-1])
        precip_flat = precip.reshape(-1, precip.shape[-1])

        state_mean = state_flat.mean(axis=0)
        state_std = state_flat.std(axis=0)
        tendency_mean = tendency_flat.mean(axis=0)
        tendency_std = tendency_flat.std(axis=0)
        precip_mean = precip_flat.mean(axis=0)
        precip_std = precip_flat.std(axis=0)

        state_std = np.maximum(state_std, EPS)
        tendency_std = np.maximum(tendency_std, EPS)
        precip_std = np.maximum(precip_std, EPS)

        return cls(
            state_mean=state_mean.astype(np.float32),
            state_std=state_std.astype(np.float32),
            tendency_mean=tendency_mean.astype(np.float32),
            tendency_std=tendency_std.astype(np.float32),
            precip_mean=precip_mean.astype(np.float32),
            precip_std=precip_std.astype(np.float32),
        )

    # ----- Torch version for on-device application -----
    def to_torch(self, device=None) -> "TorchNormalizer":
        return TorchNormalizer(
            state_mean=torch.as_tensor(self.state_mean, device=device),
            state_std=torch.as_tensor(self.state_std, device=device),
            tendency_mean=torch.as_tensor(self.tendency_mean, device=device),
            tendency_std=torch.as_tensor(self.tendency_std, device=device),
            precip_mean=torch.as_tensor(self.precip_mean, device=device),
            precip_std=torch.as_tensor(self.precip_std, device=device),
        )


class TorchNormalizer:
    """Torch version of Normalizer, applies on device."""

    def __init__(self,
                 state_mean, state_std,
                 tendency_mean, tendency_std,
                 precip_mean, precip_std):
        self.state_mean = state_mean
        self.state_std = state_std
        self.tendency_mean = tendency_mean
        self.tendency_std = tendency_std
        self.precip_mean = precip_mean
        self.precip_std = precip_std

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, L, V) -> (B, T, L, V)"""
        return (x - self.state_mean) / self.state_std

    def normalize_tendency(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.tendency_mean) / self.tendency_std

    def normalize_precip(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V) -> (B, T, V)"""
        return (x - self.precip_mean) / self.precip_std

    def denormalize_tendency(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.tendency_std + self.tendency_mean

    def denormalize_precip(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.precip_std + self.precip_mean
