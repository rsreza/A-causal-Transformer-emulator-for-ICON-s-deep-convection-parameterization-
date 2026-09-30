"""Feed-forward MLP baseline for convection tendencies.

Treats each timestep independently — no temporal memory. Each (B, T, L, V)
input is reshaped to (B*T*L, V), passed through a shared MLP, then
reshaped back to (B, T, L, N_OUTPUT_VARS).

This establishes the "no memory" baseline. If a more sophisticated model
(Transformer, BiLSTM) can't beat this, something is wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
)


class MLPConvectionEmulator(nn.Module):
    """Level-shared feed-forward MLP.

    Parameters
    ----------
    n_input_vars : int
    n_output_vars : int
    n_levels : int
    t_past : int
    hidden_dims : list of int  hidden layer sizes for tendency head
    precip_hidden : int  hidden size for precipitation head
    dropout : float
    """

    def __init__(self,
                 n_input_vars: int = N_INPUT_VARS,
                 n_output_vars: int = N_OUTPUT_VARS,
                 n_levels: int = N_LEVELS,
                 t_past: int = 12,
                 hidden_dims: tuple = (128, 128),
                 precip_hidden: int = 64,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_levels = n_levels
        self.t_past = t_past

        # Shared tendency MLP (applied per level)
        layers = []
        in_dim = n_input_vars
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_output_vars))
        self.tendency_mlp = nn.Sequential(*layers)

        # Precipitation head: pool over levels, then MLP
        self.precip_mlp = nn.Sequential(
            nn.Linear(n_input_vars, precip_hidden),
            nn.GELU(),
            nn.Linear(precip_hidden, N_PRECIP_VARS),
        )

    def forward(self, state: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        state : (B, T, L, V_in)

        Returns
        -------
        dict with keys:
            tendency : (B, T, L, V_out)
            precip   : (B, T, V_precip)
            attention : None
        """
        B, T, L, V = state.shape
        assert L == self.n_levels
        assert V == self.n_input_vars

        # Reshape to (B*T*L, V) for per-level MLP
        x = state.reshape(B * T * L, V)

        # Predict tendency per level
        tendency_flat = self.tendency_mlp(x)                    # (B*T*L, V_out)
        tendency = tendency_flat.reshape(B, T, L, self.n_output_vars)

        # Predict precipitation: pool (mean + max) over levels, then MLP
        pooled_mean = state.mean(dim=2)                         # (B, T, V)
        pooled_max = state.max(dim=2).values                    # (B, T, V)
        pooled = 0.5 * (pooled_mean + pooled_max)               # (B, T, V)
        precip = self.precip_mlp(pooled.reshape(B * T, V))
        precip = precip.reshape(B, T, N_PRECIP_VARS)

        return {
            "tendency": tendency,
            "precip": precip,
            "attention": None,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
