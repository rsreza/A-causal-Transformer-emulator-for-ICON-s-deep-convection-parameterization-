"""Bidirectional LSTM baseline for convection tendencies.

Direct comparison to Heuer et al. (2025), who used BiLSTMs for ICON
convection emulation. Processes the full 12-step sequence bidirectionally
and outputs a prediction at each timestep.

Key difference from the Transformer:
  - Recurrent memory (gated cell state) instead of attention
  - Bidirectional — sees the future in the input window
  - Not causal: attention weights can't be interpreted the same way

This gives us a strong non-Transformer reference point.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
)


class BiLSTMConvectionEmulator(nn.Module):
    """Bidirectional LSTM emulator.

    Treats each vertical level independently through the LSTM (weights shared
    across levels), then mixes levels with a small MLP.

    Parameters
    ----------
    n_input_vars, n_output_vars, n_levels, t_past : as in MLP
    hidden_size : int  LSTM hidden dimension
    n_layers : int  number of stacked LSTM layers
    dropout : float
    cross_level_hidden : int  hidden size of level-mixing MLP
    precip_hidden : int  hidden size of precipitation head
    """

    def __init__(self,
                 n_input_vars: int = N_INPUT_VARS,
                 n_output_vars: int = N_OUTPUT_VARS,
                 n_levels: int = N_LEVELS,
                 t_past: int = 12,
                 hidden_size: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 cross_level_hidden: int = 64,
                 precip_hidden: int = 64,
                 ):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_levels = n_levels
        self.t_past = t_past
        self.hidden_size = hidden_size

        # Input embedding per level
        self.input_proj = nn.Linear(n_input_vars, hidden_size)

        # Bidirectional LSTM — applied to each level independently
        # We batch levels into the batch dimension: (B*L, T, H)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Per-level tendency head (post-LSTM, sees both directions)
        self.tendency_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_output_vars),
        )

        # Cross-level mixing + precipitation head
        self.cross_level = nn.Sequential(
            nn.Linear(n_input_vars * n_levels + 2 * hidden_size, cross_level_hidden),
            nn.GELU(),
            nn.Linear(cross_level_hidden, cross_level_hidden),
            nn.GELU(),
        )
        self.precip_head = nn.Linear(cross_level_hidden, N_PRECIP_VARS)

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

        # Reshape to per-level sequences: (B*L, T, V)
        x = state.permute(0, 2, 1, 3).reshape(B * L, T, V)

        # Project to hidden size
        x = self.input_proj(x)                       # (B*L, T, H)

        # BiLSTM over time
        lstm_out, _ = self.lstm(x)                   # (B*L, T, 2H)

        # Tendency per level
        tendency = self.tendency_head(lstm_out)      # (B*L, T, V_out)
        tendency = tendency.reshape(B, L, T, self.n_output_vars)
        tendency = tendency.permute(0, 2, 1, 3)      # (B, T, L, V_out)

        # Cross-level + precipitation
        # Flatten the full input state plus the LSTM summary
        state_flat = state.reshape(B, T, L * V)                      # (B, T, L*V)
        lstm_summary = lstm_out.mean(dim=0).mean(dim=0)              # (2H,)
        lstm_summary = lstm_summary.unsqueeze(0).expand(B * T, -1)   # (B*T, 2H)
        combined = torch.cat([state_flat.reshape(B * T, L * V),
                              lstm_summary], dim=-1)                 # (B*T, L*V + 2H)
        hidden = self.cross_level(combined)                          # (B*T, H_cross)
        precip = self.precip_head(hidden).reshape(B, T, N_PRECIP_VARS)

        return {
            "tendency": tendency,
            "precip": precip,
            "attention": None,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
