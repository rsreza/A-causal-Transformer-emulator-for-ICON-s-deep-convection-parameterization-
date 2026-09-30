"""Causal Transformer emulator for ICON convection.

Architecture:
  - Per-level linear embedding: (V_in -> d_model)
  - Learnable positional encoding on the time axis
  - Stack of causal Transformer encoder blocks (multi-head self-attention)
  - Take the last-timestep output (causal: only sees past)
  - Cross-level mixing MLP
  - Parallel heads for tendencies (per-level) and precipitation (pooled)

The attention weights from every layer are returned as a list of tensors
with shape (B, n_heads, T, T). These are the interpretable quantity:
for each prediction, we can plot which past timesteps the model relied on,
stratified by convective regime (shallow/deep/organized/suppressed).

Parameter count target: ~250K-600K, comparable to the BiLSTM baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .column_spec import (
    N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS, N_LEVELS,
)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    """Learnable additive positional encoding on the time axis."""

    def __init__(self, t_past: int, d_model: int, init_scale: float = 0.02):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(t_past, d_model) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.embed[:T].unsqueeze(0)


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Returns (output, attention_weights).
    attention_weights : (B*L, n_heads, T, T)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out, attn


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: attention + feed-forward."""

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        h, attn = self.attn(self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, attn


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CausalTransformerEmulator(nn.Module):
    """Causal Transformer emulator for convective tendencies."""

    def __init__(self,
                 n_input_vars: int = N_INPUT_VARS,
                 n_output_vars: int = N_OUTPUT_VARS,
                 n_levels: int = N_LEVELS,
                 t_past: int = 12,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ff: int = 128,
                 dropout: float = 0.1,
                 cross_level_hidden: int = 128,
                 precip_hidden: int = 64,
                 ):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_levels = n_levels
        self.t_past = t_past
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.input_proj = nn.Linear(n_input_vars, d_model)
        self.pos_enc = LearnablePositionalEncoding(t_past, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Per-level tendency head
        self.tendency_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_output_vars),
        )

        # Cross-level + precipitation head
        self.cross_level = nn.Sequential(
            nn.Linear(d_model * 2, cross_level_hidden),
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
            tendency   : (B, T, L, V_out)
            precip     : (B, T, V_precip)
            attention  : list of (B, n_heads, T, T), one per layer
        """
        B, T, L, V = state.shape
        assert L == self.n_levels
        assert V == self.n_input_vars
        assert T == self.t_past

        # Per-level sequences
        x = state.permute(0, 2, 1, 3).reshape(B * L, T, V)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        attn_list = []
        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        x = self.final_norm(x)

        # Tendency per level
        tendency = self.tendency_head(x)
        tendency = tendency.reshape(B, L, T, self.n_output_vars)
        tendency = tendency.permute(0, 2, 1, 3)

        # Precip: pool over levels
        x_levels = x.reshape(B, L, T, self.d_model).permute(0, 2, 1, 3)
        x_mean = x_levels.mean(dim=2)
        x_max = x_levels.max(dim=2).values
        x_pool = torch.cat([x_mean, x_max], dim=-1)

        h = self.cross_level(x_pool)
        precip = self.precip_head(h)

        # Reduce attention over levels for interpretability
        attn_summary = []
        for a in attn_list:
            a = a.reshape(B, L, self.n_heads, T, T).mean(dim=1)
            attn_summary.append(a)

        return {
            "tendency": tendency,
            "precip": precip,
            "attention": attn_summary,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
