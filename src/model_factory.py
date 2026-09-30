"""Model factory: build a model from config.

Reads cfg['model'] and returns the right model class with the right
hyperparameters. To add a new architecture, just add a branch here.
"""
from __future__ import annotations

import torch.nn as nn

from .column_spec import (
    N_INPUT_VARS, N_OUTPUT_VARS, N_LEVELS,
)
from .model_mlp import MLPConvectionEmulator
from .model_bilstm import BiLSTMConvectionEmulator


def build_model(cfg: dict) -> nn.Module:
    """Build a model from config['model'].

    Parameters
    ----------
    cfg : full config dict

    Returns
    -------
    nn.Module
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()

    common = dict(
        n_input_vars=N_INPUT_VARS,
        n_output_vars=N_OUTPUT_VARS,
        n_levels=model_cfg.get("n_levels", N_LEVELS),
        t_past=model_cfg.get("t_past", 12),
    )

    if name == "mlp":
        return MLPConvectionEmulator(
            **common,
            hidden_dims=tuple(model_cfg.get("hidden_dims", (128, 128))),
            precip_hidden=model_cfg.get("precip_hidden", 64),
            dropout=model_cfg.get("dropout", 0.0),
        )

    elif name == "bilstm":
        return BiLSTMConvectionEmulator(
            **common,
            hidden_size=model_cfg.get("hidden_size", 64),
            n_layers=model_cfg.get("n_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            cross_level_hidden=model_cfg.get("cross_level_hidden", 64),
            precip_hidden=model_cfg.get("precip_hidden", 64),
        )

    elif name == "transformer":
        from .model_transformer import CausalTransformerEmulator
        return CausalTransformerEmulator(
            **common,
            d_model=model_cfg.get("d_model", 64),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 4),
            d_ff=model_cfg.get("d_ff", 128),
            dropout=model_cfg.get("dropout", 0.1),
            cross_level_hidden=model_cfg.get("cross_level_hidden", 128),
        )

    else:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Choose from 'mlp', 'bilstm', 'transformer'."
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
