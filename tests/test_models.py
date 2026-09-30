"""Unit tests for baseline models (MLP and BiLSTM).

Both must:
  - Accept (B, T, L, V) input
  - Return tendency (B, T, L, V_out), precip (B, T, V_precip), attention=None
  - Produce finite outputs
  - Have a num_parameters() method
"""
import pytest
import torch

from src.column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
)
from src.model_mlp import MLPConvectionEmulator
from src.model_bilstm import BiLSTMConvectionEmulator


MODELS = [
    MLPConvectionEmulator,
    BiLSTMConvectionEmulator,
]


@pytest.fixture
def dummy_input():
    B, T = 2, 12
    return torch.randn(B, T, N_LEVELS, N_INPUT_VARS)


@pytest.mark.parametrize("model_cls", MODELS)
def test_output_shapes(model_cls, dummy_input):
    model = model_cls()
    out = model(dummy_input)
    B, T = dummy_input.shape[:2]
    assert out["tendency"].shape == (B, T, N_LEVELS, N_OUTPUT_VARS)
    assert out["precip"].shape == (B, T, N_PRECIP_VARS)


@pytest.mark.parametrize("model_cls", MODELS)
def test_attention_is_none(model_cls, dummy_input):
    """MLP and BiLSTM don't return attention weights."""
    model = model_cls()
    out = model(dummy_input)
    assert out["attention"] is None


@pytest.mark.parametrize("model_cls", MODELS)
def test_outputs_finite(model_cls, dummy_input):
    model = model_cls()
    out = model(dummy_input)
    assert torch.isfinite(out["tendency"]).all()
    assert torch.isfinite(out["precip"]).all()


@pytest.mark.parametrize("model_cls", MODELS)
def test_num_parameters(model_cls):
    model = model_cls()
    n = model.num_parameters()
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.parametrize("model_cls", MODELS)
def test_gradients_flow(model_cls, dummy_input):
    """Backward pass should populate gradients on all parameters."""
    model = model_cls()
    out = model(dummy_input)
    loss = out["tendency"].pow(2).mean() + out["precip"].pow(2).mean()
    loss.backward()

    n_with_grad = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    # At least most parameters should get gradients (some may be unused)
    assert n_with_grad > 0.8 * n_trainable


@pytest.mark.parametrize("model_cls", MODELS)
def test_batch_size_1(model_cls, dummy_input):
    """Batch size 1 should work (common during inference)."""
    model = model_cls()
    single = dummy_input[:1]
    out = model(single)
    assert out["tendency"].shape[0] == 1
    assert out["precip"].shape[0] == 1
