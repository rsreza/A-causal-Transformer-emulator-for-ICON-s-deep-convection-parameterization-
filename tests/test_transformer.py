"""Unit tests for the Causal Transformer emulator.

Key tests:
  - Forward pass produces expected shapes
  - Attention weights have correct shape and causality
  - Causality: future inputs do NOT affect earlier outputs
  - Gradients flow through all parameters
"""
import pytest
import torch

from src.column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
)
from src.model_transformer import (
    CausalTransformerEmulator,
    CausalSelfAttention,
    TransformerBlock,
    LearnablePositionalEncoding,
)


T_PAST = 12


@pytest.fixture
def dummy_input():
    torch.manual_seed(0)
    B = 2
    return torch.randn(B, T_PAST, N_LEVELS, N_INPUT_VARS)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return CausalTransformerEmulator(t_past=T_PAST)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_forward_shapes(model, dummy_input):
    out = model(dummy_input)
    B = dummy_input.shape[0]
    assert out["tendency"].shape == (B, T_PAST, N_LEVELS, N_OUTPUT_VARS)
    assert out["precip"].shape == (B, T_PAST, N_PRECIP_VARS)


def test_attention_shape(model, dummy_input):
    out = model(dummy_input)
    B = dummy_input.shape[0]
    assert isinstance(out["attention"], list)
    assert len(out["attention"]) == model.n_layers
    for a in out["attention"]:
        assert a.shape == (B, model.n_heads, T_PAST, T_PAST)


def test_outputs_finite(model, dummy_input):
    out = model(dummy_input)
    assert torch.isfinite(out["tendency"]).all()
    assert torch.isfinite(out["precip"]).all()


# ---------------------------------------------------------------------------
# Causality — the key tests
# ---------------------------------------------------------------------------
def test_causality_permuting_future_does_not_change_past(model, dummy_input):
    """Permuting future timesteps must not change outputs at earlier timesteps."""
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)

    permuted = dummy_input.clone()
    permuted[:, 6:] = dummy_input[:, torch.randperm(6) + 6]

    with torch.no_grad():
        out2 = model(permuted)

    diff_early = (out1["tendency"][:, :6] - out2["tendency"][:, :6]).abs().max()
    assert diff_early < 1e-5, (
        f"Causality violated: early timesteps changed by {float(diff_early):.2e}"
    )


def test_attention_is_causal(model, dummy_input):
    """Upper triangle of attention (future positions) must be 0."""
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    for i, a in enumerate(out["attention"]):
        upper = torch.triu(a, diagonal=1)
        max_upper = float(upper.abs().max())
        assert max_upper < 1e-6, (
            f"Layer {i} attends to future positions: max = {max_upper:.2e}"
        )


def test_attention_rows_sum_to_one(model, dummy_input):
    """Softmax over time axis — each row sums to 1."""
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    for i, a in enumerate(out["attention"]):
        row_sums = a.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-5
        ), f"Layer {i}: attention rows don't sum to 1"


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
def test_gradients_flow(model, dummy_input):
    out = model(dummy_input)
    loss = out["tendency"].pow(2).mean() + out["precip"].pow(2).mean()
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    assert n_with_grad == n_total, (
        f"Only {n_with_grad}/{n_total} params have gradients"
    )


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------
def test_num_parameters(model):
    n = model.num_parameters()
    assert isinstance(n, int)
    assert n > 0
    assert 50_000 < n < 2_000_000, f"Unexpected param count: {n}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_batch_size_1(model, dummy_input):
    out = model(dummy_input[:1])
    assert out["tendency"].shape[0] == 1
    assert out["precip"].shape[0] == 1


def test_eval_mode_no_dropout(model, dummy_input):
    """Two eval-mode forward passes should be identical."""
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)
    assert torch.allclose(out1["tendency"], out2["tendency"], atol=1e-6)


def test_different_inputs_different_outputs(model):
    torch.manual_seed(0)
    x1 = torch.randn(2, T_PAST, N_LEVELS, N_INPUT_VARS)
    x2 = torch.randn(2, T_PAST, N_LEVELS, N_INPUT_VARS)
    model.eval()
    with torch.no_grad():
        y1 = model(x1)["tendency"]
        y2 = model(x2)["tendency"]
    assert not torch.allclose(y1, y2), \
        "Model produces identical outputs for different inputs"


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------
def test_causal_self_attention_causality():
    """CausalSelfAttention alone must be causal (in eval mode, no dropout)."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    attn.eval()  # disable dropout for deterministic comparison
    x = torch.randn(2, 8, 32)

    x_perm = x.clone()
    x_perm[:, 4:] = x[:, torch.randperm(4) + 4]

    with torch.no_grad():
        out1, _ = attn(x)
        out2, _ = attn(x_perm)

    diff = (out1[:, :4] - out2[:, :4]).abs().max()
    assert diff < 1e-6, f"Attention not causal: diff={float(diff):.2e}"


def test_positional_encoding_shapes():
    pe = LearnablePositionalEncoding(t_past=T_PAST, d_model=16)
    x = torch.randn(4, T_PAST, 16)
    out = pe(x)
    assert out.shape == x.shape


def test_transformer_block_shapes():
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64)
    x = torch.randn(4, T_PAST, 32)
    out, attn = block(x)
    assert out.shape == x.shape
    assert attn.shape == (4, 4, T_PAST, T_PAST)
