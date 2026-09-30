"""Unit tests for the physics-constrained loss.

Tests cover:
  - Individual loss terms (data, mass, energy, positivity)
  - Composite loss
  - Gradient flow through all terms
  - Edge cases (zero tendencies, zero precip)
"""
import pytest
import torch

from src.column_spec import (
    N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
    Z_INTERFACES, IDX_QC, IDX_QI,
)
from src.physics_loss import (
    LossConfig, PhysicsConstrainedLoss,
    data_loss, mass_conservation_loss, energy_conservation_loss, positivity_loss,
)
from src.model_mlp import MLPConvectionEmulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def batch():
    B, T, L = 4, 12, 30
    torch.manual_seed(0)
    return {
        "state": torch.randn(B, T, L, N_INPUT_VARS) * 0.1,
        "pred_tend": torch.randn(B, T, L, N_OUTPUT_VARS) * 1e-5,
        "true_tend": torch.randn(B, T, L, N_OUTPUT_VARS) * 1e-5,
        "pred_precip": torch.rand(B, T, N_PRECIP_VARS) * 5.0,
        "true_precip": torch.rand(B, T, N_PRECIP_VARS) * 5.0,
    }


@pytest.fixture
def dz():
    dz_full = torch.as_tensor(Z_INTERFACES, dtype=torch.float32)
    return torch.diff(dz_full)


@pytest.fixture
def loss_fn(dz):
    return PhysicsConstrainedLoss(LossConfig(), dz)


# ---------------------------------------------------------------------------
# Data loss
# ---------------------------------------------------------------------------
def test_data_loss_zero_when_identical(batch):
    """Data loss is 0 when prediction equals target."""
    loss = data_loss(batch["pred_tend"], batch["pred_tend"])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_data_loss_positive_when_different(batch):
    loss = data_loss(batch["pred_tend"], batch["true_tend"])
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_data_loss_with_precip(batch):
    """Data loss with precip includes both tendency and precip."""
    loss_no_precip = data_loss(batch["pred_tend"], batch["true_tend"])
    loss_with_precip = data_loss(
        batch["pred_tend"], batch["true_tend"],
        pred_precip=batch["pred_precip"],
        target_precip=batch["true_precip"],
    )
    assert loss_with_precip.item() > loss_no_precip.item()


# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------
def test_mass_loss_finite(batch, dz):
    loss = mass_conservation_loss(
        batch["pred_tend"], batch["state"], batch["true_precip"], dz,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_mass_loss_zero_for_zero_tendencies(batch, dz):
    """If tendencies are zero and precip is zero, mass loss is 0."""
    zero_tend = torch.zeros_like(batch["pred_tend"])
    zero_precip = torch.zeros_like(batch["true_precip"])
    loss = mass_conservation_loss(
        zero_tend, batch["state"], zero_precip, dz,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mass_loss_accepts_layer_or_interface_dz(batch):
    """Loss should accept dz of length L or L+1."""
    dz_full = torch.as_tensor(Z_INTERFACES, dtype=torch.float32)
    dz_layers = torch.diff(dz_full)

    loss_layers = mass_conservation_loss(
        batch["pred_tend"], batch["state"], batch["true_precip"], dz_layers,
    )
    loss_iface = mass_conservation_loss(
        batch["pred_tend"], batch["state"], batch["true_precip"], dz_full,
    )
    # Should be similar (interface dz gets averaged to layers)
    assert torch.isfinite(loss_layers)
    assert torch.isfinite(loss_iface)


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------
def test_energy_loss_finite(batch, dz):
    loss = energy_conservation_loss(batch["pred_tend"], batch["state"], dz)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_energy_loss_zero_for_zero_dT(batch, dz):
    """Energy loss is 0 when dT/dt is zero everywhere."""
    zero_tend = torch.zeros_like(batch["pred_tend"])
    loss = energy_conservation_loss(zero_tend, batch["state"], dz)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Positivity
# ---------------------------------------------------------------------------
def test_positivity_zero_for_positive_condensate(batch):
    """If qc + dt*dqc is positive everywhere, no penalty."""
    # Set state qc/qi to large positive values, tendencies to small positives
    state = batch["state"].clone()
    state[..., IDX_QC] = 1.0
    state[..., IDX_QI] = 1.0
    pred_tend = torch.ones_like(batch["pred_tend"]) * 1e-6

    loss = positivity_loss(pred_tend, state)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_positivity_positive_for_negative_condensate(batch):
    """If tendencies drive qc negative, penalty is > 0."""
    state = batch["state"].clone()
    state[..., IDX_QC] = 1e-5      # small qc
    state[..., IDX_QI] = 1e-5
    pred_tend = torch.zeros_like(batch["pred_tend"])
    pred_tend[..., 2] = -1e-3      # dqc/dt strongly negative

    loss = positivity_loss(pred_tend, state)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------
def test_composite_loss_returns_all_terms(batch, loss_fn):
    pred = {"tendency": batch["pred_tend"], "precip": batch["pred_precip"]}
    target = {"tendency": batch["true_tend"], "precip": batch["true_precip"]}
    out = loss_fn(pred, target, batch["state"])

    assert set(out.keys()) == {"total", "data", "mass", "energy", "positivity"}
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} not finite"
        assert v.item() >= 0.0, f"{k} is negative"


def test_composite_loss_total_is_weighted_sum(batch, loss_fn):
    """Verify total = λ_data·data + λ_mass·mass + λ_energy·energy + λ_pos·pos."""
    pred = {"tendency": batch["pred_tend"], "precip": batch["pred_precip"]}
    target = {"tendency": batch["true_tend"], "precip": batch["true_precip"]}
    out = loss_fn(pred, target, batch["state"])

    cfg = loss_fn.config
    expected = (cfg.lambda_data * out["data"]
                + cfg.lambda_mass * out["mass"]
                + cfg.lambda_energy * out["energy"]
                + cfg.lambda_pos * out["positivity"])
    assert out["total"].item() == pytest.approx(expected.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow through a model
# ---------------------------------------------------------------------------
def test_gradients_flow_through_model(loss_fn):
    """A full model + loss should produce gradients on all parameters."""
    torch.manual_seed(0)
    B, T, L = 4, 12, 30
    state = torch.randn(B, T, L, N_INPUT_VARS) * 0.1
    true_tend = torch.randn(B, T, L, N_OUTPUT_VARS) * 1e-5
    true_precip = torch.rand(B, T, N_PRECIP_VARS) * 5.0

    model = MLPConvectionEmulator()
    pred = model(state)
    target = {"tendency": true_tend, "precip": true_precip}

    out = loss_fn(pred, target, state)
    out["total"].backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    assert n_with_grad == n_total, (
        f"Only {n_with_grad}/{n_total} params have gradients"
    )


# ---------------------------------------------------------------------------
# LossConfig
# ---------------------------------------------------------------------------
def test_loss_config_from_dict():
    cfg = {"loss": {"lambda_mass": 0.5, "lambda_energy": 0.3}}
    loss_cfg = LossConfig.from_config(cfg)
    assert loss_cfg.lambda_mass == 0.5
    assert loss_cfg.lambda_energy == 0.3
    # Defaults for unspecified
    assert loss_cfg.lambda_data == 1.0
    assert loss_cfg.lambda_pos == 0.05
