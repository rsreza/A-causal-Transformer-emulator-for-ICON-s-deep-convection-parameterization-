"""Physics-constrained loss for convection emulation.

Total loss:
    L = L_data + λ_mass · L_mass + λ_energy · L_energy + λ_pos · L_pos

Following Beucler et al. (2020) and Sarauer et al. (2025), we add
physical constraints as soft penalties. The network is encouraged to
predict tendencies that satisfy conservation laws.

All loss terms are normalized to be O(1) so the λ weights are stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .column_spec import (
    N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
    C_P, IDX_QC, IDX_QI,
    IDX_DT, IDX_DQ, IDX_DQC, IDX_DQI,
)


DT_SECONDS = 3600.0   # 1 hour per timestep (same as generator)


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Weights and settings for the multi-term loss."""
    lambda_data: float = 1.0
    lambda_mass: float = 0.1
    lambda_energy: float = 0.1
    lambda_pos: float = 0.05
    huber_delta: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict) -> "LossConfig":
        loss = cfg.get("loss", {})
        return cls(
            lambda_data=loss.get("lambda_data", 1.0),
            lambda_mass=loss.get("lambda_mass", 0.1),
            lambda_energy=loss.get("lambda_energy", 0.1),
            lambda_pos=loss.get("lambda_pos", 0.05),
            huber_delta=loss.get("huber_delta", 1.0),
        )


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------
def data_loss(pred_tendency: torch.Tensor,
              target_tendency: torch.Tensor,
              pred_precip: Optional[torch.Tensor] = None,
              target_precip: Optional[torch.Tensor] = None,
              delta: float = 1.0) -> torch.Tensor:
    """Huber loss between predicted and target tendencies (and precip).

    Parameters
    ----------
    pred_tendency, target_tendency : (B, T, L, V_out)
    pred_precip, target_precip     : (B, T, V_precip), optional
    delta : Huber transition point

    Returns
    -------
    scalar
    """
    loss = F.huber_loss(pred_tendency, target_tendency, delta=delta)
    if pred_precip is not None and target_precip is not None:
        loss = loss + F.huber_loss(pred_precip, target_precip, delta=delta)
    return loss


def mass_conservation_loss(pred_tendency: torch.Tensor,
                            state: torch.Tensor,
                            target_precip: torch.Tensor,
                            dz: torch.Tensor,
                            ) -> torch.Tensor:
    """Column-integrated water budget residual.

    In a closed column, the water mass tendency integrated over height
    should equal the precipitation flux at the surface:

        ∫ (dq/dt + dqc/dt + dqi/dt) dz + precip_column = 0

    The residual is normalized by the column tendency scale, giving an
    O(1) dimensionless residual when the imbalance is comparable to
    the tendency itself.

    Parameters
    ----------
    pred_tendency : (B, T, L, V_out)  predicted tendencies
    state         : (B, T, L, V_in)   (unused, kept for API symmetry)
    target_precip : (B, T, V_precip)  target precipitation [mm/day]
    dz            : (L,) or (L+1,)    layer thicknesses [m]

    Returns
    -------
    scalar (mean squared normalized residual)
    """
    B, T, L, _ = pred_tendency.shape
    assert dz.numel() in (L, L + 1), f"dz shape mismatch: {dz.shape} vs L={L}"

    if dz.numel() == L + 1:
        dz = 0.5 * (dz[1:] + dz[:-1])

    dz = dz.to(pred_tendency.dtype).to(pred_tendency.device)

    dq   = pred_tendency[..., IDX_DQ]
    dqc  = pred_tendency[..., IDX_DQC]
    dqi  = pred_tendency[..., IDX_DQI]
    dq_total = dq + dqc + dqi

    col_dq = (dq_total * dz.view(1, 1, L)).sum(dim=-1)   # (B, T) [kg/m^2/s]

    precip_kg = target_precip.sum(dim=-1) / 86400.0      # (B, T) [kg/m^2/s]

    residual = col_dq + precip_kg                        # (B, T)

    # Normalize by the mean column tendency scale, not by precip (which
    # is orders of magnitude smaller). Gives O(1) when imbalance is
    # comparable to the mean tendency.
    scale = (col_dq.abs().mean() + precip_kg.abs().mean()).clamp(min=1e-9)
    residual_norm = residual / scale

    return (residual_norm ** 2).mean()


def energy_conservation_loss(pred_tendency: torch.Tensor,
                              state: torch.Tensor,
                              dz: torch.Tensor,
                              ) -> torch.Tensor:
    """Column-integrated enthalpy budget residual.

    The column-integrated Cp·dT/dt is normalized by its own scale, giving
    an O(1) loss when the column is significantly heated/cooled relative
    to the mean magnitude.

    Parameters
    ----------
    pred_tendency : (B, T, L, V_out)
    state         : (B, T, L, V_in)  (unused, kept for API symmetry)
    dz            : (L,) or (L+1,)

    Returns
    -------
    scalar
    """
    B, T, L, _ = pred_tendency.shape
    if dz.numel() == L + 1:
        dz = 0.5 * (dz[1:] + dz[:-1])
    dz = dz.to(pred_tendency.dtype).to(pred_tendency.device)

    dT = pred_tendency[..., IDX_DT]
    col_dT = (C_P * dT * dz.view(1, 1, L)).sum(dim=-1)   # (B, T) [J/m^2/s]

    scale = col_dT.abs().mean().clamp(min=1e-9)
    residual_norm = col_dT / scale

    return (residual_norm ** 2).mean()


def positivity_loss(pred_tendency: torch.Tensor,
                     state: torch.Tensor,
                     dt_seconds: float = DT_SECONDS,
                     ) -> torch.Tensor:
    """Penalize negative cloud liquid / ice after one Euler step.

    For each level, we compute:
        qc_new = qc_current + Δt · dqc/dt_pred
        qi_new = qi_current + Δt · dqi/dt_pred

    And penalize the negative part.

    Parameters
    ----------
    pred_tendency : (B, T, L, V_out)
    state         : (B, T, L, V_in)
    dt_seconds    : timestep for Euler update

    Returns
    -------
    scalar
    """
    qc = state[..., IDX_QC]
    qi = state[..., IDX_QI]
    dqc_dt = pred_tendency[..., IDX_DQC]
    dqi_dt = pred_tendency[..., IDX_DQI]

    qc_new = qc + dt_seconds * dqc_dt
    qi_new = qi + dt_seconds * dqi_dt

    loss_qc = F.relu(-qc_new).mean()
    loss_qi = F.relu(-qi_new).mean()

    return loss_qc + loss_qi


# ---------------------------------------------------------------------------
# Composite loss module
# ---------------------------------------------------------------------------
class PhysicsConstrainedLoss(nn.Module):
    """Composite loss with data + physics terms.

    Parameters
    ----------
    config : LossConfig
    dz : torch.Tensor  layer thicknesses [m], shape (L,) or (L+1,)
    """

    def __init__(self, config: LossConfig, dz: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer("dz", dz.float())

    def forward(self,
                pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                state: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pred : dict with keys 'tendency', 'precip'
        target : dict with same keys
        state : (B,T,L,V_in) input state

        Returns
        -------
        dict with keys: total, data, mass, energy, positivity
        """
        cfg = self.config

        # Data loss includes both tendency and precipitation
        l_data = data_loss(
            pred["tendency"], target["tendency"],
            pred_precip=pred["precip"], target_precip=target["precip"],
            delta=cfg.huber_delta,
        )

        l_mass = mass_conservation_loss(
            pred["tendency"], state, target["precip"], self.dz,
        )

        l_energy = energy_conservation_loss(
            pred["tendency"], state, self.dz,
        )

        l_pos = positivity_loss(pred["tendency"], state)

        total = (cfg.lambda_data * l_data
                 + cfg.lambda_mass * l_mass
                 + cfg.lambda_energy * l_energy
                 + cfg.lambda_pos * l_pos)

        # Return individual terms detached (for logging); total keeps graph
        return {
            "total": total,
            "data": l_data.detach(),
            "mass": l_mass.detach(),
            "energy": l_energy.detach(),
            "positivity": l_pos.detach(),
        }
