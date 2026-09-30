"""Convective regime definitions.

Four regimes, each with a characteristic profile shape and memory length.
Used by:
  - the synthetic generator (to plant ground truth)
  - the evaluation (to stratify metrics)
  - the attention analysis (to compare memory across regimes)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import numpy as np

from .column_spec import Z_CENTERS


class Regime(IntEnum):
    SHALLOW = 0
    DEEP = 1
    ORGANIZED = 2
    SUPPRESSED = 3


REGIME_NAMES: Dict[int, str] = {
    Regime.SHALLOW: "shallow",
    Regime.DEEP: "deep",
    Regime.ORGANIZED: "organized",
    Regime.SUPPRESSED: "suppressed",
}

REGIME_FROM_NAME: Dict[str, Regime] = {v: k for k, v in REGIME_NAMES.items()}


@dataclass(frozen=True)
class RegimeProfile:
    name: str
    cloud_top_km: float
    mass_flux_peak_km: float
    mass_flux_width_km: float
    precip_scale: float
    memory_hours: float
    entrainment_rate: float


REGIME_PROFILES: Dict[Regime, RegimeProfile] = {
    Regime.SHALLOW: RegimeProfile(
        name="shallow",
        cloud_top_km=2.5,
        mass_flux_peak_km=1.0,
        mass_flux_width_km=1.5,
        precip_scale=0.0,
        memory_hours=2.0,
        entrainment_rate=1.0e-3,
    ),
    Regime.DEEP: RegimeProfile(
        name="deep",
        cloud_top_km=13.0,
        mass_flux_peak_km=5.0,
        mass_flux_width_km=6.0,
        precip_scale=1.0,
        memory_hours=6.0,
        entrainment_rate=2.0e-4,
    ),
    Regime.ORGANIZED: RegimeProfile(
        name="organized",
        cloud_top_km=14.0,
        mass_flux_peak_km=6.0,
        mass_flux_width_km=8.0,
        precip_scale=1.4,
        memory_hours=10.0,
        entrainment_rate=1.5e-4,
    ),
    Regime.SUPPRESSED: RegimeProfile(
        name="suppressed",
        cloud_top_km=0.0,
        mass_flux_peak_km=0.0,
        mass_flux_width_km=0.1,
        precip_scale=0.0,
        memory_hours=0.5,
        entrainment_rate=2.0e-3,
    ),
}


def regime_mass_flux_shape(regime: Regime,
                           z_centers: np.ndarray = Z_CENTERS) -> np.ndarray:
    """Normalized vertical shape of the mass flux [dimensionless, peak ~1]."""
    p = REGIME_PROFILES[regime]
    if regime == Regime.SUPPRESSED:
        return np.zeros_like(z_centers)
    z_km = z_centers / 1000.0
    shape = np.exp(-((z_km - p.mass_flux_peak_km) ** 2) /
                   (2.0 * p.mass_flux_width_km ** 2))
    shape[z_km > p.cloud_top_km] = 0.0
    shape[z_km < 0.3] = 0.0
    if shape.max() > 0:
        shape = shape / shape.max()
    return shape


def regime_memory_length(regime: Regime, t_past: int = 12) -> int:
    """Expected memory length in timesteps (1 h each), clipped to t_past."""
    hours = REGIME_PROFILES[regime].memory_hours
    return int(min(t_past, max(1, round(hours))))
