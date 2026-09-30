"""Shared column specification: vertical grid and variable ordering.

Both the synthetic generator and the real-data preprocessor must produce
data in exactly this layout. This is the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Vertical grid
# ---------------------------------------------------------------------------
N_LEVELS = 30
Z_SURFACE_M = 0.0
Z_TOP_M = 20_000.0


def build_vertical_grid(n_levels: int = N_LEVELS,
                        z_top: float = Z_TOP_M) -> np.ndarray:
    """Return layer-interface heights [m], length n_levels+1.

    Geometric stretch: denser near the surface, sparser aloft.
    """
    eta = np.linspace(0.0, 1.0, n_levels + 1)
    z = z_top * (eta ** 1.6)
    return z


Z_INTERFACES = build_vertical_grid()
Z_CENTERS = 0.5 * (Z_INTERFACES[:-1] + Z_INTERFACES[1:])

# ---------------------------------------------------------------------------
# Variable ordering (MUST match the NetCDF canonical format)
# ---------------------------------------------------------------------------
INPUT_VARS: List[str] = [
    "T",       # temperature [K]
    "q",       # specific humidity [kg/kg]
    "q_c",     # cloud liquid water [kg/kg]
    "q_i",     # cloud ice [kg/kg]
    "u",       # zonal wind [m/s]
    "v",       # meridional wind [m/s]
    "p",       # pressure [Pa]
    "z",       # geopotential height [m]
    "Q_rad",   # radiative heating rate [K/s]
    "Q_q",     # large-scale moisture forcing [kg/kg/s]
]
N_INPUT_VARS = len(INPUT_VARS)

OUTPUT_VARS: List[str] = [
    "dT_dt",
    "dq_dt",
    "dqc_dt",
    "dqi_dt",
    "du_dt",
    "dv_dt",
]
N_OUTPUT_VARS = len(OUTPUT_VARS)

PRECIP_VARS: List[str] = ["rain_rate", "snow_rate"]
N_PRECIP_VARS = len(PRECIP_VARS)

META_VARS: List[str] = ["lat", "lon", "time", "sst", "ps"]
N_META_VARS = len(META_VARS)

# Indices for physics losses
IDX_T = INPUT_VARS.index("T")
IDX_Q = INPUT_VARS.index("q")
IDX_QC = INPUT_VARS.index("q_c")
IDX_QI = INPUT_VARS.index("q_i")

IDX_DT = OUTPUT_VARS.index("dT_dt")
IDX_DQ = OUTPUT_VARS.index("dq_dt")
IDX_DQC = OUTPUT_VARS.index("dqc_dt")
IDX_DQI = OUTPUT_VARS.index("dqi_dt")

# Physical constants (SI)
C_P = 1005.0          # specific heat of dry air [J/kg/K]
L_V = 2.501e6         # latent heat of vaporization [J/kg]
L_S = 2.834e6         # latent heat of sublimation [J/kg]
RHO_REF = 1.2         # reference air density [kg/m^3]
G = 9.81


@dataclass
class ColumnSpec:
    """Convenience container for shared spec."""
    n_levels: int = N_LEVELS
    n_input_vars: int = N_INPUT_VARS
    n_output_vars: int = N_OUTPUT_VARS
    n_precip_vars: int = N_PRECIP_VARS
    input_vars: List[str] = field(default_factory=lambda: list(INPUT_VARS))
    output_vars: List[str] = field(default_factory=lambda: list(OUTPUT_VARS))
    z_interfaces: np.ndarray = field(default_factory=lambda: Z_INTERFACES.copy())
    z_centers: np.ndarray = field(default_factory=lambda: Z_CENTERS.copy())
