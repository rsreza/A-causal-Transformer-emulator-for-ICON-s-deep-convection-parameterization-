"""Simplified 1-D mass-flux physics for synthetic convection generation.

This is NOT ICON's Tiedtke-Bechtold scheme. It is a deliberately simplified,
regime-parameterized mass-flux model designed to produce data with:

  - learnable, regime-dependent tendencies
  - a known ground-truth regime label
  - realistic vertical structure
  - temporal memory (moisture preconditioning builds up over hours)

Design choices for numerical stability at dt = 3600 s:

  1. Condensate (qc, qi) is DIAGNOSTIC, not prognostic.
     Physics: cloud liquid/ice adjusts to local source/sink on a ~1000 s
     timescale, far shorter than our 1-hour step. Setting qc = cond/K_eff
     (equilibrium) every step is physically correct and eliminates the
     stiff-ODE oscillation that plagued an explicit Euler integration.

  2. Temperature, moisture, and momentum are prognostically integrated
     via Euler, but their tendencies are physically scaled so state
     changes per step are small compared to the state itself.

Magnitude calibration (target = tropical convection):
  M          ~ 0.05 kg/m^2/s
  dT/dt      ~ 1e-5 to 1e-4 K/s
  dq/dt      ~ 1e-8 to 1e-7 kg/kg/s
  rain       ~ 10 to 30 mm/day
  qc         ~ 1e-4 kg/kg peak
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.column_spec import (
    N_LEVELS, Z_CENTERS, C_P, L_V,
)
from src.regimes import (
    Regime, REGIME_PROFILES, regime_mass_flux_shape,
)

DT_SECONDS = 3600.0   # 1 hour per timestep

# Realistic condensate caps
QC_MAX = 5.0e-3       # 5 g/kg cloud liquid water
QI_MAX = 3.0e-3       # 3 g/kg cloud ice

# Autoconversion rate constants (physical)
K_AUTO = 1.0e-3       # [1/s] for cloud liquid -> rain
K_ICE_AUTO = 5.0e-4   # [1/s] for cloud ice -> snow


def _stiff_decay_rate(k: float, dt: float) -> float:
    """Effective decay rate for a stiff first-order sink.

    For dX/dt = -k*X, the exact solution over dt is X*exp(-k*dt).
    Matching to an Euler form X*(1 - K_eff*dt) gives
        K_eff = (1 - exp(-k*dt)) / dt,
    which satisfies K_eff * dt < 1 for all k, dt > 0.
    """
    return (1.0 - float(np.exp(-k * dt))) / dt


# Pre-compute the effective rates (constant for our fixed timestep)
K_AUTO_EFF = _stiff_decay_rate(K_AUTO, DT_SECONDS)
K_ICE_AUTO_EFF = _stiff_decay_rate(K_ICE_AUTO, DT_SECONDS)


# ---------------------------------------------------------------------------
# Background atmosphere
# ---------------------------------------------------------------------------
def background_state(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Return a plausible tropical background profile.

    qc and qi are initialized to zero; they are diagnostic and will be
    set on the first call to mass_flux_step.
    """
    z = Z_CENTERS
    z_km = z / 1000.0

    T = 300.0 - 6.5 * np.minimum(z_km, 11.0)
    T[z_km > 11.0] = T[z_km > 11.0] - 1.0 * (z_km[z_km > 11.0] - 11.0)
    T = T + rng.normal(0.0, 1.0, size=N_LEVELS)

    p = 101300.0 * np.exp(-z / 8000.0)
    p = p + rng.normal(0.0, 50.0, size=N_LEVELS)

    q = 0.018 * np.exp(-z_km / 2.5)
    q = q * (1.0 + rng.normal(0.0, 0.05, size=N_LEVELS))
    q = np.clip(q, 1e-6, None)

    u = 2.0 + 0.3 * z_km + rng.normal(0.0, 1.0, size=N_LEVELS)
    v = -1.0 + 0.1 * z_km + rng.normal(0.0, 1.0, size=N_LEVELS)

    # Diagnostic condensate — will be recomputed each step. Start at zero.
    qc = np.zeros(N_LEVELS)
    qi = np.zeros(N_LEVELS)

    return dict(T=T, q=q, qc=qc, qi=qi, u=u, v=v, p=p, z=z)


def large_scale_forcing(rng: np.random.Generator,
                        regime: Regime,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Radiative heating [K/s] and moisture forcing [kg/kg/s]."""
    Q_rad = -1.5e-5 * np.exp(-((Z_CENTERS / 1000.0 - 8.0) ** 2) / 50.0)
    Q_rad = Q_rad + rng.normal(0.0, 1e-6, size=N_LEVELS)

    Q_q = -1.0e-8 * np.ones(N_LEVELS)
    if regime in (Regime.DEEP, Regime.ORGANIZED):
        Q_q = Q_q - 2.0e-8 * np.exp(-((Z_CENTERS / 1000.0 - 4.0) ** 2) / 20.0)
    Q_q = Q_q + rng.normal(0.0, 1e-9, size=N_LEVELS)

    return Q_rad, Q_q


def surface_fluxes(rng: np.random.Generator,
                   regime: Regime,
                   ) -> Tuple[float, float]:
    """Sensible and latent heat fluxes [W/m^2]."""
    if regime == Regime.SUPPRESSED:
        shf = rng.uniform(5.0, 20.0)
        lhf = rng.uniform(20.0, 60.0)
    elif regime == Regime.SHALLOW:
        shf = rng.uniform(15.0, 40.0)
        lhf = rng.uniform(80.0, 140.0)
    elif regime == Regime.DEEP:
        shf = rng.uniform(10.0, 30.0)
        lhf = rng.uniform(100.0, 180.0)
    else:  # ORGANIZED
        shf = rng.uniform(5.0, 25.0)
        lhf = rng.uniform(120.0, 200.0)
    return float(shf), float(lhf)


# ---------------------------------------------------------------------------
# Single mass-flux timestep
# ---------------------------------------------------------------------------
def mass_flux_step(state: Dict[str, np.ndarray],
                   regime: Regime,
                   shf: float,
                   lhf: float,
                   Q_rad: np.ndarray,
                   Q_q: np.ndarray,
                   rng: np.random.Generator,
                   ) -> Tuple[Dict[str, np.ndarray], float, float, np.ndarray, np.ndarray]:
    """Compute tendencies for one timestep.

    Returns:
        tendencies  dict with keys dT_dt, dq_dt, dqc_dt, dqi_dt, du_dt, dv_dt
        rain        surface rain rate [mm/day]
        snow        surface snow rate [mm/day]
        qc_diag     diagnostic cloud liquid water at this step [kg/kg]
        qi_diag     diagnostic cloud ice at this step [kg/kg]

    qc and qi are computed diagnostically (instantaneous equilibrium),
    not integrated. Their tendencies dqc_dt and dqi_dt are reported as
    the difference from the current state value, so the model has a
    well-defined target to learn.
    """
    T, q, qc_old, qi_old = state["T"], state["q"], state["qc"], state["qi"]
    u, v = state["u"], state["v"]
    z = state["z"]

    shape = regime_mass_flux_shape(regime)
    prof = REGIME_PROFILES[regime]

    # Base mass flux driven by latent heat flux (CAPE proxy).
    conv_trigger = np.tanh((lhf - 60.0) / 60.0)
    M_base = 0.05 * max(0.0, float(conv_trigger))
    M_base = M_base * (1.0 + 0.05 * rng.normal())
    M_base = max(M_base, 0.0)

    M = M_base * shape

    T_plume = T + 1.5 * shape
    q_plume = q * (1.0 + 0.3 * shape)

    dM_dz = np.gradient(M, z, edge_order=1)

    # Condensation source [kg/kg/s]
    # Magnitude calibrated so peak column rain ~ 10-15 mm/day.
    cond_rate = 4.0e-7 * M * shape

    # ---- Temperature tendency ----
    dT_subsidence = -1.0 * dM_dz * (T_plume - T)
    dT_latent = (L_V / C_P) * cond_rate
    dT_dt = dT_subsidence + dT_latent + Q_rad
    dT_dt = dT_dt + rng.normal(0.0, 1e-7, size=N_LEVELS)

    # ---- Moisture tendency ----
    dq_subsidence = -dM_dz * (q_plume - q)
    dq_cond = -cond_rate
    dq_dt = dq_subsidence + dq_cond + Q_q
    dq_dt = dq_dt + rng.normal(0.0, 1e-10, size=N_LEVELS)

    # ---- Diagnostic cloud liquid ----
    # At equilibrium: cond_rate = K_AUTO_EFF * qc * (1 + qc/QC_MAX)
    # Solve the quadratic for qc:
    #   qc = (-1 + sqrt(1 + 4*cond_rate/(K_AUTO_EFF*QC_MAX))) * QC_MAX/2
    a = cond_rate / K_AUTO_EFF
    qc_diag = (-1.0 + np.sqrt(1.0 + 4.0 * a / QC_MAX)) * QC_MAX / 2.0
    qc_diag = np.clip(qc_diag, 0.0, QC_MAX)

    # Autoconversion at diagnostic qc — this is the rain source
    auto = K_AUTO_EFF * qc_diag * (1.0 + qc_diag / QC_MAX)

    dqc_dt = (qc_diag - qc_old) / DT_SECONDS

    # ---- Diagnostic cloud ice ----
    ice_factor = (T < 273.0).astype(float)
    K_ice_form = 1.0e-7
    ice_source = K_ice_form * M * shape * ice_factor
    b = ice_source / K_ICE_AUTO_EFF
    qi_diag = (-1.0 + np.sqrt(1.0 + 4.0 * b / QI_MAX)) * QI_MAX / 2.0
    qi_diag = np.clip(qi_diag, 0.0, QI_MAX)

    ice_auto = K_ICE_AUTO_EFF * qi_diag * (1.0 + qi_diag / QI_MAX)
    dqi_dt = (qi_diag - qi_old) / DT_SECONDS

    # ---- Momentum ----
    eps = 1.0e-3
    du_dt = -eps * M * (u - u.mean())
    dv_dt = -eps * M * (v - v.mean())

    # ---- Precipitation [mm/day = kg/m^2/day] ----
    if regime == Regime.SUPPRESSED:
        rain = 0.0
        snow = 0.0
    else:
        col_rain = float(np.trapz(auto, z))
        col_snow = float(np.trapz(ice_auto, z))
        rain = prof.precip_scale * col_rain * 86400.0
        snow = prof.precip_scale * col_snow * 86400.0
        rain = max(0.0, rain * (1.0 + 0.02 * rng.normal()))
        snow = max(0.0, snow * (1.0 + 0.02 * rng.normal()))

    tendencies = dict(
        dT_dt=dT_dt, dq_dt=dq_dt, dqc_dt=dqc_dt,
        dqi_dt=dqi_dt, du_dt=du_dt, dv_dt=dv_dt,
    )
    return tendencies, rain, snow, qc_diag, qi_diag


# ---------------------------------------------------------------------------
# Advance state (Euler for T, q, u, v; DIAGNOSTIC for qc, qi)
# ---------------------------------------------------------------------------
def advance_state(state: Dict[str, np.ndarray],
                  tendencies: Dict[str, np.ndarray],
                  qc_diag: np.ndarray,
                  qi_diag: np.ndarray) -> Dict[str, np.ndarray]:
    """Advance the column by DT_SECONDS.

    T, q, u, v are integrated with Euler.
    qc, qi are set DIAGNOSTICALLY (instantaneous equilibrium).
    """
    new = dict(state)
    new["T"] = state["T"] + DT_SECONDS * tendencies["dT_dt"]
    new["q"] = np.clip(state["q"] + DT_SECONDS * tendencies["dq_dt"], 1e-8, None)
    new["u"] = state["u"] + DT_SECONDS * tendencies["du_dt"]
    new["v"] = state["v"] + DT_SECONDS * tendencies["dv_dt"]
    # Diagnostic condensate: NOT integrated
    new["qc"] = qc_diag
    new["qi"] = qi_diag
    return new


# ---------------------------------------------------------------------------
# Preconditioning
# ---------------------------------------------------------------------------
def apply_preconditioning(q: np.ndarray,
                          regime: Regime,
                          precond: float) -> np.ndarray:
    """Return moisture enhanced by accumulated precond in [0, 1]."""
    if regime == Regime.SUPPRESSED:
        return q.copy()
    z_km = Z_CENTERS / 1000.0
    if regime == Regime.SHALLOW:
        bump = np.exp(-((z_km - 1.0) ** 2) / 2.0)
    else:  # DEEP, ORGANIZED
        bump = np.exp(-((z_km - 2.0) ** 2) / 4.0)
    return q * (1.0 + 0.3 * precond * bump)
