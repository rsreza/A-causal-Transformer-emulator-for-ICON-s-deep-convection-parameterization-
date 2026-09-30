"""Synthetic convection generator with planted regimes.

Runs on CPU in ~2-3 minutes for 50,000 sequences (12 timesteps, 30 levels).

Each sequence is a single atmospheric column evolving over 12 hours.
A regime is chosen at the start; the column state evolves according to the
simplified 1-D mass-flux model in data.convection_physics.

Output: data/synthetic/{train,val,test}.nc

Canonical NetCDF layout (dimensions: sample, time, level):
    state       (n_sample, T_PAST, N_LEVELS, N_INPUT_VARS)   float32
    tendency    (n_sample, T_PAST, N_LEVELS, N_OUTPUT_VARS)  float32
    precip      (n_sample, T_PAST, N_PRECIP_VARS)            float32
    metadata    (n_sample, N_META_VARS)                      float32
    regime      (n_sample,)                                  int8
    mass_flux   (n_sample, N_LEVELS)                         float32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.column_spec import (  # noqa: E402
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
    INPUT_VARS, OUTPUT_VARS, PRECIP_VARS, META_VARS,
    Z_INTERFACES, Z_CENTERS,
)
from src.regimes import (  # noqa: E402
    Regime, REGIME_NAMES, REGIME_FROM_NAME,
    regime_mass_flux_shape, regime_memory_length,
)
from src.utils import load_config, ensure_dir, set_seed  # noqa: E402
from data.convection_physics import (  # noqa: E402
    background_state, large_scale_forcing, surface_fluxes,
    mass_flux_step, advance_state, apply_preconditioning,
)

T_PAST = 12


def generate_sequence(rng: np.random.Generator,
                      regime: Regime,
                      t_past: int = T_PAST) -> dict:
    """Generate one sequence + metadata."""
    state = background_state(rng)
    Q_rad, Q_q = large_scale_forcing(rng, regime)
    shf, lhf = surface_fluxes(rng, regime)

    memory = regime_memory_length(regime, t_past)

    states = np.zeros((t_past, N_LEVELS, N_INPUT_VARS), dtype=np.float32)
    tendencies = np.zeros((t_past, N_LEVELS, N_OUTPUT_VARS), dtype=np.float32)
    precips = np.zeros((t_past, N_PRECIP_VARS), dtype=np.float32)

    precond = 0.0
    target = 1.0 if regime != Regime.SUPPRESSED else 0.0

    for t in range(t_past):
        # Update preconditioning scalar
        precond = precond * (1.0 - 1.0 / memory) + target / memory

        # Apply regime-dependent moisture buildup to what the model sees
        q_eff = apply_preconditioning(state["q"], regime, precond)
        state_eff = dict(state)
        state_eff["q"] = q_eff

        # Compute tendencies + diagnostic condensate
        tend, rain, snow, qc_diag, qi_diag = mass_flux_step(
            state_eff, regime, shf, lhf, Q_rad, Q_q, rng,
        )

        # Fill input tensor (order MUST match INPUT_VARS)
        states[t, :, 0] = state["T"]
        states[t, :, 1] = q_eff
        states[t, :, 2] = state["qc"]
        states[t, :, 3] = state["qi"]
        states[t, :, 4] = state["u"]
        states[t, :, 5] = state["v"]
        states[t, :, 6] = state["p"]
        states[t, :, 7] = state["z"]
        states[t, :, 8] = Q_rad
        states[t, :, 9] = Q_q

        # Fill output tensor (order MUST match OUTPUT_VARS)
        tendencies[t, :, 0] = tend["dT_dt"]
        tendencies[t, :, 1] = tend["dq_dt"]
        tendencies[t, :, 2] = tend["dqc_dt"]
        tendencies[t, :, 3] = tend["dqi_dt"]
        tendencies[t, :, 4] = tend["du_dt"]
        tendencies[t, :, 5] = tend["dv_dt"]

        precips[t, 0] = rain
        precips[t, 1] = snow

        # Advance: T, q, u, v prognostically; qc, qi diagnostically
        state = advance_state(state, tend, qc_diag, qi_diag)

    # Metadata
    lat = rng.uniform(-30.0, 30.0)
    lon = rng.uniform(-180.0, 180.0)
    time_day = rng.uniform(0.0, 365.0)
    if regime == Regime.SUPPRESSED:
        sst = rng.uniform(295.0, 299.0)
    else:
        sst = rng.uniform(298.0, 303.0)
    ps = 101300.0 + rng.normal(0.0, 200.0)
    metadata = np.array([lat, lon, time_day, sst, ps], dtype=np.float32)

    mass_flux = regime_mass_flux_shape(regime).astype(np.float32)

    return dict(
        state=states, tendency=tendencies, precip=precips,
        metadata=metadata, regime=int(regime), mass_flux=mass_flux,
    )


def generate_dataset(n_samples: int,
                     regimes: list,
                     regime_probs: list,
                     seed: int,
                     log_every: int = 2000,
                     ) -> dict:
    """Generate n_samples sequences with planted regimes."""
    assert len(regimes) == len(regime_probs)
    assert abs(sum(regime_probs) - 1.0) < 1e-6

    regime_enum = [REGIME_FROM_NAME[name] for name in regimes]
    probs = np.asarray(regime_probs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    all_state = np.zeros((n_samples, T_PAST, N_LEVELS, N_INPUT_VARS),
                         dtype=np.float32)
    all_tend = np.zeros((n_samples, T_PAST, N_LEVELS, N_OUTPUT_VARS),
                        dtype=np.float32)
    all_precip = np.zeros((n_samples, T_PAST, N_PRECIP_VARS), dtype=np.float32)
    all_meta = np.zeros((n_samples, len(META_VARS)), dtype=np.float32)
    all_regime = np.zeros((n_samples,), dtype=np.int8)
    all_mass_flux = np.zeros((n_samples, N_LEVELS), dtype=np.float32)

    regime_choices = rng.choice(len(regime_enum), size=n_samples, p=probs)

    t0 = time.time()
    for i in tqdm(range(n_samples), desc="generating", ncols=80):
        reg = regime_enum[int(regime_choices[i])]
        seq = generate_sequence(rng, reg)
        all_state[i] = seq["state"]
        all_tend[i] = seq["tendency"]
        all_precip[i] = seq["precip"]
        all_meta[i] = seq["metadata"]
        all_regime[i] = seq["regime"]
        all_mass_flux[i] = seq["mass_flux"]

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            tqdm.write(f"  [{i+1:>6d}/{n_samples}]  "
                       f"{rate:6.0f} samples/s   ETA {remaining:5.1f}s")

    return dict(
        state=all_state, tendency=all_tend, precip=all_precip,
        metadata=all_meta, regime=all_regime, mass_flux=all_mass_flux,
    )


def build_dataset_dict(data: dict, n_sample: int) -> xr.Dataset:
    """Build an xarray Dataset in canonical layout."""
    ds = xr.Dataset(
        data_vars={
            "state":    (("sample", "time", "level", "input_var"), data["state"]),
            "tendency": (("sample", "time", "level", "output_var"), data["tendency"]),
            "precip":   (("sample", "time", "precip_var"), data["precip"]),
            "metadata": (("sample", "meta_var"), data["metadata"]),
            "regime":   (("sample",), data["regime"].astype(np.int8)),
            "mass_flux": (("sample", "level"), data["mass_flux"]),
        },
        coords={
            "sample": np.arange(n_sample),
            "time": np.arange(T_PAST),
            "level": np.arange(N_LEVELS),
            "input_var": INPUT_VARS,
            "output_var": OUTPUT_VARS,
            "precip_var": PRECIP_VARS,
            "meta_var": META_VARS,
            "z_centers": ("level", Z_CENTERS.astype(np.float32)),
            "z_interfaces": (("level_iface",), Z_INTERFACES.astype(np.float32)),
            "regime_names": (("regime_id",),
                             [REGIME_NAMES[r] for r in Regime]),
        },
        attrs={
            "description": "Synthetic convection dataset (planted regimes)",
            "source": "data.generate_synthetic_convection",
            "t_past": T_PAST,
            "n_levels": N_LEVELS,
            "n_input_vars": N_INPUT_VARS,
            "n_output_vars": N_OUTPUT_VARS,
        },
    )
    return ds


def split_and_write(data: dict,
                    n_train: int, n_val: int, n_test: int,
                    out_dir: Path) -> None:
    """Split into train/val/test and write NetCDF files."""
    n_total = data["state"].shape[0]
    assert n_train + n_val + n_test <= n_total, "split exceeds total"

    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (0, n_train),
        "val":   (n_train, n_train + n_val),
        "test":  (n_train + n_val, n_train + n_val + n_test),
    }

    for name, (i0, i1) in splits.items():
        n_split = i1 - i0
        sub = {k: v[i0:i1] for k, v in data.items()}
        ds = build_dataset_dict(sub, n_split)
        path = out_dir / f"{name}.nc"
        ds.to_netcdf(path, engine="netcdf4")
        size_mb = path.stat().st_size / 1e6
        print(f"  wrote {path}   ({n_split} samples, {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic convection dataset with planted regimes."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    syn = cfg["data"]["synthetic"]

    n_samples = args.n_samples or syn["n_samples"]
    if args.n_samples is not None:
        total_cfg = syn["n_train"] + syn["n_val"] + syn["n_test"]
        scale = args.n_samples / total_cfg
        n_train = int(syn["n_train"] * scale)
        n_val = int(syn["n_val"] * scale)
        n_test = args.n_samples - n_train - n_val
    else:
        n_train = syn["n_train"]
        n_val = syn["n_val"]
        n_test = syn["n_test"]

    regimes = syn["regimes"]
    regime_probs = syn["regime_probs"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    out_dir = Path(args.out_dir or syn["output_dir"])

    set_seed(seed)
    ensure_dir(out_dir)

    print("=" * 72)
    print("Synthetic convection dataset generation")
    print("=" * 72)
    print(f"  n_samples     : {n_samples}")
    print(f"  n_train/val/test: {n_train}/{n_val}/{n_test}")
    print(f"  regimes       : {regimes}")
    print(f"  regime_probs  : {regime_probs}")
    print(f"  seed          : {seed}")
    print(f"  out_dir       : {out_dir}")
    print()

    data = generate_dataset(
        n_samples=n_samples, regimes=regimes,
        regime_probs=regime_probs, seed=seed,
    )

    print()
    print("Splitting and writing NetCDF...")
    split_and_write(data, n_train, n_val, n_test, out_dir)

    profiles = {REGIME_NAMES[r]: regime_mass_flux_shape(r) for r in Regime}
    np.savez(out_dir / "regime_profiles.npz", **profiles)

    print()
    print("Done.")
    print(f"  {out_dir}/train.nc")
    print(f"  {out_dir}/val.nc")
    print(f"  {out_dir}/test.nc")
    print(f"  {out_dir}/regime_profiles.npz")


if __name__ == "__main__":
    main()
