"""Compute normalization statistics from the training set.

Reads data/synthetic/train.joblib, computes per-variable mean and std
for state, tendency, and precip, saves to data/synthetic/normalization.npz.

Run this once after generating the dataset:
    python -m data.compute_normalization

The normalizer is then loaded automatically by SyntheticColumnDataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.normalize import Normalizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data/synthetic/train.joblib",
                        help="Path to training data (.joblib).")
    parser.add_argument("--out", type=str,
                        default="data/synthetic/normalization.npz",
                        help="Output path for normalization stats.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"Loading {data_path}...")
    data = joblib.load(data_path)

    state = data["state"]
    tendency = data["tendency"]
    precip = data["precip"]

    print(f"  state shape    : {state.shape}")
    print(f"  tendency shape : {tendency.shape}")
    print(f"  precip shape   : {precip.shape}")

    print("Computing statistics...")
    norm = Normalizer.compute(state, tendency, precip)

    print(f"  state_mean shape    : {norm.state_mean.shape}")
    print(f"  state_mean [T, q]   : {norm.state_mean[0, 0]:.2f}, "
          f"{norm.state_mean[0, 1]:.6f}")
    print(f"  state_std  [T, q]   : {norm.state_std[0, 0]:.2f}, "
          f"{norm.state_std[0, 1]:.6f}")
    print(f"  tendency_mean[dT/dt]: {norm.tendency_mean[0, 0]:.3e}")
    print(f"  tendency_std [dT/dt]: {norm.tendency_std[0, 0]:.3e}")
    print(f"  precip_mean         : {norm.precip_mean}")
    print(f"  precip_std          : {norm.precip_std}")

    norm.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
