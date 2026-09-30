"""Dataset loaders for synthetic and real ICON convection data.

Two classes with identical __getitem__ signatures:

  SyntheticColumnDataset  - reads .joblib files (fast, robust, CPU-friendly)
  StreamingColumnDataset  - lazy chunked reads via xarray/NetCDF (large files, DDP)

Both return dicts with keys: state, tendency, precip, regime.

Why .joblib for synthetic:
  - NetCDF/HDF5 conflicts with PyTorch's C runtime on some Linux systems
    (segfaults during file read).
  - .npz (compressed or uncompressed) triggers CRC errors on systems with
    limited RAM (< 8 GB), because the write buffer competes with the
    in-memory dataset arrays.
  - joblib uses memory-mapped I/O and is the standard ML persistence format.

Real ICON/ClimSim data still uses NetCDF via StreamingColumnDataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .column_spec import (
    N_LEVELS, N_INPUT_VARS, N_OUTPUT_VARS, N_PRECIP_VARS,
)


T_PAST_DEFAULT = 12


class SyntheticColumnDataset(Dataset):
    """In-memory dataset reading .joblib files.

    Loads the full file into RAM. Fast, no HDF5 dependency.
    Suitable for synthetic data (~1 GB for 50,000 samples).
    """

    def __init__(self,
                 path: str | Path,
                 split: str = "train",
                 t_past: int = T_PAST_DEFAULT,
                 dtype: torch.dtype = torch.float32,
                 ):
        self.path = Path(path)
        self.split = split
        self.t_past = t_past
        self.dtype = dtype

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        # Load all arrays into memory via joblib (memory-mapped, robust)
        import joblib
        data = joblib.load(self.path)

        # Validate shapes
        state = data["state"]
        tendency = data["tendency"]
        precip = data["precip"]

        assert state.shape[1] == t_past, (
            f"Expected t_past={t_past}, got {state.shape[1]}"
        )
        assert state.shape[2] == N_LEVELS
        assert state.shape[3] == N_INPUT_VARS
        assert tendency.shape[3] == N_OUTPUT_VARS
        assert precip.shape[2] == N_PRECIP_VARS

        # Materialize as numpy arrays
        self.state = np.ascontiguousarray(state, dtype=np.float32)
        self.tendency = np.ascontiguousarray(tendency, dtype=np.float32)
        self.precip = np.ascontiguousarray(precip, dtype=np.float32)
        self.regime = np.asarray(data["regime"], dtype=np.int64)
        self.metadata = np.asarray(data["metadata"], dtype=np.float32)
        self.n_samples = self.state.shape[0]

        # Regime names if present
        if "regime_names" in data:
            self.regime_names = [str(x) for x in data["regime_names"]]
        else:
            self.regime_names = []

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "state":    torch.as_tensor(self.state[idx], dtype=self.dtype),
            "tendency": torch.as_tensor(self.tendency[idx], dtype=self.dtype),
            "precip":   torch.as_tensor(self.precip[idx], dtype=self.dtype),
            "regime":   torch.as_tensor(self.regime[idx], dtype=torch.long),
        }


class StreamingColumnDataset(Dataset):
    """Lazy dataset for large NetCDF files (real ICON/ClimSim).

    Uses xarray chunked reads so we never load the full file into RAM.
    With DDP, each rank reads a shard: samples [rank::world_size].

    NOTE: This class requires netCDF4 or h5netcdf/h5py. On systems where
    HDF5 conflicts with PyTorch, use SyntheticColumnDataset with .joblib
    instead, or run on a cluster where the conflict does not occur.
    """

    def __init__(self,
                 path: str | Path,
                 split: str = "train",
                 t_past: int = T_PAST_DEFAULT,
                 chunk_size: int = 10000,
                 rank: int = 0,
                 world_size: int = 1,
                 dtype: torch.dtype = torch.float32,
                 ):
        self.path = Path(path)
        self.split = split
        self.t_past = t_past
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        # Lazy import — only load xarray if this class is actually used
        import xarray as xr

        try:
            self._ds = xr.open_dataset(
                self.path,
                engine="h5netcdf",
                chunks={"sample": chunk_size},
            )
        except Exception:
            self._ds = xr.open_dataset(
                self.path,
                engine="netcdf4",
                chunks={"sample": chunk_size},
            )

        assert self._ds.state.shape[1] == t_past
        assert self._ds.state.shape[2] == N_LEVELS
        assert self._ds.state.shape[3] == N_INPUT_VARS

        self.n_total = self._ds.sizes["sample"]
        self.indices = np.arange(self.rank, self.n_total, self.world_size)
        self.n_samples = len(self.indices)

        if "regime_names" in self._ds:
            self.regime_names = [str(x) for x in self._ds.regime_names.values]
        else:
            self.regime_names = []

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        global_idx = int(self.indices[idx])
        sample = self._ds.isel(sample=global_idx)

        state = np.asarray(sample.state.values, dtype=np.float32)
        tendency = np.asarray(sample.tendency.values, dtype=np.float32)
        precip = np.asarray(sample.precip.values, dtype=np.float32)

        if "regime" in sample:
            regime_val = int(sample.regime.values)
        else:
            regime_val = -1

        return {
            "state":    torch.as_tensor(state, dtype=self.dtype),
            "tendency": torch.as_tensor(tendency, dtype=self.dtype),
            "precip":   torch.as_tensor(precip, dtype=self.dtype),
            "regime":   torch.tensor(regime_val, dtype=torch.long),
        }

    def __del__(self):
        try:
            self._ds.close()
        except Exception:
            pass


def build_dataset(cfg: dict,
                  split: str,
                  rank: int = 0,
                  world_size: int = 1,
                  ) -> Dataset:
    """Factory: read config and return the right dataset for the split."""
    source = cfg["data"]["source"]

    if source == "synthetic":
        syn = cfg["data"]["synthetic"]
        out_dir = Path(syn["output_dir"])
        path = out_dir / f"{split}.joblib"
        return SyntheticColumnDataset(path=path, split=split)

    elif source == "real":
        real = cfg["data"]["real"]
        key = f"{split}_file"
        path = Path(real[key])
        return StreamingColumnDataset(
            path=path,
            split=split,
            chunk_size=real.get("chunk_size", 10000),
            rank=rank,
            world_size=world_size,
        )

    else:
        raise ValueError(f"Unknown data.source: {source}")


def build_dataloader(cfg: dict,
                     split: str,
                     rank: int = 0,
                     world_size: int = 1,
                     ) -> torch.utils.data.DataLoader:
    """Factory: return a DataLoader for the split."""
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = build_dataset(cfg, split, rank=rank, world_size=world_size)

    train_cfg = cfg["training"]
    is_train = (split == "train")

    sampler = None
    shuffle = is_train

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 0) if is_train else 0,
        pin_memory=False,
        drop_last=is_train,
        persistent_workers=(
            train_cfg.get("num_workers", 0) > 0 and is_train
        ),
    )
    return loader
