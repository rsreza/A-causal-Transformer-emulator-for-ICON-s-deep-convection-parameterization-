"""Dataset loaders for synthetic and real ICON convection data.

SyntheticColumnDataset reads .joblib files and normalizes inputs/targets
using precomputed statistics. Returns both normalized and raw tensors,
so the training loop can use normalized values for the model and raw
values for the physics-constrained loss.

StreamingColumnDataset does not normalize (real data path — normalization
is applied in the training loop when the real pipeline is wired up).
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
from .normalize import Normalizer, EPS


T_PAST_DEFAULT = 12


class SyntheticColumnDataset(Dataset):
    """In-memory dataset reading .joblib files, with normalization.

    Each __getitem__ returns:
        state_norm     : normalized input  (T, L, V_in)
        tendency_norm  : normalized target (T, L, V_out)
        precip_norm    : normalized target (T, V_precip)
        state          : raw input         (T, L, V_in)
        tendency       : raw target        (T, L, V_out)
        precip         : raw target        (T, V_precip)
        regime         : int64

    The 'state'/'tendency'/'precip' keys (raw) are used by the physics loss.
    The '_norm' keys are used by the model and data loss.
    """

    def __init__(self,
                 path: str | Path,
                 split: str = "train",
                 t_past: int = T_PAST_DEFAULT,
                 normalizer_path: Optional[str | Path] = None,
                 dtype: torch.dtype = torch.float32,
                 ):
        self.path = Path(path)
        self.split = split
        self.t_past = t_past
        self.dtype = dtype

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        # Load all arrays into memory via joblib
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

        self.state = np.ascontiguousarray(state, dtype=np.float32)
        self.tendency = np.ascontiguousarray(tendency, dtype=np.float32)
        self.precip = np.ascontiguousarray(precip, dtype=np.float32)
        self.regime = np.asarray(data["regime"], dtype=np.int64)
        self.metadata = np.asarray(data["metadata"], dtype=np.float32)
        self.n_samples = self.state.shape[0]

        if "regime_names" in data:
            self.regime_names = [str(x) for x in data["regime_names"]]
        else:
            self.regime_names = []

        # Load or compute normalizer
        if normalizer_path is None:
            normalizer_path = self.path.parent / "normalization.npz"
        normalizer_path = Path(normalizer_path)

        if normalizer_path.exists():
            self.normalizer = Normalizer.load(normalizer_path)
        else:
            if split == "train":
                # Compute and cache for future splits
                self.normalizer = Normalizer.compute(
                    self.state, self.tendency, self.precip,
                )
                self.normalizer.save(normalizer_path)
            else:
                raise FileNotFoundError(
                    f"Normalizer not found at {normalizer_path}. "
                    f"Run `python -m data.compute_normalization` first, "
                    f"or load the train split first to compute it."
                )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        state = self.state[idx]              # (T, L, V_in)
        tendency = self.tendency[idx]        # (T, L, V_out)
        precip = self.precip[idx]            # (T, V_precip)

        n = self.normalizer
        state_norm = (state - n.state_mean) / (n.state_std + EPS)
        tendency_norm = (tendency - n.tendency_mean) / (n.tendency_std + EPS)
        precip_norm = (precip - n.precip_mean) / (n.precip_std + EPS)

        return {
            # Normalized (model input and data-loss target)
            "state_norm": torch.as_tensor(state_norm, dtype=self.dtype),
            "tendency_norm": torch.as_tensor(tendency_norm, dtype=self.dtype),
            "precip_norm": torch.as_tensor(precip_norm, dtype=self.dtype),
            # Raw (physics-loss inputs)
            "state": torch.as_tensor(state, dtype=self.dtype),
            "tendency": torch.as_tensor(tendency, dtype=self.dtype),
            "precip": torch.as_tensor(precip, dtype=self.dtype),
            # Label
            "regime": torch.as_tensor(self.regime[idx], dtype=torch.long),
        }


class StreamingColumnDataset(Dataset):
    """Lazy dataset for large NetCDF files (real ICON/ClimSim).

    Same interface as SyntheticColumnDataset, but streams from disk.
    Normalization is computed on the fly from the first chunk (approximate)
    if not provided.
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

        import xarray as xr

        try:
            self._ds = xr.open_dataset(self.path, engine="h5netcdf",
                                        chunks={"sample": chunk_size})
        except Exception:
            self._ds = xr.open_dataset(self.path, engine="netcdf4",
                                        chunks={"sample": chunk_size})

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

        regime_val = int(sample.regime.values) if "regime" in sample else -1

        # Real-data path: no normalization applied here (applied later when
        # the real pipeline is wired up). Return raw tensors as both keys.
        return {
            "state_norm": torch.as_tensor(state, dtype=self.dtype),
            "tendency_norm": torch.as_tensor(tendency, dtype=self.dtype),
            "precip_norm": torch.as_tensor(precip, dtype=self.dtype),
            "state": torch.as_tensor(state, dtype=self.dtype),
            "tendency": torch.as_tensor(tendency, dtype=self.dtype),
            "precip": torch.as_tensor(precip, dtype=self.dtype),
            "regime": torch.tensor(regime_val, dtype=torch.long),
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
        normalizer_path = out_dir / "normalization.npz"
        return SyntheticColumnDataset(
            path=path, split=split,
            normalizer_path=normalizer_path if normalizer_path.exists() else None,
        )
    elif source == "real":
        real = cfg["data"]["real"]
        key = f"{split}_file"
        path = Path(real[key])
        return StreamingColumnDataset(
            path=path, split=split,
            chunk_size=real.get("chunk_size", 10000),
            rank=rank, world_size=world_size,
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
            dataset, num_replicas=world_size, rank=rank, shuffle=is_train,
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
