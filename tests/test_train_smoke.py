"""Automated smoke test for the training pipeline.

Runs 1 epoch on 3 batches of the real dataset (fast, ~5 seconds) and
asserts that:
  - The training loop completes without error
  - All loss terms are finite
  - A checkpoint is saved
  - Training history is saved

This is intentionally NOT a test of model quality — it verifies the
wiring works end-to-end. Full training happens outside pytest.
"""
import argparse
import json
from pathlib import Path

import pytest
import torch

from src.train import train
from src.utils import load_config


@pytest.fixture
def smoke_config():
    """Minimal config for the smoke test."""
    cfg = load_config("configs/default.yaml")
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 32
    cfg["training"]["num_workers"] = 0
    cfg["training"]["checkpoint_dir"] = "results/checkpoints_test"
    cfg["training"]["early_stopping_patience"] = 5
    return cfg


@pytest.fixture
def smoke_args():
    """Simulate argparse output for --max-batches 3."""
    return argparse.Namespace(
        config="configs/default.yaml",
        max_batches=3,
        epochs=1,
        seed=42,
    )


def test_smoke_train_runs(smoke_config, smoke_args):
    """Full training loop runs without errors and saves artifacts."""
    # Skip if data not present (e.g. fresh clone without generation)
    train_path = Path(smoke_config["data"]["synthetic"]["output_dir"]) / "train.joblib"
    if not train_path.exists():
        pytest.skip(f"Training data not found: {train_path}")

    history = train(smoke_config, smoke_args)

    # History should have 1 epoch (or fewer if early-stopped, unlikely)
    assert len(history["train"]) >= 1
    assert len(history["val"]) >= 1

    # All losses in the first epoch should be finite
    for term in ["total", "data", "mass", "energy", "positivity"]:
        v = history["train"][0][term]
        assert torch.isfinite(torch.tensor(v)), f"Train {term} not finite: {v}"
        v = history["val"][0][term]
        assert torch.isfinite(torch.tensor(v)), f"Val {term} not finite: {v}"

    # Total loss should be positive
    assert history["train"][0]["total"] > 0.0
    assert history["val"][0]["total"] > 0.0


def test_smoke_checkpoint_saved(smoke_config, smoke_args):
    """Checkpoint and history files are written."""
    train_path = Path(smoke_config["data"]["synthetic"]["output_dir"]) / "train.joblib"
    if not train_path.exists():
        pytest.skip(f"Training data not found: {train_path}")

    train(smoke_config, smoke_args)

    ckpt_dir = Path(smoke_config["training"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "best.pt"
    hist_path = ckpt_dir / "history.json"

    assert ckpt_path.exists(), f"Checkpoint not saved: {ckpt_path}"
    assert hist_path.exists(), f"History not saved: {hist_path}"

    # Checkpoint should have expected keys
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "epoch" in ckpt
    assert "model_state" in ckpt
    assert "best_val" in ckpt

    # History should be valid JSON
    with open(hist_path) as f:
        history = json.load(f)
    assert "train" in history
    assert "val" in history


def test_smoke_loss_reasonable(smoke_config, smoke_args):
    """After 1 epoch, the loss should be O(1) — not NaN, not 1e6."""
    train_path = Path(smoke_config["data"]["synthetic"]["output_dir"]) / "train.joblib"
    if not train_path.exists():
        pytest.skip(f"Training data not found: {train_path}")

    history = train(smoke_config, smoke_args)

    total = history["train"][0]["total"]
    # Loss should be within a sane range. With normalization, this is O(1).
    assert 0.001 < total < 100.0, f"Unreasonable total loss: {total}"

    # Physics terms should also be bounded
    assert history["train"][0]["mass"] < 100.0, \
        f"Mass loss too large: {history['train'][0]['mass']}"
    assert history["train"][0]["energy"] < 100.0, \
        f"Energy loss too large: {history['train'][0]['energy']}"
    assert history["train"][0]["positivity"] < 100.0, \
        f"Positivity loss too large: {history['train'][0]['positivity']}"
