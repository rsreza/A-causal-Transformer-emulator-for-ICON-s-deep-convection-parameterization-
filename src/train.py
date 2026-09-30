"""Training loop for ICON convection emulators.

Key design:
  - Model sees NORMALIZED inputs and produces NORMALIZED outputs.
  - Data loss (tendency + precip) is computed in NORMALIZED space.
  - Physics losses (mass, energy, positivity) are computed in PHYSICAL
    space, after denormalizing the model's output.
  - This gives stable gradient magnitudes across all loss terms.

Usage
-----
CPU (quick):
    python -m src.train --config configs/default.yaml

GPU (single):
    python -m src.train --config configs/real.yaml

DDP (multi-GPU):
    torchrun --nproc_per_node=8 -m src.train --config configs/real.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .column_spec import Z_INTERFACES
from .dataset import build_dataloader
from .model_factory import build_model, count_parameters
from .normalize import Normalizer
from .physics_loss import (
    LossConfig, mass_conservation_loss,
    energy_conservation_loss, positivity_loss,
)
from .utils import (
    load_config, set_seed, select_device, setup_logging, ensure_dir,
    is_distributed, get_rank, get_world_size, get_local_rank,
)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def init_distributed() -> bool:
    if not is_distributed():
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Composite loss with normalization handling
# ---------------------------------------------------------------------------
class NormalizedPhysicsLoss(nn.Module):
    """Wraps the physics loss to handle normalization correctly.

    pred_norm : normalized model output
    target    : dict with normalized AND raw targets
    state_raw : physical state (for positivity loss)

    Computes:
        L_data   in normalized space  (tendency + precip)
        L_mass   in physical space    (column water budget)
        L_energy in physical space    (column enthalpy budget)
        L_pos    in physical space    (condensate positivity)
    """

    def __init__(self, config: LossConfig, dz: torch.Tensor,
                 normalizer: Normalizer, device: torch.device):
        super().__init__()
        self.config = config
        self.register_buffer("dz", dz.float())
        tn = normalizer.to_torch(device=device)
        self.tendency_mean = tn.tendency_mean
        self.tendency_std = tn.tendency_std
        self.precip_mean = tn.precip_mean
        self.precip_std = tn.precip_std

    def forward(self, pred_norm, target, state_raw):
        cfg = self.config

        # ----- Data loss in normalized space -----
        l_data = F.huber_loss(pred_norm["tendency"],
                              target["tendency_norm"],
                              delta=cfg.huber_delta)
        l_data = l_data + F.huber_loss(pred_norm["precip"],
                                        target["precip_norm"],
                                        delta=cfg.huber_delta)

        # ----- Denormalize predictions for physics losses -----
        pred_tend_phys = (pred_norm["tendency"] * self.tendency_std
                          + self.tendency_mean)

        # ----- Physics losses in physical space -----
        l_mass = mass_conservation_loss(
            pred_tend_phys, state_raw, target["precip"], self.dz,
        )
        l_energy = energy_conservation_loss(
            pred_tend_phys, state_raw, self.dz,
        )
        l_pos = positivity_loss(pred_tend_phys, state_raw)

        total = (cfg.lambda_data * l_data
                 + cfg.lambda_mass * l_mass
                 + cfg.lambda_energy * l_energy
                 + cfg.lambda_pos * l_pos)

        return {
            "total": total,
            "data": l_data.detach(),
            "mass": l_mass.detach(),
            "energy": l_energy.detach(),
            "positivity": l_pos.detach(),
        }


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    scaler, grad_clip, amp_enabled, max_batches=-1,
                    rank=0, logger=None):
    model.train()
    totals = {"total": 0.0, "data": 0.0, "mass": 0.0,
              "energy": 0.0, "positivity": 0.0}
    n_batches = 0
    n_skipped = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        state_norm = batch["state_norm"].to(device, non_blocking=True)
        state_raw = batch["state"].to(device, non_blocking=True)
        target = {
            "tendency_norm": batch["tendency_norm"].to(device, non_blocking=True),
            "precip_norm": batch["precip_norm"].to(device, non_blocking=True),
            "precip": batch["precip"].to(device, non_blocking=True),
        }

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred_norm = model(state_norm)
                losses = loss_fn(pred_norm, target, state_raw)
                loss = losses["total"]
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_norm = model(state_norm)
            losses = loss_fn(pred_norm, target, state_raw)
            loss = losses["total"]
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in losses.items():
            totals[k] += float(v.detach().cpu())
        n_batches += 1

    if n_skipped > 0 and rank == 0 and logger is not None:
        logger.warning(f"Skipped {n_skipped} non-finite batches during training")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device, max_batches=-1):
    model.eval()
    totals = {"total": 0.0, "data": 0.0, "mass": 0.0,
              "energy": 0.0, "positivity": 0.0}
    n_batches = 0
    n_skipped = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        state_norm = batch["state_norm"].to(device, non_blocking=True)
        state_raw = batch["state"].to(device, non_blocking=True)
        target = {
            "tendency_norm": batch["tendency_norm"].to(device, non_blocking=True),
            "precip_norm": batch["precip_norm"].to(device, non_blocking=True),
            "precip": batch["precip"].to(device, non_blocking=True),
        }

        pred_norm = model(state_norm)
        losses = loss_fn(pred_norm, target, state_raw)

        if not torch.isfinite(losses["total"]):
            n_skipped += 1
            continue

        for k, v in losses.items():
            totals[k] += float(v.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        # All batches were non-finite — return inf so early stopping triggers
        return {k: float("inf") for k in totals}

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: dict, args: argparse.Namespace) -> dict:
    ddp_active = init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    logger = setup_logging(rank=rank)
    set_seed(cfg["seed"] + rank)

    if ddp_active:
        device = torch.device(
            f"cuda:{get_local_rank()}"
            if torch.cuda.is_available() else "cpu"
        )
    else:
        device = select_device(cfg["training"].get("device", "auto"))

    if rank == 0:
        logger.info(f"Training on device: {device}")
        logger.info(f"DDP active: {ddp_active}, world_size: {world_size}")

    # Data
    train_loader = build_dataloader(cfg, "train",
                                     rank=rank, world_size=world_size)
    val_loader = build_dataloader(cfg, "val",
                                   rank=rank, world_size=world_size)

    if rank == 0:
        logger.info(f"Train batches: {len(train_loader)}, "
                    f"Val batches: {len(val_loader)}")

    # Normalizer
    out_dir = Path(cfg["data"]["synthetic"]["output_dir"])
    normalizer = Normalizer.load(out_dir / "normalization.npz")
    if rank == 0:
        logger.info(f"Loaded normalizer from {out_dir / 'normalization.npz'}")

    # Model
    model = build_model(cfg).to(device)
    if rank == 0:
        logger.info(f"Model: {cfg['model']['name']}  "
                    f"params: {count_parameters(model):,}")

    if ddp_active:
        model = DDP(
            model,
            device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
        )

    # Loss
    dz = torch.as_tensor(Z_INTERFACES, dtype=torch.float32)
    loss_cfg = LossConfig.from_config(cfg)
    loss_fn = NormalizedPhysicsLoss(loss_cfg, dz, normalizer, device).to(device)

    # Optimizer
    train_cfg = cfg["training"]
    optimizer = AdamW(model.parameters(),
                       lr=train_cfg["lr"],
                       weight_decay=train_cfg["weight_decay"])

    # LR scheduler — only meaningful for epochs > 2
    if train_cfg["epochs"] > 2:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"],
            eta_min=train_cfg["lr"] * 0.01,
        )
    else:
        scheduler = None

    # Mixed precision
    amp_enabled = (train_cfg.get("mixed_precision", False)
                   and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    if rank == 0:
        logger.info(f"Mixed precision: {amp_enabled}")

    # Checkpointing
    ckpt_dir = ensure_dir(train_cfg["checkpoint_dir"])
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    history = {"train": [], "val": [], "lr": [], "epoch_time": []}

    max_batches = getattr(args, "max_batches", -1)
    if max_batches > 0 and rank == 0:
        logger.info(f"Smoke mode: max {max_batches} batches per epoch")

    # ---- Main epoch loop ----
    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        if ddp_active and hasattr(train_loader, "sampler"):
            try:
                train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler, train_cfg["grad_clip"], amp_enabled,
            max_batches=max_batches, rank=rank, logger=logger,
        )
        val_losses = validate(
            model, val_loader, loss_fn, device, max_batches=max_batches,
        )

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        history["train"].append(train_losses)
        history["val"].append(val_losses)
        history["lr"].append(lr)
        history["epoch_time"].append(epoch_time)

        if rank == 0:
            logger.info(
                f"epoch {epoch+1:>3}/{train_cfg['epochs']}  "
                f"lr={lr:.2e}  time={epoch_time:5.1f}s  |  "
                f"train: {train_losses['total']:.4f}  "
                f"(d={train_losses['data']:.4f} "
                f"m={train_losses['mass']:.4f} "
                f"e={train_losses['energy']:.4f} "
                f"p={train_losses['positivity']:.4f})  |  "
                f"val: {val_losses['total']:.4f}"
            )

        # Best-model checkpointing (skip if val is non-finite)
        val_finite = torch.isfinite(torch.tensor(val_losses["total"])).item()
        if rank == 0 and val_finite and val_losses["total"] < best_val - 1e-6:
            best_val = val_losses["total"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch + 1,
                "model_state": (model.module.state_dict() if ddp_active
                                else model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "best_val": best_val,
                "config": cfg,
            }
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"  -> new best val loss {best_val:.4f}, "
                        f"saved to {ckpt_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_cfg["early_stopping_patience"]:
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if rank == 0:
        logger.info(f"Training done. Best val loss: {best_val:.4f} "
                    f"at epoch {best_epoch}")
        hist_path = Path(train_cfg["checkpoint_dir"]) / "history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"History saved to {hist_path}")

    cleanup_distributed()
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a convection emulator."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    train(cfg, args)

    # Force-exit to skip C-library teardown, which can segfault on some
    # systems due to HDF5/OpenMP conflicts between torch and system libs.
    # All results are already saved at this point.
    os._exit(0)


if __name__ == "__main__":
    main()
