"""Evaluation script for trained convection emulators.

Loads a checkpoint from training, runs inference on the test set, and
computes:
  - Per-variable offline metrics (RMSE, bias, correlation) in physical units
  - Regime-stratified metrics (shallow, deep, organized, suppressed)
  - Conservation residuals (mass and energy imbalance in physical units)
  - Visualizations: profile plots and precipitation scatter

Usage
-----
Evaluate the model in configs/default.yaml:
    python -m src.evaluate --config configs/default.yaml

Override checkpoint:
    python -m src.evaluate --config configs/default.yaml \\
        --checkpoint results/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .column_spec import (
    N_LEVELS, N_OUTPUT_VARS, N_PRECIP_VARS,
    OUTPUT_VARS, PRECIP_VARS, Z_CENTERS, Z_INTERFACES, C_P,
    IDX_DQ, IDX_DQC, IDX_DQI, IDX_DT,
)
from .dataset import build_dataloader
from .model_factory import build_model
from .normalize import Normalizer
from .regimes import Regime, REGIME_NAMES
from .utils import load_config, select_device, setup_logging, ensure_dir


def _safe_corr(p, t):
    if p.size == 0:
        return float("nan")
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def compute_per_variable_metrics(pred, target, var_names):
    metrics = {}
    for v, name in enumerate(var_names):
        p = pred[..., v].reshape(-1)
        t = target[..., v].reshape(-1)
        if p.size == 0:
            metrics[name] = {"rmse": float("nan"),
                             "bias": float("nan"),
                             "corr": float("nan")}
            continue
        err = p - t
        rmse = float(np.sqrt(np.mean(err ** 2))) if np.isfinite(err).all() else float("nan")
        bias = float(np.mean(err)) if np.isfinite(err).all() else float("nan")
        corr = _safe_corr(p, t)
        metrics[name] = {"rmse": rmse, "bias": bias, "corr": corr}
    return metrics


def aggregate_metrics(per_regime):
    if not per_regime:
        return {}
    var_names = list(next(iter(per_regime.values())).keys())
    out = {}
    for var in var_names:
        rmse_vals = [per_regime[r][var]["rmse"] for r in per_regime]
        bias_vals = [per_regime[r][var]["bias"] for r in per_regime]
        corr_vals = [per_regime[r][var]["corr"] for r in per_regime]
        out[var] = {
            "rmse": float(np.nanmean(rmse_vals)),
            "bias": float(np.nanmean(bias_vals)),
            "corr": float(np.nanmean(corr_vals)),
        }
    return out


def compute_conservation_residuals(pred_tendency, state, target_precip, dz):
    dq = pred_tendency[..., IDX_DQ]
    dqc = pred_tendency[..., IDX_DQC]
    dqi = pred_tendency[..., IDX_DQI]
    dq_total = dq + dqc + dqi
    col_dq = (dq_total * dz).sum(axis=-1)
    precip_kg = target_precip.sum(axis=-1) / 86400.0
    mass_residual = col_dq + precip_kg
    mass_imbalance = float(np.nanmean(np.abs(mass_residual)))
    dT = pred_tendency[..., IDX_DT]
    col_dT = (C_P * dT * dz).sum(axis=-1)
    energy_imbalance = float(np.nanmean(np.abs(col_dT)))
    return {
        "mass_imbalance_kg_m2_s": mass_imbalance,
        "energy_imbalance_W_m2": energy_imbalance,
    }


@torch.no_grad()
def run_inference(model, loader, device, normalizer, max_batches=-1):
    model.eval()
    pred_t_list, pred_p_list = [], []
    tgt_t_list, tgt_p_list = [], []
    state_list, regime_list = [], []

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        state_norm = batch["state_norm"].to(device)
        out_norm = model(state_norm)
        pred_t = (out_norm["tendency"].cpu().numpy()
                  * normalizer.tendency_std[None, None]
                  + normalizer.tendency_mean[None, None])
        pred_p = (out_norm["precip"].cpu().numpy()
                  * normalizer.precip_std[None]
                  + normalizer.precip_mean[None])
        pred_t_list.append(pred_t)
        pred_p_list.append(pred_p)
        state_list.append(batch["state"].numpy())
        tgt_t_list.append(batch["tendency"].numpy())
        tgt_p_list.append(batch["precip"].numpy())
        regime_list.append(batch["regime"].numpy())

    return {
        "pred_tendency": np.concatenate(pred_t_list, axis=0),
        "pred_precip": np.concatenate(pred_p_list, axis=0),
        "state": np.concatenate(state_list, axis=0),
        "target_tendency": np.concatenate(tgt_t_list, axis=0),
        "target_precip": np.concatenate(tgt_p_list, axis=0),
        "regime": np.concatenate(regime_list, axis=0),
    }


def plot_tendency_profiles(data, out_dir, var_idx=0):
    var_name = OUTPUT_VARS[var_idx]
    pred = data["pred_tendency"]
    tgt = data["target_tendency"]
    regime = data["regime"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, r in zip(axes, Regime):
        mask = (regime == int(r))
        if mask.sum() == 0:
            ax.set_title(f"{REGIME_NAMES[r]} (no samples)")
            continue
        p_mean = pred[mask, :, :, var_idx].mean(axis=(0, 1))
        t_mean = tgt[mask, :, :, var_idx].mean(axis=(0, 1))
        ax.plot(p_mean, Z_CENTERS / 1000.0, label="pred", lw=2)
        ax.plot(t_mean, Z_CENTERS / 1000.0, label="true", lw=2, ls="--")
        ax.set_title(f"{REGIME_NAMES[r]}  (N={int(mask.sum())})")
        ax.set_xlabel(f"{var_name}")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("height [km]")
    plt.tight_layout()
    out = out_dir / f"profile_{var_name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {out}")


def plot_precip_scatter(data, out_dir):
    pred = data["pred_precip"][..., 0].flatten()
    tgt = data["target_precip"][..., 0].flatten()
    regime = np.repeat(data["regime"], data["pred_precip"].shape[1])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:gray"]
    for r in Regime:
        mask = (regime == int(r))
        if mask.sum() == 0:
            continue
        ax.scatter(tgt[mask], pred[mask], s=2, alpha=0.3,
                   color=colors[int(r)], label=REGIME_NAMES[r])
    lims = [0, max(tgt.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("true rain [mm/day]")
    ax.set_ylabel("predicted rain [mm/day]")
    ax.set_title("Precipitation: predicted vs true")
    ax.legend(markerscale=4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "precip_scatter.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {out}")


def evaluate(cfg, checkpoint_path, max_batches=-1):
    device = select_device(cfg["training"].get("device", "auto"))
    logger = setup_logging()
    logger.info(f"Evaluating on device: {device}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    out_dir = Path(cfg["data"]["synthetic"]["output_dir"])
    normalizer = Normalizer.load(out_dir / "normalization.npz")

    test_loader = build_dataloader(cfg, split="test")
    logger.info(f"Test batches: {len(test_loader)}")

    logger.info("Running inference...")
    data = run_inference(model, test_loader, device, normalizer,
                          max_batches=max_batches)

    logger.info("Computing metrics...")
    regime_metrics = {}
    for r in Regime:
        mask = (data["regime"] == int(r))
        n = int(mask.sum())
        if n == 0:
            continue
        regime_metrics[REGIME_NAMES[r]] = {
            "n_samples": n,
            "tendency": compute_per_variable_metrics(
                data["pred_tendency"][mask],
                data["target_tendency"][mask],
                OUTPUT_VARS,
            ),
            "precip": compute_per_variable_metrics(
                data["pred_precip"][mask],
                data["target_precip"][mask],
                PRECIP_VARS,
            ),
        }

    tendency_metrics = aggregate_metrics(
        {name: rm["tendency"] for name, rm in regime_metrics.items()}
    )
    precip_metrics = aggregate_metrics(
        {name: rm["precip"] for name, rm in regime_metrics.items()}
    )

    dz_full = np.diff(Z_INTERFACES).astype(np.float32)
    residuals = compute_conservation_residuals(
        data["pred_tendency"], data["state"], data["target_precip"], dz_full,
    )

    all_metrics = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("best_val", float("nan"))),
        "overall_tendency": tendency_metrics,
        "overall_precip": precip_metrics,
        "regime_stratified": regime_metrics,
        "conservation_residuals": residuals,
    }
    metrics_dir = ensure_dir(cfg["eval"]["metrics_dir"])
    model_name = cfg["model"]["name"]
    metrics_path = metrics_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    print()
    print("=" * 72)
    print(f"Evaluation summary — model: {model_name}")
    print("=" * 72)
    print()
    print("Overall tendency metrics (nanmean over regimes):")
    for name, m in tendency_metrics.items():
        print(f"  {name:8s}  RMSE={m['rmse']:.3e}  "
              f"bias={m['bias']:+.3e}  corr={m['corr']:+.4f}")
    print()
    print("Overall precipitation metrics (nanmean over regimes):")
    for name, m in precip_metrics.items():
        print(f"  {name:10s}  RMSE={m['rmse']:.3f}  "
              f"bias={m['bias']:+.3f}  corr={m['corr']:+.4f}")
    print()
    print("Conservation residuals:")
    print(f"  mass   : {residuals['mass_imbalance_kg_m2_s']:.3e} kg/m²/s")
    print(f"  energy : {residuals['energy_imbalance_W_m2']:.3f} W/m²")
    print()
    print("Regime-stratified metrics (correlation of dT_dt):")
    for regime_name, rm in regime_metrics.items():
        corr = rm["tendency"]["dT_dt"]["corr"]
        print(f"  {regime_name:11s}  N={rm['n_samples']:5d}  "
              f"corr={corr:+.4f}")

    fig_dir = ensure_dir(cfg["eval"]["figures_dir"])
    print()
    print(f"Saving figures to {fig_dir}/")
    plot_tendency_profiles(data, fig_dir, var_idx=0)
    plot_precip_scatter(data, fig_dir)

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best.pt")
    parser.add_argument("--max-batches", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, max_batches=args.max_batches)
    os._exit(0)


if __name__ == "__main__":
    main()
