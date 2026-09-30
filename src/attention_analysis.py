"""Attention analysis for the causal Transformer.

Extracts per-regime attention weights from a trained Transformer and
quantifies the effective memory length of each convective regime.

Physical question: do deep/organized convection have longer memory than
shallow/suppressed? Our synthetic generator plants these memories, so
we can check whether the model recovered them without supervision.

Outputs
-------
- results/figures/attention_by_regime.png  : mean attention vs lag per regime
- results/figures/attention_layer_<i>.png  : per-layer attention per regime
- results/metrics/attention_analysis.json  : numerical summary

Usage
-----
    python -m src.attention_analysis --config configs/default.yaml
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

from .dataset import build_dataloader
from .model_factory import build_model
from .regimes import Regime, REGIME_NAMES, regime_memory_length
from .utils import load_config, select_device, setup_logging, ensure_dir


# ---------------------------------------------------------------------------
# Attention collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_attention(model, loader, device, max_batches=-1):
    """Run the model and collect attention weights + regime labels.

    Returns
    -------
    attention : (N, n_layers, n_heads, T, T)
    regime    : (N,)
    """
    model.eval()

    attn_list = []
    regime_list = []

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        state_norm = batch["state_norm"].to(device)
        out = model(state_norm)

        if out["attention"] is None:
            raise ValueError(
                "Model does not return attention weights. "
                "Set model.name: transformer in the config."
            )

        # out["attention"] is a list of (B, n_heads, T, T) per layer
        stack = torch.stack(out["attention"], dim=0)         # (L, B, H, T, T)
        stack = stack.permute(1, 0, 2, 3, 4).cpu().numpy()   # (B, L, H, T, T)

        attn_list.append(stack)
        regime_list.append(batch["regime"].numpy())

    attention = np.concatenate(attn_list, axis=0)
    regime = np.concatenate(regime_list, axis=0)

    return attention, regime


# ---------------------------------------------------------------------------
# Lag-based reduction
# ---------------------------------------------------------------------------
def attention_by_lag(attn, regime, r):
    """Average attention as a function of lag time for a given regime.

    attn   : (N, n_layers, n_heads, T, T)
    regime : (N,)
    r      : int regime id

    Returns
    -------
    mean_by_lag : (T,)
    std_by_lag  : (T,)
    """
    mask = (regime == r)
    if mask.sum() == 0:
        return np.zeros(attn.shape[-1]), np.zeros(attn.shape[-1])

    a = attn[mask]                          # (N_r, L, H, T, T)
    T = a.shape[-1]
    by_lag = np.zeros(T)
    std_lag = np.zeros(T)
    for k in range(T):
        idx = np.arange(k, T)
        vals = a[:, :, :, idx, idx - k]     # entries at (t, t-k)
        by_lag[k] = vals.mean()
        std_lag[k] = vals.std()
    return by_lag, std_lag


def effective_memory_length(by_lag, threshold=0.5):
    """Lag at which cumulative attention exceeds threshold."""
    total = by_lag.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(by_lag) / total
    return int(np.searchsorted(cum, threshold))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_attention_by_regime(attn, regime, out_path):
    """One panel per regime, showing mean attention vs lag."""
    T = attn.shape[-1]
    lags = np.arange(T)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:gray"]
    summary = {}

    for ax, r in zip(axes, Regime):
        by_lag, std_lag = attention_by_lag(attn, regime, int(r))
        mem_50 = effective_memory_length(by_lag, 0.5)
        mem_90 = effective_memory_length(by_lag, 0.9)
        n_samples = int((regime == int(r)).sum())

        if by_lag[0] > 0:
            by_lag_norm = by_lag / by_lag[0]
        else:
            by_lag_norm = by_lag

        ax.plot(lags, by_lag_norm, "o-",
                color=colors[int(r)], lw=2, markersize=6)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.axhline(0.1, color="gray", ls=":", alpha=0.3)
        ax.set_title(f"{REGIME_NAMES[r]}  (N={n_samples})")
        ax.set_xlabel("lag [timesteps]")
        ax.set_xticks(lags)
        ax.grid(alpha=0.3)
        ax.legend([f"mem50={mem_50}, mem90={mem_90}"], loc="upper right")

        summary[REGIME_NAMES[r]] = {
            "n_samples": n_samples,
            "mean_attention_by_lag": by_lag.tolist(),
            "std_attention_by_lag": std_lag.tolist(),
            "memory_lag_50pct": mem_50,
            "memory_lag_90pct": mem_90,
        }

    axes[0].set_ylabel("normalized attention (lag 0 = 1)")
    fig.suptitle(
        "Causal Transformer attention vs lag time, per regime",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")

    return summary


def plot_attention_heatmaps(attn, regime, out_dir):
    """One heatmap per layer: mean attention (heads and samples averaged)."""
    L = attn.shape[1]

    for layer in range(L):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, r in zip(axes, Regime):
            mask = (regime == int(r))
            if mask.sum() == 0:
                ax.set_title(f"{REGIME_NAMES[r]} (no samples)")
                continue
            m = attn[mask, layer].mean(axis=(0, 1))     # (T, T)
            im = ax.imshow(m, cmap="viridis", aspect="auto",
                            vmin=0, vmax=m.max() if m.max() > 0 else 1)
            ax.set_title(f"{REGIME_NAMES[r]}  layer {layer}")
            ax.set_xlabel("key (past)")
            ax.set_ylabel("query (current)")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        out = out_dir / f"attention_layer_{layer}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze(cfg, checkpoint_path, max_batches=-1):
    device = select_device(cfg["training"].get("device", "auto"))
    logger = setup_logging()
    logger.info(f"Analyzing on device: {device}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    test_loader = build_dataloader(cfg, split="test")
    logger.info(f"Test batches: {len(test_loader)}")

    logger.info("Collecting attention weights...")
    attention, regime = collect_attention(
        model, test_loader, device, max_batches=max_batches,
    )
    logger.info(f"Collected attention shape: {attention.shape}")

    fig_dir = ensure_dir(cfg["eval"]["figures_dir"])
    metrics_dir = ensure_dir(cfg["eval"]["metrics_dir"])

    logger.info("Plotting attention vs regime...")
    summary = plot_attention_by_regime(
        attention, regime, fig_dir / "attention_by_regime.png",
    )

    logger.info("Plotting per-layer heatmaps...")
    plot_attention_heatmaps(attention, regime, fig_dir)

    # Compare to planted memory
    for name, s in summary.items():
        r = [r for r in Regime if REGIME_NAMES[r] == name][0]
        planted = regime_memory_length(r, t_past=attention.shape[-1])
        s["planted_memory_timesteps"] = planted
        s["memory_50pct_minus_planted"] = s["memory_lag_50pct"] - planted

    # Print summary
    print()
    print("=" * 72)
    print("Attention memory analysis")
    print("=" * 72)
    print(f"{'regime':12s}  {'N':>6s}  {'planted':>8s}  "
          f"{'mem50':>6s}  {'mem90':>6s}  {'diff':>6s}")
    for name, s in summary.items():
        print(f"{name:12s}  {s['n_samples']:>6d}  "
              f"{s['planted_memory_timesteps']:>8d}  "
              f"{s['memory_lag_50pct']:>6d}  "
              f"{s['memory_lag_90pct']:>6d}  "
              f"{s['memory_50pct_minus_planted']:>+6d}")

    metrics_path = metrics_dir / "attention_analysis.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Metrics saved to {metrics_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best.pt")
    parser.add_argument("--max-batches", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    analyze(cfg, args.checkpoint, max_batches=args.max_batches)
    os._exit(0)


if __name__ == "__main__":
    main()
