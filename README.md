# ICON Convection Transformer

A causal Transformer emulator for ICON's deep convection parameterization.

**Two data paths, one codebase:**

- **Synthetic** (default) — runs on a single CPU in ~35 minutes
- **Real** — ClimSim / NARVAL / QUBICC / ICON dumps on multi-GPU clusters

Only the config changes between paths. The emulator, physics constraints, and
evaluation metrics are identical.

## Quick Start (CPU, synthetic)

```bash
git clone <repo-url>
cd "Causal Transformer Emulator  Convection Parameterization"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m data.generate_synthetic_convection
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.attention_analysis --config configs/default.yaml
```

## Real Data (GPU / DDP)

```bash
pip install -r requirements-gpu.txt

python -m data.download_climsim --out data/raw/climsim/
python -m data.prepare_real_data --input data/raw/climsim/ --out data/prepared/

torchrun --nproc_per_node=8 -m src.train --config configs/real.yaml
sbatch scripts/train_slurm.sh        # SLURM alternative
```

See `docs/real_data.md` for the full production guide.

## Layout

```
data/       synthetic generator, real-data prep, ClimSim downloader
src/        emulator, training, evaluation, attention analysis
configs/    default.yaml (synthetic), real.yaml (DDP)
app/        Streamlit regime + attention explorer
tests/      pytest suite
```

## The Physical Question

Traditional mass-flux schemes are **diagnostic** — no memory. Do convective
regimes have memory? The causal Transformer's attention weights make this
measurable: we can plot *which past timesteps* the model relies on, stratified
by regime. We expect deep/organized convection to attend further back than
shallow convection — a physical prediction, not a hyperparameter.

## Citation

```bibtex
@software{icon_convection_transformer,
  title  = {ICON Convection Transformer},
  year   = {2026},
  author = {Reza}
}
```

Built on:
- Heuer et al. (2024, 2025) — ICON ML parameterization
- Beucler et al. (2020) — physics-constrained ML
- Vaswani et al. (2017) — Transformer

## License

MIT (see LICENSE).
