# PriorStock

PriorStock combines grouped OHLCV technical representations with attention over five text-derived market factors. Training follows Base, Factor, and noise-aware calibration stages for CMIN-US and CMIN-CN.

## Repository Layout

- `configs/` contains the six market and stage configurations.
- `data/cmin_us/` and `data/cmin_cn/` contain price frames, sample indices, OHLCV-124 features, factor responses, and factor embedding caches.
- `priorstock/` contains data loading, grouped technical modeling, factor attention, calibration losses, metrics, and training utilities.
- `scripts/base.py`, `scripts/factor.py`, and `scripts/noise_aware.py` implement the three training stages.
- `weights/<market>/` contains Base, Factor, and noise-aware weights for fast one-command reproduction.
- `train.py` and `train.ps1` run every stage from the beginning.
- `reproduce.py` and `reproduce.ps1` evaluate the bundled final weights.

## Environment

```powershell
git lfs pull
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

## Full Training

```powershell
.\train.ps1
```

Use `python train.py --market cmin_us` or `python train.py --market cmin_cn` to train one market.

## Fast Reproduction

The bundled weights are provided to reproduce the final evaluations without retraining.

```powershell
.\reproduce.ps1
```

Evaluation files are written under `runs/<market>/quick_reproduction/`. Full-training files are written under `runs/<market>/`.
