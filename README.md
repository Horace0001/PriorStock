# PriorStock Reproduction

This repository reproduces the two reported full-test results with Python 3.12.4, PyTorch 2.5.1+cu121, CUDA 12.1, and seed 7.

| Market | Factor selection | Noise selection | Full ACC | Full MCC |
|---|---|---|---:|---:|
| CMIN-US | fixed epoch 2 | fixed epoch 10 | 0.5532 | 0.0956 |
| CMIN-CN | validation significant | fixed epoch 1 | 0.5773 | 0.1494 |

Install the pinned environment and Git LFS assets, then run:

```powershell
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
git lfs pull
.\reproduce.ps1
```

The first two stages use samples with absolute return at least 0.01. The Factor stage uses BCE plus a 0.10 soft-MCC term. The final stage keeps the Base model frozen, trains on the full training split, applies soft targets and boundary calibration to low-return samples, and applies weak soft metrics to significant and 0.005 to 0.01 absolute-return samples.

Final metrics are written to `runs/<market>/noise/test_significant_metrics.json` and `runs/<market>/noise/test_full_metrics.json`.

The complete Base training entry is:

```powershell
.\train_base.ps1 -Market cmin_us
.\train_base.ps1 -Market cmin_cn
```

Both Base scripts use validation `MCC + 0.5 balanced_accuracy + 0.25 macro_f1` for checkpoint selection, matching the source runs.

The CMIN-US Base full-test result is ACC 0.5328 and MCC 0.0684. The CMIN-CN Base full-test result is ACC 0.5499 and MCC 0.0934. The bundled checkpoints reproduce these values. Fresh CUDA training can vary slightly across devices because kernel-level numerical differences can change a close validation checkpoint ordering.

The tracked assets require approximately 22 GB of local storage and Git LFS capacity. No API access is required.
