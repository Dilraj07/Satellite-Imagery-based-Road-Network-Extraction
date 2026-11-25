# Baseline Training Outputs

Organized per-experiment outputs downloaded from Kaggle (`notebooks/train_compare.ipynb`).
Every run uses the exact paper split (`5,323` train / `280` selection / `623` test; test SHA-1 `2811fca6c9e44e6cdf713455a422502e4e5e4431`).

| Folder | Model | Params (M) | Epochs Run | Best Selection IoU | Status |
|---|---|---:|---:|---:|---|
| `lraspp_mbv3/` | LR-ASPP MobileNetV3-Large (scratch) | 3.22 | 160 / 160 | **58.01%** (ep 148, raw) | Complete (`DONE`) |
| `deeplabv3_mbv3/` | DeepLabV3 MobileNetV3-Large (scratch) | 11.02 | 80 (early stop) | **51.91%** (ep 37, EMA) | Complete (`DONE`) |
| `unet/` | U-Net (scratch) | 31.04 | 55 (NaN ep 44; ep 0-43 clean) | **56.24%** (ep 40, EMA) | Partial backup *(re-run recommended with `0a08832`)* |
| `dlinknet34_scratch/` | D-LinkNet34 (scratch) | 31.10 | Pending | Pending | Drop downloaded checkpoint & logs here |

> Note: Checkpoints (`*.pth`) are gitignored via root `.gitignore` (`*.pth`), while `history.csv`, `summary.json`, `split_sha1.json`, and `training_curves.png` are lightweight records.
