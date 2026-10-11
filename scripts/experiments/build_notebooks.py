"""Builds the three Kaggle notebooks for the comparison experiments:

  notebooks/train_compare.ipynb     train one baseline or ablation (set EXPERIMENT)
  notebooks/evaluate_compare.ipynb  score every trained model on the paper's 623 test tiles
  notebooks/massachusetts.ipynb     v4 on Massachusetts Roads without retraining, then trained on it

The notebooks clone the repository and import scripts/experiments/exp_data.py and exp_torch.py,
so the code that runs on Kaggle is the code in the repository.

Usage (repo root): python scripts/experiments/build_notebooks.py
"""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_URL = "https://github.com/Dilraj07/Satellite-Imagery-based-Road-Network-Extraction.git"

SETUP = r'''# Code from the repository
import os, sys, glob, json, subprocess, shutil, zipfile
REPO_URL = "%s"
REPO_REF = "main"          # a branch, or a commit hash to pin the exact code
REPO = "/kaggle/working/repo"
if not os.path.isdir(REPO):
    if subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO]).returncode:
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)          # REPO_REF is a commit
        subprocess.run(["git", "-C", REPO, "checkout", "-q", REPO_REF], check=True)
V4 = os.path.join(REPO, "models", "best_model_v4.pth")   # the paper's model, included in the repository
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%%h %%s"], capture_output=True, text=True).stdout)
try:
    import skimage
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-image"], check=True)
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments"))
import numpy as np, pandas as pd, torch
import exp_data as D
import exp_torch as T
DEVICE = torch.device("cuda")
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))''' % REPO_URL

TABLE = r'''def table(rows):
    """rows: [(name, params, summary dict)] -> printed table and LaTeX lines."""
    keys = [("iou", "IoU"), ("f1", "F1"), ("relaxed_f1", "Rel. F1"), ("cldice", "clDice"), ("canopy_recall", "Canopy R")]
    df = pd.DataFrame([{"model": n, "params (M)": round(p / 1e6, 2) if p else None,
                        **{h: round(100 * s[k], 1) for k, h in keys},
                        "parts": s["median_components_per_tile"]} for n, p, s in rows])
    print(df.to_string(index=False))
    latex = [f"{n} & {p / 1e6:.2f} & " + " & ".join(f"{100 * s[k]:.1f}" for k, _ in keys) + " \\\\" for n, p, s in rows]
    return df, latex'''


def md(text):
    return ("markdown", text)


def code(text):
    return ("code", text)


TRAIN = [
    md("""# Train one comparison model on the paper's split

Each run trains **one** model with the v4 training code: the same 5,323 training tiles, the same
280 selection tiles for choosing the best epoch, and the same 623 test tiles left untouched.

| `EXPERIMENT` | What it trains | Recipe |
|---|---|---|
| `dlinknet34` | D-LinkNet34 with ImageNet ResNet-34 encoder (as in the original paper) | full (160 epochs) |
| `unet` | U-Net, random initialisation | full |
| `lraspp_mbv3` | LR-ASPP MobileNetV3-Large with ImageNet backbone | full |
| `abl_full` | Our network, unchanged (reference row of the ablation) | short (60 epochs) |
| `abl_no_cldice` | Our network without clDice | short |
| `abl_no_gates` | Our network without attention gates | short |
| `abl_no_strip_shift` | Our network with plain 3x3 convolutions and no channel shift | short |
| `abl_no_transformer` | Our network without transformer layers | short |
| `dlinknet34_scratch` | Optional: D-LinkNet34 without pretraining | full |

**How to run**
1. Kaggle, New Notebook, add the dataset `balraj98/deepglobe-road-extraction-dataset`.
2. File, Import Notebook, choose this file. Settings: GPU T4, Internet on.
3. Set `EXPERIMENT` below. For the first run of a new model, set `SMOKE_TEST = True` and run it once
   (a few minutes); if it ends with `DONE`, set it back to `False`.
4. Save Version, Save & Run All. Make one Kaggle notebook per experiment (name it after `EXPERIMENT`).
5. If it ends with `RESUME NEEDED` (Kaggle's 12 hour limit): Add Input, Your Work, this notebook's
   output, and run a new version. It continues from the last epoch.
6. Download `<EXPERIMENT>/best_<EXPERIMENT>.pth`. After 20 training steps the log prints the
   expected time per epoch and in total."""),
    code(r'''# Settings
EXPERIMENT = "dlinknet34"
SMOKE_TEST = False       # True: 2 short epochs on a few tiles, to check that everything runs
TIME_LIMIT_H = 11.0      # stop and save before Kaggle's 12 hour limit
NUM_WORKERS = 4'''),
    code(SETUP),
    code(r'''# Split (identical to train_v4; checks the 623 test tiles against the paper)
spec = T.EXPERIMENTS[EXPERIMENT]
assert spec.get("dataset", "deepglobe") == "deepglobe", "use notebooks/massachusetts.ipynb for this experiment"
DG = D.find_deepglobe()
split = D.deepglobe_split(DG)
train_items, select_items = D.deepglobe_items(DG, split["train"]), D.deepglobe_items(DG, split["select"])
print({k: len(v) for k, v in split.items()}, "| test sha1", D.sha1(split["test"]))
OUT = f"/kaggle/working/{EXPERIMENT}" + ("_smoke" if SMOKE_TEST else "")   # a smoke run is never resumed
os.makedirs(OUT, exist_ok=True)
json.dump({k: D.sha1(v) for k, v in split.items()}, open(f"{OUT}/split_sha1.json", "w"), indent=1)'''),
    code(r'''# Train
status = T.train_experiment(EXPERIMENT, train_items, select_items, OUT, test_sha1=D.TEST_SHA1,
                            time_limit_h=TIME_LIMIT_H, num_workers=NUM_WORKERS, smoke_test=SMOKE_TEST)'''),
    code(r'''# Curves
import matplotlib.pyplot as plt
h = pd.read_csv(f"{OUT}/history.csv")
fig, a = plt.subplots(1, 2, figsize=(10, 3.5))
a[0].plot(h.epoch, h.train_loss); a[0].set_xlabel("epoch"); a[0].set_ylabel("training loss")
a[1].plot(h.epoch, 100 * h.val_iou, label="model"); a[1].plot(h.epoch, 100 * h.ema_iou, label="EMA")
a[1].set_xlabel("epoch"); a[1].set_ylabel("selection IoU (%)"); a[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150)
print(status, sorted(os.listdir(OUT)))'''),
]

EVAL = [
    md("""# Score the comparison models on the paper's 623 test tiles

**Inputs:** the DeepGlobe dataset, plus one Kaggle dataset or model holding the checkpoints:
every `best_<experiment>.pth` produced by `train_compare.ipynb` (v4 comes with the repository).
Settings: GPU T4, Internet on. Run All (about 10 minutes per model).

The notebook first re-measures v4 and **stops with an error** if it does not reproduce the paper
(TTA IoU 61.07%, single pass IoU 59.81%), so every other row is scored exactly like the paper.
Output: `/kaggle/working/compare_results.zip` (JSON, per tile CSV, LaTeX table rows)."""),
    code(r'''# Settings
PAPER_V4 = {"tta_raw": 0.61073, "single_raw": 0.59807}   # results.json of the paper
TOL = 0.002
VARIANTS = {"single": ["raw"], "tta": ["raw", "full"]}
OUT = "/kaggle/working/compare_results"'''),
    code(SETUP),
    code(r'''# Test tiles and canopy threshold (as in the paper)
os.makedirs(OUT, exist_ok=True)
DG = D.find_deepglobe()
test_items = D.deepglobe_items(DG, D.deepglobe_split(DG)["test"])
VEG_T = D.otsu_vegetation_threshold(test_items)
print(len(test_items), "test tiles | vegetation threshold", VEG_T, "(paper: 0.1191)")
assert abs(VEG_T - 0.119140625) < 1e-6'''),
    code(r'''# Checkpoints: v4 plus one file per experiment (the copy with the best selection score)
found = {}
for path in sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True)):
    base = os.path.basename(path)
    if not base.startswith("best_") or base == "best_model_v4.pth" or "train_state" in base:
        continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "experiment" not in ck:
        continue
    name = ck["experiment"]
    if ck["spec"].get("dataset", "deepglobe") != "deepglobe":
        continue
    assert ck.get("test_ids_sha1") == D.TEST_SHA1, f"{path}: trained on a different split"
    if name not in found or ck["val_iou"] > found[name][1]["val_iou"]:
        found[name] = (path, {k: v for k, v in ck.items() if k != "model_state_dict"})
found["v4"] = (V4, {})
order = ["v4"] + [e for e in T.EXPERIMENTS if e in found]
for n in order:
    print(f"{n:20s} {found[n][0]}  epoch {found[n][1].get('epoch', 158)}")'''),
    code(r'''# Evaluate
RECORDS, PARAMS = {}, {}
for n in order:
    model, meta = T.load_any_checkpoint(found[n][0], DEVICE)
    PARAMS[n] = T.n_params(model)
    print(f"{n}: {PARAMS[n]:,} parameters")
    RECORDS[n] = T.evaluate(model, test_items, VARIANTS, VEG_T, DEVICE)
    del model; torch.cuda.empty_cache()
    if n == "v4":
        got = {v: D.summarize(D.count_matrix(RECORDS[n], v))["iou"] for v in PAPER_V4}
        print("v4 reproduction:", got, "paper:", PAPER_V4)
        assert all(abs(got[v] - PAPER_V4[v]) <= TOL for v in PAPER_V4), "v4 does not reproduce the paper; stop here"
        print("V4 REPRODUCED")'''),
    code(TABLE),
    code(r'''# Results
SUMMARY = {n: {f"{s}_{v}": D.summary(RECORDS[n], f"{s}_{v}") for s, vl in VARIANTS.items() for v in vl} for n in order}
TESTS = {}
for n in order:
    ref = "v4" if not n.startswith("abl_") else ("abl_full" if n != "abl_full" else None)
    if n == "v4" or ref is None or ref not in RECORDS:
        continue
    for var in ("tta_raw", "tta_full"):
        TESTS[f"{ref}_vs_{n}_{var}"] = D.paired_test(RECORDS[ref], RECORDS[n], var)
for var in ("tta_raw", "tta_full", "single_raw"):
    print(f"\n=== {var} ===")
    _, latex = table([(n, PARAMS[n], SUMMARY[n][var]) for n in order])
    open(f"{OUT}/table_{var}.tex", "w").write("\n".join(latex) + "\n")
print("\nPaired Wilcoxon tests on per tile IoU (positive mean_diff: the first model is better):")
for k, t in TESTS.items():
    print(f"  {k:45s} mean diff {100 * t['mean_diff']:+.2f} points | better on {100 * t['frac_tiles_better']:.0f}% of tiles | p = {t['wilcoxon_p']}")'''),
    code(r'''# Save
rows = []
for n in order:
    for var in SUMMARY[n]:
        for tile, m in D.per_tile(RECORDS[n], var).items():
            rows.append({"model": n, "variant": var, "tile": tile, **m})
pd.DataFrame(rows).to_csv(f"{OUT}/per_tile_metrics.csv", index=False)
json.dump({"summary": SUMMARY, "paired_tests": TESTS, "params": PARAMS, "vegetation_threshold": VEG_T,
           "checkpoints": {n: {"path": found[n][0], **{k: v for k, v in found[n][1].items()
                                                         if isinstance(v, (int, float, str, dict, list))}}
                           for n in order}},
          open(f"{OUT}/compare_results.json", "w"), indent=1, default=str)
shutil.make_archive("/kaggle/working/compare_results", "zip", OUT)
print("saved /kaggle/working/compare_results.zip")'''),
]

MASS = [
    md("""# Massachusetts Roads: a second dataset

**Inputs:** the dataset `balraj98/massachusetts-roads-dataset` (the v4 model comes with the repository).
Settings: GPU T4, Internet on. Save Version, Save & Run All.

* **Part A** (about 10 minutes): the v4 model, trained only on DeepGlobe, applied to the 49 official
  test tiles without retraining. Massachusetts has 1 m pixels and v4 was trained on 0.5 m pixels, so
  it is run twice: on the tiles as they are, and on tiles enlarged two times to 0.5 m pixels.
* **Part B** (several hours): our network trained on the 1,108 official training tiles with the
  v4 recipe, best epoch chosen on the 14 official validation tiles, then scored on the 49 test tiles.
  If it ends with `RESUME NEEDED`, attach this notebook's output and run a new version.

The official split is used by most published Massachusetts results, so part B can be compared
with them directly. Output: `/kaggle/working/massachusetts_results.zip`."""),
    code(r'''# Settings
RUN_ZERO_SHOT = True
TRAIN_EXPERIMENT = "mass_scratch"   # or "mass_finetune" (start from the v4 weights)
RUN_TRAINING = True
SMOKE_TEST = False
TIME_LIMIT_H = 11.0
GSD_SCALE = 0.5   # 0.5 m / 1 m: gap bridging distances halve in pixels
VARIANTS = {"single": ["raw"], "tta": ["raw", "full"]}
OUT = "/kaggle/working/massachusetts_results"'''),
    code(SETUP),
    code(TABLE),
    code(r'''# Dataset
os.makedirs(OUT, exist_ok=True)
ROOT = D.find_massachusetts()
split = D.massachusetts_split(ROOT)
print(ROOT, {k: len(v) for k, v in split.items()}, "(official: train 1108, select/val 14, test 49)")
print("image size", D.read_rgb(split["test"][0][1]).shape, "| label pixels", int(D.read_mask(split["test"][0][2]).sum()))
assert split["train"] and split["test"] and split["select"]
VEG_T = D.otsu_vegetation_threshold(split["test"])
RESULTS = {"vegetation_threshold": VEG_T, "n": {k: len(v) for k, v in split.items()}}'''),
    code(r'''# Part A: v4 without retraining
if RUN_ZERO_SHOT:
    model, _ = T.load_any_checkpoint(V4, DEVICE)
    rows = []
    for scale in (1.0, 2.0):
        rec = T.evaluate(model, split["test"], VARIANTS, VEG_T, DEVICE, scale=scale, gsd_scale=GSD_SCALE)
        RESULTS[f"zero_shot_scale{scale:g}"] = {v: D.summary(rec, v) for v in ("single_raw", "tta_raw", "tta_full")}
        rows += [(f"v4, scale {scale:g}, {v}", T.n_params(model), RESULTS[f"zero_shot_scale{scale:g}"][v])
                 for v in ("tta_raw", "tta_full")]
    del model; torch.cuda.empty_cache()
    table(rows)
else:
    print("part A skipped")'''),
    code(r'''# Part B: train on Massachusetts
TRAIN_OUT = f"/kaggle/working/{TRAIN_EXPERIMENT}" + ("_smoke" if SMOKE_TEST else "")
status = None
if RUN_TRAINING:
    init = V4 if T.EXPERIMENTS[TRAIN_EXPERIMENT].get("init") == "v4" else None
    status = T.train_experiment(TRAIN_EXPERIMENT, split["train"], split["select"], TRAIN_OUT,
                                init_checkpoint=init, time_limit_h=TIME_LIMIT_H, smoke_test=SMOKE_TEST)'''),
    code(r'''# Part B: test
best = f"{TRAIN_OUT}/best_{TRAIN_EXPERIMENT}.pth"
if status == "DONE" and os.path.exists(best):
    model, meta = T.load_any_checkpoint(best, DEVICE)
    rec = T.evaluate(model, split["test"], VARIANTS, VEG_T, DEVICE, gsd_scale=GSD_SCALE)
    RESULTS[TRAIN_EXPERIMENT] = {v: D.summary(rec, v) for v in ("single_raw", "tta_raw", "tta_full")}
    RESULTS[TRAIN_EXPERIMENT]["epoch"] = meta.get("epoch")
    table([(f"{TRAIN_EXPERIMENT}, {v}", T.n_params(model), RESULTS[TRAIN_EXPERIMENT][v])
           for v in ("single_raw", "tta_raw", "tta_full")])
    pd.DataFrame([{"tile": t, **m} for t, m in D.per_tile(rec, "tta_raw").items()]).to_csv(
        f"{OUT}/per_tile_{TRAIN_EXPERIMENT}.csv", index=False)
    shutil.copy(best, OUT)
else:
    print("part B test skipped:", status)'''),
    code(r'''# Save
json.dump(RESULTS, open(f"{OUT}/massachusetts_results.json", "w"), indent=1, default=str)
shutil.make_archive("/kaggle/working/massachusetts_results", "zip", OUT)
print("saved /kaggle/working/massachusetts_results.zip")'''),
]


def write(cells, name):
    nb = {"cells": [], "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            compile(src, name, "exec")
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = os.path.join(REPO, "notebooks", name)
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, f"({sum(k == 'code' for k, _ in cells)} code cells, all compile)")


if __name__ == "__main__":
    write(TRAIN, "train_compare.ipynb")
    write(EVAL, "evaluate_compare.ipynb")
    write(MASS, "massachusetts.ipynb")
