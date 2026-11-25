"""Checks for exp_data.py that run without PyTorch or the datasets.

The metric and postprocessing functions in exp_data.py are copies of those in
scripts/paper_figures/kaggle_eval.py (which produced the paper's numbers); these tests run
both versions on the same synthetic masks and require identical results.

Run: python -m pytest scripts/experiments/test_exp_data.py -q
"""
import ast
import os
import random
import sys
import tempfile

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_data as D  # noqa: E402

KAGGLE_EVAL = os.path.join(D.REPO, "scripts", "paper_figures", "kaggle_eval.py")


def kaggle_eval_functions(*names):
    """The named top level functions of kaggle_eval.py, compiled without running the script."""
    tree = ast.parse(open(KAGGLE_EVAL, encoding="utf-8").read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {"np": np, "cv2": cv2, "skeletonize": skeletonize, "RHO": D.RHO,
          "hysteresis_threshold": D.hysteresis_threshold, "connect_canopy_gaps": D.connect_canopy_gaps}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), KAGGLE_EVAL, "exec"), ns)
    return [ns[n] for n in names]


def synthetic_tile(seed, size=256):
    rng = np.random.default_rng(seed)
    gt = np.zeros((size, size), np.uint8)
    for _ in range(4):
        p1, p2 = tuple(int(v) for v in rng.integers(0, size, 2)), tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.line(gt, p1, p2, 1, int(rng.integers(3, 9)))
    probs = np.clip(cv2.GaussianBlur(gt.astype(np.float32), (0, 0), 3) + rng.normal(0, 0.15, gt.shape), 0, 1)
    probs[rng.random(gt.shape) < 0.002] = 0.9          # speckle, so postprocessing has work to do
    rgb = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
    return gt.astype(bool), probs.astype(np.float32), rgb


@pytest.mark.parametrize("seed", range(5))
def test_metrics_match_kaggle_eval(seed):
    count_k, derive_k, exg_k = kaggle_eval_functions("count_metrics", "derive", "exg")
    G, probs, rgb = synthetic_tile(seed)
    V = D.exg(rgb) > 0.05
    assert np.array_equal(D.exg(rgb), exg_k(rgb))
    near_g = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 3) <= D.RHO
    sk_g = skeletonize(G)
    for P in (probs > 0.5, probs > 0.3, np.zeros_like(G)):
        a, b = D.count_metrics(P, G, V, near_g, sk_g), count_k(P, G, V, near_g, sk_g)
        assert a == b
        da, db = D.derive(a), derive_k(b)
        assert all((np.isnan(da[k]) and np.isnan(db[k])) or da[k] == db[k] for k in da)


@pytest.mark.parametrize("seed", range(3))
def test_postprocessing_matches_kaggle_eval(seed):
    (binarize_k,) = kaggle_eval_functions("binarize_all")
    _, probs, _ = synthetic_tile(seed)
    variants = ["raw", "hyst", "hyst_close", "full"]
    ours, theirs = D.binarize_all(probs, variants, gsd_scale=1.0), binarize_k(probs, variants)
    for v in variants:
        assert np.array_equal(ours[v], theirs[v]), v


def test_shuffle_matches_train_v4():
    ids = [str(i) for i in range(500)]
    a, b = list(ids), list(ids)
    random.seed(42); random.shuffle(a)        # how train_v4 shuffled
    random.Random(42).shuffle(b)              # how exp_data shuffles
    assert a == b


def test_split_and_tile_record_on_fake_dataset():
    with tempfile.TemporaryDirectory() as root:
        train = os.path.join(root, "deepglobe", "train")
        os.makedirs(train)
        for i in range(40):
            G, _, rgb = synthetic_tile(i, 64)
            cv2.imwrite(os.path.join(train, f"{1000 + i}_sat.jpg"), rgb)
            cv2.imwrite(os.path.join(train, f"{1000 + i}_mask.png"), G.astype(np.uint8) * 255)
        assert D.find_deepglobe(root) == train
        split = D.deepglobe_split(train, select_tiles=5, check_sha=False)
        assert (len(split["train"]), len(split["select"]), len(split["test"])) == (31, 5, 4)
        assert not set(split["test"]) & (set(split["train"]) | set(split["select"]))
        tid, img, lbl = D.deepglobe_items(train, split["test"])[0]
        probs = D.read_mask(lbl).astype(np.float16)            # a perfect prediction
        rec = D.tile_record((tid, lbl, img, {"tta": probs}, {"tta": ["raw", "full"]}, 0.1, 1.0))
        s = D.derive(rec["variants"]["tta_raw"])
        assert s["iou"] == 1.0 and s["f1"] == 1.0
        summary = D.summary([rec, rec], "tta_raw")
        assert summary["iou"] == 1.0 and summary["n_tiles"] == 2


def test_split_json_loads_correctly():
    split_path = os.path.join(D.REPO, "models", "v4_training", "split.json")
    assert os.path.isfile(split_path)
    import json
    with open(split_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    assert len(s["test_ids"]) == 623
    assert len(s["select_val_ids"]) == 280
    assert D.sha1(s["test_ids"]) == D.TEST_SHA1
    assert D.sha1(s["select_val_ids"]) == s["select_val_sha1"]
    assert not set(s["select_val_ids"]) & set(s["test_ids"])


def test_massachusetts_discovery():
    with tempfile.TemporaryDirectory() as root:
        for fmt in ("png", "tiff"):
            for sub in ("train", "train_labels", "val", "val_labels", "test", "test_labels"):
                os.makedirs(os.path.join(root, "mass", fmt, sub))
        tiff = os.path.join(root, "mass", "tiff")
        for sub, ext in (("train", ".tiff"), ("train_labels", ".tif"), ("test", ".tiff"), ("test_labels", ".tif"),
                         ("val", ".tiff"), ("val_labels", ".tif")):
            for k in ("10078660_15", "10228690_15"):
                cv2.imwrite(os.path.join(tiff, sub, k + ext), np.zeros((8, 8, 3), np.uint8))
        assert D.find_massachusetts(root) == tiff
        split = D.massachusetts_split(tiff)
        assert [len(split[k]) for k in ("train", "select", "test")] == [2, 2, 2]
        assert split["test"][0][0] == "10078660_15"


def test_find_checkpoints_variants():
    import zipfile
    with tempfile.TemporaryDirectory() as root:
        cache = os.path.join(root, "cache")
        inp = os.path.join(root, "input")
        os.makedirs(inp)
        # 1. Plain .pth
        p_pth = os.path.join(inp, "best_a.pth")
        open(p_pth, "wb").write(b"x")
        # 2. Browser-renamed PyTorch .zip
        p_zip = os.path.join(inp, "best_b.zip")
        with zipfile.ZipFile(p_zip, "w") as zf:
            zf.writestr("best_b/data.pkl", b"x")
        # 3. Bundle .zip containing a .pth
        with zipfile.ZipFile(os.path.join(inp, "bundle.zip"), "w") as zf:
            zf.writestr("sub/best_c.pth", b"x")
        # 4. Unpacked PyTorch folder (Kaggle auto-unzip)
        unpacked = os.path.join(inp, "ds", "best_d")
        os.makedirs(unpacked)
        open(os.path.join(unpacked, "data.pkl"), "wb").write(b"x")
        open(os.path.join(unpacked, "version"), "wb").write(b"2")

        found = D.find_checkpoints(inp, prefix="best_", cache_dir=cache)
        bases = {os.path.basename(p) for p in found}
        assert bases == {"best_a.pth", "best_b.zip", "best_c.pth", "best_d.pth"}
        repacked = [p for p in found if os.path.basename(p) == "best_d.pth"][0]
        with zipfile.ZipFile(repacked) as zf:
            assert "best_d/data.pkl" in zf.namelist()
