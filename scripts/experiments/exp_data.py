"""Data splits and accuracy metrics shared by the experiment notebooks (NumPy only).

The metric code (exg, count_metrics, derive, the postprocessing variants) is copied from
scripts/paper_figures/kaggle_eval.py, which produced every accuracy number in the paper,
so baselines, ablations and the published v4 results are scored the same way.
scripts/experiments/test_exp_data.py checks that the copies still agree with kaggle_eval.py.
"""
import glob
import hashlib
import importlib.util
import os
import random

import cv2
import numpy as np
from scipy import stats as sstats
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(rel_path, name):
    """Import one repository file directly, without running its package __init__ (which needs PyTorch)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pp = _load("backend/src/utils/graph_postprocess.py", "graph_postprocess")
hysteresis_threshold, connect_canopy_gaps = _pp.hysteresis_threshold, _pp.connect_canopy_gaps

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TEST_SHA1 = "2811fca6c9e44e6cdf713455a422502e4e5e4431"   # the 623 test tiles of the paper
RHO = 3                                                  # relaxed tolerance in pixels


def sha1(ids):
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()


# ── DeepGlobe ────────────────────────────────────────────────────────────────

def find_deepglobe(root="/kaggle/input"):
    """The DeepGlobe 'train' folder (the only part with masks), found the same way as in train_v4."""
    for dirpath, dirs, _ in os.walk(root):
        if "train" in dirs:
            try:
                if any("sat.jpg" in f for f in os.listdir(os.path.join(dirpath, "train"))):
                    return os.path.join(dirpath, "train")
            except OSError:
                pass
    raise FileNotFoundError("Attach the DeepGlobe road extraction dataset (balraj98/deepglobe-road-extraction-dataset).")


def deepglobe_split(train_dir, select_tiles=280, check_sha=True):
    """The v4 split: identical 5,323 train / 280 selection / 623 test tiles as recorded in split.json."""
    split_candidates = [
        os.path.join(REPO, "models", "v4_training", "split.json"),
        os.path.join(os.getcwd(), "models", "v4_training", "split.json"),
        "/kaggle/working/repo/models/v4_training/split.json",
    ]
    split_json = next((p for p in split_candidates if os.path.isfile(p)), None)
    all_ids = {os.path.basename(f).replace("_sat.jpg", "") for f in glob.glob(os.path.join(train_dir, "*_sat.jpg"))}
    all_ids = {i for i in all_ids if os.path.exists(os.path.join(train_dir, f"{i}_mask.png"))}

    if split_json is not None:
        import json
        with open(split_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        test = saved.get("test_ids", [])
        select = saved.get("select_val_ids", [])
        if set(test).issubset(all_ids) and set(select).issubset(all_ids):
            train = sorted(all_ids - set(test) - set(select))
            if check_sha:
                assert sha1(test) == TEST_SHA1, "test tiles differ from the paper's 623 tiles"
                assert sha1(select) == saved.get("select_val_sha1"), "selection tiles differ from split.json"
                if "train_sha1" in saved and len(train) == saved.get("train", 5323):
                    assert sha1(train) == saved.get("train_sha1"), "training tiles differ from split.json"
            assert not (set(train) | set(select)) & set(test)
            return {"train": train, "select": select, "test": test}

    # Fallback for synthetic/custom datasets where split.json IDs do not exist
    ids = sorted(all_ids)
    random.Random(42).shuffle(ids)
    cut = int(len(ids) * 0.9)
    pool, test = ids[:cut], ids[cut:]
    if check_sha:
        assert sha1(test) == TEST_SHA1, "test tiles differ from the paper's 623 tiles"
    select = sorted(random.Random(7).sample(pool, min(select_tiles, len(pool))))
    train = sorted(set(pool) - set(select))
    assert not (set(train) | set(select)) & set(test)
    return {"train": train, "select": select, "test": test}


def deepglobe_items(train_dir, ids):
    """[(tile id, image path, mask path)] in the given order."""
    return [(i, os.path.join(train_dir, f"{i}_sat.jpg"), os.path.join(train_dir, f"{i}_mask.png")) for i in ids]


# ── Massachusetts Roads ──────────────────────────────────────────────────────

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def find_massachusetts(root="/kaggle/input"):
    """Folder holding train/, train_labels/, val/, val_labels/, test/, test_labels/ (TIFF version preferred)."""
    found = []
    for dirpath, dirs, _ in os.walk(root):
        if {"train", "train_labels", "test", "test_labels"} <= set(dirs):
            found.append(dirpath)
    if not found:
        raise FileNotFoundError("Attach the Massachusetts Roads dataset (balraj98/massachusetts-roads-dataset); "
                                "expected folders train, train_labels, test, test_labels.")
    return sorted(found, key=lambda d: ("tif" not in d.lower(), d))[0]


def paired_items(img_dir, label_dir):
    """[(stem, image path, label path)] for files present in both folders, sorted by name."""
    stem = lambda f: os.path.splitext(f)[0]
    images = {stem(f): os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXT)}
    labels = {stem(f): os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(IMAGE_EXT)}
    return [(k, images[k], labels[k]) for k in sorted(set(images) & set(labels))]


def massachusetts_split(root):
    val = next((v for v in ("val", "valid", "validation") if os.path.isdir(os.path.join(root, v))), None)
    split = {"train": paired_items(os.path.join(root, "train"), os.path.join(root, "train_labels")),
             "test": paired_items(os.path.join(root, "test"), os.path.join(root, "test_labels"))}
    split["select"] = paired_items(os.path.join(root, val), os.path.join(root, val + "_labels")) if val else []
    return split


# ── Reading tiles ────────────────────────────────────────────────────────────

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    """Boolean road mask; labels stored as 0/255 or 0/1."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"could not read {path}")
    return m > (127 if m.max() > 1 else 0)


# ── Canopy mask (from kaggle_eval.py) ────────────────────────────────────────

def exg(rgb):
    rgb = rgb.astype(np.float32)
    s = rgb.sum(axis=2) + 1e-6
    return (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / s


def otsu_vegetation_threshold(items):
    """Otsu threshold of the excess green index, sampled exactly as in kaggle_eval.py (seed 0, 4000 px per tile)."""
    rng = np.random.default_rng(0)
    samples = []
    for _, img_path, _ in items:
        e = exg(read_rgb(img_path)).ravel()
        samples.append(e[rng.integers(0, e.size, 4000)])
    return float(threshold_otsu(np.concatenate(samples)))


# ── Postprocessing variants (from kaggle_eval.py) ────────────────────────────

def binarize_all(probs, variants, gsd_scale=1.0):
    """raw: p > 0.5; hyst: hysteresis 0.35/0.12; hyst_close: + 5x5 closing; full: + gap bridging.
    gsd_scale = 0.5 m / pixel size; the bridging distances are in pixels and scale with it
    (1.0 for DeepGlobe, which reproduces kaggle_eval.py exactly; 0.5 for 1 m imagery)."""
    out = {}
    if "raw" in variants:
        out["raw"] = probs > 0.5
    if any(v in variants for v in ("hyst", "hyst_close", "full")):
        hyst = hysteresis_threshold(probs, high_thresh=0.35, low_thresh=0.12)
        if "hyst" in variants:
            out["hyst"] = hyst > 0
        closed = cv2.morphologyEx(hyst, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        if "hyst_close" in variants:
            out["hyst_close"] = closed > 0
        if "full" in variants:
            out["full"] = connect_canopy_gaps(closed, max_gap_dist=220.0 * gsd_scale, max_angle_deg=65.0,
                                              road_width=max(2, int(round(6 * gsd_scale))),
                                              border_margin=int(round(16 * gsd_scale))) > 0
    return out


# ── Pixel counts and derived metrics (from kaggle_eval.py) ───────────────────

COUNT_KEYS = ["tp", "fp", "fn", "npred", "rel_tp_pred", "rel_tp_gt", "sk_p", "sk_p_in_g", "sk_g", "sk_g_in_p",
              "can_gt", "can_hit", "can_hit_rel", "open_gt", "open_hit", "open_hit_rel"]


def count_metrics(P, G, V, near_g, sk_g):
    if P.any():
        near_p = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_L2, 3) <= RHO
    else:
        near_p = np.zeros_like(P)
    sk_p = skeletonize(P)
    gv, go = G & V, G & ~V
    return {
        "tp": int((P & G).sum()), "fp": int((P & ~G).sum()), "fn": int((~P & G).sum()), "npred": int(P.sum()),
        "rel_tp_pred": int((P & near_g).sum()), "rel_tp_gt": int((G & near_p).sum()),
        "sk_p": int(sk_p.sum()), "sk_p_in_g": int((sk_p & G).sum()),
        "sk_g": int(sk_g.sum()), "sk_g_in_p": int((sk_g & P).sum()),
        "cc_pred": int(cv2.connectedComponents(P.astype(np.uint8), connectivity=8)[0] - 1),
        "can_gt": int(gv.sum()), "can_hit": int((gv & P).sum()), "can_hit_rel": int((gv & near_p).sum()),
        "open_gt": int(go.sum()), "open_hit": int((go & P).sum()), "open_hit_rel": int((go & near_p).sum()),
    }


def derive(s):
    div = lambda a, b: a / b if b else float("nan")
    gt = s["tp"] + s["fn"]
    iou = div(s["tp"], s["tp"] + s["fp"] + s["fn"])
    p, r = div(s["tp"], s["tp"] + s["fp"]), div(s["tp"], gt)
    rp, rr = div(s["rel_tp_pred"], s["npred"]), div(s["rel_tp_gt"], gt)
    tprec, tsens = div(s["sk_p_in_g"], s["sk_p"]), div(s["sk_g_in_p"], s["sk_g"])
    f = lambda a, b: div(2 * a * b, a + b)
    return {"iou": iou, "precision": p, "recall": r, "f1": f(p, r),
            "relaxed_precision": rp, "relaxed_recall": rr, "relaxed_f1": f(rp, rr),
            "cldice": f(tprec, tsens), "topo_precision": tprec, "topo_sensitivity": tsens,
            "canopy_recall": div(s["can_hit"], s["can_gt"]), "canopy_recall_relaxed": div(s["can_hit_rel"], s["can_gt"]),
            "open_recall": div(s["open_hit"], s["open_gt"]), "open_recall_relaxed": div(s["open_hit_rel"], s["open_gt"])}


def tile_record(job):
    """Scores one tile for every requested variant. job = (tile id, label path, image path,
    {source: float16 probabilities}, {source: [variants]}, vegetation threshold, gsd_scale)."""
    tid, label_path, img_path, probs_by_src, variants, veg_t, gsd_scale = job
    G = read_mask(label_path)
    V = exg(read_rgb(img_path)) > veg_t
    near_g = (cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 3) <= RHO) if G.any() else np.zeros_like(G)
    sk_g = skeletonize(G)
    rec = {"tile": tid, "gt_pixels": int(G.sum()), "pixels": int(G.size),
           "cc_gt": int(cv2.connectedComponents(G.astype(np.uint8), connectivity=8)[0] - 1), "variants": {}}
    for src, vlist in variants.items():
        probs = probs_by_src[src].astype(np.float32)
        for v, P in binarize_all(probs, vlist, gsd_scale).items():
            rec["variants"][f"{src}_{v}"] = count_metrics(P, G, V, near_g, sk_g)
    return rec


# ── Aggregation ──────────────────────────────────────────────────────────────

def count_matrix(records, variant):
    return np.array([[r["variants"][variant][k] for k in COUNT_KEYS] for r in records], dtype=np.float64)


def summarize(mat):
    return derive(dict(zip(COUNT_KEYS, mat.sum(0))))


def bootstrap_ci(mat, keys=("iou", "f1", "relaxed_f1", "cldice"), reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    draws = {k: [] for k in keys}
    for _ in range(reps):
        d = summarize(mat[rng.integers(0, n, n)])
        for k in keys:
            draws[k].append(d[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def per_tile(records, variant):
    """{tile: metrics} for one variant; tile IoU is NaN when the tile has no road in label or prediction."""
    out = {}
    for r in records:
        c = r["variants"][variant]
        d = derive(c)
        union = c["tp"] + c["fp"] + c["fn"]
        out[r["tile"]] = {"iou": d["iou"] if union else float("nan"), "f1": d["f1"],
                          "relaxed_f1": d["relaxed_f1"], "cldice": d["cldice"],
                          "components_pred": c["cc_pred"], "components_gt": r["cc_gt"]}
    return out


def summary(records, variant):
    mat = count_matrix(records, variant)
    pt = per_tile(records, variant)
    return {**summarize(mat), "ci95": bootstrap_ci(mat),
            "median_components_per_tile": float(np.median([v["components_pred"] for v in pt.values()])),
            "median_gt_components_per_tile": float(np.median([v["components_gt"] for v in pt.values()])),
            "n_tiles": len(records)}


def paired_test(records_a, records_b, variant, metric="iou"):
    """Wilcoxon signed rank test of per tile scores, model a against model b, on the tiles both scored."""
    a, b = per_tile(records_a, variant), per_tile(records_b, variant)
    tiles = [t for t in a if t in b and not (np.isnan(a[t][metric]) or np.isnan(b[t][metric]))]
    da = np.array([a[t][metric] for t in tiles])
    db = np.array([b[t][metric] for t in tiles])
    diff = da - db
    w = sstats.wilcoxon(da, db) if len(tiles) > 10 and (diff != 0).any() else None
    return {"n_tiles": len(tiles), "mean_diff": float(diff.mean()) if len(tiles) else float("nan"),
            "frac_tiles_better": float((diff > 0).mean()) if len(tiles) else float("nan"),
            "wilcoxon_p": float(w.pvalue) if w else None}


def find_checkpoints(root="/kaggle/input", prefix="best_", cache_dir="/tmp/ckpt_cache"):
    """Find checkpoints under root (.pth, browser-renamed .zip, bundle .zip, or unpacked data.pkl dirs)."""
    import shutil
    import zipfile

    out = []
    if not os.path.isdir(root):
        return out

    def _tag_path(tag, fname):
        sub = os.path.join(cache_dir, hashlib.md5(tag.encode()).hexdigest()[:8])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, fname)

    for dirpath, _, files in os.walk(root):
        fnset = set(files)
        dname = os.path.basename(dirpath)
        if "data.pkl" in fnset and (".format_version" in fnset or "version" in fnset) and dname.startswith(prefix):
            dst = _tag_path(dirpath, dname + ".pth")
            if not os.path.exists(dst):
                base_parent = os.path.dirname(dirpath)
                with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
                    for sub_root, _, sub_files in os.walk(dirpath):
                        for sf in sub_files:
                            fp = os.path.join(sub_root, sf)
                            zf.write(fp, os.path.relpath(fp, base_parent).replace("\\", "/"))
            out.append(dst)
            continue

        for fn in sorted(files):
            fp = os.path.join(dirpath, fn)
            if fn.endswith(".pth") and fn.startswith(prefix):
                out.append(fp)
            elif fn.endswith(".zip"):
                try:
                    with zipfile.ZipFile(fp) as zf:
                        names = zf.namelist()
                        if fn.startswith(prefix) and any(n == "data.pkl" or (n.count("/") == 1 and n.endswith("/data.pkl")) for n in names):
                            out.append(fp)
                        else:
                            for m in sorted(names):
                                mb = os.path.basename(m)
                                if mb.startswith(prefix) and mb.endswith(".pth") and not m.endswith("/"):
                                    dst = _tag_path(f"{fp}:{m}", mb)
                                    if not os.path.exists(dst):
                                        with zf.open(m) as src, open(dst, "wb") as dst_f:
                                            shutil.copyfileobj(src, dst_f)
                                    out.append(dst)
                except zipfile.BadZipFile:
                    pass
    return sorted(set(out))
