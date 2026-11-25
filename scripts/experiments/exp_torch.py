"""PyTorch side of the experiment notebooks: experiment list, training loop, prediction.

Every experiment reuses the v4 training code (notebooks/train_v4.ipynb): the same split,
loss, augmentation, optimiser, schedule, EMA, collapse gate, and best epoch chosen on the
280 selection tiles. Only one thing changes per experiment: the network (baselines), one
part of our design (ablations), or the dataset (Massachusetts).
"""
import copy
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(REPO, "backend"), os.path.join(REPO, "scripts", "paper_figures"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp_data as D                                                    # noqa: E402
import reference_models as R                                            # noqa: E402
from src.data.dataset import CanopyShadowDropout                        # noqa: E402
from src.models.mobilevit_v2 import MobileViT_v2, load_mobilevit_checkpoint  # noqa: E402
from src.utils.loss import RoadExtractionLoss, soft_skel                # noqa: E402

torch.backends.cudnn.benchmark = False   # as in the paper's evaluation

# ── Recipes ──────────────────────────────────────────────────────────────────
# FULL is the v4 recipe of the paper (Table "Training Configuration").
FULL = {"epochs": 160, "warmup_epochs": 5, "alpha_decay_epochs": 60, "min_epochs": 80, "patience": 40,
        "crop": 512, "batch": 8, "accum": 1, "lr": 1e-3, "min_lr": 1e-5, "weight_decay": 1e-4,
        "ema_decay": 0.999, "max_pos_frac": 0.20, "repeat": 1}
# SHORT: 60 epochs for the ablations, schedule scaled to the length, no early stopping.
# The full model is retrained with SHORT too (abl_full), so every ablation row is compared
# with a model trained for the same time.
SHORT = {**FULL, "epochs": 60, "warmup_epochs": 2, "alpha_decay_epochs": 23, "min_epochs": 60, "patience": 60}
# Massachusetts has 1,108 training tiles; each is used twice per epoch (two random crops).
MASS = {**FULL, "repeat": 2}

EXPERIMENTS = {
    # Baselines: full recipe on the paper's split.
    "unet":        {"model": "unet", "recipe": {**FULL, "batch": 4, "accum": 2},
                    "desc": "U-Net (Ronneberger et al., 2015), random initialization"},
    "dlinknet34":  {"model": "dlinknet34", "pretrained": True, "recipe": FULL,
                    "desc": "D-LinkNet34 (Zhou et al., 2018), ImageNet ResNet-34 encoder as in original"},
    "dlinknet34_scratch": {"model": "dlinknet34", "pretrained": False, "recipe": FULL,
                           "desc": "D-LinkNet34 without pretraining (from scratch, matching proposed model)"},
    "deeplabv3_mbv3": {"model": "deeplabv3", "pretrained": False, "recipe": FULL,
                       "desc": "DeepLabV3 MobileNetV3-Large (Chen et al., 2017), random initialization"},
    "lraspp_mbv3": {"model": "lraspp", "pretrained": False, "recipe": FULL,
                    "desc": "LR-ASPP MobileNetV3-Large (Howard et al., 2019), random initialization"},
    "lraspp_mbv3_pretrained": {"model": "lraspp", "pretrained": True, "recipe": FULL,
                               "desc": "LR-ASPP MobileNetV3-Large with ImageNet backbone"},
    # Ablations: SHORT recipe, one change each.
    "abl_full":           {"model": "mobilevit", "arch": {}, "recipe": SHORT,
                           "desc": "Proposed network, short recipe (reference row of the ablation)"},
    "abl_no_cldice":      {"model": "mobilevit", "arch": {}, "cldice": False, "recipe": SHORT,
                           "desc": "Without clDice: BCE + Dice only"},
    "abl_no_gates":       {"model": "mobilevit", "arch": {"attention_gates": False}, "recipe": SHORT,
                           "desc": "Without attention gates (plain skip connections)"},
    "abl_no_strip_shift": {"model": "mobilevit", "arch": {"strip_conv": False, "channel_shift": False},
                           "recipe": SHORT, "desc": "Without strip convolutions and channel shift (3x3 convolutions)"},
    "abl_no_transformer": {"model": "mobilevit", "arch": {"transformer": False}, "recipe": SHORT,
                           "desc": "Without transformer layers (local convolutions only)"},
    # Second dataset.
    "mass_scratch":   {"model": "mobilevit", "arch": {}, "recipe": MASS, "dataset": "massachusetts",
                       "desc": "Proposed network trained on Massachusetts Roads from random weights"},
    "mass_finetune":  {"model": "mobilevit", "arch": {}, "recipe": {**SHORT, "repeat": 2, "lr": 3e-4},
                       "dataset": "massachusetts", "init": "v4",
                       "desc": "Optional: v4 weights fine tuned on Massachusetts Roads"},
}


def build_model(spec, pretrained=None):
    pre = spec.get("pretrained", False) if pretrained is None else pretrained
    kind = spec["model"]
    if kind == "mobilevit":
        return MobileViT_v2(num_classes=1, **spec.get("arch", {}))
    if kind == "unet":
        return R.UNet()
    if kind == "dlinknet34":
        return R.DLinkNet34(pretrained=pre)
    if kind == "lraspp":
        return R._TorchvisionSeg(torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone="IMAGENET1K_V1" if pre else None, num_classes=1))
    if kind == "deeplabv3":
        return R._TorchvisionSeg(torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone="IMAGENET1K_V1" if pre else None, num_classes=1, aux_loss=False))
    raise ValueError(kind)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def make_loss(spec, recipe):
    return RoadExtractionLoss(total_epochs=recipe["epochs"], alpha_start=0.5, alpha_end=0.15, dice_weight=0.35,
                              pos_weight=2.0, decay_power=0.5, alpha_decay_epochs=recipe["alpha_decay_epochs"],
                              use_cldice=spec.get("cldice", True))


# ── Data ─────────────────────────────────────────────────────────────────────

def train_transform(crop):
    """The v4 augmentation."""
    return A.Compose([
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.3),
        CanopyShadowDropout(p=0.5),
        A.Normalize(mean=D.MEAN, std=D.STD, max_pixel_value=255.0), ToTensorV2()])


NORMALIZE = A.Compose([A.Normalize(mean=D.MEAN, std=D.STD, max_pixel_value=255.0), ToTensorV2()])


class TileDataset(Dataset):
    def __init__(self, items, tf, repeat=1):
        self.items, self.tf, self.repeat = items, tf, repeat

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        _, img_path, mask_path = self.items[i % len(self.items)]
        out = self.tf(image=D.read_rgb(img_path), mask=D.read_mask(mask_path).astype(np.uint8))
        return out["image"], out["mask"].float().unsqueeze(0)


def pad32(x):
    """Reflect pad the last two dims to multiples of 32; returns (padded, H, W)."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


# ── Validation during training (the v4 selection score) ─────────────────────

@torch.no_grad()
def validate(net, loader, device, amp=True):
    net.eval()
    tp = fp = fn = 0.0
    pos, cl, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xp, h, w = pad32(x)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            p = torch.sigmoid(net(xp)).float()[..., :h, :w]
        b = (p > 0.5).float()
        tp += (b * y).sum().item(); fp += (b * (1 - y)).sum().item(); fn += ((1 - b) * y).sum().item()
        pos += b.mean((1, 2, 3)).sum().item()
        sp, sy = soft_skel(p, 10), soft_skel(y, 10)
        tprec = ((sp * y).sum((1, 2, 3)) + 1) / (sp.sum((1, 2, 3)) + 1)
        tsens = ((sy * p).sum((1, 2, 3)) + 1) / (sy.sum((1, 2, 3)) + 1)
        cl += (2 * tprec * tsens / (tprec + tsens + 1e-7)).sum().item()
        n += x.shape[0]
    return {"iou": tp / (tp + fp + fn + 1e-9), "precision": tp / (tp + fp + 1e-9),
            "recall": tp / (tp + fn + 1e-9), "positive_frac": pos / n, "soft_cldice": cl / n}


# ── Training ─────────────────────────────────────────────────────────────────

def train_experiment(name, train_items, select_items, out_dir, test_sha1=None, init_checkpoint=None,
                     time_limit_h=11.0, num_workers=4, smoke_test=False, input_root="/kaggle/input", seed=42):
    """Trains EXPERIMENTS[name]; resumable across Kaggle sessions. Returns 'DONE' or 'RESUME NEEDED'."""
    import pandas as pd
    spec = EXPERIMENTS[name]
    rc = dict(spec["recipe"])
    if smoke_test:   # a few minutes: proves every step runs before hours are spent
        rc.update(epochs=2, min_epochs=2, warmup_epochs=1, alpha_decay_epochs=1)
        train_items, select_items = train_items[:16 * rc["batch"]], select_items[:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    best_path, state_path = os.path.join(out_dir, f"best_{name}.pth"), os.path.join(out_dir, "train_state.pth")

    train_loader = DataLoader(TileDataset(train_items, train_transform(rc["crop"]), rc["repeat"]),
                              batch_size=rc["batch"], shuffle=True, drop_last=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0)
    val_batch = 4 if max(D.read_rgb(select_items[0][1]).shape[:2]) <= 1024 else 1   # Massachusetts: 1500 px
    val_loader = DataLoader(TileDataset(select_items, NORMALIZE), batch_size=val_batch,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

    model = build_model(spec).to(device)
    if init_checkpoint:
        init_model, _ = load_any_checkpoint(init_checkpoint, device)
        model.load_state_dict(init_model.state_dict())
        print("initialised from", init_checkpoint)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    loss_fn = make_loss(spec, rc)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["weight_decay"])
    steps_per_epoch = len(train_loader) // rc["accum"]
    total, warm = rc["epochs"] * steps_per_epoch, rc["warmup_epochs"] * steps_per_epoch
    min_ratio = rc["min_lr"] / rc["lr"]

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best, history, ema_steps = 0, {"iou": -1.0, "epoch": 0}, [], 0

    # Resume from the attached output that got furthest (rows in its history.csv), or from this session.
    previous = glob.glob(os.path.join(input_root, "**", name, "train_state.pth"), recursive=True)
    previous += [state_path] if os.path.exists(state_path) else []
    epochs_done = lambda p: sum(1 for _ in open(os.path.join(os.path.dirname(p), "history.csv"))) \
        if os.path.exists(os.path.join(os.path.dirname(p), "history.csv")) else 0
    previous.sort(key=epochs_done)
    if previous and not smoke_test:
        st = torch.load(previous[-1], map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        use_amp = st.get("use_amp", use_amp)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best, history, ema_steps = st["epoch"] + 1, st["best"], st["history"], st["ema_steps"]
        src = os.path.join(os.path.dirname(previous[-1]), f"best_{name}.pth")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(best_path):
            import shutil
            shutil.copy(src, best_path)
        print(f"RESUMED from {previous[-1]} at epoch {start_epoch}; best so far {best}")
    last_healthy_path = state_path if os.path.exists(state_path) else (previous[-1] if previous and not smoke_test else None)

    print(f"{name}: {spec['desc']}\nparameters {n_params(model):,} | {len(train_items)} training tiles "
          f"x{rc['repeat']} | {len(select_items)} selection tiles | {steps_per_epoch} steps per epoch | recipe {rc}")
    if start_epoch == 0:   # preflight: the selection pass runs before any long training
        sub = DataLoader(TileDataset(select_items[:2], NORMALIZE), batch_size=1)
        print("preflight selection pass:", {k: round(v, 4) for k, v in validate(model, sub, device, amp=use_amp).items()})

    meta = {"experiment": name, "spec": spec, "arch": spec.get("arch", {}), "test_ids_sha1": test_sha1,
            "n_train": len(train_items), "n_select": len(select_items), "params": n_params(model)}
    # Snapshot buffers (e.g. BatchNorm running stats) so an FP16 overflow never poisons running_mean/var.
    bn_bufs = {k: v.clone() for k, v in model.named_buffers()}
    t_start, stopped_for_time = time.time(), False
    for epoch in range(start_epoch, rc["epochs"]):
        t0 = time.time()
        model.train()
        alpha = loss_fn.update_alpha(epoch)
        run_loss, steps, micro, epoch_opt_steps, amp_fallbacks, fp32_bad_steps = 0.0, 0, 0, 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits, y)
            loss_val = loss.item()
            if use_amp and not (math.isfinite(loss_val) and torch.isfinite(logits).all()):
                with torch.no_grad():
                    for k, b in model.named_buffers():
                        b.copy_(bn_bufs[k])
                with torch.autocast(device.type, enabled=False):
                    logits = model(x)
                loss = loss_fn(logits, y)
                loss_val = loss.item()
                amp_fallbacks += 1
                if amp_fallbacks <= 3:
                    print(f"  [amp-fallback] epoch {epoch} step {i}: non-finite in FP16, retried in FP32 (loss={loss_val:.4f})")
            if not math.isfinite(loss_val):
                with torch.no_grad():
                    for k, b in model.named_buffers():
                        b.copy_(bn_bufs[k])
                opt.zero_grad(set_to_none=True)
                micro = 0
                fp32_bad_steps += 1
                if fp32_bad_steps >= 5:
                    print(f"  [stability] epoch {epoch} step {i}: {fp32_bad_steps} non-finite batches in FP32; aborting epoch early for rollback.")
                    steps = 0
                    break
                continue
            scaler.scale(loss / rc["accum"]).backward()
            run_loss += loss_val; steps += 1; micro += 1
            with torch.no_grad():
                for k, b in model.named_buffers():
                    bn_bufs[k].copy_(b)
            if micro < rc["accum"]:
                continue
            micro = 0
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                sched.step()
                ema_steps += 1
                epoch_opt_steps += 1
                d = min(rc["ema_decay"], (1 + ema_steps) / (10 + ema_steps))
                with torch.no_grad():
                    for ve, vm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if ve.dtype.is_floating_point:
                            ve.mul_(d).add_(vm.detach(), alpha=1 - d)
                        else:
                            ve.copy_(vm)
                if epoch == start_epoch and epoch_opt_steps == 20:
                    per_step = (time.time() - t0) / 20
                    print(f"timing: {per_step:.2f} s per step -> about {per_step * steps_per_epoch / 60:.1f} min of "
                          f"training per epoch, {per_step * total / 3600:.1f} h for {rc['epochs']} epochs (plus selection passes)")

        if use_amp and amp_fallbacks >= 20:
            use_amp = False
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            print(f"  [amp-fallback] epoch {epoch}: {amp_fallbacks} FP16 overflows; switching remaining epochs to FP32.")

        healthy = (steps > 0 and all(torch.isfinite(v).all() for v in model.state_dict().values())
                   and all(torch.isfinite(v).all() for v in ema.state_dict().values()))
        if healthy:
            v_raw, v_ema = validate(model, val_loader, device, amp=use_amp), validate(ema, val_loader, device, amp=use_amp)
            row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "alpha": alpha,
                   "train_loss": run_loss / steps,
                   **{f"val_{k}": v for k, v in v_raw.items()}, **{f"ema_{k}": v for k, v in v_ema.items()},
                   "epoch_min": (time.time() - t0) / 60}
            healthy = math.isfinite(row["train_loss"])
        if not healthy:
            print(f"WARNING: epoch {epoch} ended non-finite (steps={steps}); restoring previous healthy state and disabling AMP.")
            if last_healthy_path and os.path.exists(last_healthy_path):
                st = torch.load(last_healthy_path, map_location=device, weights_only=False)
                model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
                opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
                bn_bufs = {k: v.clone() for k, v in model.named_buffers()}
                use_amp = False
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                continue
            break

        history.append(row)
        for tag, net, v in (("raw", model, v_raw), ("ema", ema, v_ema)):
            if v["positive_frac"] <= rc["max_pos_frac"] and v["iou"] > best["iou"]:
                best = {"iou": v["iou"], "epoch": epoch, "weights": tag}
                torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "weights": tag,
                            **{f"val_{k}": val for k, val in v.items()}, **meta}, best_path)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "use_amp": use_amp,
                    "epoch": epoch, "best": best, "history": history, "ema_steps": ema_steps}, state_path)
        last_healthy_path = state_path
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        print(f"epoch {epoch:3d} | {row['epoch_min']:.1f} min | loss {row['train_loss']:.4f} | selection IoU "
              f"raw {v_raw['iou']:.4f} ema {v_ema['iou']:.4f} | pos {v_ema['positive_frac']:.3f} | best {best}")
        if epoch + 1 >= rc["min_epochs"] and epoch - best["epoch"] >= rc["patience"]:
            print(f"EARLY STOP at epoch {epoch}: no improvement for {rc['patience']} epochs.")
            break
        if (time.time() - t_start) / 3600 + 1.3 * row["epoch_min"] / 60 > time_limit_h:
            stopped_for_time = True
            print(f"TIME LIMIT after epoch {epoch}. Run a new version with this notebook's output attached to resume.")
            break
    json.dump({"experiment": name, "best": best, "epochs_run": len(history), "recipe": rc},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    status = "RESUME NEEDED" if stopped_for_time else "DONE"
    print(status, "| best:", best)
    return status


# ── Prediction and evaluation ────────────────────────────────────────────────

def load_any_checkpoint(path, device):
    """(model in eval mode, metadata) for an experiment checkpoint or a v4/v3 MobileViT checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "experiment" in ck:
        model = build_model(ck["spec"], pretrained=False)
        model.load_state_dict(ck["model_state_dict"])
        meta = {k: v for k, v in ck.items() if k != "model_state_dict"}
    else:
        model, meta = load_mobilevit_checkpoint(path, device=device)
    return model.to(device).eval(), meta


@torch.no_grad()
def predict(model, rgb, device, tta=True, scale=1.0):
    """Road probabilities (float16) for one image: (single pass, four flip TTA or None).
    scale resamples the image before the network (2.0 turns 1 m pixels into 0.5 m) and the
    probabilities back to the original size afterwards."""
    h0, w0 = rgb.shape[:2]
    if scale != 1.0:
        rgb = cv2.resize(rgb, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_CUBIC)
    x = (rgb.astype(np.float32) / 255.0 - np.array(D.MEAN, np.float32)) / np.array(D.STD, np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    x, h, w = pad32(x)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        p = torch.sigmoid(model(x)).float()
        pt = None
        if tta:
            acc = p.clone()
            for dims in ([3], [2], [2, 3]):
                acc += torch.flip(torch.sigmoid(model(torch.flip(x, dims))).float(), dims)
            pt = acc / 4.0
    out = []
    for q in (p, pt):
        if q is None:
            out.append(None)
            continue
        q = q[0, 0, :h, :w].cpu().numpy()
        if scale != 1.0:
            q = cv2.resize(q, (w0, h0), interpolation=cv2.INTER_AREA)
        out.append(q.astype(np.float16))
    return out[0], out[1]


def evaluate(model, items, variants, veg_t, device, scale=1.0, gsd_scale=1.0):
    """Per tile pixel counts (exp_data.tile_record) for every item and requested variant.
    variants: {"single": [...], "tta": [...]}, e.g. {"single": ["raw"], "tta": ["raw", "full"]}."""
    need_tta = bool(variants.get("tta"))
    pool = mp.get_context("fork").Pool(os.cpu_count()) if "fork" in mp.get_all_start_methods() else None
    records, jobs, t0 = [], [], time.time()

    def flush():
        records.extend(pool.map(D.tile_record, jobs) if pool else [D.tile_record(j) for j in jobs])
        jobs.clear()

    for k, (tid, img_path, label_path) in enumerate(items):
        p, pt = predict(model, D.read_rgb(img_path), device, tta=need_tta, scale=scale)
        srcs = {"single": p, **({"tta": pt} if pt is not None else {})}
        jobs.append((tid, label_path, img_path, srcs, {s: v for s, v in variants.items() if v}, veg_t, gsd_scale))
        if len(jobs) >= 32:
            flush()
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(items)} tiles, {(time.time() - t0) / 60:.1f} min")
    if jobs:
        flush()
    if pool:
        pool.close()
    return records
