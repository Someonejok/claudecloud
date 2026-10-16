"""Fixed-schedule trainer with per-epoch resumable checkpoints.

There is no early stopping and no checkpoint selection: the learning rate warms up and
then decays (cosine) to zero over a fixed number of epochs, and the final weights are
the model. Validation curves are recorded for the figures only, so no reported number
was used to choose a checkpoint (reviewer #7, R5, R9).

Determinism and resumption: every training sample is drawn from a generator seeded by
(run seed, epoch, sample position), so the data stream does not depend on how many
times the process was restarted. The full training state is written atomically after
every epoch; a killed run resumes from the last completed epoch.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from . import metrics as M
from .augment import Augmenter
from .config import RunConfig
from .models import build, encoder_parameters, n_params, normalise_input
from .preprocess import render_slice, to_channels, warp_to_native, zscore_stats

EPS = 1e-6


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
class SliceData(Dataset):
    """Items are (row, sample_seed). With sample_seed < 0 no augmentation is applied."""

    def __init__(self, rows: pd.DataFrame, cache, cfg: RunConfig, aug_spec: str, zstats=None):
        self.rows = rows.reset_index(drop=True)[["case", "slice", "has_lesion"]]
        self.cache, self.cfg, self.zstats = cache, cfg, zstats
        self.aug = Augmenter(aug_spec)
        self.metas = {c: cache.meta(c) for c in self.rows.case.unique()}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        i, s = item if isinstance(item, tuple) else (item, -1)
        r = self.rows.iloc[i]
        case, k = r.case, int(r.slice)
        hu = np.asarray(self.cache.hu(case)[k], np.float32)
        mk = np.asarray(self.cache.lab(case)[k]) > 0
        meta, S = self.metas[case], self.cfg.image_size
        if s >= 0:
            rng = np.random.default_rng(s)
            A = self.aug.affine(S, rng)
            h, m, _ = render_slice(hu, mk, meta, self.cfg.fov_mm, S, A)
            h, m = self.aug.elastic(h, m, rng)
            h = self.aug.intensity_hu(h, rng)
            x = to_channels(h, self.cfg.window, self.zstats)
            if self.cfg.window != "zscore":
                x = self.aug.gamma(x, rng)
        else:
            h, m, _ = render_slice(hu, mk, meta, self.cfg.fov_mm, S)
            x = to_channels(h, self.cfg.window, self.zstats)
        return (torch.from_numpy(np.ascontiguousarray(x, np.float32)),
                torch.from_numpy(m[None].astype(np.float32)),
                torch.tensor(float(r.has_lesion)))


def sampling_weights(rows: pd.DataFrame, cfg: RunConfig, seed: int) -> np.ndarray:
    pos = rows.has_lesion.values == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if cfg.sampling == "balanced":
        w = np.where(pos, cfg.pos_fraction / max(npos, 1), (1 - cfg.pos_fraction) / max(nneg, 1))
    elif cfg.sampling == "all":
        w = np.ones(len(rows))
    elif cfg.sampling.startswith("under"):
        ratio = float(cfg.sampling[5:].rstrip("x"))
        keep = np.zeros(len(rows), bool)
        keep[pos] = True
        neg_idx = np.where(~pos)[0]
        rng = np.random.RandomState(seed)
        k = min(len(neg_idx), int(round(npos * ratio)))
        keep[rng.choice(neg_idx, size=k, replace=False)] = True
        w = keep.astype(float)
    else:
        raise ValueError(f"unknown sampling {cfg.sampling}")
    return w / w.sum()


class EpochSampler(Sampler):
    def __init__(self, weights, n, seed):
        self.w, self.n, self.seed, self.epoch = weights, n, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, 1])
        idx = rng.choice(len(self.w), size=self.n, replace=True, p=self.w)
        seeds = np.random.default_rng([self.seed, self.epoch, 2]).integers(0, 2**31 - 1, self.n)
        return iter(zip(idx.tolist(), seeds.tolist()))


def class_balance_table(rows: pd.DataFrame, cfg: RunConfig, seed=0) -> dict:
    """Slices before and after the sampling scheme (expected per epoch) - reviewer #5."""
    w = sampling_weights(rows, cfg, seed)
    pos = rows.has_lesion.values == 1
    return {"slices_total": len(rows), "pos_slices": int(pos.sum()), "neg_slices": int((~pos).sum()),
            "neg_slices_ever_used": int(((w > 0) & ~pos).sum()),
            "neg_discarded_fraction": float(1 - ((w > 0) & ~pos).sum() / max((~pos).sum(), 1)),
            "expected_pos_fraction_per_batch": float(w[pos].sum())}


# ----------------------------------------------------------------------------- loss
def make_loss(cfg: RunConfig):
    def fn(logits, y, has_lesion):
        logits = logits.float()
        p = torch.sigmoid(logits)
        w = None
        if cfg.neg_slice_weight != 1.0:
            w = torch.where(has_lesion > 0.5, torch.ones_like(has_lesion),
                            torch.full_like(has_lesion, cfg.neg_slice_weight))
        pw = torch.tensor(cfg.pos_weight, device=logits.device) if cfg.pos_weight != 1.0 else None
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pw)
        bce = bce.flatten(1).mean(1)
        tp, fn_, fp = (p * y).flatten(1).sum(1), ((1 - p) * y).flatten(1).sum(1), (p * (1 - y)).flatten(1).sum(1)
        if w is not None:
            bce = (bce * w).sum() / w.sum()
            tp, fn_, fp = (tp * w).sum(), (fn_ * w).sum(), (fp * w).sum()
        else:
            bce, tp, fn_, fp = bce.mean(), tp.sum(), fn_.sum(), fp.sum()
        s = cfg.dice_smooth
        if cfg.loss == "bce_dice":
            dice = (2 * tp + s) / (2 * tp + fn_ + fp + s)
            return cfg.bce_weight * bce + (1 - cfg.bce_weight) * (1 - dice)
        if cfg.loss == "tversky":
            ti = (tp + s) / (tp + cfg.tversky_alpha * fn_ + cfg.tversky_beta * fp + s)
            return cfg.bce_weight * bce + (1 - cfg.bce_weight) * (1 - ti)
        raise ValueError(cfg.loss)
    return fn


# ----------------------------------------------------------------------------- optimiser
def make_optimizer(model, cfg: RunConfig):
    enc = encoder_parameters(model)
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in ids]
    groups = [{"params": rest, "lr": cfg.lr}]
    if enc:
        groups.append({"params": enc, "lr": cfg.lr * cfg.encoder_lr_mult})
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(groups, lr=cfg.lr)
    raise ValueError(cfg.optimizer)


def lr_factor(step, total_steps, warmup_steps, schedule):
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


# ----------------------------------------------------------------------------- checkpoints
def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"]); np.random.set_state(st["numpy"]); random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_case(model, cfg: RunConfig, cache, case: str, zstats=None, batch=16, tta=None):
    """Probability map on the case's native grid, float32 (Z, H, W)."""
    dev = device()
    tta = cfg.tta_flip if tta is None else tta
    model.eval()
    hu_vol, meta = cache.hu(case), cache.meta(case)
    Z, H, W = hu_vol.shape
    out = np.zeros((Z, H, W), np.float32)
    xs, Ms = [], []
    for k in range(Z):
        h, _, Mx = render_slice(np.asarray(hu_vol[k], np.float32), None, meta, cfg.fov_mm, cfg.image_size)
        xs.append(to_channels(h, cfg.window, zstats)); Ms.append(Mx)
    use_amp = cfg.amp and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    for i in range(0, Z, batch):
        xb = torch.from_numpy(np.stack(xs[i:i + batch])).to(dev)
        xb = normalise_input(xb, cfg)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
            lg = model(xb).float()
            if tta:
                lg = 0.5 * (lg + torch.flip(model(torch.flip(xb, [3])).float(), [3]))
        pr = torch.sigmoid(lg)[:, 0].cpu().numpy()
        for j, p in enumerate(pr):
            out[i + j] = warp_to_native(p, Ms[i + j], (H, W))
    return np.clip(out, 0, 1)


def save_prob(path: Path, prob: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, prob=np.round(prob * 255).astype(np.uint8))
    os.replace(tmp, path)


def load_prob(path: Path) -> np.ndarray:
    return np.load(path)["prob"].astype(np.float32) / 255.0


def evaluate_cases(probs: dict, cache, thr=0.5, preds: dict | None = None):
    scans, slices, lesions = [], [], []
    for case, prob in probs.items():
        meta = cache.meta(case)
        s, sl, les = M.evaluate_scan(prob, np.asarray(cache.lab(case)), meta["spacing"], case, thr,
                                     pred=None if preds is None else preds[case],
                                     codes_are_subtypes=meta.get("label_kind") == "subtype")
        scans.append(s); slices.append(sl); lesions += les
    S, SL, L = pd.DataFrame(scans), pd.concat(slices, ignore_index=True), pd.DataFrame(lesions)
    if not len(L):
        L = pd.DataFrame(columns=["volume_ml", "detected", "covered_fraction", "case"])
    return S, SL, L


def compute_profile(model, cfg: RunConfig) -> dict:
    """Parameters and forward FLOPs for one slice (reviewer R6)."""
    out = {"params": n_params(model)}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, cfg.in_channels, cfg.image_size, cfg.image_size, device=next(model.parameters()).device)
        model.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x)
        out["gflops_per_slice"] = fc.get_total_flops() / 1e9
    except Exception as e:  # informational only
        out["gflops_per_slice"] = np.nan
        out["flops_error"] = repr(e)[:200]
    return out


# ----------------------------------------------------------------------------- training
def lesion_hu_sample(cache, cases, n_max=1_000_000, seed=0) -> np.ndarray:
    vals = []
    for c in cases:
        lab = np.asarray(cache.lab(c)) > 0
        if lab.any():
            vals.append(np.asarray(cache.hu(c))[lab].astype(np.float32))
    v = np.concatenate(vals) if vals else np.array([0.0, 100.0], np.float32)
    if len(v) > n_max:
        v = np.random.RandomState(seed).choice(v, n_max, replace=False)
    return v


def train_run(cfg: RunConfig, index: pd.DataFrame, cache, train_cases, val_cases, seed: int,
              ckpt_dir: Path, num_workers=2, log=print, stop_after_epochs=None):
    """Train (resuming if a checkpoint exists) and return (model, history, info, zstats).
    `stop_after_epochs` exists for the resume test: it simulates a disconnect."""
    dev = device()
    ckpt_dir = Path(ckpt_dir)
    last = ckpt_dir / "last.pt"
    tr_rows = index[index.case.isin(train_cases)].reset_index(drop=True)
    va_rows = index[index.case.isin(val_cases)].reset_index(drop=True)
    zstats = zscore_stats(lesion_hu_sample(cache, train_cases)) if cfg.window == "zscore" else None

    set_seed(seed)
    model = build(cfg).to(dev)
    opt = make_optimizer(model, cfg)
    use_amp = cfg.amp and dev.type == "cuda"
    bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not bf16)
    loss_fn = make_loss(cfg)

    ds_tr = SliceData(tr_rows, cache, cfg, cfg.aug, zstats)
    sampler = EpochSampler(sampling_weights(tr_rows, cfg, seed), cfg.samples_per_epoch, seed)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler, num_workers=num_workers,
                       pin_memory=dev.type == "cuda", drop_last=True,
                       persistent_workers=False)
    ds_va = SliceData(va_rows, cache, cfg, "none", zstats)
    dl_va = DataLoader(ds_va, batch_size=max(cfg.batch_size, 8), shuffle=False, num_workers=num_workers)

    steps_per_epoch = cfg.samples_per_epoch // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    warm = steps_per_epoch * cfg.warmup_epochs
    bases = [g["lr"] for g in opt.param_groups]
    start, hist, train_secs = 0, [], 0.0
    if last.exists():
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        start, hist, train_secs = st["epoch"], st["hist"], st.get("train_secs", 0.0)
        log(f"    resumed at epoch {start}/{cfg.epochs}")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(start, cfg.epochs):
        t0 = time.time()
        model.train()
        sampler.set_epoch(ep)
        run, nb = 0.0, 0
        for i, (xb, yb, hl) in enumerate(dl_tr):
            step = ep * steps_per_epoch + i
            f = lr_factor(step, total, warm, cfg.schedule)
            for g, b in zip(opt.param_groups, bases):
                g["lr"] = b * f
            xb, yb, hl = xb.to(dev, non_blocking=True), yb.to(dev), hl.to(dev)
            xb = normalise_input(xb, cfg)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                lg = model(xb)
            loss = loss_fn(lg, yb, hl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            run += float(loss.detach()); nb += 1
        # validation curve (figures only, never used for selection)
        model.eval()
        vl, vn, tp, fp, fn = 0.0, 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for xb, yb, hl in dl_va:
                xb, yb, hl = xb.to(dev), yb.to(dev), hl.to(dev)
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                    lg = model(normalise_input(xb, cfg))
                vl += float(loss_fn(lg, yb, hl)) * len(xb); vn += len(xb)
                pb = (torch.sigmoid(lg.float()) >= cfg.threshold).float()
                tp += float((pb * yb).sum()); fp += float((pb * (1 - yb)).sum()); fn += float(((1 - pb) * yb).sum())
        secs = time.time() - t0
        train_secs += secs
        hist.append({"epoch": ep + 1, "lr": opt.param_groups[0]["lr"], "train_loss": run / max(nb, 1),
                     "val_loss": vl / max(vn, 1), "val_dice_pooled": 2 * tp / (2 * tp + fp + fn + EPS),
                     "seconds": secs})
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                      "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                      "epoch": ep + 1, "hist": hist, "rng": _rng_state(), "train_secs": train_secs,
                      "config_hash": cfg.hash()}, last)
        if (ep + 1) % 10 == 0 or ep + 1 == cfg.epochs:
            h = hist[-1]
            log(f"    ep {ep+1:>3}/{cfg.epochs} loss {h['train_loss']:.4f} val {h['val_loss']:.4f} "
                f"dice(pooled) {h['val_dice_pooled']:.3f} {secs:.1f}s")
        if stop_after_epochs is not None and ep + 1 >= stop_after_epochs:
            return None, hist, {"stopped": True}, zstats

    info = {"train_seconds": train_secs, "epochs": cfg.epochs,
            "sec_per_epoch": float(np.median([h["seconds"] for h in hist])) if hist else np.nan,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else np.nan,
            "train_slices": len(tr_rows), **class_balance_table(tr_rows, cfg, seed)}
    return model, hist, info, zstats


def finish_run(model, cfg, hist, info, zstats, cache, val_cases, ledger, arm, seed, fold, paths,
               keep_weights=False, log=print):
    """Evaluate on the validation fold at the fixed threshold, write everything, clean up."""
    t0 = time.time()
    probs = {c: predict_case(model, cfg, cache, c, zstats) for c in val_cases}
    n_sl = sum(p.shape[0] for p in probs.values())
    infer_ms = (time.time() - t0) / max(n_sl, 1) * 1000
    S, SL, L = evaluate_cases(probs, cache, cfg.threshold)
    summ = M.summarise(S, SL, L)
    key = f"{arm}__s{seed}_f{fold}"
    if cfg.save_probs:
        for c, p in probs.items():
            save_prob(paths.probs / ledger.experiment / key / f"{c}.npz", p)
    tdir = ledger.dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    S.to_csv(tdir / f"{key}__scans.csv", index=False)
    SL.to_csv(tdir / f"{key}__slices.csv", index=False)
    L.to_csv(tdir / f"{key}__lesions.csv", index=False)
    pd.DataFrame(hist).to_csv(tdir / f"{key}__history.csv", index=False)
    wdir = paths.ckpt / ledger.experiment / key
    if keep_weights:
        _atomic_save({"model": {k: v.half() if v.is_floating_point() else v
                                for k, v in model.state_dict().items()},
                      "config": cfg.to_dict(), "zstats": zstats}, wdir / "final.pt")
    prof = compute_profile(model, cfg)
    rec = {"config_hash": cfg.hash(), "config": cfg.to_dict(), "metrics": summ,
           "info": {**info, **prof, "inference_ms_per_slice": infer_ms}, "zstats": zstats,
           "val_cases": list(val_cases), "kept_weights": keep_weights}
    ledger.put(arm, seed, fold, rec)
    last = wdir / "last.pt"
    if last.exists():
        last.unlink()
    log(f"    {key}: volDice {summ['dice_vol_mean']:.3f} | sliceDice {summ['dice_slice_mean']:.3f} | "
        f"sliceIoU {summ['iou_slice_mean']:.3f} | FP/scan {summ['fp_lesions_per_scan']:.1f} | "
        f"scanAUC {summ['scan_auc']:.3f} | {info.get('train_seconds', 0)/60:.1f} min")
    return rec


def load_final(path: Path):
    st = torch.load(path, map_location=device(), weights_only=False)
    cfg = RunConfig(**st["config"])
    model = build(cfg.__class__(**{**st["config"], "encoder_weights": None})).to(device())
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st["model"].items()})
    model.eval()
    return model, cfg, st.get("zstats")
