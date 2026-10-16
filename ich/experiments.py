"""Every experiment arm, the pre-registration, and resumable runners.

All arms are written to preregistration.json (with their configuration hashes) before
anything trains. Re-running checks the file instead of rewriting it: an arm whose
definition changed after registration is refused unless the change is recorded with
`amend(...)`, which appends a timestamped entry instead of erasing history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits as SP
from .config import RunConfig, override, quick
from .ledger import Ledger

# ----------------------------------------------------------------------------- base recipe
# Lean, fast and identical across arms except for the factor an experiment varies.
BASE = RunConfig(window="three_window", fov_mm=230.0, image_size=320, arch="unet",
                 encoder="resnet34", encoder_weights="imagenet", epochs=50,
                 samples_per_epoch=800, batch_size=16, optimizer="adamw", lr=3e-4,
                 weight_decay=1e-4, warmup_epochs=2, schedule="cosine", loss="bce_dice",
                 bce_weight=0.5, dice_smooth=1.0, sampling="balanced", pos_fraction=0.5,
                 aug="base", threshold=0.5)

# The original manuscript's model and recipe, under a correct protocol. Same sample
# budget as every other arm so the comparison is about the recipe, not training length.
PAPER = override(BASE, arch="paper_unet", encoder="none", encoder_weights=None,
                 window="brain_40_120", image_size=256, optimizer="adam", lr=0.003718,
                 weight_decay=0.0, warmup_epochs=0, dice_smooth=15.0, sampling="under1x",
                 aug="paper_moderate", grad_clip=0.0)

MAGNITUDES = (0.5, 1.0, 2.0, 4.0)
FAMILIES = ("affine", "elastic", "intensity", "noise")


def arms() -> dict[str, dict]:
    """experiment -> {arm: RunConfig}; plus seeds/keep_weights per arm."""
    E = {}
    E["E1_paper"] = {"paper_recipe": PAPER}
    E["E2_imbalance"] = {
        "balanced": BASE,
        "under1x": override(BASE, sampling="under1x"),
        "all_slices": override(BASE, sampling="all"),
        "all_negweight0.2": override(BASE, sampling="all", neg_slice_weight=0.2),
        "all_posweight10": override(BASE, sampling="all", pos_weight=10.0),
        "all_tversky_a0.7": override(BASE, sampling="all", loss="tversky",
                                     tversky_alpha=0.7, tversky_beta=0.3),
    }
    E["E3_window"] = {w: override(BASE, window=w) for w in
                      ("three_window", "brain_40_80", "brain_40_120", "brain_30_120", "zscore")}
    e4 = {"none": override(BASE, aug="none"), "base": BASE,
          "paper_moderate": override(BASE, aug="paper_moderate"),
          "paper_extreme": override(BASE, aug="paper_extreme")}
    for f in FAMILIES:
        for m in MAGNITUDES:
            e4[f"{f}@{m:g}"] = override(BASE, aug=f"base+{f}@{m:g}")
    E["E4_augmentation"] = e4
    E["E5_architecture"] = {
        "paper_unet": override(BASE, arch="paper_unet", encoder="none", encoder_weights=None),
        "resnet34": BASE,
        "densenet121": override(BASE, encoder="densenet121"),
        "vgg16_bn": override(BASE, encoder="vgg16_bn"),
        "mobilenet_v2": override(BASE, encoder="mobilenet_v2"),
    }
    return E


# arms that run extra seeds, and arms whose fold weights are kept for test-set analysis
EXTRA_SEEDS = {("E4_augmentation", a): (0, 1, 2) for a in ("none", "base", "paper_moderate", "paper_extreme")}
KEEP_WEIGHTS = {("E1_paper", "paper_recipe"), ("E4_augmentation", "none"), ("E4_augmentation", "base"),
                ("E4_augmentation", "paper_moderate"), ("E4_augmentation", "paper_extreme"),
                ("E5_architecture", "resnet34")}

# run order when GPU time is limited: each prefix of this list is a coherent paper
PRIORITY = [
    ("nnunet", "P+B"),
    ("E1_paper", None),
    ("E4_augmentation", ("none", "base", "paper_moderate", "paper_extreme")),
    ("E4_augmentation", None),
    ("nnunet", "P"),
    ("E2_imbalance", None),
    ("E3_window", None),
    ("E5_architecture", None),
]

PRIMARY_METRIC = "dice_vol_mean"
DECISION_RULES = {
    "primary_endpoint": "per-scan volumetric Dice averaged over ICH-positive scans (dice_vol_mean), "
                        "threshold 0.5, no post-processing, on out-of-fold development predictions",
    "comparisons": "paired Wilcoxon signed-rank on matched (seed, fold) blocks vs the experiment's "
                   "reference arm, Holm-corrected within each experiment; Cliff's delta; paired "
                   "bootstrap 95% CI of the mean difference",
    "augmentation_dose_response": "Page's L trend test across magnitudes 0 (base), 0.5, 1, 2, 4 within "
                                  "each family, predicted decreasing; Friedman omnibus across "
                                  "none/base/paper_moderate/paper_extreme on 15 blocks",
    "references": {"E2_imbalance": "balanced", "E3_window": "three_window", "E4_augmentation": "base",
                   "E5_architecture": "resnet34"},
    "postprocessing": "operating point (threshold, min 3D component mm3, brain mask) maximising mean "
                      "per-scan score over all development scans (Dice on positive scans; 1 if empty "
                      "prediction else 0 on negative scans); nested leave-one-fold-out estimate "
                      "reported; frozen before the test sets are opened",
    "test_sets": "PhysioNet 15-scan held-out test and BHSD test, each evaluated once, after every "
                 "configuration and the post-processing recipe were frozen",
    "checkpoint": "final epoch; no early stopping; no validation-based checkpoint selection",
}


def preregistration(nnunet_epochs: dict | None = None) -> dict:
    E = arms()
    return {"base_recipe": BASE.to_dict(), "paper_recipe": PAPER.to_dict(),
            "arms": {e: {a: {"hash": c.hash(), "aug": c.aug, "describe": c.describe()}
                         for a, c in d.items()} for e, d in E.items()},
            "extra_seeds": {f"{e}/{a}": list(s) for (e, a), s in EXTRA_SEEDS.items()},
            "nnunet": {"configuration": "2d", "trainer": "nnUNetTrainer_ICH (fixed epochs)",
                       "datasets": {"P+B": "PhysioNet dev (60) + BHSD train", "P": "PhysioNet dev (60)"},
                       "epochs": nnunet_epochs or "set by calibration before fold 0 trains"},
            "decision_rules": DECISION_RULES, "primary_metric": PRIMARY_METRIC}


def write_or_check_prereg(path: Path, nnunet_epochs: dict | None = None) -> dict:
    path = Path(path)
    cur = preregistration(nnunet_epochs)
    if not path.exists():
        cur["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cur["amendments"] = []
        path.write_text(json.dumps(cur, indent=1))
        print(f"pre-registration written: {path}")
        return cur
    old = json.loads(path.read_text())
    changed = []
    for e, d in cur["arms"].items():
        for a, v in d.items():
            o = old["arms"].get(e, {}).get(a)
            if o is not None and o["hash"] != v["hash"]:
                changed.append(f"{e}/{a}")
    if changed:
        raise RuntimeError(f"arms changed since pre-registration: {changed}. Use amend() to record why.")
    if nnunet_epochs and old["nnunet"].get("epochs") != nnunet_epochs:
        if isinstance(old["nnunet"].get("epochs"), dict):
            raise RuntimeError(f"nnU-Net epochs already registered as {old['nnunet']['epochs']}")
        old["nnunet"]["epochs"] = nnunet_epochs
        old["amendments"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "what": f"nnU-Net epochs fixed by calibration: {nnunet_epochs}"})
        path.write_text(json.dumps(old, indent=1))
    return old


def amend(path: Path, note: str):
    old = json.loads(Path(path).read_text())
    cur = preregistration(old["nnunet"].get("epochs") if isinstance(old["nnunet"].get("epochs"), dict) else None)
    old["amendments"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "what": note,
                              "previous_arms": old["arms"]})
    old["arms"] = cur["arms"]
    Path(path).write_text(json.dumps(old, indent=1))


# ----------------------------------------------------------------------------- runner
def seeds_for(exp, arm):
    return EXTRA_SEEDS.get((exp, arm), (0,))


def keep_weights_for(cfg: RunConfig) -> bool:
    """Arms with identical configurations share runs, so weights are kept for a
    configuration if ANY arm using it needs them."""
    E = arms()
    return any(a in E.get(e, {}) and E[e][a].hash() == cfg.hash() for e, a in KEEP_WEIGHTS)


def weights_path(paths, exp, arm, seed, fold) -> Path:
    """Final weights of a run, following an alias to the run that actually trained."""
    r = Ledger(paths.ledger, exp).get(arm, seed, fold)
    if r is None:
        raise FileNotFoundError(f"{exp}/{arm} s{seed} f{fold} has not run")
    if r.get("alias_of"):
        src = Path(r["alias_of"])
        return paths.ckpt / src.parent.name / src.stem / "final.pt"
    return paths.ckpt / exp / f"{arm}__s{seed}_f{fold}" / "final.pt"


def _find_alias(paths, cfg_hash, seed, fold):
    """A finished run of an identical configuration in another experiment."""
    for p in paths.ledger.glob(f"*/*__s{seed}_f{fold}.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("config_hash") == cfg_hash and not r.get("alias_of"):
            return r, p
    return None, None


def run_arm(exp, arm, cfg: RunConfig, paths, index, cache, split, folds=range(SP.N_FOLDS), seeds=None,
            quick_mode=False, num_workers=2, log=print):
    from .train import finish_run, train_run
    keep = keep_weights_for(cfg)            # decided on the registered config, not the quick one
    if quick_mode:
        cfg = override(quick(cfg), encoder_weights=None)
    led = Ledger(paths.ledger, exp)
    seeds = seeds or seeds_for(exp, arm)
    for seed in seeds:
        for k in folds:
            if led.done(arm, seed, k, cfg.hash()):
                continue
            alias, src = _find_alias(paths, cfg.hash(), seed, k)
            if alias is not None:
                rec = dict(alias, alias_of=str(src))
                for key in ("experiment", "arm", "seed", "fold"):
                    rec.pop(key, None)
                led.put(arm, seed, k, rec)
                log(f"  {exp}/{arm} s{seed} f{k}: identical to {src.parent.name}/{src.stem}, reused")
                for name in ("scans", "slices", "lesions", "history"):
                    s_ = src.parent / "tables" / f"{src.stem}__{name}.csv"
                    if s_.exists():
                        d_ = led.dir / "tables" / f"{arm}__s{seed}_f{k}__{name}.csv"
                        d_.parent.mkdir(parents=True, exist_ok=True)
                        d_.write_bytes(s_.read_bytes())
                continue
            tr = SP.cases(split, partition="pn_dev", not_fold=k)
            va = SP.cases(split, partition="pn_dev", fold=k)
            log(f"  {exp}/{arm} s{seed} f{k}: {cfg.describe()}  ({len(tr)} train / {len(va)} val scans)")
            ck = paths.ckpt / exp / f"{arm}__s{seed}_f{k}"
            model, hist, info, z = train_run(cfg, index, cache, tr, va, seed, ck, num_workers, log)
            finish_run(model, cfg, hist, info, z, cache, va, led, arm, seed, k, paths,
                       keep_weights=keep, log=log)
            del model
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def run_experiment(exp, paths, index, cache, split, only=None, quick_mode=False, num_workers=2, log=print):
    for arm, cfg in arms()[exp].items():
        if only and arm not in only:
            continue
        run_arm(exp, arm, cfg, paths, index, cache, split, quick_mode=quick_mode,
                num_workers=num_workers, log=log)


def budget(paths) -> pd.DataFrame:
    """Remaining work, costed from MEASURED minutes per finished run."""
    from .ledger import all_ledgers
    done = all_ledgers(paths.ledger)
    mins = None
    if len(done) and "config_hash" in done.columns:
        rows = []
        for p in paths.ledger.glob("*/*.json"):
            r = json.loads(p.read_text())
            if not r.get("alias_of"):
                rows.append((r["config"]["image_size"], r["info"].get("train_seconds", np.nan) / 60))
        if rows:
            mins = pd.DataFrame(rows, columns=["size", "min"]).groupby("size")["min"].median().to_dict()
    out = []
    for exp, d in arms().items():
        for arm, cfg in d.items():
            n_total = len(seeds_for(exp, arm)) * SP.N_FOLDS
            led = Ledger(paths.ledger, exp)
            n_done = sum(led.get(arm, s, k) is not None for s in seeds_for(exp, arm) for k in range(SP.N_FOLDS))
            m = (mins or {}).get(cfg.image_size)
            out.append({"experiment": exp, "arm": arm, "runs": n_total, "done": n_done,
                        "min_per_run (measured)": round(m, 1) if m else None,
                        "hours_left": round((n_total - n_done) * m / 60, 1) if m else None})
    return pd.DataFrame(out)
