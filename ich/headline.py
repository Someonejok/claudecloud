"""Headline models: nnU-Net 2D on PhysioNet+BHSD ("P+B") and on PhysioNet only ("P")."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import nnunet_io as NN
from . import splits as SP
from .experiments import write_or_check_prereg
from .ledger import Ledger
from .train import evaluate_cases

DATASETS = {"P+B": (501, "PhysioBHSD"), "P": (502, "Physio")}


def experiment_name(which: str) -> str:
    return f"N_{which.replace('+', '')}"


def training_cases(split, which):
    tr = SP.cases(split, partition="pn_dev")
    if which == "P+B":
        tr += SP.cases(split, partition="bh_train")
    return tr


def nnunet_folds(split, which) -> list[dict]:
    tr = training_cases(split, which)
    return [{"train": [c for c in tr if split["fold"][c] != k], "val": [c for c in tr if split["fold"][c] == k]}
            for k in range(SP.N_FOLDS)]


def config_hash(which, epochs) -> str:
    return f"nnunet2d_{DATASETS[which][0]}_{int(epochs)}ep"


def run(paths, cache, split, which: str, local_pre: Path, hours_per_fold: float,
        epoch_bounds=(50, 400), folds=range(SP.N_FOLDS), quick_mode=False):
    did, name = DATASETS[which]
    NN.install_trainer()
    NN.export_dataset(paths, cache, did, name, training_cases(split, which))
    NN.plan_and_preprocess(paths, local_pre, did, name, nnunet_folds(split, which))
    prereg = paths.root / "preregistration.json"
    reg = json.loads(prereg.read_text()) if prereg.exists() else {}
    ep = (reg.get("nnunet", {}).get("epochs") or {})
    ep = ep if isinstance(ep, dict) else {}
    if which not in ep:
        if quick_mode:
            epochs = 1
        else:
            spe = NN.calibrate(paths, local_pre, did, name)
            epochs = int(np.clip(hours_per_fold * 3600 / spe, *epoch_bounds))
            print(f"  calibration: {spe:.1f} s/epoch -> {epochs} epochs per fold "
                  f"(~{epochs * spe / 3600:.1f} h/fold)")
        ep[which] = epochs
        write_or_check_prereg(prereg, ep)
    epochs = ep[which]
    led = Ledger(paths.ledger, experiment_name(which))
    for k in folds:
        d = NN.train_fold(paths, local_pre, did, name, k, epochs)
        if not led.done("nnunet2d", 0, k, config_hash(which, epochs)):
            score_fold(paths, cache, split, which, k, d, epochs, led)
    return epochs


def score_fold(paths, cache, split, which, k, fold_dir: Path, epochs, led):
    va = nnunet_folds(split, which)[k]["val"]
    probs = {c: NN.load_probability(fold_dir / "validation" / f"{c}.npz") for c in va}
    for c, p in probs.items():
        assert p.shape == cache.hu(c).shape, f"{c}: nnU-Net probability {p.shape} vs cache {cache.hu(c).shape}"
    tdir = led.dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    rec = {"config_hash": config_hash(which, epochs), "config": {"model": "nnU-Net 2d", "epochs": epochs,
                                                                 "dataset": which},
           "metrics": {}, "info": _training_info(fold_dir)}
    for src in ("physionet", "bhsd"):
        sub = {c: p for c, p in probs.items() if c.startswith("pn" if src == "physionet" else "bh_")}
        if not sub:
            continue
        S, SL, L = evaluate_cases(sub, cache, 0.5)
        tag = "" if src == "physionet" else "bhsd_"
        rec["metrics"].update({tag + kk: v for kk, v in M.summarise(S, SL, L).items()})
        key = f"nnunet2d__s0_f{k}" + ("" if src == "physionet" else "__bhsd")
        S.to_csv(tdir / f"{key}__scans.csv", index=False)
        SL.to_csv(tdir / f"{key}__slices.csv", index=False)
        L.to_csv(tdir / f"{key}__lesions.csv", index=False)
    led.put("nnunet2d", 0, k, rec)
    m = rec["metrics"]
    print(f"  {experiment_name(which)} fold {k}: PhysioNet volDice {m.get('dice_vol_mean', np.nan):.3f} "
          f"sliceDice {m.get('dice_slice_mean', np.nan):.3f} scanAUC {m.get('scan_auc', np.nan):.3f}"
          + (f" | BHSD volDice {m['bhsd_dice_vol_mean']:.3f}" if "bhsd_dice_vol_mean" in m else ""))


def _training_info(fold_dir: Path) -> dict:
    logs = sorted(Path(fold_dir).glob("training_log_*.txt"))
    if not logs:
        return {}
    import re
    txt = "\n".join(p.read_text(errors="ignore") for p in logs)
    t = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", txt)]
    return {"sec_per_epoch": float(np.median(t)) if t else np.nan,
            "train_seconds": float(np.sum(t)) if t else np.nan, "epochs_logged": len(t)}


def oof_probabilities(paths, split, which, epochs):
    """(case, fold, prob) for every development scan of this model."""
    did, name = DATASETS[which]
    for k, f in enumerate(nnunet_folds(split, which)):
        d = NN.fold_dir(paths, did, name, k)
        for c in f["val"]:
            yield c, k, NN.load_probability(d / "validation" / f"{c}.npz")


def test_probabilities(paths, cache, split, which, local_pre, epochs):
    did, name = DATASETS[which]
    test = SP.cases(split, partition=("pn_test", "bh_test"))
    in_dir = NN.export_test_images(paths, cache, test)
    out = NN.predict(paths, local_pre, did, name, in_dir, epochs=epochs)
    return {c: NN.load_probability(out / f"{c}.npz") for c in test}
