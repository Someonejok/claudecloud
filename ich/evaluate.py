"""Phase 4 (post-processing chosen on development data) and Phase 5 (test sets, once).

Models evaluated on the test sets:
    nnU-Net P+B   headline: nnU-Net 2D trained on PhysioNet dev + BHSD train, 5-fold ensemble
    nnU-Net P     nnU-Net 2D trained on PhysioNet dev only (BHSD test = external)
    paper         the original manuscript's U-Net and recipe, 5-fold ensemble
    aug_none / aug_base / aug_moderate / aug_extreme    custom U-Net, one per regime
The recipe for each model is chosen on its own development predictions and written to
disk before its test predictions are computed; a recipe file is never recomputed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import postproc as PP
from . import splits as SP
from . import stats as ST
from .config import SUBTYPES
from .experiments import weights_path
from .preprocess import brain_mask, skull_distance_mm
from .train import evaluate_cases, load_final, load_prob, predict_case, save_prob

CUSTOM = {"paper": ("E1_paper", "paper_recipe"), "aug_none": ("E4_augmentation", "none"),
          "aug_base": ("E4_augmentation", "base"), "aug_moderate": ("E4_augmentation", "paper_moderate"),
          "aug_extreme": ("E4_augmentation", "paper_extreme")}
NNUNET = {"nnU-Net P+B": "P+B", "nnU-Net P": "P"}


def brain_volume(cache, case) -> np.ndarray:
    hu = np.asarray(cache.hu(case))
    sp = cache.meta(case)["spacing"]
    return np.stack([brain_mask(hu[k], (sp[1], sp[2])) for k in range(hu.shape[0])])


# ----------------------------------------------------------------------------- probabilities
def _custom_models(paths, exp, arm, seed=0):
    out = []
    for k in range(SP.N_FOLDS):
        p = weights_path(paths, exp, arm, seed, k)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - {exp}/{arm} fold {k} must run with kept weights")
        out.append(p)
    return out


def oof(paths, cache, split, name, nn_epochs=None):
    """Yield (case, fold, prob) over the development scans of model `name`."""
    if name in NNUNET:
        from .headline import oof_probabilities
        yield from oof_probabilities(paths, split, NNUNET[name], nn_epochs)
        return
    exp, arm = CUSTOM[name]
    for k, wp in enumerate(_custom_models(paths, exp, arm)):
        cache_dir = paths.probs / "oof" / name / f"f{k}"
        model = None
        for c in SP.cases(split, partition="pn_dev", fold=k):
            f = cache_dir / f"{c}.npz"
            if not f.exists():
                if model is None:
                    model, cfg, z = load_final(wp)
                save_prob(f, predict_case(model, cfg, cache, c, z))
            yield c, k, load_prob(f)


def test_probs(paths, cache, split, name, local_pre=None, nn_epochs=None) -> dict:
    test = SP.cases(split, partition=("pn_test", "bh_test"))
    d = paths.probs / "test" / name
    if all((d / f"{c}.npz").exists() for c in test):
        return {c: load_prob(d / f"{c}.npz") for c in test}
    if name in NNUNET:
        from .headline import test_probabilities
        probs = test_probabilities(paths, cache, split, NNUNET[name], local_pre, nn_epochs)
    else:
        exp, arm = CUSTOM[name]
        acc = {c: None for c in test}
        wps = _custom_models(paths, exp, arm)
        for wp in wps:
            model, cfg, z = load_final(wp)
            for c in test:
                p = predict_case(model, cfg, cache, c, z)
                acc[c] = p if acc[c] is None else acc[c] + p
        probs = {c: v / len(wps) for c, v in acc.items()}
    for c, p in probs.items():
        save_prob(d / f"{c}.npz", p)
    return probs


# ----------------------------------------------------------------------------- phase 4
def select_postprocessing(paths, cache, split, name, nn_epochs=None) -> dict:
    out = paths.tables / f"postproc_{_slug(name)}.json"
    if out.exists():
        return json.loads(out.read_text())
    rows = []
    for c, k, prob in oof(paths, cache, split, name, nn_epochs):
        rows.append(PP.sweep_scan(prob, np.asarray(cache.lab(c)), cache.meta(c)["spacing"],
                                  brain_volume(cache, c), c, k))
    sw = pd.concat(rows, ignore_index=True)
    sw.to_csv(paths.tables / f"postproc_sweep_{_slug(name)}.csv", index=False)
    per, summ = PP.nested_selection(sw)
    per.to_csv(paths.tables / f"postproc_nested_{_slug(name)}.csv", index=False)
    summ["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summ["dev_scans"] = int(sw.case.nunique())
    out.write_text(json.dumps(summ, indent=1))
    print(f"  {name}: recipe {summ['recipe']} | nested dev score {summ['nested_score']:.3f} "
          f"(default {summ['default_score']:.3f}, optimism {summ['selection_optimism']:+.3f})")
    return summ


# ----------------------------------------------------------------------------- phase 5
def fp_near_skull(probs_bin: dict, cache, near_mm=3.0) -> pd.DataFrame:
    """For every false-positive predicted component: how much of it lies within
    `near_mm` of bone (the partial-volume failure the manuscript described)."""
    rows = []
    for c, pr in probs_bin.items():
        gt = np.asarray(cache.lab(c)) > 0
        lab, n = M.components(pr)
        if not n:
            continue
        hit = np.bincount(lab[gt], minlength=n + 1)
        hu = np.asarray(cache.hu(c))
        sp = cache.meta(c)["spacing"]
        for i in np.where(hit[1:] == 0)[0] + 1:
            vox = np.argwhere(lab == i)
            zs = np.unique(vox[:, 0])
            near = 0
            for z in zs:
                dmap = skull_distance_mm(hu[z], (sp[1], sp[2]))
                v = vox[vox[:, 0] == z]
                near += int((dmap[v[:, 1], v[:, 2]] <= near_mm).sum())
            rows.append({"case": c, "voxels": len(vox), "volume_ml": len(vox) * np.prod(sp) / 1000,
                         "frac_within_3mm_of_bone": near / len(vox)})
    return pd.DataFrame(rows)


def test_evaluation(paths, cache, split, index, name, local_pre=None, nn_epochs=None) -> dict:
    """Evaluate one model on both test sets, once. The recipe must already be frozen."""
    out = paths.tables / f"test_{_slug(name)}.json"
    if out.exists():
        return json.loads(out.read_text())
    rec_p = paths.tables / f"postproc_{_slug(name)}.json"
    if not rec_p.exists():
        raise RuntimeError(f"post-processing for {name} is not frozen yet - run Phase 4 first")
    recipe = json.loads(rec_p.read_text())["recipe"]
    probs = test_probs(paths, cache, split, name, local_pre, nn_epochs)
    res = {"model": name, "recipe": recipe, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tdir = paths.tables / f"test_{_slug(name)}"
    tdir.mkdir(parents=True, exist_ok=True)
    for part, label in (("pn_test", "physionet"), ("bh_test", "bhsd")):
        cs = SP.cases(split, partition=part)
        if not cs:
            continue
        sub = {c: probs[c] for c in cs}
        preds = {c: PP.apply(p, recipe["threshold"], recipe["min_mm3"],
                             brain_volume(cache, c) if recipe["brain_mask"] else None,
                             cache.meta(c)["spacing"]) for c, p in sub.items()}
        for tag, pr in (("default", None), ("recipe", preds)):
            S, SL, L = evaluate_cases(sub, cache, 0.5, pr)
            s = M.summarise(S, SL, L)
            pos = S[S.positive]
            for col, key in (("dice", "dice_vol_mean"), ("iou", "iou_vol_mean")):
                s[key + "_ci95"] = ST.cluster_bootstrap(pos, col)
            s["dice_slice_mean_ci95"] = ST.cluster_bootstrap(SL[SL.gt_mm2 > 0], "dice")
            res[f"{label}_{tag}"] = s
            S.to_csv(tdir / f"{label}_{tag}_scans.csv", index=False)
            SL.to_csv(tdir / f"{label}_{tag}_slices.csv", index=False)
            L.to_csv(tdir / f"{label}_{tag}_lesions.csv", index=False)
            if tag == "recipe":
                M.size_strata(SL, L).to_csv(tdir / f"{label}_size_strata.csv", index=False)
                M.subtype_strata(S, SL, L, index, list(SUBTYPES)).to_csv(
                    tdir / f"{label}_subtype_strata.csv", index=False)
                fp = fp_near_skull(preds, cache)
                fp.to_csv(tdir / f"{label}_fp_components.csv", index=False)
                res[f"{label}_fp_near_bone_fraction"] = float((fp.frac_within_3mm_of_bone > 0.5).mean()) \
                    if len(fp) else np.nan
    out.write_text(json.dumps(res, indent=1, default=float))
    h = res.get("physionet_recipe", {})
    print(f"  {name}: PhysioNet test volDice {h.get('dice_vol_mean', np.nan):.3f} "
          f"CI {h.get('dice_vol_mean_ci95')} | scan AUC {h.get('scan_auc', np.nan):.3f}"
          + (f" | BHSD volDice {res['bhsd_recipe']['dice_vol_mean']:.3f}" if "bhsd_recipe" in res else ""))
    return res


def _slug(name):
    return name.replace(" ", "_").replace("+", "plus").replace("-", "")
