"""Extensions beyond the reviewer-driven v3 pipeline (November 2026).

STATUS: written without being executed (the author asked that no code be run in the
session that produced it). Every function here is additive: nothing in the existing
pipeline imports this module, so the pre-registered arms and their hashes are
unchanged. Run `pytest tests/` and the QUICK notebook path before trusting any output.

X1  leakage_demo        quantify how much the original slice-level protocol inflated
                        the result: the same recipe scored on held-out *slices* versus
                        on unseen *scans* (Yagis et al. 2021; Wen et al. 2020).
X2  affinity_diversity  measure, instead of asserting via the "manifold hypothesis", how
                        far each augmentation policy moves data from the clean
                        distribution (affinity) and how much it raises training
                        difficulty (diversity) (Gontijo-Lopes et al., ICLR 2021).
X3  volumetry           clinically meaningful agreement of haemorrhage volume (ICC,
                        Bland-Altman, agreement at the 30 mL epidural surgical threshold
                        of Bullock et al. 2006).
X4  fold_uncertainty    5-fold ensemble disagreement as a failure flag, plus voxel
                        calibration (ECE, reliability curve).
X5  run_scaling         data-scaling curve (25/50/75/100 % of training scans, and
                        +BHSD) for the custom U-Net: is data the binding constraint?
X6  augmentation_shift  in-domain (PhysioNet) versus external (BHSD) effect of each
                        augmentation regime: does strong augmentation that hurts
                        in-domain help out of domain (Zhang et al. 2020, BigAug)?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import splits as SP
from .config import override
from .experiments import PAPER, arms, weights_path
from .ledger import Ledger


# ============================================================================ X1 leakage
def slice_level_split(index: pd.DataFrame, dev_cases, seed: int, test_frac=0.2,
                      undersample_first=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the ORIGINAL protocol on purpose: 1:1 undersampling of negative slices
    over the whole pool, then a random slice-level 80/20 split stratified by lesion
    presence. Slices of one scan end up on both sides."""
    rows = index[index.case.isin(dev_cases)].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    pos = rows[rows.has_lesion == 1]
    neg = rows[rows.has_lesion == 0]
    if undersample_first:
        neg = neg.iloc[rng.choice(len(neg), size=min(len(neg), len(pos)), replace=False)]
    pool = pd.concat([pos, neg])
    te_idx = []
    for _, g in pool.groupby("has_lesion"):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        te_idx += list(idx[:int(round(len(idx) * test_frac))])
    te = pool.loc[te_idx]
    tr = pool.drop(index=te_idx)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def _slice_scores(probs: dict, cache, keep: dict[str, set] | None, thr=0.5) -> dict:
    """Per-slice lesion Dice / IoU and the original class-averaged BinaryIoU, restricted
    to `keep[case]` slice indices (None = all slices). Also the tf.keras BinaryIoU
    exactly as the original code computed it: confusion counts pooled over every pixel
    of every evaluated slice, IoU per class, then the mean of the two classes."""
    sl_all = []
    for c, p in probs.items():
        _, sl, _ = M.evaluate_scan(p, np.asarray(cache.lab(c)), cache.meta(c)["spacing"], c, thr)
        if keep is not None:
            sl = sl[sl.slice.isin(keep.get(c, set()))]
        sl_all.append(sl)
    SL = pd.concat(sl_all, ignore_index=True)
    tp, fp, fn = SL.tp.sum(), SL.fp.sum(), SL.fn.sum()
    # pixels per slice differ by scan; recover tn per slice from the grid size
    n_px = np.array([np.prod(cache.lab(c).shape[1:]) for c in SL.case])
    tn = (n_px - SL.tp - SL.fp - SL.fn).sum()
    iou_fg, iou_bg = tp / max(tp + fp + fn, 1), tn / max(tn + fp + fn, 1)
    pos = SL[SL.gt_mm2 > 0]
    return {"slices": len(SL), "pos_slices": len(pos),
            "dice_slice_mean": float(pos.dice.mean()), "iou_slice_mean": float(pos.fg_iou.mean()),
            "binary_iou_tf_pooled": float(0.5 * (iou_fg + iou_bg)), "lesion_iou_pooled": float(iou_fg),
            "binary_iou_classavg_slice_mean": float(SL.binary_iou_classavg.mean())}


def leakage_demo(paths, index, cache, split, seeds=(0, 1, 2), quick_mode=False, num_workers=2, log=print):
    """For each seed: train the original recipe on a slice-level split of the 60
    development scans; score it (a) on its held-out slices - the original, leaky
    estimate - and (b) on the 15 untouched PhysioNet test scans, whose patients it never
    saw. The gap between (a) and (b), with the clean scan-level estimate from E1 beside
    it, is the size of the leak. The PhysioNet test set is opened here only for a model
    that is NOT a candidate for any headline result and whose settings are fixed, so it
    does not influence any decision; state this in the methods."""
    from .config import quick
    from .train import evaluate_cases, predict_case, train_run
    out = paths.tables / "X1_leakage_demo.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    cfg = override(quick(PAPER), encoder_weights=None) if quick_mode else PAPER
    dev = SP.cases(split, partition="pn_dev")
    test = SP.cases(split, partition="pn_test")
    rows = [] if not len(done) else done.to_dict("records")
    for seed in seeds:
        if len(done) and (done.seed == seed).any():
            continue
        tr, te = slice_level_split(index, dev, seed)
        shared = sorted(set(tr.case) & set(te.case))
        log(f"  X1 seed {seed}: {len(tr)} train / {len(te)} held-out slices; "
            f"{len(shared)} of {len(set(te.case))} held-out scans also contribute training slices")
        ck = paths.ckpt / "X1_leakage" / f"s{seed}"
        # train_run selects rows with index.case in train_cases, so pass the training
        # slices as the index; the empty validation list disables the curve
        model, hist, info, z = train_run(cfg, tr, cache, sorted(set(tr.case)), [], seed, ck,
                                         num_workers, log)
        keep = {c: set(g.slice.astype(int)) for c, g in te.groupby("case")}
        p_te = {c: predict_case(model, cfg, cache, c, z) for c in keep}
        leaky = _slice_scores(p_te, cache, keep)
        p_test = {c: predict_case(model, cfg, cache, c, z) for c in test}
        clean = _slice_scores(p_test, cache, None)
        S, _, _ = evaluate_cases(p_test, cache, 0.5)
        r = {"seed": seed, "heldout_scans_shared_with_train": len(shared),
             "heldout_scans": len(set(te.case))}
        r.update({f"leaky_{k}": v for k, v in leaky.items()})
        r.update({f"unseen_{k}": v for k, v in clean.items()})
        r["unseen_dice_vol_mean"] = float(S[S.positive].dice.mean())
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        last = ck / "last.pt"
        if last.exists():
            last.unlink()
        del model
    return pd.DataFrame(rows)


# ============================================================================ X2 affinity / diversity
def affinity_diversity(paths, index, cache, split, ref_arm="base", n_draws=3, max_slices=400,
                       seed=0, log=print) -> pd.DataFrame:
    """Affinity (Gontijo-Lopes et al. 2021), adapted to segmentation: take the model
    trained WITHOUT the policy (flip-only `base`, fold k) and score its own validation
    fold twice - clean, and with the policy applied to image and mask. Affinity is
    Dice(augmented) - Dice(clean): 0 means the policy stays on the data distribution as
    the model sees it, strongly negative means the augmented slices look unlike real
    data. Diversity is the policy's final-epoch training loss relative to `base`
    (higher = harder, more varied training data). Both are measured, so the extreme
    regime's failure can be described with numbers rather than with the manifold
    hypothesis (reviewer major 12, recommended 8)."""
    import torch
    from .models import normalise_input
    from .train import SliceData, device, load_final

    E4 = arms()["E4_augmentation"]
    led = Ledger(paths.ledger, "E4_augmentation")
    dev = device()
    rows = []
    for k in range(SP.N_FOLDS):
        model, cfg, z = load_final(weights_path(paths, "E4_augmentation", ref_arm, 0, k))
        va = index[index.case.isin(SP.cases(split, partition="pn_dev", fold=k)) & (index.has_lesion == 1)]
        if len(va) > max_slices:
            va = va.sample(max_slices, random_state=seed)
        va = va.reset_index(drop=True)
        for arm, acfg in E4.items():
            spec = acfg.aug
            ds = SliceData(va, cache, cfg, spec, z)
            items = [(i, -1) for i in range(len(va))] if spec == "none" else \
                [(i, int(s)) for d in range(n_draws)
                 for i, s in enumerate(np.random.default_rng([seed, k, d]).integers(0, 2**31 - 1, len(va)))]
            tp = fp = fn = 0.0
            per = []
            with torch.no_grad():
                for j in range(0, len(items), 16):
                    batch = [ds[it] for it in items[j:j + 16]]      # rendered lazily, 16 at a time
                    xb = torch.stack([b[0] for b in batch]); yb = torch.stack([b[1] for b in batch])
                    lg = model(normalise_input(xb.to(dev), cfg))
                    pb = (torch.sigmoid(lg.float()) >= 0.5).float().cpu()
                    t = (pb * yb).sum((1, 2, 3)); f_ = (pb * (1 - yb)).sum((1, 2, 3))
                    n_ = ((1 - pb) * yb).sum((1, 2, 3))
                    per += (2 * t / (2 * t + f_ + n_).clamp(min=1)).tolist()
                    tp += float(t.sum()); fp += float(f_.sum()); fn += float(n_.sum())
            rows.append({"fold": k, "arm": arm, "aug": spec, "dice_slice_mean": float(np.mean(per)),
                         "dice_pooled": 2 * tp / max(2 * tp + fp + fn, 1)})
        log(f"  X2 fold {k}: {len(va)} lesion slices x {len(E4)} policies")
        del model
    A = pd.DataFrame(rows)
    clean = A[A.arm == "none"].set_index("fold").dice_slice_mean
    A["affinity"] = A.dice_slice_mean - A.fold.map(clean)
    # diversity from the training histories already on disk
    div = []
    for arm in E4:
        for k in range(SP.N_FOLDS):
            h = led.dir / "tables" / f"{arm}__s0_f{k}__history.csv"
            if h.exists():
                div.append({"arm": arm, "fold": k, "final_train_loss": float(pd.read_csv(h).train_loss.iloc[-1])})
    D = pd.DataFrame(div)
    if len(D):
        ref = D[D.arm == ref_arm].set_index("fold").final_train_loss
        D["diversity"] = D.final_train_loss / D.fold.map(ref)
        A = A.merge(D, on=["arm", "fold"], how="left")
    A.to_csv(paths.tables / "X2_affinity_diversity.csv", index=False)
    summ = A.groupby("arm")[["affinity", "diversity"]].mean()
    perf = _arm_perf(paths, "E4_augmentation")
    if perf is not None:
        summ = summ.join(perf)
    summ.to_csv(paths.tables / "X2_affinity_diversity_summary.csv")
    return summ


def _arm_perf(paths, exp, metric="dice_vol_mean"):
    led = Ledger(paths.ledger, exp)
    rows = []
    for p in led.dir.glob("*__s*_f*.json"):
        r = json.loads(p.read_text())
        arm = p.stem.split("__")[0]
        if "metrics" in r:
            rows.append({"arm": arm, metric: r["metrics"].get(metric)})
    return pd.DataFrame(rows).groupby("arm")[metric].mean().to_frame() if rows else None


# ============================================================================ X3 volumetry
def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1): two-way random effects, absolute agreement, single measurement
    (McGraw & Wong 1996), for two raters (reference, model)."""
    d = np.column_stack([x, y]).astype(float)
    n, k = d.shape
    if n < 2:
        return np.nan
    gm = d.mean()
    msr = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def volumetry(scans: pd.DataFrame, threshold_ml=30.0) -> dict:
    """`scans` is a per-scan table from evaluate_cases (columns gt_ml, pred_ml, positive).
    Uses haemorrhage-positive scans for agreement and all scans for the threshold table.
    The 30 mL cut is the epidural-haematoma surgical guideline (Bullock et al. 2006); on
    PhysioNet the mask pools subtypes, so report it as total-volume agreement, and as an
    EDH-specific analysis only on BHSD where voxel subtype labels exist."""
    from scipy.stats import spearmanr
    pos = scans[scans.positive]
    g, p = pos.gt_ml.values, pos.pred_ml.values
    diff = p - g
    big_g, big_p = scans.gt_ml >= threshold_ml, scans.pred_ml >= threshold_ml
    return {"n_pos_scans": len(pos), "icc_a1": icc_a1(g, p),
            "spearman_r": float(spearmanr(g, p).correlation) if len(pos) > 2 else np.nan,
            "bland_altman_bias_ml": float(diff.mean()) if len(pos) else np.nan,
            "bland_altman_loa_ml": [float(diff.mean() - 1.96 * diff.std(ddof=1)),
                                    float(diff.mean() + 1.96 * diff.std(ddof=1))] if len(pos) > 1 else None,
            "median_abs_error_ml": float(np.median(np.abs(diff))) if len(pos) else np.nan,
            f"scans_ref_ge_{threshold_ml:g}ml": int(big_g.sum()),
            f"agree_at_{threshold_ml:g}ml": int((big_g == big_p).sum()),
            f"sens_at_{threshold_ml:g}ml": float((big_g & big_p).sum() / big_g.sum()) if big_g.any() else np.nan,
            f"spec_at_{threshold_ml:g}ml": float((~big_g & ~big_p).sum() / (~big_g).sum()) if (~big_g).any() else np.nan}


def fig_bland_altman(scans: pd.DataFrame, out: Path, title=""):
    import matplotlib.pyplot as plt
    pos = scans[scans.positive]
    m, d = (pos.gt_ml + pos.pred_ml) / 2, pos.pred_ml - pos.gt_ml
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
    ax.scatter(m, d, s=14, color="#2a6f97", alpha=0.85)
    b, s = d.mean(), d.std(ddof=1)
    for v, ls in ((b, "-"), (b - 1.96 * s, "--"), (b + 1.96 * s, "--")):
        ax.axhline(v, color="#555555", lw=0.9, ls=ls)
    ax.set_xlabel("mean of reference and predicted volume (mL)")
    ax.set_ylabel("predicted - reference (mL)")
    ax.set_title(title, fontsize=9)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ============================================================================ X4 uncertainty / calibration
def fold_uncertainty(paths, cache, split, exp="E4_augmentation", arm="base", partition="pn_test",
                     thr=0.5, n_bins=10, max_vox=2_000_000, seed=0) -> tuple[pd.DataFrame, dict]:
    """Per test scan: mean ensemble probability, fold disagreement (voxel SD across the
    five fold models, averaged inside the predicted-or-reference region), and the
    scan's Dice. Reports whether disagreement ranks the failures (Spearman rho of
    disagreement vs Dice; AUROC of disagreement for Dice < 0.5), plus expected
    calibration error of the ensemble on voxels inside the head."""
    from scipy.stats import spearmanr
    from .train import load_final, predict_case
    test = SP.cases(split, partition=partition)
    per_fold = {c: [] for c in test}
    for k in range(SP.N_FOLDS):
        model, cfg, z = load_final(weights_path(paths, exp, arm, 0, k))
        for c in test:
            per_fold[c].append(predict_case(model, cfg, cache, c, z).astype(np.float16))
        del model
    rng = np.random.default_rng(seed)
    rows, cal_p, cal_y = [], [], []
    for c in test:
        P = np.stack(per_fold[c]).astype(np.float32)
        mean, sd = P.mean(0), P.std(0)
        gt = np.asarray(cache.lab(c)) > 0
        pr = mean >= thr
        roi = gt | pr
        tp, fp, fn = (gt & pr).sum(), (~gt & pr).sum(), (gt & ~pr).sum()
        rows.append({"case": c, "positive": bool(gt.any()),
                     "dice": 2 * tp / (2 * tp + fp + fn) if gt.any() else np.nan,
                     "disagreement_roi": float(sd[roi].mean()) if roi.any() else 0.0,
                     "disagreement_max": float(sd.max())})
        head = np.asarray(cache.hu(c)) > -200
        idx = np.flatnonzero(head)
        if len(idx) > max_vox // max(len(test), 1):
            idx = rng.choice(idx, max_vox // max(len(test), 1), replace=False)
        cal_p.append(mean.ravel()[idx]); cal_y.append(gt.ravel()[idx])
    U = pd.DataFrame(rows)
    p, y = np.concatenate(cal_p), np.concatenate(cal_y)
    bins = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rel = [{"bin_lo": bins[i], "bin_hi": bins[i + 1], "n": int((b == i).sum()),
            "mean_p": float(p[b == i].mean()) if (b == i).any() else np.nan,
            "frac_pos": float(y[b == i].mean()) if (b == i).any() else np.nan} for i in range(n_bins)]
    R = pd.DataFrame(rel)
    ece = float(np.nansum(R.n * np.abs(R.mean_p - R.frac_pos)) / R.n.sum())
    # positive-voxel-weighted ECE: the plain ECE is dominated by easy background voxels
    lesion_bins = R[R.bin_lo >= 0.1]
    ece_hi = float(np.nansum(lesion_bins.n * np.abs(lesion_bins.mean_p - lesion_bins.frac_pos))
                   / max(lesion_bins.n.sum(), 1))
    pos = U[U.positive]
    out = {"ece_all_head_voxels": ece, "ece_prob_ge_0.1": ece_hi,
           "spearman_disagreement_vs_dice": float(spearmanr(pos.disagreement_roi, pos.dice).correlation)
           if len(pos) > 2 else np.nan,
           "auroc_disagreement_flags_dice_lt_0.5": M.roc_auc((pos.dice < 0.5).values, pos.disagreement_roi.values)}
    U.to_csv(paths.tables / f"X4_uncertainty_{exp}_{arm}_{partition}.csv", index=False)
    R.to_csv(paths.tables / f"X4_reliability_{exp}_{arm}_{partition}.csv", index=False)
    return U, out


# ============================================================================ X5 data scaling
SCALING_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _subsample_scans(index, cases, frac, seed):
    """Stratified by scan positivity so every fraction keeps haemorrhage-positive scans."""
    if frac >= 1.0:
        return sorted(cases)
    pos = index[index.case.isin(cases)].groupby("case").has_lesion.max()
    rng = np.random.RandomState(seed)
    out = []
    for v in (1, 0):
        pool = sorted(pos[pos == v].index)
        rng.shuffle(pool)
        out += pool[:max(1, int(round(len(pool) * frac)))]
    return sorted(out)


def run_scaling(paths, index, cache, split, fractions=SCALING_FRACTIONS, add_bhsd=True, quick_mode=False,
                num_workers=2, log=print):
    """Custom U-Net (BASE recipe) on a fraction of each fold's PhysioNet training scans,
    scored on the same full validation fold, so the learning curve is paired by fold.
    `+BHSD` adds the fold-matched BHSD training scans to the full PhysioNet set. Ledger
    experiment `X5_scaling`; not part of the original pre-registration, so report it as
    an exploratory extension (or amend the pre-registration before running)."""
    from .config import quick
    from .experiments import BASE
    from .train import finish_run, train_run
    cfg = override(quick(BASE), encoder_weights=None) if quick_mode else BASE
    led = Ledger(paths.ledger, "X5_scaling")
    plan = [(f"frac{f:g}", f, False) for f in fractions] + ([("frac1+BHSD", 1.0, True)] if add_bhsd else [])
    for arm, frac, bh in plan:
        for k in range(SP.N_FOLDS):
            if led.done(arm, 0, k, cfg.hash()):
                continue
            tr = _subsample_scans(index, SP.cases(split, partition="pn_dev", not_fold=k), frac, seed=k)
            if bh:
                tr += SP.cases(split, partition="bh_train", not_fold=k)
            va = SP.cases(split, partition="pn_dev", fold=k)
            log(f"  X5 {arm} f{k}: {len(tr)} train scans / {len(va)} val")
            ck = paths.ckpt / "X5_scaling" / f"{arm}__s0_f{k}"
            model, hist, info, z = train_run(cfg, index, cache, tr, va, 0, ck, num_workers, log)
            info["train_scans"] = len(tr)
            finish_run(model, cfg, hist, info, z, cache, va, led, arm, 0, k, paths, keep_weights=False, log=log)
            del model


# ============================================================================ X6 in-domain vs external
def augmentation_shift(paths, cache, split, index=None) -> pd.DataFrame:
    """For the four regimes whose fold weights are kept (none / flip-only / original
    moderate / original extreme), compare test performance on PhysioNet (in-domain) with
    BHSD (external, for these PhysioNet-only models). Requires evaluate.test_evaluation
    to have run for aug_* (it writes test_aug_*.json). Reports the in-domain minus
    external gap per regime: if stronger augmentation narrows the gap while lowering
    in-domain Dice, the original 'extreme is worse' conclusion is domain-dependent."""
    rows = []
    for name in ("aug_none", "aug_base", "aug_moderate", "aug_extreme"):
        f = paths.tables / f"test_{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        pn, bh = r.get("physionet_recipe", {}), r.get("bhsd_recipe", {})
        rows.append({"regime": name, "physionet_dice_vol": pn.get("dice_vol_mean"),
                     "physionet_ci95": pn.get("dice_vol_mean_ci95"),
                     "bhsd_dice_vol": bh.get("dice_vol_mean"), "bhsd_ci95": bh.get("dice_vol_mean_ci95"),
                     "gap_in_minus_ext": (pn.get("dice_vol_mean") or np.nan) - (bh.get("dice_vol_mean") or np.nan)})
    D = pd.DataFrame(rows)
    D.to_csv(paths.tables / "X6_augmentation_domain_shift.csv", index=False)
    return D
