"""Metrics, with every definition fixed here and quoted in the paper.

All metrics are computed on the NATIVE voxel grid of each scan, lesion (foreground)
class only, from a binary prediction obtained by thresholding the probability map at a
stated threshold (and optional stated post-processing).

Headline:  per-scan volumetric Dice, averaged over ICH-positive scans.
Secondary: micro (pooled-voxel) Dice/IoU over all scans; per-slice Dice/IoU averaged over
           lesion-positive slices (the earlier, harsher protocol); precision, recall and
           specificity (pooled voxels); lesion-wise sensitivity (a ground-truth 3D
           component counts as detected if >= 10 % of its voxels are predicted) and false-
           positive components per scan; HD95 and ASSD in mm; normalised surface Dice at
           2 mm; scan-level detection (positive call = any predicted component; ROC-AUC
           from the maximum probability in the scan); slice-level AUC.
Empty-mask rules: Dice is undefined on scans/slices with an empty ground truth and is
not averaged there (their false positives are counted by specificity, FP components
and the detection metrics). Surface distances are undefined if either mask is empty;
the number of such scans is reported next to the mean.
The class-averaged "binary IoU" of the original manuscript ((IoU_lesion + IoU_bg) / 2,
what tf.keras.metrics.BinaryIoU returns by default) is computed once for contrast only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

EPS = 1e-9
HIT_FRACTION = 0.10
NSD_TOL_MM = 2.0
STRUCT3D = ndi.generate_binary_structure(3, 3)      # 26-connectivity


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, structure=ndi.generate_binary_structure(m.ndim, 1),
                                   border_value=0)


def surface_metrics(gt: np.ndarray, pr: np.ndarray, spacing) -> dict:
    """HD95, ASSD (mm) and NSD at NSD_TOL_MM. Undefined (nan) if either mask is empty."""
    if not gt.any() or not pr.any():
        return {"hd95_mm": np.nan, "assd_mm": np.nan, "nsd": np.nan}
    sg, sp = _surface(gt), _surface(pr)
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dt_g[sp], dt_p[sg]           # pred surface -> gt surface, and back
    d = np.concatenate([d_pg, d_gp])
    nsd = ((d_pg <= NSD_TOL_MM).sum() + (d_gp <= NSD_TOL_MM).sum()) / (len(d_pg) + len(d_gp))
    return {"hd95_mm": float(np.percentile(d, 95)), "assd_mm": float(d.mean()), "nsd": float(nsd)}


def components(m: np.ndarray):
    lab, n = ndi.label(m, structure=STRUCT3D if m.ndim == 3 else None)
    return lab, n


def lesion_level(gt: np.ndarray, pr: np.ndarray, spacing, gt_codes: np.ndarray | None = None):
    """Per ground-truth component: volume, detected?, subtype (if labels carry it).
    Returns (component rows, number of false-positive predicted components)."""
    gl, gn = components(gt)
    pl, pn = components(pr)
    vox_ml = float(np.prod(spacing)) / 1000.0
    rows = []
    if gn:
        sizes = np.bincount(gl.ravel(), minlength=gn + 1)
        cov = np.bincount(gl[pr], minlength=gn + 1)
        for i in range(1, gn + 1):
            r = {"volume_ml": sizes[i] * vox_ml, "detected": bool(cov[i] / sizes[i] >= HIT_FRACTION),
                 "covered_fraction": float(cov[i] / sizes[i])}
            if gt_codes is not None:
                codes = gt_codes[gl == i]
                codes = codes[codes > 0]
                r["code"] = int(np.bincount(codes).argmax()) if codes.size else 0
            rows.append(r)
    fp = 0
    if pn:
        hit = np.bincount(pl[gt], minlength=pn + 1)
        fp = int((hit[1:] == 0).sum())
    return rows, fp


def evaluate_scan(prob: np.ndarray, gt_codes: np.ndarray, spacing, case: str,
                  thr: float = 0.5, pred: np.ndarray | None = None, codes_are_subtypes: bool = False):
    """prob (Z,H,W) float, gt_codes (Z,H,W) uint8 (>0 = lesion; 1..5 subtype codes if
    available). `pred` overrides thresholding (post-processed binary mask).
    codes_are_subtypes: True only for BHSD, whose label values are subtype codes;
    PhysioNet stores every lesion as 1, which must not be read as a subtype.
    Returns (scan_row, slice_rows DataFrame, lesion_rows list)."""
    gt = gt_codes > 0
    pr = (prob >= thr) if pred is None else pred.astype(bool)
    tp = int((gt & pr).sum()); fp = int((~gt & pr).sum()); fn = int((gt & ~pr).sum())
    tn = int(gt.size - tp - fp - fn)
    vox_ml = float(np.prod(spacing)) / 1000.0
    pos = bool(gt.any())
    les, fp_comp = lesion_level(gt, pr, spacing, gt_codes if codes_are_subtypes else None)
    scan = {"case": case, "positive": pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "gt_ml": gt.sum() * vox_ml, "pred_ml": pr.sum() * vox_ml,
            "dice": 2 * tp / (2 * tp + fp + fn) if pos else np.nan,
            "iou": tp / (tp + fp + fn) if pos else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if pos else np.nan,
            "gt_lesions": len(les), "detected_lesions": int(sum(r["detected"] for r in les)),
            "fp_lesions": fp_comp, "max_prob": float(prob.max()),
            "called_positive": bool(pr.any())}
    scan.update(surface_metrics(gt, pr, spacing) if pos else
                {"hd95_mm": np.nan, "assd_mm": np.nan, "nsd": np.nan})
    # per slice
    g2, p2 = gt.reshape(gt.shape[0], -1), pr.reshape(pr.shape[0], -1)
    stp = (g2 & p2).sum(1); sfp = (~g2 & p2).sum(1); sfn = (g2 & ~p2).sum(1)
    stn = g2.shape[1] - stp - sfp - sfn
    gpx = g2.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fg_iou = np.where(gpx > 0, stp / (stp + sfp + sfn), np.nan)
        dice = np.where(gpx > 0, 2 * stp / (2 * stp + sfp + sfn), np.nan)
        # class-averaged "binary IoU" (original manuscript), defined on every slice
        u_fg = stp + sfp + sfn
        iou_fg_all = np.where(u_fg > 0, stp / np.maximum(u_fg, 1), 1.0)
        iou_bg = stn / np.maximum(stn + sfp + sfn, 1)
    px_mm2 = spacing[1] * spacing[2]
    sl = pd.DataFrame({"case": case, "slice": np.arange(gt.shape[0]),
                       "gt_mm2": gpx * px_mm2, "pred_mm2": p2.sum(1) * px_mm2,
                       "tp": stp, "fp": sfp, "fn": sfn, "fg_iou": fg_iou, "dice": dice,
                       "binary_iou_classavg": 0.5 * (iou_fg_all + iou_bg),
                       "max_prob": prob.reshape(prob.shape[0], -1).max(1)})
    for r in les:
        r["case"] = case
    return scan, sl, les


# ----------------------------------------------------------------------------- aggregation
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summarise(scans: pd.DataFrame, slices: pd.DataFrame, lesions: pd.DataFrame) -> dict:
    pos = scans[scans.positive]
    neg = scans[~scans.positive]
    tp, fp, fn, tn = (scans[c].sum() for c in ("tp", "fp", "fn", "tn"))
    sp = slices[slices.gt_mm2 > 0]
    out = {
        "n_scans": len(scans), "n_pos_scans": len(pos), "n_slices": len(slices),
        "n_pos_slices": len(sp),
        # headline
        "dice_vol_mean": float(pos.dice.mean()) if len(pos) else np.nan,
        "dice_vol_median": float(pos.dice.median()) if len(pos) else np.nan,
        "iou_vol_mean": float(pos.iou.mean()) if len(pos) else np.nan,
        # pooled voxels
        "dice_micro": 2 * tp / (2 * tp + fp + fn + EPS), "iou_micro": tp / (tp + fp + fn + EPS),
        "precision_micro": tp / (tp + fp + EPS), "recall_micro": tp / (tp + fn + EPS),
        "specificity_micro": tn / (tn + fp + EPS),
        # per slice (earlier protocol)
        "dice_slice_mean": float(sp.dice.mean()) if len(sp) else np.nan,
        "iou_slice_mean": float(sp.fg_iou.mean()) if len(sp) else np.nan,
        "binary_iou_classavg_ALLSLICES": float(slices.binary_iou_classavg.mean()),
        # surfaces
        "hd95_mm_mean": float(pos.hd95_mm.mean()), "hd95_mm_median": float(pos.hd95_mm.median()),
        "assd_mm_mean": float(pos.assd_mm.mean()), "nsd_mean": float(pos.nsd.mean()),
        "n_pos_scans_empty_pred": int((pos.pred_ml == 0).sum()),
        # lesions
        "lesion_sensitivity": float(lesions.detected.mean()) if len(lesions) else np.nan,
        "fp_lesions_per_scan": float(scans.fp_lesions.mean()),
        "fp_lesions_per_neg_scan": float(neg.fp_lesions.mean()) if len(neg) else np.nan,
        # detection
        "scan_sensitivity": float(pos.called_positive.mean()) if len(pos) else np.nan,
        "scan_specificity": float((~neg.called_positive).mean()) if len(neg) else np.nan,
        "scan_auc": roc_auc(scans.positive.values, scans.max_prob.values),
        "slice_auc": roc_auc((slices.gt_mm2 > 0).values, slices.max_prob.values),
        "slice_sensitivity": float((sp.pred_mm2 > 0).mean()) if len(sp) else np.nan,
        "slice_specificity": float((slices[slices.gt_mm2 == 0].pred_mm2 == 0).mean()),
        "volume_abs_err_ml": float((pos.pred_ml - pos.gt_ml).abs().mean()) if len(pos) else np.nan,
    }
    return out


SIZE_BINS_ML = ((0, 0.5, "small (<0.5 mL)"), (0.5, 5.0, "medium (0.5-5 mL)"),
                (5.0, np.inf, "large (>5 mL)"))
SLICE_BINS_MM2 = ((0, 50, "small (<50 mm2)"), (50, 300, "medium (50-300 mm2)"),
                  (300, np.inf, "large (>300 mm2)"))


def size_strata(slices: pd.DataFrame, lesions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sp = slices[slices.gt_mm2 > 0]
    for lo, hi, name in SLICE_BINS_MM2:
        d = sp[(sp.gt_mm2 >= lo) & (sp.gt_mm2 < hi)]
        rows.append({"stratum": f"slice lesion area {name}", "n": len(d),
                     "dice_slice_mean": d.dice.mean(), "iou_slice_mean": d.fg_iou.mean(),
                     "slice_sensitivity": (d.pred_mm2 > 0).mean() if len(d) else np.nan})
    for lo, hi, name in SIZE_BINS_ML:
        d = lesions[(lesions.volume_ml >= lo) & (lesions.volume_ml < hi)] if len(lesions) else lesions
        rows.append({"stratum": f"lesion volume {name}", "n": len(d),
                     "lesion_sensitivity": d.detected.mean() if len(d) else np.nan,
                     "covered_fraction_mean": d.covered_fraction.mean() if len(d) else np.nan})
    return pd.DataFrame(rows)


def subtype_strata(scans, slices, lesions, slice_index: pd.DataFrame, subtypes) -> pd.DataFrame:
    """PhysioNet: slice-level subtype flags (from the CSV) define the strata.
    BHSD: voxel-level subtype codes also give per-subtype lesion sensitivity and recall."""
    m = slices.merge(slice_index[["case", "slice", *subtypes]], on=["case", "slice"], how="left")
    rows = []
    for i, st in enumerate(subtypes, start=1):
        d = m[(m[st] == 1) & (m.gt_mm2 > 0)]
        sc = scans[scans.case.isin(d.case.unique())]
        r = {"subtype": st, "pos_slices": len(d), "scans": d.case.nunique(),
             "dice_slice_mean": d.dice.mean() if len(d) else np.nan,
             "slice_sensitivity": (d.pred_mm2 > 0).mean() if len(d) else np.nan,
             "dice_vol_mean_scans_with_subtype": sc.dice.mean() if len(sc) else np.nan,
             "voxel_recall_slices": d.tp.sum() / max(d.tp.sum() + d.fn.sum(), 1) if len(d) else np.nan}
        if len(lesions) and "code" in lesions.columns:
            L = lesions[lesions.code == i]
            r["lesions_of_subtype"] = len(L)
            r["lesion_sensitivity"] = L.detected.mean() if len(L) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
