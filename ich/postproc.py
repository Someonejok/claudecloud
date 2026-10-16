"""Post-processing: threshold, minimum 3D component volume, brain-mask constraint.

The operating point is chosen on out-of-fold DEVELOPMENT predictions only, with a rule
fixed in advance:

    maximise the mean per-scan score over all development scans, where a positive scan
    scores its volumetric Dice and a negative scan scores 1 if nothing is predicted and
    0 otherwise.

Negative scans therefore penalise false positives; without them a sweep would happily
pick a threshold that paints blood on normal brains. `nested_selection` estimates how
much of the gain is selection optimism (choose on four folds, score the fifth). The
recipe applied to the test sets is the one chosen on all development scans.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import components

THRESHOLDS = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2))
MIN_MM3 = (0, 10, 25, 50, 100, 200)
BRAIN = (False, True)


def grid() -> pd.DataFrame:
    g = pd.DataFrame(list(itertools.product(THRESHOLDS, MIN_MM3, BRAIN)),
                     columns=["threshold", "min_mm3", "brain_mask"])
    g["gp"] = np.arange(len(g))
    return g


def apply(prob: np.ndarray, threshold: float, min_mm3: float = 0, brain: np.ndarray | None = None,
          spacing=(1, 1, 1)) -> np.ndarray:
    b = prob >= threshold
    if brain is not None:
        b &= brain
    if min_mm3 > 0 and b.any():
        lab, n = components(b)
        if n:
            vox = float(np.prod(spacing))
            sizes = np.bincount(lab.ravel(), minlength=n + 1) * vox
            keep = sizes >= min_mm3
            keep[0] = False
            b = keep[lab]
    return b


def sweep_scan(prob: np.ndarray, gt: np.ndarray, spacing, brain: np.ndarray | None, case: str,
               fold) -> pd.DataFrame:
    """All grid points for one scan from one labelling per (threshold, brain)."""
    g = grid()
    gt = gt > 0
    gsum = int(gt.sum())
    vox = float(np.prod(spacing))
    out = []
    for thr in THRESHOLDS:
        for use_b in BRAIN:
            if use_b and brain is None:
                continue
            b = prob >= thr
            if use_b:
                b = b & brain
            lab, n = components(b)
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            ov = np.bincount(lab[gt], minlength=n + 1) if (n and gsum) else np.zeros(n + 1, np.int64)
            for ms in MIN_MM3:
                keep = sizes * vox >= ms
                keep[0] = False
                tp = int(ov[keep].sum())
                pred = int(sizes[keep].sum())
                fp_c = int(((ov == 0) & keep).sum())
                out.append((thr, ms, use_b, tp, pred - tp, gsum - tp, int(keep.sum()), fp_c))
    df = pd.DataFrame(out, columns=["threshold", "min_mm3", "brain_mask", "tp", "fp", "fn",
                                    "n_comp", "fp_comp"])
    df = df.merge(g, on=["threshold", "min_mm3", "brain_mask"])
    df["case"], df["fold"], df["positive"] = case, fold, gsum > 0
    pos = df.positive
    df["dice"] = np.where(pos, 2 * df.tp / np.maximum(2 * df.tp + df.fp + df.fn, 1), np.nan)
    df["score"] = np.where(pos, df.dice, (df.n_comp == 0).astype(float))
    return df


def choose(sw: pd.DataFrame) -> dict:
    s = sw.groupby("gp").score.mean()
    best = int(s.idxmax())
    row = sw[sw.gp == best].iloc[0]
    return {"threshold": float(row.threshold), "min_mm3": float(row.min_mm3),
            "brain_mask": bool(row.brain_mask), "dev_score": float(s.max())}


def nested_selection(sw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leave-one-fold-out choice of the operating point. Returns per-fold rows and a
    summary with the nested (honest) and naive (optimistic) dev scores."""
    rows = []
    for k in sorted(sw.fold.unique()):
        pick = choose(sw[sw.fold != k])
        held = sw[(sw.fold == k)]
        hit = held[(held.threshold == pick["threshold"]) & (held.min_mm3 == pick["min_mm3"])
                   & (held.brain_mask == pick["brain_mask"])]
        base = held[(held.threshold == 0.5) & (held.min_mm3 == 0) & (~held.brain_mask)]
        rows.append({"fold": k, **{f"chosen_{a}": b for a, b in pick.items() if a != "dev_score"},
                     "held_score": float(hit.score.mean()),
                     "held_dice_pos": float(hit.dice.mean()),
                     "held_score_default": float(base.score.mean()),
                     "held_dice_pos_default": float(base.dice.mean())})
    per = pd.DataFrame(rows)
    final = choose(sw)
    summ = {"recipe": final,
            "nested_score": float(per.held_score.mean()),
            "nested_dice_pos": float(per.held_dice_pos.mean()),
            "default_score": float(per.held_score_default.mean()),
            "default_dice_pos": float(per.held_dice_pos_default.mean()),
            "naive_score": final["dev_score"]}
    summ["selection_optimism"] = summ["naive_score"] - summ["nested_score"]
    return per, summ
