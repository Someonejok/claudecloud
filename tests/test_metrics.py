import numpy as np
import pandas as pd
import pytest

from ich import metrics as M
from ich import postproc as PP
from ich import stats as ST


def cube(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return m


def test_scan_metrics_known_values():
    gt = cube((8, 32, 32), (2, 8, 8), (6, 16, 16))          # 4*8*8 = 256 voxels
    pr = cube((8, 32, 32), (2, 8, 10), (6, 16, 18))         # shifted 2 columns: overlap 192
    prob = pr.astype(np.float32)
    s, sl, les = M.evaluate_scan(prob, gt.astype(np.uint8), (5.0, 1.0, 1.0), "x")
    assert s["dice"] == pytest.approx(2 * 192 / 512)
    assert s["iou"] == pytest.approx(192 / 320)
    assert s["hd95_mm"] == pytest.approx(2.0, abs=0.01)
    assert les[0]["detected"] and s["fp_lesions"] == 0
    assert sl.dice.dropna().between(0.74, 0.76).all()
    # class-averaged binary IoU is far above lesion IoU on a mostly-empty volume
    assert sl.binary_iou_classavg.mean() > 0.85 > s["iou"]


def test_empty_cases_and_detection():
    shape = (4, 16, 16)
    gt = np.zeros(shape, np.uint8)
    prob = np.zeros(shape, np.float32)
    prob[1, 2:4, 2:4] = 0.9                                  # false positive on a negative scan
    s, sl, les = M.evaluate_scan(prob, gt, (5, 1, 1), "neg")
    assert np.isnan(s["dice"]) and s["fp_lesions"] == 1 and s["called_positive"]
    S = pd.DataFrame([s])
    gt2 = cube(shape, (1, 4, 4), (3, 10, 10)).astype(np.uint8)
    s2, sl2, les2 = M.evaluate_scan(gt2.astype(np.float32), gt2, (5, 1, 1), "pos")
    S = pd.DataFrame([s, s2])
    summ = M.summarise(S, pd.concat([sl, sl2]), pd.DataFrame(les + les2))
    assert summ["dice_vol_mean"] == pytest.approx(1.0)
    assert summ["scan_sensitivity"] == 1.0 and summ["scan_specificity"] == 0.0
    assert summ["fp_lesions_per_neg_scan"] == 1.0


def test_roc_auc():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc([0, 1], [0.9, 0.1]) == 0.0


def test_sweep_matches_direct_application():
    rng = np.random.default_rng(0)
    gt = cube((6, 24, 24), (1, 4, 4), (4, 12, 12)).astype(np.uint8)
    prob = np.clip(gt * 0.6 + rng.random(gt.shape) * 0.5, 0, 1).astype(np.float32)
    brain = np.ones(gt.shape, bool); brain[:, :, :3] = False
    sw = PP.sweep_scan(prob, gt, (5.0, 1.0, 1.0), brain, "c", 0)
    for _, r in sw.sample(25, random_state=0).iterrows():
        pr = PP.apply(prob, r.threshold, r.min_mm3, brain if r.brain_mask else None, (5.0, 1.0, 1.0))
        tp = int((pr & (gt > 0)).sum()); fp = int((pr & (gt == 0)).sum())
        assert (r.tp, r.fp) == (tp, fp)


def test_nested_selection_regression():
    """MAXPERF crashed here (single-key groupby). Any number of folds/keys must work."""
    rng = np.random.default_rng(1)
    rows = []
    for k in range(3):
        for i in range(4):
            gt = cube((4, 16, 16), (1, 3, 3), (3, 9, 9)).astype(np.uint8) if i % 2 == 0 else np.zeros((4, 16, 16), np.uint8)
            prob = np.clip(gt * 0.55 + rng.random(gt.shape) * 0.5, 0, 1).astype(np.float32)
            rows.append(PP.sweep_scan(prob, gt, (5, 1, 1), None, f"c{k}{i}", k))
    sw = pd.concat(rows)
    per, summ = PP.nested_selection(sw)
    assert len(per) == 3 and 0 <= summ["nested_score"] <= 1
    assert summ["naive_score"] >= summ["nested_score"] - 1e-9


def test_stats():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2):
        for f in range(5):
            base = rng.uniform(0.4, 0.6)
            for j, a in enumerate(["m0", "m1", "m2", "m4"]):
                rows.append({"arm": a, "seed": s, "fold": f, "v": base - 0.05 * j + rng.normal(0, 0.005)})
    R = pd.DataFrame(rows)
    pt = ST.page_trend(R, ["m0", "m1", "m2", "m4"], "v")
    assert pt["p"] < 0.001
    t = ST.paired_tests(R, [("m4", "m0"), ("m1", "m0")], "v")
    assert t.powered.all() and t.significant.all() and (t.mean_diff < 0).all()
    assert ST.holm(np.array([0.01, 0.04, 0.5])).tolist() == pytest.approx([0.03, 0.08, 0.5])
