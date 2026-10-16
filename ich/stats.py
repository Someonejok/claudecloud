"""Paired statistics on matched (seed, fold) blocks.

Fold-to-fold variance on 12-scan validation folds is larger than most effects, so
every comparison is paired on the blocks both arms share. Families of comparisons are
Holm-corrected. With n pairs the smallest attainable two-sided Wilcoxon p is 2^(1-n);
comparisons that cannot reach 0.05 are flagged as under-powered and never reported as
evidence of no effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _blocks(runs, arm, metric, by="arm"):
    d = runs[runs[by] == arm].copy()
    d["block"] = list(zip(d.seed, d.fold))
    return d.set_index("block")[metric]


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    p = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min((len(p) - rank) * p[i], 1.0))
        adj[i] = running
    return adj


def paired_tests(runs: pd.DataFrame, pairs, metric: str, by="arm", n_boot=10000, seed=0):
    rows = []
    rs = np.random.RandomState(seed)
    for A, B in pairs:
        da, db = _blocks(runs, A, metric, by), _blocks(runs, B, metric, by)
        common = da.index.intersection(db.index)
        a, b = da.loc[common].values.astype(float), db.loc[common].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        n = len(a)
        d = a - b
        if n >= 2 and np.any(d != 0):
            W, p = stats.wilcoxon(a, b)
        else:
            W, p = np.nan, (1.0 if n >= 2 else np.nan)
        if n >= 2:
            boots = d[rs.randint(0, n, (n_boot, n))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = np.nan
        min_p = 2.0 ** (1 - n) if n else np.nan
        rows.append({"A": A, "B": B, "n_pairs": n, "mean_A": a.mean() if n else np.nan,
                     "mean_B": b.mean() if n else np.nan, "mean_diff": d.mean() if n else np.nan,
                     "ci95_lo": lo, "ci95_hi": hi, "W": W, "p_raw": p,
                     "cliffs_delta": cliffs_delta(a, b), "min_attainable_p": min_p,
                     "powered": bool(n and min_p <= 0.05)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_raw.values)
        out["significant"] = out.p_holm < 0.05
    return out


def friedman(runs, arms, metric, by="arm") -> dict:
    tabs = [_blocks(runs, a, metric, by) for a in arms]
    common = tabs[0].index
    for t in tabs[1:]:
        common = common.intersection(t.index)
    if len(common) < 3 or len(arms) < 3:
        return {"n_blocks": len(common), "chi2": np.nan, "p": np.nan}
    chi2, p = stats.friedmanchisquare(*[t.loc[common].values for t in tabs])
    return {"n_blocks": len(common), "chi2": float(chi2), "p": float(p)}


def page_trend(runs, ordered_arms, metric, by="arm", predicted="decreasing") -> dict:
    """Page's L test for a monotone dose-response across ordered magnitudes on matched
    blocks. predicted='decreasing' tests that the metric falls as magnitude rises."""
    tabs = [_blocks(runs, a, metric, by) for a in ordered_arms]
    common = tabs[0].index
    for t in tabs[1:]:
        common = common.intersection(t.index)
    if len(common) < 2 or len(ordered_arms) < 3:
        return {"n_blocks": len(common), "L": np.nan, "p": np.nan}
    data = np.column_stack([t.loc[common].values for t in tabs])
    if predicted == "decreasing":
        data = -data
    r = stats.page_trend_test(data)
    return {"n_blocks": len(common), "L": float(r.statistic), "p": float(r.pvalue),
            "predicted": predicted}


def summarise(runs: pd.DataFrame, metric: str, by="arm") -> pd.DataFrame:
    g = runs.groupby(by)[metric]
    out = g.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["reported"] = out.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f} (n={int(r['count'])})", axis=1)
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def cluster_bootstrap(per_scan: pd.DataFrame, col: str, n=5000, seed=0, weight_col=None):
    """95 % CI of a mean over scans, resampling SCANS (slices travel with their scan)."""
    v = per_scan[[c for c in ("case", col, weight_col) if c]].dropna(subset=[col])
    if not len(v):
        return np.nan, np.nan
    groups = [g[col].values for _, g in v.groupby("case")]
    rs = np.random.RandomState(seed)
    means = []
    for _ in range(n):
        pick = rs.randint(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
