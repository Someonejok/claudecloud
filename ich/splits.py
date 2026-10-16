"""Scan-level partitions, written to disk once and never redrawn.

PhysioNet: the 15-scan held-out test set is the seed-12345 draw used since the
post-review rework (stratified by scan positivity); the remaining 60 scans form 5
development folds (StratifiedGroupKFold, seed 0). BHSD: the official test split from
dataset.json when present, otherwise a seeded stratified draw; BHSD-train is split
into 5 folds that are paired with the PhysioNet folds for the combined model.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import SUBTYPES

N_FOLDS = 5


def _scan_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("case")
    t = pd.DataFrame({"pos": g.has_lesion.max(), "source": g.source.first(),
                      "patient": g.patient.first()})
    for st in SUBTYPES:
        if st in df.columns:
            t[st] = g[st].max()
    return t.reset_index()


def physionet_test_split(df_pn: pd.DataFrame, n_test=15, seed=12345) -> list[str]:
    """Same algorithm, ordering and seed as the earlier rework, so the 15 scans are
    identical to the ones already set aside there."""
    p = df_pn.assign(PatientNumber=df_pn.patient.astype(int)) \
        .groupby("PatientNumber").has_lesion.max().rename("pos").reset_index()
    rng = np.random.RandomState(seed)
    test = []
    for v in (1, 0):
        pool = p[p.pos == v].PatientNumber.values.copy()
        rng.shuffle(pool)
        test += list(pool[:int(round(n_test * (p.pos == v).mean()))])
    if len(test) < n_test:
        extra = [i for i in p.PatientNumber.values if i not in test]
        rng.shuffle(extra)
        test += extra[:n_test - len(test)]
    return [f"pn{int(i):03d}" for i in sorted(test)]


def physionet_folds(df_dev: pd.DataFrame, seed=0) -> dict[str, int]:
    p = df_dev.assign(PatientNumber=df_dev.patient.astype(int)) \
        .groupby("PatientNumber").has_lesion.max().rename("pos").reset_index()
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    out = {}
    for k, (_, va) in enumerate(sgk.split(p, p.pos, p.PatientNumber)):
        for i in p.iloc[va].PatientNumber:
            out[f"pn{int(i):03d}"] = k
    return out


def _dominant_subtype(row) -> str:
    present = [st for st in SUBTYPES if row.get(st, 0)]
    return present[0] if len(present) == 1 else ("multi" if present else "none")


def bhsd_partition(df_bh: pd.DataFrame, test_frac=0.2, seed=12345) -> tuple[list[str], dict[str, int]]:
    t = _scan_table(df_bh)
    if "official_split" in df_bh.columns:
        off = df_bh.groupby("case").official_split.first().fillna("")
        test = sorted(off[off == "test"].index)
    else:
        test = []
    if not test:
        strat = t.apply(_dominant_subtype, axis=1).values
        # merge rare strata so every stratum can be split
        vc = pd.Series(strat).value_counts()
        strat = np.array([s if vc[s] >= 5 else "rare" for s in strat])
        rng = np.random.RandomState(seed)
        test = []
        for s in sorted(set(strat)):
            pool = t.case.values[strat == s].copy()
            rng.shuffle(pool)
            test += list(pool[:int(round(len(pool) * test_frac))])
        test = sorted(test)
    tr = t[~t.case.isin(test)].reset_index(drop=True)
    strat = tr.apply(_dominant_subtype, axis=1).values
    vc = pd.Series(strat).value_counts()
    strat = np.array([s if vc[s] >= N_FOLDS else "rare" for s in strat])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    folds = {}
    for k, (_, va) in enumerate(skf.split(tr, strat)):
        for c in tr.case.iloc[va]:
            folds[c] = k
    return test, folds


def make_splits(df: pd.DataFrame, path: Path) -> dict:
    """Create once; afterwards load and verify against the current index."""
    path = Path(path)
    if path.exists():
        s = json.loads(path.read_text())
        missing = set(s["partition"]) - set(df.case.unique())
        if missing:
            raise RuntimeError(f"splits file refers to {len(missing)} scans missing from the cache")
        return s
    pn = df[df.source == "physionet"]
    bh = df[df.source == "bhsd"]
    part, fold = {}, {}
    pn_test = physionet_test_split(pn)
    for c in pn.case.unique():
        part[c] = "pn_test" if c in pn_test else "pn_dev"
    fold.update(physionet_folds(pn[~pn.case.isin(pn_test)]))
    if len(bh):
        bh_test, bh_folds = bhsd_partition(bh)
        for c in bh.case.unique():
            part[c] = "bh_test" if c in bh_test else "bh_train"
        fold.update(bh_folds)
    s = {"partition": part, "fold": fold, "n_folds": N_FOLDS,
         "provenance": {"physionet_test": "seed 12345, stratified by scan positivity (15 scans)",
                        "physionet_folds": "StratifiedGroupKFold(5, shuffle, seed 0) on scan positivity",
                        "bhsd_test": "official dataset.json test split if present, else seeded "
                                     "stratified 20% by dominant subtype",
                        "bhsd_folds": "StratifiedKFold(5, seed 0) by dominant subtype"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1))
    return s


def cases(s: dict, *, partition=None, fold=None, not_fold=None, source=None) -> list[str]:
    out = []
    for c, p in s["partition"].items():
        if partition is not None and p not in ((partition,) if isinstance(partition, str) else partition):
            continue
        if source is not None and not c.startswith({"physionet": "pn", "bhsd": "bh_"}[source]):
            continue
        f = s["fold"].get(c)
        if fold is not None and f != fold:
            continue
        if not_fold is not None and f == not_fold:
            continue
        out.append(c)
    return sorted(out)


def leakage_audit(s: dict, cache=None) -> pd.DataFrame:
    """Two checks: no scan is in two partitions or two folds (by ID), and no two scans
    in different partitions look like the same acquisition (pixel fingerprints)."""
    part = s["partition"]
    groups = {}
    for c, p in part.items():
        key = p if p.endswith("test") else f"{p}_fold{s['fold'].get(c)}"
        groups.setdefault(key, set()).add(c)
    rows = []
    for a, b in itertools.combinations(sorted(groups), 2):
        rows.append({"A": a, "B": b, "scans A": len(groups[a]), "scans B": len(groups[b]),
                     "shared scan IDs": len(groups[a] & groups[b])})
    out = pd.DataFrame(rows)
    if cache is not None:
        ids = sorted(part)
        fp = np.stack([np.asarray(cache.meta(c)["fingerprint"], np.float32) for c in ids])
        sim = fp @ fp.T
        np.fill_diagonal(sim, -1)
        key = {c: (part[c] if part[c].endswith("test") else f"{part[c]}_fold{s['fold'].get(c)}")
               for c in ids}
        cross = np.array([[key[ids[i]] != key[ids[j]] for j in range(len(ids))]
                          for i in range(len(ids))])
        out.attrs["max_cross_partition_similarity"] = float(sim[cross].max()) if cross.any() else np.nan
        out.attrs["near_duplicates"] = [(ids[i], ids[j], float(sim[i, j]))
                                        for i, j in zip(*np.where((sim > 0.98) & cross)) if i < j]
    return out
