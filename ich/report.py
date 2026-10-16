"""Tables, figures and results_summary.json (read by the manuscript builder).

Everything here only READS results already on disk, so it can be rerun at any point;
missing results simply leave their keys out and the manuscript marks them pending.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiments as EX
from . import splits as SP
from . import stats as ST
from .config import SUBTYPES
from .ledger import Ledger

# categorical slots 1-4 of the validated reference palette (light surface)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARKERS = ("o", "s", "^", "D")
SECONDARY = ("dice_slice_mean", "iou_slice_mean", "dice_micro", "precision_micro", "recall_micro",
             "lesion_sensitivity", "fp_lesions_per_scan", "hd95_mm_mean", "scan_auc", "slice_auc")


# ----------------------------------------------------------------------------- data tables
def data_tables(index: pd.DataFrame, split: dict, cache=None) -> dict:
    part = pd.Series(split["partition"], name="partition")
    df = index.merge(part, left_on="case", right_index=True)
    rows = []
    for p, d in df.groupby("partition"):
        sc = d.groupby("case").has_lesion.max()
        r = {"partition": p, "scans": d.case.nunique(), "ICH-positive scans": int(sc.sum()),
             "slices": len(d), "lesion-positive slices": int(d.has_lesion.sum()),
             "lesion-negative slices": int((1 - d.has_lesion).sum())}
        for st in SUBTYPES:
            r[f"{st} slices"] = int(d[st].sum())
        rows.append(r)
    partitions = pd.DataFrame(rows)
    srows = []
    for src, d in df.groupby("source"):
        for st in SUBTYPES:
            x = d[(d[st] == 1) & (d.has_lesion == 1)]
            srows.append({"dataset": src, "subtype": st, "slices": len(x), "scans": x.case.nunique(),
                          "lesion area mm2 (median)": float(x.lesion_mm2.median()) if len(x) else np.nan,
                          "lesion area mm2 (IQR)": (f"{x.lesion_mm2.quantile(.25):.0f}-{x.lesion_mm2.quantile(.75):.0f}"
                                                    if len(x) else "")})
    subtypes = pd.DataFrame(srows)
    out = {"partitions": partitions, "subtypes": subtypes}
    if cache is not None:
        out["leakage"] = SP.leakage_audit(split, cache)
    return out


def class_balance(index, split) -> pd.DataFrame:
    """Slices before/after each sampling scheme, per development fold (reviewer #5)."""
    from .train import class_balance_table
    rows = []
    for k in range(SP.N_FOLDS):
        tr = index[index.case.isin(SP.cases(split, partition="pn_dev", not_fold=k))]
        for arm in ("under1x", "balanced", "all_slices"):
            cfg = EX.arms()["E2_imbalance"][arm]
            rows.append({"fold": k, "scheme": arm, **class_balance_table(tr, cfg, 0)})
    return pd.DataFrame(rows)


def demographics(raw: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Age and sex by ICH status, with every figure labelled (minor 3)."""
    if raw is None or not len(raw):
        return pd.DataFrame()
    cols = {c.lower().replace("\n", " "): c for c in raw.columns}
    pid = next((cols[c] for c in cols if "patient" in c), None)
    age = next((cols[c] for c in cols if "age" in c), None)
    sex = next((cols[c] for c in cols if "gender" in c or "sex" in c), None)
    if pid is None:
        return pd.DataFrame()
    d = raw[[c for c in (pid, age, sex) if c]].copy()
    d.columns = ["patient"] + (["age"] if age else []) + (["sex"] if sex else [])
    d["patient"] = pd.to_numeric(d.patient, errors="coerce")
    pos = index[index.source == "physionet"].groupby("patient").has_lesion.max()
    pos.index = pos.index.astype(int)
    d = d.dropna(subset=["patient"]).merge(pos.rename("ich"), left_on="patient", right_index=True)
    rows = []
    for label, g in (("all", d), ("ICH-positive", d[d.ich == 1]), ("ICH-negative", d[d.ich == 0])):
        r = {"group": label, "patients": len(g)}
        if "age" in g:
            a = pd.to_numeric(g.age, errors="coerce")
            r.update({"age mean": a.mean(), "age SD": a.std(), "age min": a.min(), "age max": a.max()})
        if "sex" in g:
            s = g.sex.astype(str).str.lower().str[0]
            r.update({"male": int((s == "m").sum()), "female": int((s == "f").sum())})
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CV tables
def experiment_tables(paths, exp: str) -> dict:
    led = Ledger(paths.ledger, exp)
    runs = led.frame()
    if not len(runs):
        return {}
    metric = EX.PRIMARY_METRIC
    summ = ST.summarise(runs, metric)
    sec = runs.groupby("arm")[[c for c in SECONDARY if c in runs.columns]].mean().reset_index()
    summ = summ.merge(sec, on="arm", how="left")
    out = {"runs": runs, "summary": summ}
    ref = EX.DECISION_RULES["references"].get(exp)
    if ref and ref in set(runs.arm):
        pairs = [(a, ref) for a in sorted(set(runs.arm)) if a != ref]
        out["tests"] = ST.paired_tests(runs, pairs, metric)
    if exp == "E4_augmentation":
        trends = []
        for fam in EX.FAMILIES:
            order = ["base"] + [f"{fam}@{m:g}" for m in EX.MAGNITUDES]
            if all(a in set(runs.arm) for a in order):
                trends.append({"family": fam, **ST.page_trend(runs, order, metric)})
        out["trend"] = pd.DataFrame(trends)
        comp = [a for a in ("none", "base", "paper_moderate", "paper_extreme") if a in set(runs.arm)]
        if len(comp) >= 3:
            out["friedman_composites"] = ST.friedman(runs, comp, metric)
    return out


def nnunet_tables(paths, which: str) -> dict:
    from .headline import experiment_name
    runs = Ledger(paths.ledger, experiment_name(which)).frame()
    if not len(runs):
        return {}
    cols = [c for c in runs.columns if c.startswith(("dice", "iou", "precision", "recall", "lesion",
                                                     "fp_", "hd95", "scan_", "slice_", "bhsd_"))]
    return {"runs": runs, "mean": runs[cols].mean(), "sd": runs[cols].std()}


def compute_table(paths) -> pd.DataFrame:
    rows = []
    for p in sorted(paths.ledger.glob("E5_architecture/*.json")):
        r = json.loads(p.read_text())
        if r.get("alias_of"):
            src = json.loads(Path(r["alias_of"]).read_text())
            r["info"] = src["info"]
        i = r["info"]
        rows.append({"arm": r["arm"], "fold": r["fold"], "params_M": i.get("params", np.nan) / 1e6,
                     "GFLOPs/slice": i.get("gflops_per_slice"), "train_min": i.get("train_seconds", np.nan) / 60,
                     "s/epoch": i.get("sec_per_epoch"), "inference_ms/slice": i.get("inference_ms_per_slice"),
                     "peak_GPU_GB": i.get("peak_gpu_mem_gb"), "dice_vol_mean": r["metrics"].get("dice_vol_mean")})
    t = pd.DataFrame(rows)
    return t.groupby("arm").mean(numeric_only=True).reset_index() if len(t) else t


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_dose_response(e4: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = e4.get("runs")
    if runs is None or not len(runs):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    xs = list(range(len(EX.MAGNITUDES) + 1))      # equal spacing: magnitudes double each step
    m = EX.PRIMARY_METRIC
    for fam, col, mk in zip(EX.FAMILIES, (C1, C2, C3, C4), MARKERS):
        arms = ["base"] + [f"{fam}@{x:g}" for x in EX.MAGNITUDES]
        mu, lo, hi = [], [], []
        for a in arms:
            v = runs[runs.arm == a][m].dropna().values
            mu.append(v.mean() if len(v) else np.nan)
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            lo.append(mu[-1] - 1.96 * se); hi.append(mu[-1] + 1.96 * se)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.10, lw=0)
        ax.plot(xs, mu, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5, label=fam)
        if np.isfinite(mu[-1]):
            ax.annotate(fam, (xs[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
    none = runs[runs.arm == "none"][m].mean()
    if np.isfinite(none):
        ax.axhline(none, color=INK2, lw=1, ls="--")
        ax.annotate("no augmentation", (xs[0], none), xytext=(2, 4), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0 (flip only)"] + [f"{x:g}" + (" (Table 1)" if x == 1 else "") for x in EX.MAGNITUDES])
    ax.set_xlabel("augmentation magnitude (multiple of the Table 1 value)", color=INK, fontsize=10)
    ax.set_ylabel("volumetric Dice, ICH-positive scans", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    ax.set_xlim(-0.3, xs[-1] + 0.9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return out


def fig_learning_curves(paths, items: list[tuple[str, str, str]], out: Path):
    """items: (experiment, arm, label). Mean over folds; the final epoch is the model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.4), dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURF)
    drawn = False
    for ax, (exp, arm, label) in zip(axes[0], items):
        _style(ax)
        hs = [pd.read_csv(p) for p in sorted((paths.ledger / exp / "tables").glob(f"{arm}__s0_f*__history.csv"))]
        if not hs:
            ax.set_title(f"{label}: not run", fontsize=9, color=INK2)
            continue
        drawn = True
        H = pd.concat(hs).groupby("epoch").mean(numeric_only=True).reset_index()
        ax.plot(H.epoch, H.train_loss, color=C1, lw=2, label="training loss")
        ax.plot(H.epoch, H.val_loss, color=C2, lw=2, label="validation loss")
        ax.axvline(H.epoch.max(), color=INK2, lw=1, ls=":")
        ax.annotate("model = final epoch", (H.epoch.max(), ax.get_ylim()[1]), xytext=(-4, -12),
                    textcoords="offset points", ha="right", fontsize=8, color=INK2)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        ax.set_ylabel("loss", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if drawn:
        fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return out if drawn else None


def fig_qualitative(paths, cache, split, model: str, out: Path, n_each=2):
    """CT / ground truth / prediction / error overlay for the best, the worst, a missed
    lesion and a false-positive slice of the PhysioNet test set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import _slug
    from .preprocess import window
    from .train import load_prob
    tdir = paths.tables / f"test_{_slug(model)}"
    sl_p = tdir / "physionet_recipe_slices.csv"
    if not sl_p.exists():
        return None
    rec = json.loads((paths.tables / f"postproc_{_slug(model)}.json").read_text())["recipe"]
    SL = pd.read_csv(sl_p)
    pos = SL[SL.gt_mm2 > 0].dropna(subset=["dice"])
    picks = []
    picks += [("best", r) for _, r in pos[pos.dice > 0].sort_values("dice", ascending=False).head(n_each).iterrows()]
    picks += [("boundary error", r) for _, r in pos[(pos.dice > 0.2) & (pos.dice < 0.6)].head(n_each).iterrows()]
    picks += [("missed", r) for _, r in pos[pos.pred_mm2 == 0].sort_values("gt_mm2", ascending=False).head(n_each).iterrows()]
    picks += [("false positive", r) for _, r in SL[(SL.gt_mm2 == 0) & (SL.pred_mm2 > 0)]
              .sort_values("pred_mm2", ascending=False).head(n_each).iterrows()]
    if not picks:
        return None
    from . import postproc as PP
    fig, axes = plt.subplots(len(picks), 4, figsize=(10, 2.6 * len(picks)), dpi=150, squeeze=False)
    fig.patch.set_facecolor("white")
    for row, (kind, r) in zip(axes, picks):
        c, k = r.case, int(r.slice)
        prob = load_prob(paths.probs / "test" / model / f"{c}.npz")
        from .evaluate import brain_volume
        bm = brain_volume(cache, c) if rec["brain_mask"] else None
        pred = PP.apply(prob, rec["threshold"], rec["min_mm3"], bm, cache.meta(c)["spacing"])[k]
        hu = np.asarray(cache.hu(c)[k], np.float32)
        gt = np.asarray(cache.lab(c)[k]) > 0
        img = window(hu, 40, 80)
        ov = np.dstack([img] * 3)
        for m, col in ((gt & pred, C3), (pred & ~gt, C2), (gt & ~pred, C1)):
            rgb = np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)])
            ov[m] = 0.35 * ov[m] + 0.65 * rgb
        titles = (f"{kind}: {c} slice {k}", "ground truth", "prediction", "overlay (aqua TP, orange FP, blue FN)")
        for ax, im, t in zip(row, (img, gt, pred, ov), titles):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
            ax.set_title(t + (f"  Dice {r.dice:.2f}" if t == "prediction" and np.isfinite(r.dice) else ""), fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- summary
def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    return v


def build(paths, index, split, cache=None, demographics_raw=None) -> dict:
    T = paths.tables
    S = {"data": {}, "cv": {}, "nnunet_cv": {}, "test": {}, "postproc": {}}
    dt = data_tables(index, split, cache)
    dt["partitions"].to_csv(T / "table_partitions.csv", index=False)
    dt["subtypes"].to_csv(T / "table_subtypes.csv", index=False)
    S["data"]["partitions"] = dt["partitions"].to_dict("records")
    S["data"]["subtypes"] = dt["subtypes"].to_dict("records")
    if "leakage" in dt:
        dt["leakage"].to_csv(T / "table_leakage.csv", index=False)
        S["data"]["leakage_max_shared"] = int(dt["leakage"]["shared scan IDs"].max())
        S["data"]["max_cross_partition_similarity"] = dt["leakage"].attrs.get("max_cross_partition_similarity")
        S["data"]["near_duplicates"] = dt["leakage"].attrs.get("near_duplicates")
    cb = class_balance(index, split)
    cb.to_csv(T / "table_class_balance.csv", index=False)
    S["data"]["class_balance"] = cb.groupby("scheme").mean(numeric_only=True).reset_index().to_dict("records")
    if demographics_raw is not None:
        dm = demographics(demographics_raw, index)
        dm.to_csv(T / "table_demographics.csv", index=False)
        S["data"]["demographics"] = dm.to_dict("records")
    for exp in EX.arms():
        t = experiment_tables(paths, exp)
        if not t:
            continue
        t["summary"].to_csv(T / f"cv_{exp}_summary.csv", index=False)
        S["cv"][exp] = {"summary": t["summary"].to_dict("records")}
        if "tests" in t:
            t["tests"].to_csv(T / f"cv_{exp}_tests.csv", index=False)
            S["cv"][exp]["tests"] = t["tests"].to_dict("records")
        if "trend" in t:
            t["trend"].to_csv(T / "cv_E4_trend.csv", index=False)
            S["cv"][exp]["trend"] = t["trend"].to_dict("records")
        if "friedman_composites" in t:
            S["cv"][exp]["friedman_composites"] = t["friedman_composites"]
        if exp == "E4_augmentation":
            f = fig_dose_response(t, paths.figures / "fig_dose_response.png")
            S["cv"][exp]["figure"] = str(f) if f else None
    for which in ("P+B", "P"):
        t = nnunet_tables(paths, which)
        if t:
            t["runs"].to_csv(T / f"cv_nnunet_{which.replace('+', '')}.csv", index=False)
            S["nnunet_cv"][which] = {"mean": t["mean"].to_dict(), "sd": t["sd"].to_dict(),
                                     "folds": int(len(t["runs"]))}
    ct = compute_table(paths)
    if len(ct):
        ct.to_csv(T / "table_compute.csv", index=False)
        S["compute"] = ct.to_dict("records")
    for p in sorted(T.glob("postproc_*.json")):
        S["postproc"][p.stem.replace("postproc_", "")] = json.loads(p.read_text())
    for p in sorted(T.glob("test_*.json")):
        S["test"][p.stem.replace("test_", "")] = json.loads(p.read_text())
    mp = T / "table_morphology.csv"
    if mp.exists():
        S["morphology"] = pd.read_csv(mp).to_dict("records")
    lc = fig_learning_curves(paths, [("E1_paper", "paper_recipe", "original model and recipe"),
                                     ("E4_augmentation", "base", "U-Net-ResNet34, flip only")],
                             paths.figures / "fig_learning_curves.png")
    S["figures"] = {"learning_curves": str(lc) if lc else None}
    out = paths.root / "results_summary.json"
    out.write_text(json.dumps(S, indent=1, default=_num))
    return S


def morphology_table(index, split, cache, n_slices=40, n_draws=10) -> pd.DataFrame:
    """Morphology-preservation measures for every E4 policy, on development lesion slices."""
    from .augment import morphology_effect
    dev = index[index.case.isin(SP.cases(split, partition="pn_dev")) & (index.has_lesion == 1)]
    dev = dev.sample(n=min(n_slices, len(dev)), random_state=0)
    samples = [(np.asarray(cache.hu(r.case)[r.slice], np.float32), np.asarray(cache.lab(r.case)[r.slice]),
                cache.meta(r.case)) for r in dev.itertuples()]
    rows = []
    for arm, cfg in EX.arms()["E4_augmentation"].items():
        rows.append({"arm": arm, **morphology_effect(cfg.aug, samples, cfg.fov_mm, cfg.image_size, n_draws)})
    return pd.DataFrame(rows)
