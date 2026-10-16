"""Generate notebooks/ICH_v3.ipynb (run: python tools/make_notebook.py)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r"""
# ICH v3 - traumatic intracranial haemorrhage segmentation, done properly

Every number this notebook produces comes from a **scan-level** split, a **fixed training
schedule** (final weights, no checkpoint picking), and metrics computed on the **native voxel
grid**, lesion class only. The two test sets (15 held-out PhysioNet scans, BHSD official test
split) are opened **once**, in Section 8, after every setting has been frozen.

**Everything is cached on Google Drive and resumable.** If Colab disconnects, reconnect,
run Section 0 again, then re-run the cell that was running: finished runs are skipped,
an interrupted run continues from its last completed epoch, and nnU-Net continues from its
last 5-epoch checkpoint.

**How to run**
1. Runtime > Change runtime type > **L4** (or A100). Put the PhysioNet zip
   (`computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip`)
   anywhere in MyDrive.
2. Set `QUICK = True`, run everything top to bottom (about 30-45 min). This exercises every
   path with 1-epoch models in a separate folder (`ich_v3_quick`); its numbers mean nothing.
3. Set `QUICK = False`, restart the runtime, run Sections 0-3, then the Phase cells in order.
   The budget cell (Section 5) shows hours left from **measured** run times. Stopping after
   any phase still leaves a coherent paper; later phases only add to it.
4. When every training phase you want is done, run Section 7 (frozen post-processing), then
   set `OPEN_TEST_SETS = True` and run Section 8 **once**. Then Sections 9-10 build the
   tables, figures, revised manuscript and response letter on Drive.
""")

md("## 0. Setup")
code(r"""
QUICK = True                     # True = smoke test of every path; False = the real study
REPO = "https://github.com/Someonejok/claudecloud"
BRANCH = "claude/brave-cerf-56rz12"

import os, sys, subprocess, shutil
from pathlib import Path
from google.colab import drive
if not Path("/content/gdrive/MyDrive").exists():
    drive.mount("/content/gdrive")

CODE = Path("/content/claudecloud")
if not CODE.exists():
    r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO, str(CODE)])
    if r.returncode != 0:
        # private repo: upload the repository zip to MyDrive/ich_v3/claudecloud.zip instead
        z = Path("/content/gdrive/MyDrive/ich_v3/claudecloud.zip")
        assert z.exists(), f"clone failed and {z} not found"
        subprocess.run(["unzip", "-q", str(z), "-d", "/content/"], check=True)
        cand = [p for p in Path("/content").glob("claudecloud*") if (p / "ich").exists()]
        cand[0].rename(CODE)
else:
    subprocess.run(["git", "-C", str(CODE), "pull", "-q"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE / "requirements.txt")], check=True)
sys.path.insert(0, str(CODE))

os.environ["ICH_ROOT"] = "/content/gdrive/MyDrive/" + ("ich_v3_quick" if QUICK else "ich_v3")
os.environ["ICH_CACHE"] = "/content/gdrive/MyDrive/ich_v3/cache"      # one HU cache for both modes
os.environ["ICH_LOCAL"] = "/content/ich_local"

import numpy as np, pandas as pd, torch
from IPython.display import display, Image
import ich
from ich import cache as C, data_physionet, data_bhsd, splits as SP, experiments as EX
from ich import headline as HL, evaluate as EV, report as RP
from ich.config import default_paths
P = default_paths().make()
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) - 1))
LOCAL_PRE = Path("/content/nnunet_preprocessed")
print("ich", ich.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("results ->", P.root, "| quick mode" if QUICK else "| REAL RUN")
""")

md("## 1. Data: HU caches for PhysioNet CT-ICH and BHSD (built once, kept on Drive)")
code(r"""
def find_zip(keys):
    for p in Path("/content/gdrive/MyDrive").rglob("*.zip"):
        if any(k in p.name.lower() for k in keys):
            return p
    return None

RAW = Path("/content/raw"); RAW.mkdir(exist_ok=True)
LC = C.Cache(P.local_cache).make()
if any(P.cache.glob("index_*.csv")):
    print("restoring cache from Drive:", C.sync_cache(P.cache, P.local_cache), "files copied")

# PhysioNet
if not (P.local_cache / "index_physionet.csv").exists():
    pn_root = next((p.parent for p in RAW.rglob(data_physionet.MARKER)), None)
    if pn_root is None:
        z = find_zip(("hemorrhage", "intracranial", "1.3.1"))
        assert z, "put the PhysioNet zip somewhere in MyDrive"
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(RAW)], check=True)
        pn_root = next(p.parent for p in RAW.rglob(data_physionet.MARKER))
    data_physionet.build(pn_root, LC, P.local_cache / "index_physionet.csv")
PN_ROOT = next((p.parent for p in RAW.rglob(data_physionet.MARKER)), None)

# BHSD (192 voxel-labelled scans). The zip is kept on Drive after the first download.
if not (P.local_cache / "index_bhsd.csv").exists():
    keep = Path("/content/gdrive/MyDrive/ich_v3/raw"); keep.mkdir(parents=True, exist_ok=True)
    z = data_bhsd.download(keep)
    bh_root = data_bhsd.extract(z, RAW / "bhsd")
    data_bhsd.print_licence(bh_root)
    data_bhsd.build(bh_root, LC, P.local_cache / "index_bhsd.csv")
print("persisting cache to Drive:", C.sync_cache(P.local_cache, P.cache), "files copied")
INDEX = C.load_index(P.local_cache)
display(INDEX.groupby("source").agg(scans=("case", "nunique"), slices=("case", "size"),
                                    lesion_slices=("has_lesion", "sum")))
""")

md(r"""
## 2. Audit gate - training is blocked if a fatal check fails

HU calibration, orientation (columns must be the left-right axis, checked on the pixels),
CSV-to-pixel slice alignment after reorientation, scan-level disjointness of every partition
and fold, near-duplicate scans across partitions, and which axis the original code's
"HorizontalFlip" actually mirrored.
""")
code(r"""
SPLIT = SP.make_splits(INDEX, P.splits / "splits.json")
metas = pd.DataFrame([{**{k: v for k, v in LC.meta(c).items() if not isinstance(v, (list, dict))},
                       **LC.meta(c)["hu_check"], **LC.meta(c)["orientation"]} for c in LC.cases()])
pn = INDEX[INDEX.source == "physionet"]
checks = {
    "all scans in Hounsfield units": bool(metas.is_hu.all()),
    "CSV haemorrhage flags agree with masks on every PhysioNet slice":
        float((pn.csv_no_hemorrhage == 1 - pn.has_lesion).mean()) > 0.995,
    "no scan shared between partitions/folds": None,
}
aud = SP.leakage_audit(SPLIT, LC)
checks["no scan shared between partitions/folds"] = int(aud["shared scan IDs"].max()) == 0
dups = aud.attrs.get("near_duplicates", [])
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"scans whose in-plane axes the symmetry audit swapped: {int(metas.axes_swapped_by_audit.sum())}")
print(f"HU-shifted scans (stored without intercept): {int((metas.shifted_by != 0).sum())}")
print(f"max fingerprint similarity across partitions: {aud.attrs.get('max_cross_partition_similarity'):.3f}")
# Pixel fingerprints flag CANDIDATE duplicates for a human to look at; they are not proof.
print(f"candidate near-duplicate pairs (similarity > 0.98) to inspect by eye: {len(dups)}")
for a_, b_, s_ in dups[:20]:
    print(f"   {a_} ~ {b_}  r = {s_:.3f}")
fl = data_physionet.original_flip_axis(PN_ROOT, LC, "pn049") if PN_ROOT else None
if fl:
    print(f"original code's HorizontalFlip mirrored the {fl['original_horizontal_flip_mirrored']} axis "
          f"(match r = {fl['match_correlation']:.3f})")
    EX.PAPER_EXTREME_SPEC = fl["paper_extreme_spec"]
display(pd.DataFrame([{"partition": k, "scans": v} for k, v in
                      pd.Series(SPLIT["partition"]).value_counts().items()]))
assert all(checks.values()), "AUDIT FAILED - fix before training"

import matplotlib.pyplot as plt
from ich.preprocess import window
pos = INDEX[INDEX.has_lesion == 1].sample(8, random_state=0)
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for a, r in zip(ax.ravel(), pos.itertuples()):
    a.imshow(window(np.asarray(LC.hu(r.case)[r.slice]), 40, 80), cmap="gray")
    a.contour(np.asarray(LC.lab(r.case)[r.slice]) > 0, levels=[0.5], colors="r", linewidths=0.7)
    a.set_title(f"{r.case} slice {r.slice}", fontsize=8); a.axis("off")
fig.suptitle("masks must hug the blood; the face must be at the top; left-right along columns")
fig.tight_layout(); fig.savefig(P.figures / "audit_overlays.png", dpi=110); plt.show()
""")

md("## 3. Data characterisation and pre-registration (before anything trains)")
code(r"""
DT = RP.data_tables(INDEX, SPLIT, LC)
display(DT["partitions"]); display(DT["subtypes"])
display(RP.class_balance(INDEX, SPLIT).groupby("scheme").mean(numeric_only=True))
if PN_ROOT:
    display(RP.demographics(data_physionet.demographics(PN_ROOT), INDEX))
PREREG = EX.write_or_check_prereg(P.root / "preregistration.json")
print("arms registered:", sum(len(v) for v in PREREG["arms"].values()))
""")

md(r"""
## 4. Phase 2a - headline: nnU-Net 2D on PhysioNet dev + BHSD train (5 folds)

The first call measures seconds per epoch on this GPU and fixes the epoch count so one
fold takes about `HOURS_PER_FOLD`; the number is written into the pre-registration before
fold 0 trains and never changes afterwards. Re-run this cell after a disconnect.
""")
code(r"""
HOURS_PER_FOLD = 1.7
EP_PB = HL.run(P, LC, SPLIT, "P+B", LOCAL_PRE, HOURS_PER_FOLD, quick_mode=QUICK,
               folds=range(5))
""")

md("## 5. Budget (hours left, from measured run times)")
code(r"""
B = EX.budget(P)
display(B.groupby("experiment")[["runs", "done", "hours_left"]].sum())
""")

md(r"""
## 6. Phase 3 - ablations with the custom U-Net (PhysioNet dev, 5 scan-level folds)

Run in this order; each cell is resumable. E1 is the original manuscript's own model and
recipe under a correct protocol. E4 is the augmentation-intensity study: one family at a
time at 0.5x/1x/2x/4x the Table 1 magnitudes, plus the original "moderate" and "extreme"
policies.
""")
code(r"""
QSEEDS = (0,) if QUICK else None
EX.run_experiment("E1_paper", P, INDEX, LC, SPLIT, quick_mode=QUICK, num_workers=NUM_WORKERS, seeds=QSEEDS)
""")
code(r"""
EX.run_experiment("E4_augmentation", P, INDEX, LC, SPLIT, quick_mode=QUICK, num_workers=NUM_WORKERS,
                  only=("none", "base", "paper_moderate", "paper_extreme"), seeds=QSEEDS)
""")
code(r"""
if not QUICK:
    EX.run_experiment("E4_augmentation", P, INDEX, LC, SPLIT, num_workers=NUM_WORKERS)
""")
md("### Phase 2b - nnU-Net 2D on PhysioNet only (BHSD becomes a truly external test set)")
code(r"""
EP_P = HL.run(P, LC, SPLIT, "P", LOCAL_PRE, HOURS_PER_FOLD, quick_mode=QUICK)
""")
code(r"""
if not QUICK:
    for exp in ("E2_imbalance", "E3_window", "E5_architecture"):
        EX.run_experiment(exp, P, INDEX, LC, SPLIT, num_workers=NUM_WORKERS)
""")

md(r"""
## 7. Phase 4 - post-processing chosen on development predictions only, then frozen

For each model: threshold, minimum 3D component volume and brain-mask constraint are chosen
to maximise the mean per-scan score over all development scans (Dice on ICH-positive scans;
1 if nothing is predicted on a negative scan, else 0). The nested estimate shows how much of
the gain is selection optimism. Recipes are written to disk and never recomputed.
""")
code(r"""
import json
MODELS = ["nnU-Net P+B", "nnU-Net P", "paper", "aug_none", "aug_base", "aug_moderate", "aug_extreme"]
REG = json.loads((P.root / "preregistration.json").read_text())["nnunet"]["epochs"]
REG = REG if isinstance(REG, dict) else {}            # placeholder text until calibration ran
NN_EP = {"nnU-Net P+B": REG.get("P+B"), "nnU-Net P": REG.get("P")}
READY = []
for m in MODELS:
    try:
        EV.select_postprocessing(P, LC, SPLIT, m, NN_EP.get(m)); READY.append(m)
    except FileNotFoundError as e:
        print(f"  {m}: not trained yet ({str(e)[:80]}...)")
print("frozen:", READY)
""")

md(r"""
## 8. Phase 5 - the test sets, opened ONCE

Set `OPEN_TEST_SETS = True` only when every training phase you intend to report is done.
Results are written to disk and a second run only reloads them.
""")
code(r"""
OPEN_TEST_SETS = QUICK
if OPEN_TEST_SETS:
    TEST = {m: EV.test_evaluation(P, LC, SPLIT, INDEX, m, LOCAL_PRE, NN_EP.get(m)) for m in READY}
""")

md("## 9. Tables, figures, morphology measures")
code(r"""
mp = P.tables / "table_morphology.csv"
if not mp.exists():
    RP.morphology_table(INDEX, SPLIT, LC, n_slices=10 if QUICK else 120, n_draws=3 if QUICK else 20).to_csv(mp, index=False)
display(pd.read_csv(mp))
S = RP.build(P, INDEX, SPLIT, LC, data_physionet.demographics(PN_ROOT) if PN_ROOT else None)
for m in ("nnU-Net P+B", "aug_base"):
    if (P.tables / f"test_{EV._slug(m)}.json").exists():
        RP.fig_qualitative(P, LC, SPLIT, m, P.figures / f"fig_qualitative_{EV._slug(m)}.png")
for f in sorted(P.figures.glob("*.png")):
    print(f.name); display(Image(str(f), width=700))
""")

md("## 10. Revised manuscript and response to reviewers (.docx, written to Drive)")
code(r"""
from ich import manuscript
out = manuscript.build_all(P)
print(out)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
out = Path(__file__).resolve().parents[1] / "notebooks" / "ICH_v3.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
