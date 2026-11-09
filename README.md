# ICH v3: traumatic intracranial haemorrhage segmentation, rebuilt to survive review

> **Read before submitting anything: NHSJS AI policy.** NHSJS
> ([AI Usage Policy](https://nhsjs.com/?p=50884), checked November 2026) allows AI for writing and
> debugging code, running or interpreting analyses, and searching literature. It does
> **not** allow AI to draft, paraphrase, expand or rewrite prose, even if you edit the output afterwards. All AI use must be
> disclosed in a statement at the end of the manuscript. The prose in
> `ich/manuscript.py` and in `docs/preview/TEMPLATE_*.docx` was drafted by an AI assistant.
> Use those files only as a checklist and a table and figure generator. The authors must write
> the manuscript text and the response letter themselves. The tables, figures, numbers and
> reference list can be used as they are.

Code, Colab notebook and document builders for the revision of *Deep Learning Models for
Intracranial Haemorrhage Detection and Masking* (Choudhary & Poore). Every reviewer point is
addressed by an experiment, a table or a rewrite; the mapping is in the generated
response letter.

## What went wrong before, and what this fixes

| Problem | Where | Fix here |
|---|---|---|
| Validation slices were a subset of the training slices (`train_generator` used `balanced_df`, and `val_df` was split from `balanced_df`) | original notebook | scan-level splits written once to `splits.json`; automated leakage audit |
| Slice-level split; 1:1 undersampling before splitting | original notebook | undersampling only inside training folds |
| "Binary IoU" averaged lesion and background IoU, inflating 0.85 | original notebook/manuscript | lesion-only metrics on the native grid; class-averaged figure shown once for contrast |
| Optuna and checkpoints chosen on the (leaked) validation loss | original notebook | pre-registration; fixed schedules; final-epoch weights; no checkpoint selection |
| Window 30/120 HU stated, 40/120 HU used | manuscript vs code | both tested in the windowing ablation |
| Never finished: Phase B crashed (`TypeError` in a single-key `groupby`), no test table | ICH_MAXPERF notebook | post-processing selection rewritten on integer grid IDs, with a regression test; every phase resumable |
| Cosine schedule over 250 epochs killed by early stopping at epochs 30-70 (LR never decayed) | ICH_MAXPERF | fixed schedule, cosine to zero over the epochs actually run |
| 3 folds x 1 seed screening + greedy carry-forward (noise chasing) | ICH_MAXPERF | all arms pre-registered; 5 folds; paired tests with Holm correction |
| Fabricated/misattributed citations ("Kwon et al. MELBA 2023" = Rasoulian et al.; "Wu et al. Frontiers 2023" = Spahr et al.) | ICH_MAXPERF | every reference checked; see `docs/literature.md` |
| Orientation never canonicalised (the "horizontal flip" may have mirrored front-back) | both | canonical RAS orientation + pixel-level left-right symmetry audit; the audit also measures which axis the original flip mirrored |

## What "unbelievable performance" can honestly mean on this data

The per-slice lesion IoU of the original pipeline was about 0.29-0.36 once the leak was
removed. The largest legitimate gains come from **more training data** (BHSD, 192
voxel-labelled scans) and from **reporting the standard metric**, per-patient volumetric
Dice, alongside the harsher per-slice view. For reference, HemaContour (npj Digit Med
2025) trained on INSTANCE-2022 reached per-patient Dice 0.843 on PhysioNet as an external
test. Detection metrics (scan-level AUC, sensitivity) are usually much higher than
segmentation overlap and are reported separately. Nothing in the generated manuscript is a
number that was not measured: missing values print as highlighted `[pending: ...]`.

## Running it (Google Colab, L4 or A100)

1. Put the PhysioNet zip
   (`computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip`)
   anywhere in MyDrive. BHSD downloads itself (or put `label_192.zip` in `MyDrive/ich_v3/raw/`).
2. Open `notebooks/ICH_v3.ipynb` in Colab. With `QUICK = True`, run all cells (~30-45 min):
   every path runs with 1-epoch models in `MyDrive/ich_v3_quick`.
3. Set `QUICK = False`, restart, run Sections 0-3, then the phase cells in order.
   **Everything is resumable**: rerun Section 0-3 after a disconnect, then the cell that was
   running. Finished runs are skipped; an interrupted custom run resumes from its last epoch;
   nnU-Net resumes from its last 5-epoch checkpoint.
4. Section 7 freezes post-processing; set `OPEN_TEST_SETS = True` in Section 8 once; Sections
   9-10 write tables, figures, `ICH_manuscript_revised.docx` and `Response_to_reviewers.docx`
   to `MyDrive/ich_v3/manuscript/`.

GPU budget (L4, measured times replace these estimates as runs finish; see Section 5):

| order | phase | runs | estimate |
|---|---|---|---|
| 1 | nnU-Net 2D, PhysioNet+BHSD, 5 folds (headline) | 5 | ~1.7 h/fold, set by `HOURS_PER_FOLD` |
| 2 | E1 original model and recipe | 5 | ~0.5 h |
| 3 | E4 none / flip / moderate / extreme, 3 seeds | 60 | ~4-5 h |
| 4 | E4 family x magnitude ladder | 80 | ~6 h |
| 5 | nnU-Net 2D, PhysioNet only | 5 | ~1.7 h/fold |
| 6 | E2 imbalance, E3 windows, E5 encoders | ~70 new | ~5 h |

Stopping after any phase still leaves a coherent paper; later phases only add tables.

## Layout

```
ich/            package (data, splits, preprocessing, augmentation, models, training,
                nnU-Net, metrics, post-processing, statistics, report, manuscript)
notebooks/      ICH_v3.ipynb (generated by tools/make_notebook.py)
tests/          pytest on synthetic phantoms, incl. exact resume, a real CPU nnU-Net round
                trip, the full quick pipeline, and (ICH_RUN_NOTEBOOK=1) the notebook itself
docs/           literature.md: citation audit and verified reference list
```

Run the tests with `pip install -r requirements.txt torch pytest && pytest tests/`.

## Extensions (November 2026, `ich/extensions.py`, untested)

Written without being run, at the author's request. Nothing in the pipeline imports the
module, so registered arms and hashes are unchanged. Run the tests and a QUICK pass first.
Arms X1 and X5 were not in the pre-registration: either amend it before running
(`experiments.amend`) or report them as exploratory.

| id | question | cost | needs |
|---|---|---|---|
| X1 `leakage_demo` | How much did the original slice-level split inflate the result? Same recipe, scored on held-out slices vs unseen scans | 3 runs of the paper recipe (~0.3 h) | splits, index, cache |
| X2 `affinity_diversity` | Measures, instead of asserting, how far each augmentation policy moves data off-distribution (Gontijo-Lopes 2021). Replaces the manifold-hypothesis argument | inference only | E4 `base` weights + E4 histories |
| X3 `volumetry` | Volume agreement (ICC, Bland-Altman, agreement at the 30 mL EDH threshold) | none | any per-scan table |
| X4 `fold_uncertainty` | Does disagreement between the 5 fold models flag failed scans? Voxel calibration | inference only | kept fold weights |
| X5 `run_scaling` | Learning curve over 25/50/75/100 % of scans, + BHSD: is data the constraint? | 25 runs (~2 h) | splits |
| X6 `augmentation_shift` | Does strong augmentation that hurts in-domain help on external BHSD? | none | `test_aug_*.json` |

Notebook usage, after Section 8 (test sets opened):

```python
from ich import extensions as X
X.leakage_demo(paths, index, cache, split)             # X1
X.affinity_diversity(paths, index, cache, split)       # X2
S = pd.read_csv(paths.tables / "test_nnUNet_PplusB/physionet_recipe_scans.csv")
X.volumetry(S); X.fig_bland_altman(S, paths.figures / "fig_bland_altman.png")   # X3
X.fold_uncertainty(paths, cache, split)                # X4
X.run_scaling(paths, index, cache, split)              # X5
X.augmentation_shift(paths, cache, split)              # X6
```

## Things to check by hand before resubmitting

- **Was the 15-scan PhysioNet test set ever scored in the earlier ICH_MAXPERF notebook?**
  `splits.py` deliberately reuses the same 15 scans. If any earlier run reported numbers on
  them, they are not untouched. Say so in the methods, or draw a new test set before training.
- **Commit `preregistration.json` to git before the first real run.** The commit
  timestamp is what makes "fixed before training" checkable. Call it an analysis plan
  fixed in advance, not a formal pre-registration, since it is not on a registry.
- **Frey & Hazenfield 2018 is real.** Restore it (see `docs/literature.md`, row 11).

- BHSD licence (sources disagree: MIT vs CC BY-NC); both allow academic use, cite Wu et al. 2023.
- Full author list for Inkeaw et al. 2022 is now in `docs/literature.md` section 3.
- O'Neill et al. 2021: copy the full author list from the RSNA page (the Crossref record would not load).
- The INSTANCE overview is an arXiv preprint; replace if a peer-reviewed version exists.
- Page count after the real tables are filled (limit 20 pages single-spaced).
