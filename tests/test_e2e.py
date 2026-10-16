"""Quick-mode path through Phases 3-5 and the report: E1 + the four E4 regimes on all
folds, post-processing chosen on development predictions, both test sets once, tables
and figures."""
import json

import numpy as np
import pytest

from ich import evaluate as EV
from ich import experiments as EX
from ich import report as RP


def test_pipeline_quick(built):
    paths, cache, index, split = built["paths"], built["cache"], built["index"], built["split"]
    EX.write_or_check_prereg(paths.root / "preregistration.json")
    EX.run_experiment("E1_paper", paths, index, cache, split, quick_mode=True, num_workers=0)
    EX.run_experiment("E4_augmentation", paths, index, cache, split, quick_mode=True, num_workers=0,
                      only=("none", "base", "paper_moderate", "paper_extreme", "affine@1", "affine@4"))
    assert len(list((paths.ledger / "E4_augmentation").glob("*.json"))) == 4 * 3 * 5 + 2 * 5
    for name in ("paper", "aug_base"):
        rec = EV.select_postprocessing(paths, cache, split, name)
        assert 0.1 <= rec["recipe"]["threshold"] <= 0.9
        # frozen: a second call returns the stored recipe unchanged
        assert EV.select_postprocessing(paths, cache, split, name) == rec
        res = EV.test_evaluation(paths, cache, split, index, name)
        assert "physionet_recipe" in res and "bhsd_recipe" in res
        assert res["physionet_recipe"]["n_scans"] == 15
    morph = RP.morphology_table(index, split, cache, n_slices=4, n_draws=2)
    morph.to_csv(paths.tables / "table_morphology.csv", index=False)
    S = RP.build(paths, index, split, cache, None)
    assert S["data"]["leakage_max_shared"] == 0
    assert "E4_augmentation" in S["cv"] and "E1_paper" in S["cv"]
    assert (paths.figures / "fig_dose_response.png").exists()
    assert (paths.figures / "fig_learning_curves.png").exists()
    q = RP.fig_qualitative(paths, cache, split, "aug_base", paths.figures / "fig_qualitative.png")
    assert q is None or q.exists()
    json.loads((paths.root / "results_summary.json").read_text())
    b = EX.budget(paths)
    assert b.done.sum() > 0
