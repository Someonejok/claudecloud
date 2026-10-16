"""Execute the Colab notebook's code cells in quick mode against the synthetic data.

Section 0 (Colab/Drive/git) and the data-download half of Section 1 are replaced by local
equivalents; every other cell runs as written. nnU-Net runs on CPU with 2 iterations per
epoch. Slow: run with  pytest tests/test_notebook.py  (it is skipped by default)."""
import json
import os
import shutil
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks" / "ICH_v3.ipynb"


@pytest.mark.skipif(not os.environ.get("ICH_RUN_NOTEBOOK"), reason="set ICH_RUN_NOTEBOOK=1")
def test_notebook_quick(synth, tmp_path, monkeypatch):
    monkeypatch.setenv("ICH_ROOT", str(tmp_path / "drive" / "ich_v3_quick"))
    monkeypatch.setenv("ICH_CACHE", str(tmp_path / "drive" / "ich_v3" / "cache"))
    monkeypatch.setenv("ICH_LOCAL", str(tmp_path / "local"))
    monkeypatch.setenv("ICH_NNUNET_ITERS", "2")
    cells = [''.join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    setup = f'''
QUICK = True
import os, sys, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
def display(*a): [print(x) for x in a]
class Image:
    def __init__(self, *a, **k): pass
import ich
from ich import cache as C, data_physionet, data_bhsd, splits as SP, experiments as EX
from ich import headline as HL, evaluate as EV, report as RP
from ich.config import default_paths
P = default_paths().make()
NUM_WORKERS = 0
LOCAL_PRE = Path(r"{tmp_path}") / "pre"
'''
    data = f'''
LC = C.Cache(P.local_cache).make()
data_physionet.build(Path(r"{synth['physionet']}"), LC, P.local_cache / "index_physionet.csv")
data_bhsd.build(Path(r"{synth['bhsd']}"), LC, P.local_cache / "index_bhsd.csv")
C.sync_cache(P.local_cache, P.cache)
INDEX = C.load_index(P.local_cache)
PN_ROOT = Path(r"{synth['physionet']}")
'''
    import matplotlib
    matplotlib.use("Agg")
    g = {}
    exec(setup, g)
    exec(data, g)
    for src in cells[2:]:
        src = src.replace("plt.show()", "plt.close('all')")
        exec(src, g)
    P = g["P"]
    assert (P.root / "results_summary.json").exists()
    S = json.loads((P.root / "results_summary.json").read_text())
    assert S["test"], "no test results"
    assert any(P.root.joinpath("manuscript").glob("*.docx"))
