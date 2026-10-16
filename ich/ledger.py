"""Results ledger: one JSON file per finished run.

A run is skipped when its JSON exists and records the same configuration hash. If an
arm's configuration changes after it ran, the stale record is reported instead of
being silently reused or overwritten.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def run_key(arm: str, seed: int, fold: int | str) -> str:
    return f"{arm}__s{seed}_f{fold}"


class Ledger:
    def __init__(self, root: Path, experiment: str):
        self.dir = Path(root) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment

    def path(self, arm, seed, fold) -> Path:
        return self.dir / f"{run_key(arm, seed, fold)}.json"

    def get(self, arm, seed, fold) -> dict | None:
        p = self.path(arm, seed, fold)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def done(self, arm, seed, fold, cfg_hash: str) -> bool:
        r = self.get(arm, seed, fold)
        if r is None:
            return False
        if r.get("config_hash") != cfg_hash:
            raise RuntimeError(
                f"{self.experiment}/{run_key(arm, seed, fold)} was run with config "
                f"{r.get('config_hash')} but the arm is now {cfg_hash}. The arm definition "
                f"changed after it ran. Either restore it, or delete {self.path(arm, seed, fold)} "
                f"(and its checkpoint) to rerun deliberately.")
        return True

    def put(self, arm, seed, fold, record: dict):
        rec = dict(record, experiment=self.experiment, arm=arm, seed=seed, fold=fold)
        _atomic_write_text(self.path(arm, seed, fold), json.dumps(rec, indent=1, default=_json_default))

    def frame(self) -> pd.DataFrame:
        rows = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                pass
        if not rows:
            return pd.DataFrame()
        flat = []
        for r in rows:
            f = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            for k, v in (r.get("metrics") or {}).items():
                f[k] = v
            flat.append(f)
        return pd.DataFrame(flat)

    def table_path(self, arm, seed, fold, name: str) -> Path:
        return self.dir / "tables" / f"{run_key(arm, seed, fold)}__{name}.csv"


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def all_ledgers(root: Path) -> pd.DataFrame:
    root = Path(root)
    frames = [Ledger(root, d.name).frame() for d in sorted(root.iterdir()) if d.is_dir()] \
        if root.exists() else []
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
