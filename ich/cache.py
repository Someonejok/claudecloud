"""Common on-disk format for every dataset.

Per scan:
    hu/<case>.npy    int16 (Z, H, W)  Hounsfield units, display orientation
    lab/<case>.npy   uint8 (Z, H, W)  0 = background; PhysioNet: 1 = haemorrhage (subtype
                                       unknown per pixel); BHSD: 1..5 = SUBTYPES order
    meta/<case>.json spacing (sz, sy, sx) mm, head centre, provenance, HU checks
index.csv          one row per slice

Display orientation: rows run anterior -> posterior, columns run along the patient's
left-right axis, so a column flip is a genuine left-right mirror. This is enforced by
canonicalising the NIfTI affine and then checked on the pixels (`orientation_audit`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from .config import SUBTYPES

HU_MIN, HU_MAX = -1024, 3071


# ----------------------------------------------------------------------------- orientation
def canonical_zyx(img: nib.Nifti1Image, data: np.ndarray | None = None):
    """Reorient to RAS and return (array (Z, H, W), spacing (sz, sy, sx), z_index_map).

    z_index_map[k] = index of the ORIGINAL slice that became display slice k, so
    per-slice CSV labels follow their pixels even when the z axis is flipped.
    The in-plane axes are returned as rows = A->P (y reversed), cols = R/L (x)."""
    if data is None:
        data = np.asanyarray(img.dataobj)
    ornt = nib.orientations.io_orientation(img.affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    tr = nib.orientations.ornt_transform(ornt, target)
    if int(tr[2, 0]) != 2 or set(tr[:2, 0].astype(int)) != {0, 1}:
        # the slice axis would be swapped with an in-plane axis: these are axial
        # series, so this means a broken affine. Do not reorient; keep the voxel order.
        tr = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    ras = nib.orientations.apply_orientation(data, tr)
    nz = data.shape[2]
    zmap = np.arange(nz)
    if tr[2, 1] < 0:
        zmap = zmap[::-1]
    sp = np.empty(3)
    for new_ax in range(3):
        old_ax = int(np.where(tr[:, 0] == new_ax)[0][0])
        sp[new_ax] = zooms[old_ax]
    # (X, Y, Z) RAS  ->  (Z, H=rows, W=cols) with anterior at the top
    zyx = np.ascontiguousarray(np.transpose(ras, (2, 1, 0))[:, ::-1, :])
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
    return zyx, spacing, zmap.copy()


def lr_symmetry_score(hu_slice: np.ndarray) -> tuple[float, float]:
    """Correlation of a head slice with its column-mirror and with its row-mirror.
    On an axial head CT the left-right mirror correlates far better."""
    m = (hu_slice > -300).astype(np.float32)
    if m.sum() < 1000:
        return np.nan, np.nan
    rows, cols = np.nonzero(m)
    cy, cx = rows.mean(), cols.mean()
    sh = (int(round(m.shape[0] / 2 - cy)), int(round(m.shape[1] / 2 - cx)))
    x = np.roll(np.clip(hu_slice, -100, 200).astype(np.float32), sh, (0, 1))

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
    return corr(x, x[:, ::-1]), corr(x, x[::-1, :])


# ----------------------------------------------------------------------------- HU checks
def fix_hu(vol: np.ndarray) -> tuple[np.ndarray, dict]:
    """Bring a volume into Hounsfield units and clamp padding. Returns (int16, report)."""
    v = np.asarray(vol, dtype=np.float32)
    rep = {"raw_min": float(v.min()), "raw_max": float(v.max()),
           "raw_p01": float(np.percentile(v, 1)), "shifted_by": 0.0}
    # unsigned storage (0..4095) without rescale intercept: air sits near 0, not -1000
    if v.min() >= 0 and np.percentile(v, 1) < 50 and v.max() > 1000:
        v = v - 1024.0
        rep["shifted_by"] = -1024.0
    v = np.clip(v, HU_MIN, HU_MAX)
    head = v > -300
    brain = v[(v > 0) & (v < 80)]
    rep.update({"air_p05": float(np.percentile(v[~head], 5)) if (~head).any() else np.nan,
                "brain_median": float(np.median(brain)) if brain.size else np.nan,
                "is_hu": bool(brain.size and 15 <= np.median(brain) <= 60)})
    return np.round(v).astype(np.int16), rep


# ----------------------------------------------------------------------------- head geometry
def head_mask_slice(hu: np.ndarray) -> np.ndarray:
    m = ndi.binary_opening(hu > -300, np.ones((3, 3)))
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(np.argmax(sizes)) + 1)


def head_geometry(hu_zyx: np.ndarray) -> dict:
    """Bounding box of the head over all slices (largest > -300 HU component per
    slice, so the scanner table and stray pads drop out)."""
    union = np.zeros(hu_zyx.shape[1:], bool)
    for k in range(hu_zyx.shape[0]):
        h = head_mask_slice(hu_zyx[k])
        if h.sum() > 500:
            union |= h
    if not union.any():
        H, W = hu_zyx.shape[1:]
        return {"head_cy": H / 2, "head_cx": W / 2, "head_h": H, "head_w": W}
    r, c = np.nonzero(union)
    return {"head_cy": float((r.min() + r.max()) / 2), "head_cx": float((c.min() + c.max()) / 2),
            "head_h": int(r.max() - r.min() + 1), "head_w": int(c.max() - c.min() + 1)}


def fingerprint(hu_zyx: np.ndarray) -> np.ndarray:
    """Small normalised thumbnail of the three middle slices, for duplicate-scan
    detection across partitions and datasets."""
    import cv2
    z = hu_zyx.shape[0] // 2
    sl = [hu_zyx[max(0, min(hu_zyx.shape[0] - 1, z + o))] for o in (-1, 0, 1)]
    t = np.concatenate([cv2.resize(np.clip(s, -100, 200).astype(np.float32), (24, 24),
                                   interpolation=cv2.INTER_AREA).ravel() for s in sl])
    t = t - t.mean()
    return (t / (np.linalg.norm(t) + 1e-6)).astype(np.float32)


# ----------------------------------------------------------------------------- writer/reader
class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.hu_dir, self.lab_dir, self.meta_dir = (self.root / d for d in ("hu", "lab", "meta"))
        self._mm = {}

    def make(self):
        for d in (self.hu_dir, self.lab_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def has(self, case: str) -> bool:
        return all((d / f"{case}{ext}").exists() for d, ext in
                   ((self.hu_dir, ".npy"), (self.lab_dir, ".npy"), (self.meta_dir, ".json")))

    def write(self, case: str, hu: np.ndarray, lab: np.ndarray, meta: dict):
        assert hu.shape == lab.shape and hu.dtype == np.int16 and lab.dtype == np.uint8
        self.make()
        for d, arr in ((self.hu_dir, hu), (self.lab_dir, lab)):
            tmp = d / f"{case}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, d / f"{case}.npy")
        meta = dict(meta, shape=list(hu.shape),
                    sha1=hashlib.sha1(hu.tobytes()[:: max(1, hu.nbytes // 2_000_000)]).hexdigest())
        tmp = self.meta_dir / f"{case}.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, self.meta_dir / f"{case}.json")

    def meta(self, case: str) -> dict:
        return json.loads((self.meta_dir / f"{case}.json").read_text())

    def hu(self, case: str) -> np.ndarray:
        return self._get("hu", case)

    def lab(self, case: str) -> np.ndarray:
        return self._get("lab", case)

    def _get(self, kind, case):
        key = (kind, case)
        if key not in self._mm:
            d = self.hu_dir if kind == "hu" else self.lab_dir
            self._mm[key] = np.load(d / f"{case}.npy", mmap_mode="r")
            if len(self._mm) > 400:
                self._mm.pop(next(iter(self._mm)))
        return self._mm[key]

    def cases(self) -> list[str]:
        return sorted(p.stem for p in self.meta_dir.glob("*.json"))


def slice_rows(case: str, source: str, patient: str, lab: np.ndarray, meta: dict,
               slice_flags: list[dict] | None = None) -> list[dict]:
    sz, sy, sx = meta["spacing"]
    rows = []
    for k in range(lab.shape[0]):
        m = lab[k]
        px = int((m > 0).sum())
        r = {"case": case, "source": source, "patient": patient, "slice": k,
             "n_slices": lab.shape[0], "has_lesion": int(px > 0), "lesion_px": px,
             "lesion_mm2": px * sy * sx, "sz": sz, "sy": sy, "sx": sx,
             "H": lab.shape[1], "W": lab.shape[2],
             "head_cy": meta["head_cy"], "head_cx": meta["head_cx"]}
        if slice_flags is not None:
            r.update(slice_flags[k])
        else:  # subtype per pixel available
            for i, st in enumerate(SUBTYPES, start=1):
                r[st] = int((m == i).any())
        rows.append(r)
    return rows


def sync_cache(src: Path, dst: Path):
    """Copy the persistent cache (Drive) to fast local disk; only missing/changed files."""
    src, dst = Path(src), Path(dst)
    n = 0
    for p in src.rglob("*"):
        if p.is_dir() or p.suffix == ".tmp" or ".tmp." in p.name:
            continue
        q = dst / p.relative_to(src)
        if not q.exists() or q.stat().st_size != p.stat().st_size:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            n += 1
    return n


def load_index(cache_root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(Path(cache_root).glob("index_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no index_*.csv under {cache_root}; build the cache first")
    df = pd.concat(frames, ignore_index=True)
    for st in SUBTYPES:
        if st in df.columns:
            df[st] = df[st].fillna(0).astype(int)
    return df
