"""PhysioNet CT-ICH (Hssayeni et al.) -> common cache.

75 scans from traumatic brain injury patients, 5 mm axial slices, per-slice subtype
labels in hemorrhage_diagnosis_raw_ct.csv and binary haemorrhage masks.
"""
from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .cache import Cache, canonical_zyx, fingerprint, fix_hu, head_geometry, lr_symmetry_score, slice_rows
from .config import SUBTYPES

MARKER = "hemorrhage_diagnosis_raw_ct.csv"


def find_root(base: Path) -> Path:
    base = Path(base)
    if (base / MARKER).exists():
        return base
    for p in base.rglob(MARKER):
        if "gdrive" not in p.parts:
            return p.parent
    raise FileNotFoundError(f"{MARKER} not found under {base}")


def orient_check_and_fix(hu: np.ndarray, lab: np.ndarray, spacing):
    """Pixel-level check that columns are the left-right axis. Returns the (possibly
    transposed) arrays, spacing and the evidence."""
    zs = np.linspace(hu.shape[0] * 0.3, hu.shape[0] * 0.8, 6).astype(int)
    sc = np.array([lr_symmetry_score(hu[z]) for z in zs], dtype=float)
    col_sym, row_sym = np.nanmedian(sc[:, 0]), np.nanmedian(sc[:, 1])
    swapped = bool(np.isfinite(row_sym) and row_sym > col_sym + 0.05)
    if swapped:
        hu = np.ascontiguousarray(hu.transpose(0, 2, 1))
        lab = np.ascontiguousarray(lab.transpose(0, 2, 1))
        spacing = (spacing[0], spacing[2], spacing[1])
    return hu, lab, spacing, {"lr_symmetry_cols": float(col_sym), "lr_symmetry_rows": float(row_sym),
                              "axes_swapped_by_audit": swapped}


def build(root: Path, cache: Cache, out_index: Path, force: bool = False) -> pd.DataFrame:
    if out_index.exists() and not force:
        df = pd.read_csv(out_index)
        print(f"PhysioNet cache present: {df.case.nunique()} scans, {len(df)} slices")
        return df
    root = find_root(root)
    cache.make()
    dx = pd.read_csv(root / MARKER)
    dx.columns = [c.strip() for c in dx.columns]
    rows, t0 = [], time.time()
    for ct_p in sorted((root / "ct_scans").glob("*.nii*")):
        pid = int(ct_p.name.split(".")[0])
        mk_p = next((root / "masks").glob(f"{pid:03d}.nii*"))
        case = f"pn{pid:03d}"
        ct_img, mk_img = nib.load(str(ct_p)), nib.load(str(mk_p))
        hu_raw = ct_img.get_fdata(dtype=np.float32)          # applies scl_slope / inter
        mk_raw = (np.asanyarray(mk_img.dataobj) > 0).astype(np.uint8)
        assert hu_raw.shape == mk_raw.shape, f"{case}: CT {hu_raw.shape} vs mask {mk_raw.shape}"
        hu_zyx, spacing, zmap = canonical_zyx(ct_img, hu_raw)
        mk_zyx, _, zmap2 = canonical_zyx(mk_img, mk_raw)
        assert np.array_equal(zmap, zmap2), f"{case}: CT and mask affines disagree on z order"
        hu16, hu_rep = fix_hu(hu_zyx)
        hu16, mk_zyx, spacing, orient = orient_check_and_fix(hu16, mk_zyx.astype(np.uint8), spacing)
        sub = dx[dx.PatientNumber == pid].sort_values("SliceNumber").reset_index(drop=True)
        if len(sub) != hu16.shape[0]:
            print(f"  warning {case}: CSV has {len(sub)} slices, volume {hu16.shape[0]}")
        flags = []
        for k in range(hu16.shape[0]):
            orig = int(zmap[k])
            r = sub.iloc[orig] if orig < len(sub) else None
            f = {st: int(r[st]) if r is not None and st in r else 0 for st in SUBTYPES}
            f["Fracture"] = int(r["Fracture_Yes_No"]) if r is not None and "Fracture_Yes_No" in r else 0
            f["csv_no_hemorrhage"] = int(r["No_Hemorrhage"]) if r is not None else -1
            f["orig_slice_number"] = int(r["SliceNumber"]) if r is not None else -1
            flags.append(f)
        geo = head_geometry(hu16)
        meta = {"case": case, "source": "physionet", "patient": str(pid), "spacing": list(spacing),
                "z_index_map": zmap.tolist(), "hu_check": hu_rep, "orientation": orient,
                "label_kind": "binary", **geo, "fingerprint": fingerprint(hu16).tolist()}
        cache.write(case, hu16, mk_zyx.astype(np.uint8), meta)
        rows += slice_rows(case, "physionet", str(pid), mk_zyx, meta, flags)
    df = pd.DataFrame(rows)
    df.to_csv(out_index, index=False)
    print(f"PhysioNet: built {df.case.nunique()} scans / {len(df)} slices in {time.time()-t0:.0f}s")
    return df


def demographics(root: Path) -> pd.DataFrame:
    """Patient_demographics.csv, with its columns named for what they are."""
    root = find_root(root)
    p = root / "Patient_demographics.csv"
    if not p.exists():
        return pd.DataFrame()
    raw = pd.read_csv(p)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw
