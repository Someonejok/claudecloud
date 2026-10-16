"""BHSD (Wu et al., MLMI 2023) -> common cache.

192 head CT volumes with voxel-level labels for five haemorrhage subtypes. The archive
ships a dataset.json giving the label names and the official train/test split; both are
read from the file rather than assumed. Labels are remapped onto SUBTYPES order.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .cache import Cache, canonical_zyx, fingerprint, fix_hu, head_geometry, slice_rows
from .config import SUBTYPE_ABBR, SUBTYPES
from .data_physionet import orient_check_and_fix

MIRRORS = (
    "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip",
    "https://huggingface.co/datasets/Wendy-Fly/BHSD/resolve/main/label_192.zip",
)
ABBR_TO_CODE = {SUBTYPE_ABBR[s]: i for i, s in enumerate(SUBTYPES, start=1)}


def download(dest_dir: Path) -> Path:
    """Fetch label_192.zip unless it (or an extraction) is already on disk. Put the zip
    in dest_dir by hand if the mirrors are unavailable."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    z = dest_dir / "label_192.zip"
    if z.exists() and z.stat().st_size > 10_000_000:
        return z
    for url in MIRRORS:
        print("trying", url)
        if z.exists() and z.stat().st_size < 10_000_000:
            z.unlink()               # an error page from a failed mirror, not a partial zip
        r = subprocess.run(["wget", "-q", "-c", "-O", str(z), url])
        if r.returncode == 0 and z.exists() and z.stat().st_size > 10_000_000:
            return z
    raise FileNotFoundError(
        "Could not download BHSD. Download label_192.zip from https://github.com/White65534/BHSD "
        f"(or its Hugging Face mirror) and place it at {z}")


def extract(zip_path: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if not any(out_dir.rglob("*.nii*")):
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)], check=True)
    return out_dir


def print_licence(root: Path):
    for p in list(Path(root).rglob("LICENSE*")) + list(Path(root).rglob("licen*")):
        print(f"--- {p} ---\n{p.read_text(errors='ignore')[:800]}")
        return
    print("No licence file inside the archive. Check https://github.com/White65534/BHSD before "
          "publishing; the paper must cite Wu et al. (MLMI 2023) in any case.")


def _stem(p: Path) -> str:
    s = p.name
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return re.sub(r"_0000$", "", s)


def find_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    """Pair image and label volumes by file stem. Directories whose name mentions
    label/mask/ground/seg hold labels; everything else holds images."""
    root = Path(root)
    imgs, labs = {}, {}
    for p in root.rglob("*.nii*"):
        # only the immediate folder decides: the archive itself is called label_192
        d = p.parent.name.lower()
        is_img = any(k in d for k in ("image", "img"))
        is_lab = not is_img and any(k in d for k in ("label", "mask", "ground", "seg", "gt"))
        if is_lab:
            labs[_stem(p)] = p
        else:
            imgs[_stem(p)] = p
    pairs = [(k, imgs[k], labs[k]) for k in sorted(imgs) if k in labs]
    if not pairs:
        raise FileNotFoundError(f"no image/label pairs found under {root}")
    return pairs


def read_dataset_json(root: Path) -> dict:
    cands = list(Path(root).rglob("dataset.json"))
    return json.loads(cands[0].read_text()) if cands else {}


def label_mapping(dj: dict) -> dict[int, int]:
    """File label value -> our code (SUBTYPES order). Handles both nnU-Net v1 style
    {"0": "background", "1": "EDH"} and v2 style {"background": 0, "EDH": 1}."""
    labels = dj.get("labels") or {}
    pairs = {}
    for k, v in labels.items():
        name, val = (v, k) if str(k).isdigit() else (k, v)
        pairs[int(val)] = str(name)
    out = {}
    for val, name in pairs.items():
        up = name.upper()
        for abbr, code in ABBR_TO_CODE.items():
            if abbr in up:
                out[val] = code
    if not out:
        # documented order in the BHSD paper: EDH, IPH, IVH, SAH, SDH
        print("  dataset.json has no usable label names; assuming 1..5 = EDH, IPH, IVH, SAH, SDH")
        out = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return out


def official_split(dj: dict) -> dict[str, str]:
    """stem -> 'train' | 'test' from dataset.json, if it carries one."""
    out = {}
    for key, part in (("training", "train"), ("train", "train"), ("test", "test"),
                      ("testing", "test"), ("validation", "test")):
        for item in dj.get(key, []) or []:
            p = item.get("image") if isinstance(item, dict) else item
            if p:
                out[_stem(Path(str(p)))] = part
    return out


def build(root: Path, cache: Cache, out_index: Path, force: bool = False) -> pd.DataFrame:
    if out_index.exists() and not force:
        df = pd.read_csv(out_index)
        print(f"BHSD cache present: {df.case.nunique()} scans, {len(df)} slices")
        return df
    root = Path(root)
    dj = read_dataset_json(root)
    mapping = label_mapping(dj)
    split = official_split(dj)
    print(f"  label mapping file->ours: {mapping}; official split entries: {len(split)}")
    cache.make()
    rows, t0, seen_vals = [], time.time(), set()
    for stem, ip, lp in find_pairs(root):
        case = "bh_" + re.sub(r"[^A-Za-z0-9]+", "_", stem)
        img, limg = nib.load(str(ip)), nib.load(str(lp))
        hu_raw = img.get_fdata(dtype=np.float32)
        lab_raw = np.asanyarray(limg.dataobj).astype(np.int16)
        if hu_raw.shape != lab_raw.shape:
            print(f"  skip {stem}: image {hu_raw.shape} vs label {lab_raw.shape}")
            continue
        seen_vals |= set(np.unique(lab_raw).tolist())
        lab = np.zeros(lab_raw.shape, np.uint8)
        for val, code in mapping.items():
            lab[lab_raw == val] = code
        hu_zyx, spacing, _ = canonical_zyx(img, hu_raw)
        lab_zyx, _, _ = canonical_zyx(limg, lab)
        hu16, hu_rep = fix_hu(hu_zyx)
        hu16, lab_zyx, spacing, orient = orient_check_and_fix(hu16, lab_zyx.astype(np.uint8), spacing)
        geo = head_geometry(hu16)
        meta = {"case": case, "source": "bhsd", "patient": stem, "spacing": list(spacing),
                "hu_check": hu_rep, "orientation": orient, "label_kind": "subtype",
                "official_split": split.get(stem, ""), "file": str(ip.relative_to(root)), **geo,
                "fingerprint": fingerprint(hu16).tolist()}
        cache.write(case, hu16, lab_zyx, meta)
        rr = slice_rows(case, "bhsd", stem, lab_zyx, meta)
        for r in rr:
            r["official_split"] = meta["official_split"]
        rows += rr
    unmapped = sorted(v for v in seen_vals if v != 0 and v not in mapping)
    if unmapped:
        print(f"  WARNING: label values {unmapped} were not mapped to a subtype and were dropped")
    df = pd.DataFrame(rows)
    df.to_csv(out_index, index=False)
    print(f"BHSD: built {df.case.nunique()} scans / {len(df)} slices in {time.time()-t0:.0f}s")
    return df
