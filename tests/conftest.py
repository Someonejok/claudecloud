"""Synthetic PhysioNet-like and BHSD-like datasets (phantom heads, CPU only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

N_PN = 27      # 15 test + 12 development scans (5 folds)
N_BH = 12
SIZE, NZ, SP_XY = 96, 10, 2.4    # 96 px x 2.4 mm = 230 mm field of view


def phantom(rng, lesion_code=1, with_lesion=True, size=SIZE, nz=NZ):
    """Return (hu (X, Y, Z), lab (X, Y, Z)) in RAS voxel order: X = left-right,
    Y = posterior->anterior. The head is symmetric in X and has a frontal bulge (+Y)."""
    x, y = np.meshgrid(np.arange(size) - size / 2, np.arange(size) - size / 2, indexing="ij")
    hu = np.full((size, size, nz), -1024.0, np.float32)
    lab = np.zeros((size, size, nz), np.uint8)
    a, b = 30 + rng.uniform(-2, 2), 36 + rng.uniform(-2, 2)
    for z in range(nz):
        r = (x / a) ** 2 + (y / b) ** 2
        bulge = ((x / 10) ** 2 + ((y - b) / 8) ** 2) < 1          # anterior only
        head = (r < 1.0) | bulge
        skull = head & ((r > 0.82) | bulge)
        brain = head & ~skull
        sl = np.where(skull, 1100.0, np.where(brain, 32.0, -1024.0))
        sl += rng.normal(0, 3, sl.shape) * head
        hu[:, :, z] = sl
        if with_lesion and 3 <= z <= 6:
            crescent = brain & (r > 0.62) & (x > 8)
            hu[:, :, z][crescent] = 68.0 + rng.normal(0, 3, crescent.sum())
            lab[:, :, z][crescent] = lesion_code
    return hu, lab


def write_nii(path, arr, affine):
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def ras_affine(flip_z=False):
    a = np.diag([SP_XY, SP_XY, 5.0, 1.0])
    if flip_z:
        a[2, 2] = -5.0
        a[2, 3] = 5.0 * (NZ - 1)
    return a


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    pn = root / "physionet"
    (pn / "ct_scans").mkdir(parents=True); (pn / "masks").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    subtypes = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
    for i in range(N_PN):
        pid = 49 + i
        pos = i % 3 != 2
        hu, lab = phantom(rng, 1, pos)
        flip_z = (i == 1)
        if flip_z:
            # stored with the z axis reversed and a negative z affine: canonicalisation
            # must restore the order AND remap the per-slice CSV labels with it
            hu, lab = hu[:, :, ::-1].copy(), lab[:, :, ::-1].copy()
        aff = ras_affine(flip_z)
        if i == 2:
            # in-plane axes stored swapped with an identity-like affine: only the pixel
            # symmetry audit can catch this one
            hu, lab = hu.transpose(1, 0, 2).copy(), lab.transpose(1, 0, 2).copy()
        write_nii(pn / "ct_scans" / f"{pid:03d}.nii", hu.astype(np.float32), aff)
        write_nii(pn / "masks" / f"{pid:03d}.nii", lab.astype(np.float32), aff)
        st = subtypes[i % 5]
        for z in range(NZ):
            has = int(lab[:, :, z].any())
            r = {"PatientNumber": pid, "SliceNumber": z + 1, **{s: 0 for s in subtypes},
                 "No_Hemorrhage": 1 - has, "Fracture_Yes_No": 0}
            r[st] = has
            rows.append(r)
    pd.DataFrame(rows).to_csv(pn / "hemorrhage_diagnosis_raw_ct.csv", index=False)
    pd.DataFrame({"Patient Number": range(49, 49 + N_PN), "Age\n(years)": rng.integers(5, 70, N_PN),
                  "Gender": ["Male", "Female"] * (N_PN // 2) + ["Male"]}).to_csv(
        pn / "Patient_demographics.csv", index=False)

    bh = root / "bhsd" / "label_192"
    (bh / "images").mkdir(parents=True); (bh / "ground truths").mkdir()
    dj = {"labels": {"0": "background", "1": "EDH", "2": "IPH", "3": "IVH", "4": "SAH", "5": "SDH"},
          "training": [], "test": []}
    for i in range(N_BH):
        hu, lab = phantom(rng, 1 + i % 5, True)
        name = f"ID_{i:04d}"
        write_nii(bh / "images" / f"{name}.nii.gz", hu.astype(np.int16), ras_affine())
        write_nii(bh / "ground truths" / f"{name}.nii.gz", lab.astype(np.uint8), ras_affine())
        (dj["test"] if i < 3 else dj["training"]).append(
            {"image": f"./images/{name}.nii.gz", "label": f"./ground truths/{name}.nii.gz"})
    (bh / "dataset.json").write_text(json.dumps(dj))
    return {"root": root, "physionet": pn, "bhsd": bh}


@pytest.fixture(scope="session")
def built(synth, tmp_path_factory):
    from ich.cache import Cache, load_index
    from ich import data_bhsd, data_physionet
    from ich.config import Paths
    from ich.splits import make_splits
    work = tmp_path_factory.mktemp("work")
    paths = Paths(work / "drive", work / "local").make()
    cache = Cache(paths.cache)
    data_physionet.build(synth["physionet"], cache, paths.cache / "index_physionet.csv")
    data_bhsd.build(synth["bhsd"], cache, paths.cache / "index_bhsd.csv")
    index = load_index(paths.cache)
    split = make_splits(index, paths.splits / "splits.json")
    return {"paths": paths, "cache": cache, "index": index, "split": split}
