"""Augmentation families with a single magnitude parameter.

Spec grammar:
    "none"                         no augmentation at all
    "base"                         left-right flip, p = 0.5 (anatomically valid on axial CT)
    "base+<family>@<m>[+<family>@<m>...]"   base plus families at magnitude m
    "paper_moderate", "paper_extreme"       Table 1 of the original manuscript, verbatim

Magnitude m = 1 reproduces the Table 1 value of each transform; m scales every range of
that family linearly and nothing else changes, so "intensity of augmentation" is varied
one family at a time (reviewer #4).

    family      m = 1 (Table 1)                         applied to
    affine      rotate +-10 deg, scale +-0.05, shift +-0.05 FOV, composed into the
                shear +-10 deg                             native->model warp (one interpolation)
    elastic     grid distortion, 5 steps, limit 0.3        remap on the model grid
    intensity   brightness +-0.10 W, contrast +-0.109,     HU before windowing (W = 120 HU,
                gamma 0.9-1.1                              the original window); gamma after
    noise       Gaussian, variance 5-10 (8-bit units at     HU, sigma = 1.05-1.49 HU at m = 1
                the original 120 HU window)

Intensity transforms act on Hounsfield units BEFORE windowing, so a brightness change is
a physical HU offset, the same whatever window or channel stack the model uses.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

W_REF = 120.0              # HU width of the original window, used to express Table 1 in HU
L_REF = 40.0
FAMILIES = ("affine", "elastic", "intensity", "noise")


def parse(spec: str) -> dict:
    spec = spec.strip()
    if spec == "none":
        return {"flip_lr": 0.0, "families": {}}
    if spec == "paper_moderate":
        return {"flip_lr": 0.0, "families": {}, "composite": "paper_moderate"}
    if spec in ("paper_extreme", "paper_extreme_lr"):
        return {"flip_lr": 0.0, "families": {}, "composite": "paper_extreme",
                "paper_flip": "lr" if spec.endswith("_lr") else "ap"}
    parts = spec.split("+")
    if parts[0] != "base":
        raise ValueError(f"augmentation spec must start with 'base', 'none' or 'paper_*': {spec}")
    fam = {}
    for p in parts[1:]:
        m = re.fullmatch(r"([a-z]+)@([0-9.]+)", p)
        if not m or m.group(1) not in FAMILIES:
            raise ValueError(f"bad family term {p!r} in {spec!r}; families: {FAMILIES}")
        fam[m.group(1)] = float(m.group(2))
    return {"flip_lr": 0.5, "families": fam}


def _affine_about_centre(size, rot_deg=0.0, scale=1.0, shear_deg=0.0, tx=0.0, ty=0.0, flip_lr=False,
                         flip_ap=False):
    c = (size - 1) / 2.0
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], float)
    a = math.radians(rot_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    Sh = np.array([[1, math.tan(math.radians(shear_deg)), 0], [0, 1, 0], [0, 0, 1]])
    F = np.diag([-1.0 if flip_lr else 1.0, -1.0 if flip_ap else 1.0, 1.0])
    T2 = np.array([[1, 0, c + tx * size], [0, 1, c + ty * size], [0, 0, 1]], float)
    return T2 @ F @ R @ Sh @ S @ T1


def _grid_maps(size, steps, limit, rng):
    """Separable grid distortion, as albumentations.GridDistortion(num_steps, distort_limit)."""
    cell = size / steps

    def axis():
        f = np.clip(1.0 + rng.uniform(-limit, limit, steps), 0.05, None)
        pos = np.concatenate([[0], np.cumsum(f * cell)])
        grid = np.arange(size, dtype=np.float32)
        return np.interp(grid / cell, np.arange(steps + 1), pos).astype(np.float32)
    mx, my = axis(), axis()
    return np.broadcast_to(mx[None, :], (size, size)).copy(), np.broadcast_to(my[:, None], (size, size)).copy()


class Augmenter:
    """Draws parameters from a numpy Generator so a run is reproducible and resumable."""

    def __init__(self, spec: str):
        self.spec = spec
        self.p = parse(spec)

    # -- geometry (composed into the native->model affine) --------------------------
    def affine(self, size, rng) -> np.ndarray | None:
        fam, comp = self.p["families"], self.p.get("composite")
        rot = sc = sh = tx = ty = 0.0
        flip_lr = flip_ap = False
        if comp == "paper_moderate":
            if rng.random() < 0.5:
                rot = rng.uniform(-10, 10)
        elif comp == "paper_extreme":
            # The original code chained every op at p = 1.0, including the flip. Its PNGs
            # were saved without reorientation, so "HorizontalFlip" mirrored the
            # anterior-posterior axis (see the orientation audit in the notebook).
            flip_ap = self.p.get("paper_flip", "ap") == "ap"
            flip_lr = not flip_ap
            rot = rng.uniform(-10, 10)
            sc = rng.uniform(-0.05, 0.05)
            tx, ty = rng.uniform(-0.05, 0.05, 2)
            sh = rng.uniform(-10, 10)
        else:
            flip_lr = rng.random() < self.p["flip_lr"]
            if "affine" in fam:
                m = fam["affine"]
                rot = rng.uniform(-10 * m, 10 * m)
                sc = rng.uniform(-0.05 * m, 0.05 * m)
                tx, ty = rng.uniform(-0.05 * m, 0.05 * m, 2)
                sh = rng.uniform(-10 * m, 10 * m)
        if not (rot or sc or sh or tx or ty or flip_lr or flip_ap):
            return None
        return _affine_about_centre(size, rot, max(1.0 + sc, 0.05), sh, tx, ty, flip_lr, flip_ap)

    # -- deformation on the model grid ------------------------------------------------
    def elastic(self, hu, mask, rng):
        fam, comp = self.p["families"], self.p.get("composite")
        lim = 0.3 * fam["elastic"] if "elastic" in fam else (0.3 if comp == "paper_extreme" else 0.0)
        if lim <= 0:
            return hu, mask
        mx, my = _grid_maps(hu.shape[0], 5, lim, rng)
        hu = cv2.remap(hu, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1024.0)
        if mask is not None:
            mask = (cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) >= 0.5).astype(np.uint8)
        return hu, mask

    # -- intensity in HU, before windowing --------------------------------------------
    def intensity_hu(self, hu, rng):
        fam, comp = self.p["families"], self.p.get("composite")
        b = c = 0.0
        sig = 0.0
        if comp == "paper_moderate":
            b, c = rng.uniform(-0.10142, 0.10142), rng.uniform(-0.10924, 0.10924)
        elif comp == "paper_extreme":
            b, c = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
            sig = math.sqrt(rng.uniform(5.0, 10.0)) / 255.0 * W_REF
        else:
            if "intensity" in fam:
                m = fam["intensity"]
                b, c = rng.uniform(-0.10 * m, 0.10 * m), rng.uniform(-0.109 * m, 0.109 * m)
            if "noise" in fam:
                m = fam["noise"]
                sig = m * math.sqrt(rng.uniform(5.0, 10.0)) / 255.0 * W_REF
        if b or c:
            hu = (hu - L_REF) * (1.0 + c) + L_REF + b * W_REF
        if sig:
            hu = hu + rng.normal(0.0, sig, hu.shape).astype(np.float32)
        return hu

    def gamma(self, x, rng):
        fam, comp = self.p["families"], self.p.get("composite")
        g = 1.0
        if comp == "paper_extreme":
            g = rng.uniform(0.9, 1.1)
        elif "intensity" in fam:
            m = fam["intensity"]
            g = max(rng.uniform(1 - 0.1 * m, 1 + 0.1 * m), 0.2)
        if g != 1.0:
            x = np.power(np.clip(x, 0, 1), g)
        return x


# ----------------------------------------------------------------------------- morphology
def morphology_effect(spec: str, samples, fov_mm: float, size: int, n_draws=20, seed=0) -> dict:
    """How far a policy moves lesions away from what blood looks like (reviewer R8).

    samples: iterable of (hu_native_slice, mask_native_slice, meta) for lesion slices.
    Compares each augmented rendering with the un-augmented rendering of the same slice.
    """
    from .preprocess import render_slice
    aug = Augmenter(spec)
    rng = np.random.default_rng(seed)
    area, hu_shift, in_blood0, in_blood1, compact0, compact1, lost = [], [], [], [], [], [], []

    def compact(m):
        m8 = m.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in cnts)
        a = m8.sum()
        return per ** 2 / (4 * math.pi * a) if a else np.nan

    for hu, mk, meta in samples:
        h0, m0, _ = render_slice(hu, mk, meta, fov_mm, size)
        if m0.sum() < 5:
            continue
        for _ in range(n_draws):
            A = aug.affine(size, rng)
            h1, m1, _ = render_slice(hu, mk, meta, fov_mm, size, A)
            h1, m1 = aug.elastic(h1, m1, rng)
            h1 = aug.intensity_hu(h1, rng)
            if m1.sum() == 0:
                lost.append(1.0)
                continue
            lost.append(0.0)
            area.append(m1.sum() / m0.sum())
            hu_shift.append(float(h1[m1 > 0].mean() - h0[m0 > 0].mean()))
            in_blood0.append(float(((h0[m0 > 0] >= 40) & (h0[m0 > 0] <= 100)).mean()))
            in_blood1.append(float(((h1[m1 > 0] >= 40) & (h1[m1 > 0] <= 100)).mean()))
            compact0.append(compact(m0)); compact1.append(compact(m1))
    f = lambda v: float(np.nanmean(v)) if len(v) else np.nan
    return {"spec": spec, "draws": len(lost), "lesion_lost_rate": f(lost),
            "area_ratio_mean": f(area), "area_ratio_p05": float(np.percentile(area, 5)) if area else np.nan,
            "area_ratio_p95": float(np.percentile(area, 95)) if area else np.nan,
            "lesion_hu_shift_mean_abs": f(np.abs(hu_shift)),
            "frac_lesion_px_in_blood_range_before": f(in_blood0),
            "frac_lesion_px_in_blood_range_after": f(in_blood1),
            "compactness_ratio": f(np.array(compact1) / np.array(compact0)) if compact0 else np.nan}
