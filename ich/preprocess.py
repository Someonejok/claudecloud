"""HU processing and the native <-> model geometry.

Each slice reaches the model through ONE affine warp from native pixels: a square
field of view of `fov_mm` centred on the head, sampled at `image_size` pixels. Because
the FOV is in millimetres, every scan is shown at the same physical scale whatever its
pixel spacing. Geometric augmentation is composed into the same affine, so the image is
interpolated once. Predictions are mapped back to native pixels with the exact inverse
warp, and every metric is computed on the native grid the annotators drew on.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndi

from .config import WINDOWS

AIR = -1024.0


def window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Linear window to [0, 1]. Returns a new array."""
    lo = level - width / 2.0
    return np.clip((hu.astype(np.float32) - lo) / float(width), 0.0, 1.0)


def to_channels(hu: np.ndarray, window_name: str, zstats: dict | None = None) -> np.ndarray:
    """(H, W) HU -> (C, H, W) float32 network input."""
    w = WINDOWS[window_name]
    if w == "zscore":
        if zstats is None:
            raise ValueError("zscore window needs dataset statistics (see zscore_stats)")
        x = np.clip(hu.astype(np.float32), zstats["p005"], zstats["p995"])
        return ((x - zstats["mean"]) / zstats["std"])[None]
    return np.stack([window(hu, lv, wd) for lv, wd in w])


def zscore_stats(hu_values: np.ndarray) -> dict:
    """nnU-Net's CT scheme: statistics of the intensities INSIDE the lesion masks of the
    training scans, clip to their 0.5-99.5 percentiles, z-score."""
    v = np.asarray(hu_values, np.float32)
    return {"p005": float(np.percentile(v, 0.5)), "p995": float(np.percentile(v, 99.5)),
            "mean": float(v.mean()), "std": float(v.std() + 1e-6)}


# ----------------------------------------------------------------------------- geometry
def base_affine(meta: dict, fov_mm: float, size: int) -> np.ndarray:
    """3x3 matrix mapping native (col, row) pixel coords to model-input coords."""
    _, sy, sx = meta["spacing"]
    cy, cx = meta["head_cy"], meta["head_cx"]
    hx, hy = fov_mm / sx, fov_mm / sy           # crop extent in native pixels
    ax, ay = size / hx, size / hy
    return np.array([[ax, 0, -(cx - hx / 2) * ax],
                     [0, ay, -(cy - hy / 2) * ay],
                     [0, 0, 1]], dtype=np.float64)


def warp_to_model(arr: np.ndarray, M: np.ndarray, size: int, fill: float, nearest=False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(np.ascontiguousarray(arr, np.float32), M[:2], (size, size),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill))


def warp_to_native(arr: np.ndarray, M: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Exact inverse of warp_to_model for a probability map."""
    return cv2.warpAffine(np.ascontiguousarray(arr, np.float32), M[:2], (shape[1], shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def render_slice(hu: np.ndarray, mask: np.ndarray | None, meta: dict, fov_mm: float, size: int,
                 aug_affine: np.ndarray | None = None):
    """Native slice -> (hu_model (S,S) float32, mask_model (S,S) uint8 or None, M)."""
    M = base_affine(meta, fov_mm, size)
    if aug_affine is not None:
        M = aug_affine @ M
    h = warp_to_model(hu, M, size, AIR)
    m = None
    if mask is not None:
        # resample the mask as a float and threshold at 0.5: unbiased area, binary result
        m = (warp_to_model((mask > 0).astype(np.float32), M, size, 0.0) >= 0.5).astype(np.uint8)
    return h, m, M


# ----------------------------------------------------------------------------- brain mask
def brain_mask(hu: np.ndarray, spacing_yx: tuple[float, float], dilate_mm: float = 3.0) -> np.ndarray:
    """Intracranial soft tissue on one native slice: -20..150 HU, closed, holes filled,
    largest component, then dilated so extra-axial blood against the inner table is
    not cut away."""
    m = (hu > -20) & (hu < 150)
    m = ndi.binary_fill_holes(ndi.binary_closing(m, np.ones((5, 5))))
    lab, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        m = lab == (int(np.argmax(sizes)) + 1)
    r = max(1, int(round(dilate_mm / min(spacing_yx))))
    return ndi.binary_dilation(m, np.ones((2 * r + 1, 2 * r + 1)))


def skull_distance_mm(hu: np.ndarray, spacing_yx) -> np.ndarray:
    """Distance (mm) from every pixel to the nearest bone pixel (> 300 HU)."""
    bone = hu > 300
    if not bone.any():
        return np.full(hu.shape, np.inf, np.float32)
    return ndi.distance_transform_edt(~bone, sampling=spacing_yx).astype(np.float32)
