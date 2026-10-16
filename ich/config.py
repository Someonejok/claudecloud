"""Paths and run configuration.

A run is identified by (experiment, arm, seed, fold). The arm's configuration is
hashed, so changing any setting of an arm after it has run is detected rather than
silently mixed with old results.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

# ----------------------------------------------------------------------------- windows
# (level, width) in Hounsfield units.
WINDOWS = {
    "brain_40_80": ((40, 80),),
    "brain_40_120": ((40, 120),),        # what the original code used (Hssayeni et al.)
    "brain_30_120": ((30, 120),),        # what the original manuscript stated
    "three_window": ((40, 80), (80, 200), (600, 2800)),   # brain / subdural / bone
    "zscore": "zscore",                   # nnU-Net style: clip to lesion 0.5-99.5 pct, z-score
}

SUBTYPES = ("Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural")
SUBTYPE_ABBR = {"Epidural": "EDH", "Intraparenchymal": "IPH", "Intraventricular": "IVH",
                "Subarachnoid": "SAH", "Subdural": "SDH"}


@dataclass(frozen=True)
class Paths:
    """Everything persistent lives under `root` (Google Drive on Colab). The HU cache
    is copied to `local` for fast memory-mapped reads."""
    root: Path
    local: Path

    @property
    def cache(self) -> Path:          # persistent copy of the HU cache (shared by quick mode)
        return Path(os.environ["ICH_CACHE"]) if os.environ.get("ICH_CACHE") else self.root / "cache"

    @property
    def local_cache(self) -> Path:    # fast working copy
        return self.local / "cache"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger"

    @property
    def ckpt(self) -> Path:
        return self.root / "checkpoints"

    @property
    def probs(self) -> Path:
        return self.root / "probs"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def nnunet(self) -> Path:
        return self.root / "nnunet"

    def make(self) -> "Paths":
        for p in (self.root, self.local, self.cache, self.local_cache, self.splits, self.ledger,
                  self.ckpt, self.probs, self.tables, self.figures, self.nnunet):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    root = Path(os.environ.get("ICH_ROOT", "/content/gdrive/MyDrive/ich_v3"))
    local = Path(os.environ.get("ICH_LOCAL", "/content/ich_local"))
    return Paths(root, local)


# ----------------------------------------------------------------------------- run config
@dataclass(frozen=True)
class RunConfig:
    # input representation
    window: str = "brain_40_80"
    fov_mm: float = 230.0           # square field of view centred on the head, in mm
    image_size: int = 320           # model input side, pixels (pixel = fov_mm / image_size mm)
    in_plane_flip_audit: bool = True

    # model
    arch: str = "unet"              # unet | paper_unet
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    paper_filters: int = 32
    paper_dropout: float = 0.01

    # optimisation (fixed schedule: no early stopping, final weights are the model)
    epochs: int = 50
    samples_per_epoch: int = 800
    batch_size: int = 16
    optimizer: str = "adamw"        # adamw | adam | rmsprop
    lr: float = 3e-4
    encoder_lr_mult: float = 1.0
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    schedule: str = "cosine"        # cosine (to zero) | constant
    amp: bool = True
    grad_clip: float = 1.0

    # loss: L = bce_weight * BCE + (1 - bce_weight) * (1 - Dice_batch)   (or Tversky)
    loss: str = "bce_dice"          # bce_dice | tversky
    bce_weight: float = 0.5
    dice_smooth: float = 1.0
    pos_weight: float = 1.0         # BCE weight on lesion pixels
    tversky_alpha: float = 0.5      # weight on false negatives
    tversky_beta: float = 0.5       # weight on false positives
    neg_slice_weight: float = 1.0   # loss weight of lesion-free slices

    # sampling of training slices
    sampling: str = "balanced"      # balanced | all | under1x | under2x ...
    pos_fraction: float = 0.5       # balanced only

    # augmentation: "none", "base" (L-R flip) or "base+<family>@<magnitude>" or a named composite
    aug: str = "base"

    # evaluation
    threshold: float = 0.5
    tta_flip: bool = False
    save_probs: bool = False

    def describe(self) -> str:
        enc = self.encoder if self.arch == "unet" else "paper"
        return (f"{self.arch}:{enc} | {self.window} | {self.image_size}px/{self.fov_mm:.0f}mm | "
                f"{self.loss} | sampling={self.sampling} | aug={self.aug} | {self.epochs}ep")

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        # evaluation-only fields do not change the trained model
        d = {k: v for k, v in asdict(self).items() if k not in ("save_probs", "tta_flip")}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def in_channels(self) -> int:
        w = WINDOWS[self.window]
        return 1 if w == "zscore" else len(w)

    @property
    def pixel_mm(self) -> float:
        return self.fov_mm / self.image_size


def override(cfg: RunConfig, **kw) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    bad = set(kw) - known
    if bad:
        raise KeyError(f"unknown RunConfig fields: {sorted(bad)}")
    return replace(cfg, **kw)


def quick(cfg: RunConfig) -> RunConfig:
    """Smoke-test version of a configuration: every code path, almost no compute."""
    return replace(cfg, epochs=1, samples_per_epoch=32, batch_size=4, warmup_epochs=0,
                   image_size=min(cfg.image_size, 128))
