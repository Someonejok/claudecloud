"""nnU-Net v2 (2D configuration) as the headline model.

nnU-Net's 2D configuration is a 2D U-Net whose preprocessing is derived from the data:
in-plane resampling to the median spacing and, for CT, clipping to the 0.5-99.5
percentiles of the intensities inside the training lesion masks followed by z-scoring
with their mean and SD. It trains on a fixed schedule (poly learning-rate decay) and the
final checkpoint is the model - no checkpoint is chosen on validation data.

Our folds are imposed through splits_final.json so nnU-Net's out-of-fold predictions are
exactly comparable with the custom U-Net ablations. Training resumes from
checkpoint_latest.pth after a disconnect; finished folds are skipped.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

TRAINER = "nnUNetTrainer_ICH"
CALIB_TRAINER = "nnUNetTrainer_ICHcalib"
CONFIG = "2d"

TRAINER_SRC = '''
import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_ICH(nnUNetTrainer):
    """Fixed epoch budget (frozen in preregistration.json) and frequent checkpoints so a
    Colab disconnect costs at most a few epochs."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ["ICH_NNUNET_EPOCHS"])
        self.save_every = 5
        if os.environ.get("ICH_NNUNET_ITERS"):          # smoke tests only
            self.num_iterations_per_epoch = int(os.environ["ICH_NNUNET_ITERS"])
            self.num_val_iterations_per_epoch = max(1, self.num_iterations_per_epoch // 5)


class nnUNetTrainer_ICHcalib(nnUNetTrainer):
    """Two epochs, only to measure seconds per epoch on this GPU."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
'''


def env(paths, local_preprocessed: Path) -> dict:
    e = dict(os.environ)
    e["nnUNet_raw"] = str(paths.nnunet / "raw")
    e["nnUNet_preprocessed"] = str(local_preprocessed)
    e["nnUNet_results"] = str(paths.nnunet / "results")
    return e


def install_trainer():
    import nnunetv2
    d = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "ich"
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text("")
    (d / "nnUNetTrainer_ICH.py").write_text(TRAINER_SRC)
    return d


def dataset_name(did: int, name: str) -> str:
    return f"Dataset{did:03d}_{name}"


def _write_nii(path: Path, arr: np.ndarray, spacing_zyx):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    sitk.WriteImage(img, str(tmp), useCompression=True)
    os.replace(tmp, path)


def export_cases(cache, cases, img_dir: Path, lab_dir: Path | None):
    for c in cases:
        ip = img_dir / f"{c}_0000.nii.gz"
        if not ip.exists():
            _write_nii(ip, np.asarray(cache.hu(c), np.int16), cache.meta(c)["spacing"])
        if lab_dir is not None:
            lp = lab_dir / f"{c}.nii.gz"
            if not lp.exists():
                _write_nii(lp, (np.asarray(cache.lab(c)) > 0).astype(np.uint8), cache.meta(c)["spacing"])


def export_dataset(paths, cache, did: int, name: str, train_cases: list[str]) -> Path:
    root = paths.nnunet / "raw" / dataset_name(did, name)
    export_cases(cache, train_cases, root / "imagesTr", root / "labelsTr")
    dj = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "haemorrhage": 1},
          "numTraining": len(train_cases), "file_ending": ".nii.gz",
          "name": name, "description": "binary ICH; built by ich.nnunet_io"}
    (root / "dataset.json").write_text(json.dumps(dj, indent=1))
    stray = {p.name.replace("_0000.nii.gz", "") for p in (root / "imagesTr").glob("*_0000.nii.gz")} - set(train_cases)
    if stray:
        raise RuntimeError(f"{root} holds cases that are not in this training set: {sorted(stray)[:5]}")
    return root


def export_test_images(paths, cache, cases) -> Path:
    d = paths.nnunet / "test_images"
    export_cases(cache, cases, d, None)
    return d


def write_splits(local_preprocessed: Path, did: int, name: str, folds: list[dict]):
    d = Path(local_preprocessed) / dataset_name(did, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits_final.json").write_text(json.dumps(folds, indent=1))


def _run(cmd, e, log_file: Path | None = None):
    print("  $", " ".join(cmd))
    with open(log_file, "a") if log_file else open(os.devnull, "w") as fh:
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last = []
        for line in p.stdout:
            fh.write(line)
            last = (last + [line])[-40:]
            if re.search(r"Epoch \d+|Current learning rate|train_loss|val_loss|Pseudo dice|Epoch time", line):
                print("   ", line.rstrip())
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(cmd)}\n{''.join(last)}")


def plan_and_preprocess(paths, local_pre: Path, did: int, name: str, folds: list[dict]):
    """Preprocess once; persist a tarball on Drive so a new runtime restores it instead
    of recomputing."""
    tar = paths.nnunet / f"preprocessed_{did:03d}.tar"
    d = Path(local_pre) / dataset_name(did, name)
    if (d / "nnUNetPlans.json").exists() and any(d.glob("nnUNetPlans_2d/*")):
        pass
    elif tar.exists():
        print(f"  restoring preprocessed data from {tar}")
        with tarfile.open(tar) as t:
            t.extractall(local_pre)
    else:
        _run(["nnUNetv2_plan_and_preprocess", "-d", str(did), "-c", CONFIG,
              "--verify_dataset_integrity"], env(paths, local_pre))
        tmp = tar.with_suffix(".tmp")
        with tarfile.open(tmp, "w") as t:
            t.add(d, arcname=d.name)
        os.replace(tmp, tar)
    write_splits(local_pre, did, name, folds)


def fold_dir(paths, did, name, fold, trainer=TRAINER):
    return paths.nnunet / "results" / dataset_name(did, name) / f"{trainer}__nnUNetPlans__{CONFIG}" / f"fold_{fold}"


def calibrate(paths, local_pre, did, name) -> float:
    """Seconds per epoch on this GPU, from a 2-epoch run of fold 0 (then deleted)."""
    d = fold_dir(paths, did, name, 0, CALIB_TRAINER)
    e = env(paths, local_pre)
    t0 = time.time()
    _run(["nnUNetv2_train", str(did), CONFIG, "0", "-tr", CALIB_TRAINER], e)
    secs = None
    logs = sorted(d.glob("training_log_*.txt"))
    if logs:
        times = [float(x) for x in re.findall(r"Epoch time: ([0-9.]+) s", logs[-1].read_text())]
        if times:
            secs = float(np.median(times))
    secs = secs or (time.time() - t0) / 2
    shutil.rmtree(d.parent, ignore_errors=True)
    return secs


def train_fold(paths, local_pre, did, name, fold, epochs: int, extra_env: dict | None = None,
               extra_args: tuple = ()):
    d = fold_dir(paths, did, name, fold)
    if (d / "checkpoint_final.pth").exists():
        print(f"  fold {fold}: finished")
        return d
    e = env(paths, local_pre)
    e["ICH_NNUNET_EPOCHS"] = str(int(epochs))
    e.update(extra_env or {})
    cmd = ["nnUNetv2_train", str(did), CONFIG, str(fold), "-tr", TRAINER, "--npz", *extra_args]
    if (d / "checkpoint_latest.pth").exists():
        cmd.append("--c")
        print(f"  fold {fold}: resuming from checkpoint_latest.pth")
    d.mkdir(parents=True, exist_ok=True)
    _run(cmd, e, d / "ich_stdout.txt")
    return d


def predict(paths, local_pre, did, name, in_dir: Path, folds=(0, 1, 2, 3, 4), epochs: int = 0,
            extra_args: tuple = ()) -> Path:
    out = paths.nnunet / "predictions" / dataset_name(did, name)
    want = [p.name.replace("_0000.nii.gz", "") for p in Path(in_dir).glob("*_0000.nii.gz")]
    if out.exists() and all((out / f"{c}.npz").exists() for c in want):
        return out
    e = env(paths, local_pre)
    e["ICH_NNUNET_EPOCHS"] = str(int(epochs))
    _run(["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out), "-d", str(did), "-c", CONFIG,
          "-tr", TRAINER, "-f", *map(str, folds), "--save_probabilities", *extra_args], e)
    return out


def load_probability(npz_path: Path) -> np.ndarray:
    """Foreground probability (Z, H, W) in the geometry of the exported image, which is
    the cache's display orientation."""
    z = np.load(npz_path)
    p = z["probabilities"] if "probabilities" in z else z[z.files[0]]
    return p[1].astype(np.float32) if p.ndim == 4 else p.astype(np.float32)
