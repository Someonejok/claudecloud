"""Real nnU-Net v2 round trip on the phantoms (CPU): export, plan/preprocess with our
folds, one tiny epoch, out-of-fold probabilities, test prediction. Slow (~minutes)."""
import json
import os
import shutil

import numpy as np
import pytest

from ich import headline as HL
from ich import nnunet_io as NN
from ich import splits as SP

pytestmark = pytest.mark.skipif(shutil.which("nnUNetv2_train") is None, reason="nnunetv2 not installed")


def test_nnunet_roundtrip(built, tmp_path):
    paths, cache, split = built["paths"], built["cache"], built["split"]
    NN.install_trainer()
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    cls = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                      NN.TRAINER, "nnunetv2.training.nnUNetTrainer")
    assert cls is not None

    which = "P+B"
    did, name = HL.DATASETS[which]
    tr = HL.training_cases(split, which)
    root = NN.export_dataset(paths, cache, did, name, tr)
    import SimpleITK as sitk
    c = tr[0]
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(root / "imagesTr" / f"{c}_0000.nii.gz")))
    assert np.array_equal(arr, np.asarray(cache.hu(c)))

    pre = tmp_path / "pre"
    folds = HL.nnunet_folds(split, which)
    assert all(set(f["train"]).isdisjoint(f["val"]) for f in folds)
    NN.plan_and_preprocess(paths, pre, did, name, folds)
    assert json.loads((pre / NN.dataset_name(did, name) / "splits_final.json").read_text()) == folds
    d = NN.train_fold(paths, pre, did, name, 0, epochs=1, extra_env={"ICH_NNUNET_ITERS": "2"},
                      extra_args=("-device", "cpu"))
    assert (d / "checkpoint_final.pth").exists()
    for c in folds[0]["val"]:
        p = NN.load_probability(d / "validation" / f"{c}.npz")
        assert p.shape == cache.hu(c).shape, c          # same orientation and grid as the cache
    # a finished fold is skipped, not retrained
    assert NN.train_fold(paths, pre, did, name, 0, epochs=1) == d
    test = SP.cases(split, partition=("pn_test", "bh_test"))[:2]
    in_dir = NN.export_test_images(paths, cache, test)
    out = NN.predict(paths, pre, did, name, in_dir, folds=(0,), epochs=1, extra_args=("-device", "cpu"))
    for c in test:
        assert NN.load_probability(out / f"{c}.npz").shape == cache.hu(c).shape
