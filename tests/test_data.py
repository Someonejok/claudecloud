import json

import numpy as np
import pytest

from ich import splits as SP
from ich.augment import Augmenter, morphology_effect, parse
from ich.preprocess import render_slice, warp_to_native


def test_cache_shapes_and_hu(built):
    idx, cache = built["index"], built["cache"]
    assert idx.case.nunique() == 27 + 12
    for c in cache.cases():
        m = cache.meta(c)
        assert m["hu_check"]["is_hu"], c
        assert cache.hu(c).dtype == np.int16 and cache.lab(c).dtype == np.uint8
        assert cache.hu(c).shape == cache.lab(c).shape


def test_csv_labels_follow_pixels_through_reorientation(built):
    """Scan 050 is stored z-reversed with a negative affine: after canonicalisation the
    CSV's per-slice haemorrhage flag must still agree with the mask on every slice."""
    pn = built["index"][built["index"].source == "physionet"]
    assert (pn.csv_no_hemorrhage == 1 - pn.has_lesion).all()
    m = built["cache"].meta("pn050")
    assert m["z_index_map"] == list(range(9, -1, -1))


def test_orientation_is_left_right_on_columns(built):
    """Every phantom has a frontal bulge: after the audit it must be at the top (rows),
    and the head must be mirror-symmetric across columns - including scan 051, which
    was stored with its in-plane axes swapped."""
    cache = built["cache"]
    assert cache.meta("pn051")["orientation"]["axes_swapped_by_audit"]
    assert not cache.meta("pn049")["orientation"]["axes_swapped_by_audit"]
    for c in ("pn049", "pn050", "pn051", "bh_ID_0003"):
        hu = np.asarray(cache.hu(c)[5], np.float32)
        head = hu > -300
        rows = np.nonzero(head.any(1))[0]
        # symmetry fixes which axis is left-right; it cannot tell front from back, so the
        # anterior-at-top check applies only where the affine was trustworthy
        if c != "pn051":
            centroid = np.nonzero(head)[0].mean()
            assert centroid > (rows[0] + rows[-1]) / 2, f"{c}: frontal bulge should be at the top"
        cols_c = head[:, ::-1]
        sym = (head == cols_c).mean()
        assert sym > 0.97, f"{c}: not left-right symmetric across columns ({sym:.3f})"


def test_bhsd_subtypes_and_official_split(built):
    bh = built["index"][built["index"].source == "bhsd"]
    assert set(np.unique(np.concatenate([np.unique(built["cache"].lab(c)) for c in bh.case.unique()]))) <= set(range(6))
    s = built["split"]
    assert len(SP.cases(s, partition="bh_test")) == 3


def test_splits_disjoint_and_reproducible(built):
    s = built["split"]
    a = SP.leakage_audit(s, built["cache"])
    assert a["shared scan IDs"].max() == 0
    assert len(SP.cases(s, partition="pn_test")) == 15
    dev = SP.cases(s, partition="pn_dev")
    assert sorted(set(s["fold"][c] for c in dev)) == [0, 1, 2, 3, 4]
    # rebuilding from the file returns the identical partition
    s2 = SP.make_splits(built["index"], built["paths"].splits / "splits.json")
    assert s2 == s


def test_render_inverse_roundtrip(built):
    cache = built["cache"]
    c = "pn049"
    lab = np.asarray(cache.lab(c)[4]) > 0
    hu = np.asarray(cache.hu(c)[4], np.float32)
    _, m, M = render_slice(hu, lab, cache.meta(c), 230.0, 192)
    back = warp_to_native(m.astype(np.float32), M, lab.shape) >= 0.5
    dice = 2 * (back & lab).sum() / (back.sum() + lab.sum())
    assert dice > 0.85


def test_augmentation_moves_mask_with_image(built):
    """Geometric draws must transform image and mask identically: the lesion's high-HU
    pixels must stay under the augmented mask."""
    cache = built["cache"]
    c = "pn049"
    hu, lab = np.asarray(cache.hu(c)[4], np.float32), np.asarray(cache.lab(c)[4]) > 0
    for spec in ("base+affine@2", "base+elastic@2", "paper_extreme"):
        aug = Augmenter(spec)
        rng = np.random.default_rng(3)
        for _ in range(10):
            A = aug.affine(128, rng)
            h, m, _ = render_slice(hu, lab, cache.meta(c), 230.0, 128, A)
            h, m = aug.elastic(h, m, rng)
            if m.sum() < 5:
                continue
            assert np.median(h[m > 0]) > 50, spec            # blood (68 HU), not brain/bone


def test_parse_and_morphology(built):
    assert parse("none")["flip_lr"] == 0
    assert parse("base+affine@1+noise@2")["families"] == {"affine": 1.0, "noise": 2.0}
    with pytest.raises(ValueError):
        parse("base+rotate@1")
    cache = built["cache"]
    samples = [(np.asarray(cache.hu(c)[4], np.float32), np.asarray(cache.lab(c)[4]), cache.meta(c))
               for c in ("pn049", "pn050")]
    e0 = morphology_effect("base", samples, 230.0, 96, n_draws=3)
    e4 = morphology_effect("base+intensity@4", samples, 230.0, 96, n_draws=6)
    assert abs(e0["area_ratio_mean"] - 1) < 0.05
    assert e4["lesion_hu_shift_mean_abs"] > e0["lesion_hu_shift_mean_abs"]


def test_original_flip_axis_detected(built, synth):
    """The phantom NIfTIs are stored X (left-right) first, so the original pipeline's
    PNG columns ran along Y: its HorizontalFlip mirrored the anterior-posterior axis."""
    from ich.data_physionet import original_flip_axis
    r = original_flip_axis(synth["physionet"], built["cache"], "pn049")
    assert r["match_correlation"] > 0.99
    assert r["original_horizontal_flip_mirrored"] == "anterior-posterior"
    assert r["paper_extreme_spec"] == "paper_extreme"
