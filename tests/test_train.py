import json

import numpy as np
import pytest
import torch

from ich import experiments as EX
from ich import splits as SP
from ich.config import override
from ich.ledger import Ledger
from ich.models import PaperUNet, n_params
from ich.train import finish_run, train_run

TINY = override(EX.BASE, image_size=64, epochs=3, samples_per_epoch=16, batch_size=4, warmup_epochs=1,
                encoder_weights=None, encoder="resnet18", amp=False)


def test_paper_unet_matches_keras_parameter_count():
    """Keras printed 25,755,649 total = 25,749,633 trainable + 6,016 non-trainable
    (BatchNorm moving statistics), for a 256x256x1 input."""
    m = PaperUNet(1, 32, 0.01)
    assert n_params(m) == 25_749_633
    bn_stats = sum(b.numel() for n, b in m.named_buffers() if "running" in n)
    assert bn_stats == 6_016
    with torch.no_grad():
        assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def _cases(built, k=0):
    s = built["split"]
    return SP.cases(s, partition="pn_dev", not_fold=k), SP.cases(s, partition="pn_dev", fold=k)


def test_resume_is_exact(built, tmp_path):
    """A run killed after epoch 2 and resumed must end with the same weights as an
    uninterrupted run."""
    tr, va = _cases(built)
    a, *_ = train_run(TINY, built["index"], built["cache"], tr, va, 0, tmp_path / "a", num_workers=0,
                      log=lambda *x: None)
    _, h, info, _ = train_run(TINY, built["index"], built["cache"], tr, va, 0, tmp_path / "b",
                              num_workers=0, log=lambda *x: None, stop_after_epochs=2)
    assert info.get("stopped") and len(h) == 2
    b, h2, *_ = train_run(TINY, built["index"], built["cache"], tr, va, 0, tmp_path / "b", num_workers=0,
                          log=lambda *x: None)
    assert len(h2) == 3
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_all_losses_samplings_and_windows_train(built, tmp_path):
    tr, va = _cases(built, 1)
    for i, cfg in enumerate([
        override(TINY, epochs=1, loss="tversky", tversky_alpha=0.7, tversky_beta=0.3, sampling="all"),
        override(TINY, epochs=1, sampling="under1x", neg_slice_weight=0.2, pos_weight=5.0, window="zscore"),
        override(TINY, epochs=1, arch="paper_unet", window="brain_40_120", aug="paper_extreme", optimizer="rmsprop"),
        override(TINY, epochs=1, aug="base+affine@4+elastic@4+intensity@4+noise@4", window="brain_30_120"),
    ]):
        m, h, info, z = train_run(cfg, built["index"], built["cache"], tr, va, 0, tmp_path / str(i),
                                  num_workers=0, log=lambda *x: None)
        assert np.isfinite(h[-1]["train_loss"]), cfg.describe()
        if cfg.sampling == "under1x":
            assert info["neg_discarded_fraction"] > 0


def test_run_arm_ledger_skip_and_alias(built, monkeypatch):
    paths, cache, index, split = built["paths"], built["cache"], built["index"], built["split"]
    fast = override(TINY, epochs=1)
    monkeypatch.setattr(EX, "arms", lambda: {"X1": {"a": fast}, "X2": {"same": fast}})
    calls = []
    import ich.train as T
    real = T.train_run

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(T, "train_run", counting)
    EX.run_arm("X1", "a", fast, paths, index, cache, split, folds=[0], seeds=[0], num_workers=0)
    EX.run_arm("X1", "a", fast, paths, index, cache, split, folds=[0], seeds=[0], num_workers=0)
    EX.run_arm("X2", "same", fast, paths, index, cache, split, folds=[0], seeds=[0], num_workers=0)
    assert len(calls) == 1                         # second call skipped, X2 aliased
    rec = Ledger(paths.ledger, "X2").get("same", 0, 0)
    assert rec["alias_of"] and "dice_vol_mean" in rec["metrics"]
    # a changed arm definition is refused, not silently mixed
    with pytest.raises(RuntimeError):
        EX.run_arm("X1", "a", override(fast, lr=1e-2), paths, index, cache, split, folds=[0], seeds=[0],
                   num_workers=0)


def test_prereg_detects_changes(tmp_path, monkeypatch):
    p = tmp_path / "prereg.json"
    EX.write_or_check_prereg(p)
    EX.write_or_check_prereg(p)                     # unchanged: fine
    reg = EX.write_or_check_prereg(p, {"P+B": 120})
    assert json.loads(p.read_text())["nnunet"]["epochs"] == {"P+B": 120}
    orig = EX.arms

    def changed():
        E = orig()
        E["E2_imbalance"]["balanced"] = override(E["E2_imbalance"]["balanced"], lr=1.0)
        return E
    monkeypatch.setattr(EX, "arms", changed)
    with pytest.raises(RuntimeError):
        EX.write_or_check_prereg(p)
    EX.amend(p, "test amendment")
    EX.write_or_check_prereg(p)
    assert len(json.loads(p.read_text())["amendments"]) == 2


def test_arm_inventory():
    E = EX.arms()
    assert len(E["E4_augmentation"]) == 4 + 4 * 4
    hashes = [c.hash() for d in E.values() for c in d.values()]
    # BASE appears in four experiments and must hash identically everywhere
    assert hashes.count(EX.BASE.hash()) == 4
