"""Tests for the ECG5000 options used by the 4-class sweep:
  1. data.drop_classes removes classes and renumbers the rest (ECG5000Dataset),
  2. the datamodule splits only the kept classes and checks data.num_classes,
  3. active.balanced=True seeds 5 labels per class, the rest of the budget random.
"""
import os
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.ecg5000_dataset import ECG5000Dataset
from run_training import get_torchvision_dm, label_active_dm

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.abspath("datasets"))
ECG_ROOT = os.path.join(DATA_ROOT, "ecg5000")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(ECG_ROOT, "ecg5000_all.npz")),
    reason="ECG5000 data not downloaded",
)


def _cfg(num_classes=4, drop_classes=(4,), num_labelled=25, balanced=True, seed=12345):
    return OmegaConf.create({
        "data": {"name": "ecg5000", "shape": [1, 140], "num_classes": num_classes,
                 "drop_classes": list(drop_classes), "val_split": 500, "mean": [0.0],
                 "std": [1.0], "transform_train": "ecg_basic", "transform_test": "basic"},
        "trainer": {"data_root": DATA_ROOT, "batch_size": 8, "num_workers": 0,
                    "seed": seed, "persistent_workers": False, "timeout": 0},
        "active": {"num_labelled": num_labelled, "min_train": num_labelled,
                   "balanced": balanced, "random_split": True},
    })


def test_drop_last_class_keeps_labels():
    full = ECG5000Dataset(root=ECG_ROOT, split="all")
    dropped = ECG5000Dataset(root=ECG_ROOT, split="all", drop_classes=[4])
    keep = full.targets != 4
    assert len(dropped) == keep.sum()
    np.testing.assert_array_equal(dropped.targets, full.targets[keep])
    np.testing.assert_array_equal(dropped.data, full.data[keep])
    assert dropped.classes == [0, 1, 2, 3]
    assert dropped.class_names[-1] == "SP/Ectopic Beat"


def test_drop_middle_class_renumbers():
    full = ECG5000Dataset(root=ECG_ROOT, split="all")
    dropped = ECG5000Dataset(root=ECG_ROOT, split="all", drop_classes=[2])
    old = full.targets[full.targets != 2]
    np.testing.assert_array_equal(dropped.targets, np.array([0, 1, -1, 2, 3])[old])


def test_datamodule_splits_kept_classes_only():
    dm = get_torchvision_dm(_cfg())
    for subset in (dm.val_set, dm.test_set):
        labels = subset.dataset.targets[subset.indices]
        assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert set(np.unique(dm.train_set.pool.targets)) == {0, 1, 2, 3}
    n_total = len(dm.val_set) + len(dm.test_set) + len(dm.train_set.pool)
    assert n_total == (ECG5000Dataset(root=ECG_ROOT, split="all").targets != 4).sum()


def test_num_classes_must_match_drop():
    with pytest.raises(AssertionError, match="num_classes"):
        get_torchvision_dm(_cfg(num_classes=5))


@pytest.mark.parametrize("num_labelled", [25, 50])
def test_balanced_initial_set_seeds_each_class(num_labelled):
    cfg = _cfg(num_labelled=num_labelled, balanced=True)
    np.random.seed(cfg.trainer.seed)
    dm = get_torchvision_dm(cfg)
    label_active_dm(cfg, num_labelled, True, dm)
    counts = np.bincount(dm.train_set.labelled_set.targets, minlength=4)
    assert counts.sum() == num_labelled
    assert (counts >= 5).all()
