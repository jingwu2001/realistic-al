"""Tests for the mimic3-benchmarks input (data=mimic3_pheno_bench[_full],
data=mimic3_ihm_bench).

Covers the converter's windowing (first/last hours, the benchmark Discretizer's
last-value + forward fill), right padding and the whole-stay (unpadded) storage,
train-only normalisation, the LSTM packing of padded rows (== Keras Masking),
and setup_mimic_input_dims (model.T / d_inp follow data.shape).
"""
import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.mimic_dataset import MIMIC3PhenotypeDataset
from data.preprocess_mimic3_benchmark import channel_header, discretize_stay, normalise_cohort, save_stays
from models.networks import build_model
from run_training import setup_mimic_input_dims

HEADER = ("Hours,Capillary refill rate,Diastolic blood pressure,Fraction inspired oxygen,"
          "Glascow coma scale eye opening,Glascow coma scale motor response,Glascow coma scale total,"
          "Glascow coma scale verbal response,Glucose,Heart Rate,Height,Mean blood pressure,"
          "Oxygen saturation,Respiratory rate,Systolic blood pressure,Temperature,Weight,pH")
HR = channel_header().index("Heart Rate")
HR_MASK = channel_header().index("mask->Heart Rate")


def _stay(tmp_path, rows):
    """rows: (hours, heart_rate) -> a benchmark timeseries CSV with only HR set."""
    p = tmp_path / "1_episode1_timeseries.csv"
    lines = [HEADER] + [f"{t}," + ",".join([""] * 8 + [str(hr)] + [""] * 8) for t, hr in rows]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_channel_layout():
    header = channel_header()
    assert len(header) == 76 and sum(h.startswith("mask->") for h in header) == 17


def test_discretize_forward_fill_and_last_value(tmp_path):
    # 5 h stay: HR 80 at 0.5 h, 90 then 95 inside hour 2 (last value wins)
    X = discretize_stay((_stay(tmp_path, [(0.5, 80), (2.2, 90), (2.8, 95)]), 5.0, None, "first"))
    assert X.shape == (5, 76)
    np.testing.assert_array_equal(X[:, HR], [80, 80, 95, 95, 95])  # forward fill
    np.testing.assert_array_equal(X[:, HR_MASK], [1, 0, 1, 0, 0])
    assert (np.abs(X).max(axis=1) > 0).all(), "a real hour is never all-zero"


@pytest.mark.parametrize("align,expected", [("first", [70, 70, 70]), ("last", [100, 100, 100])])
def test_window(tmp_path, align, expected):
    # 10 h stay = bins 0..9, window 3 h: HR 70 at hour 0, 100 at hour 7
    X = discretize_stay((_stay(tmp_path, [(0.1, 70), (7.1, 100)]), 10.0, 3, align))
    np.testing.assert_array_equal(X[:, HR], expected)


def test_normalisation_uses_train_only():
    stays = [np.array([[1.0, 1], [3, 1]]), np.full((4, 2), 5.0), np.full((4, 2), 100.0)]
    out, _, stats = normalise_cohort(stays, np.zeros((3, 2), np.float32), np.array([0, 1]), np.array([0]))
    assert stats["channel_mean"][0] == pytest.approx(np.mean([1, 3, 5, 5, 5, 5]))
    assert (out[0][:, 1] == 1).all(), "non-continuous channels are left as is"


@pytest.mark.parametrize("hours", [6, None])
def test_storage_roundtrip(tmp_path, hours):
    """Fixed window -> right-padded X.npy; full stay -> X_flat + offsets padded at
    load time. Both load back as right-padded (T, C) tensors."""
    stays = [np.full((2, 3), 1.0, np.float32), np.full((5, 3), 2.0, np.float32)]
    d = tmp_path / "all" / "train"
    d.mkdir(parents=True)
    save_stays(str(d), stays, hours)
    np.save(d / "demo.npy", np.zeros((2, 2), np.float32))
    np.save(d / "subject_ids.npy", np.array([1, 2]))
    np.save(d / "phenotype.npy", np.zeros((2, 25), np.int8))
    ds = MIMIC3PhenotypeDataset(str(tmp_path), train=True)
    T = 6 if hours else 5
    (ts0, _), _ = ds[0]
    (ts1, _), _ = ds[1]
    assert ts0.shape == ts1.shape == (T, 3)
    assert (ts0[:2] == 1).all() and not ts0[2:].any()
    assert (ts1[:5] == 2).all()


@torch.no_grad()
def test_lstm_packing_matches_unpadded():
    """A right-padded stay gives the same features as the stay on its own."""
    cfg = OmegaConf.load("src/config/model/lstm.yaml")
    cfg.update({"d_inp": 76, "use_static": False})
    torch.manual_seed(0)
    model = build_model(OmegaConf.create({"model": cfg}), num_classes=25).eval()
    g = torch.Generator().manual_seed(0)
    long, short = torch.randn(1, 10, 76, generator=g), torch.randn(1, 4, 76, generator=g)
    padded = torch.zeros(2, 10, 76)
    padded[0], padded[1, :4] = long[0], short[0]
    demo = torch.zeros(2, 2)
    batch = model.get_features((padded, demo))
    assert torch.allclose(batch[0], model.get_features((long, demo[:1]))[0], atol=1e-6)
    assert torch.allclose(batch[1], model.get_features((short, demo[:1]))[0], atol=1e-6)


def test_setup_mimic_input_dims():
    cfg = OmegaConf.create({"data": {"name": "mimic3_pheno", "shape": [48, 76]},
                            "model": {"T": 24, "d_inp": 387}})
    setup_mimic_input_dims(cfg)
    assert (cfg.model.T, cfg.model.d_inp) == (48, 76)
    other = OmegaConf.create({"data": {"name": "cifar10", "shape": [32, 32, 3]}, "model": {"T": 24}})
    setup_mimic_input_dims(other)
    assert other.model.T == 24


def test_deepcopy_shares_arrays(tmp_path):
    """ActiveLearningDataset.pool / labelled_set deepcopy the dataset on every
    access: the copies must share the (read-only) arrays, not duplicate them."""
    import copy
    d = tmp_path / "all" / "train"
    d.mkdir(parents=True)
    save_stays(str(d), [np.ones((4, 3), np.float32)] * 3, 4)
    np.save(d / "demo.npy", np.zeros((3, 2), np.float32))
    np.save(d / "subject_ids.npy", np.array([1, 2, 3]))
    np.save(d / "phenotype.npy", np.zeros((3, 25), np.int8))
    ds = MIMIC3PhenotypeDataset(str(tmp_path), train=True)
    dup = copy.deepcopy(ds)
    assert dup is not ds and dup._X is ds._X and dup.targets is ds.targets
    dup.some_pool_specific = 1                     # attributes stay per copy
    assert not hasattr(ds, "some_pool_specific")
