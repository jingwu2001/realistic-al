"""Tests for the whole-stay (unpadded) path of preprocess_mimic3_sand.py
(--hours full): the ragged builder must reproduce the dense (N, T, V) arrays
row for row, including the per-stay reset of the delta recursion."""
import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.preprocess_mimic3_sand import ragged_hourly, stay_rows


def _dense(ts_ind, hour, var_ind, value, N, T, V):
    """The --hours H construction (steps 4-5), padded to T."""
    values_raw = np.zeros((N, T, V), np.float32)
    obs = np.zeros((N, T, V), np.float32)
    values_raw[ts_ind, hour, var_ind] = value
    obs[ts_ind, hour, var_ind] = 1.0
    delta = np.zeros((N, T, V), np.float32)
    delta[:, 0, :] = obs[:, 0, :]
    for t in range(1, T):
        delta[:, t, :] = (1 - obs[:, t, :]) * (1 + delta[:, t - 1, :])
    return values_raw, obs, delta


def test_ragged_matches_dense():
    rng = np.random.default_rng(0)
    N, V = 7, 5
    n_real = rng.integers(3, 30, size=N)
    n_ev = 200
    ts_ind = rng.integers(0, N, n_ev)
    hour = np.array([rng.integers(0, n_real[i]) for i in ts_ind])
    var_ind = rng.integers(0, V, n_ev)
    value = rng.normal(size=n_ev).astype(np.float32)
    # one value per (stay, hour, variable), as upstream keeps the last in each hour
    _, keep = np.unique(np.c_[ts_ind, hour, var_ind], axis=0, return_index=True)
    ts_ind, hour, var_ind, value = ts_ind[keep], hour[keep], var_ind[keep], value[keep]

    vr, ob, de, offsets = ragged_hourly(ts_ind, hour, var_ind, value, n_real, V)
    dvr, dob, dde = _dense(ts_ind, hour, var_ind, value, N, int(n_real.max()), V)
    assert offsets[-1] == n_real.sum()
    for i in range(N):
        sl, L = slice(offsets[i], offsets[i + 1]), n_real[i]
        np.testing.assert_array_equal(vr[sl], dvr[i, :L])
        np.testing.assert_array_equal(ob[sl], dob[i, :L])
        np.testing.assert_array_equal(de[sl], dde[i, :L])


def test_stay_rows():
    offsets = np.array([0, 3, 4, 9])
    rows, lens = stay_rows(offsets, np.array([2, 0]))
    np.testing.assert_array_equal(rows, [4, 5, 6, 7, 8, 0, 1, 2])
    np.testing.assert_array_equal(lens, [5, 3])
