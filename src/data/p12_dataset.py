"""
PyTorch Datasets for the PhysioNet 2012 (P12) ICU mortality dataset.

Two dataset classes are provided:

  P12Dataset            — GRU-D format: (3, F, T_bins) per sample.
                          Channel 0 = normalised feature values (X),
                          Channel 1 = observation mask (M),
                          Channel 2 = time-since-last-observation in hours (Delta).
                          Produced by hourly binning (49 bins × 36 features).

  P12TransformerDataset — Transformer format: returns a 4-tuple
                          (ts, static, times, length) where
                            ts:      (215, 36) float32 — normalised time-series
                            static:  (9,)      float32 — normalised static features
                            times:   (215,)    float32 — timestamps in minutes
                            length:  scalar int        — actual sequence length

Both classes expose a `targets` attribute (numpy int array) for stratified splitting.

Pre-requisite: run the IrregularSampling.py preprocessing script so that
  <root>/PTdict_list.npy  and  <root>/arr_outcomes.npy  exist.

Data layout per entry in PTdict_list:
  'arr'             : (215, 36) float  — feature values, 0 where unobserved
  'time'            : (215, 1)  float  — timestamps in minutes (0-padded)
  'length'          : int               — actual number of observed timesteps
  'extended_static' : (9,)      float  — one-hot encoded static features;
                                         Age/Height/Weight may be -1 if unknown
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_FEATURES  = 36   # number of time-series features
_N_STATIC    = 9    # length of extended_static vector
_T_MAX       = 215  # maximum sequence length in PTdict_list
_N_BINS      = 49   # hourly bins for GRU-D (0 h … 48 h inclusive)
_EPS         = 1e-7


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _feature_stats(PTdict: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from all observed (non-zero) values."""
    means = np.zeros(_N_FEATURES, dtype=np.float32)
    stds  = np.ones(_N_FEATURES,  dtype=np.float32)
    for f in range(_N_FEATURES):
        vals = np.concatenate([p['arr'][:p['length'], f] for p in PTdict])
        obs  = vals[vals != 0]
        if obs.size > 0:
            means[f] = float(obs.mean())
            stds[f]  = float(obs.std())  + _EPS
    return means, stds


def _static_stats(PTdict: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-static-feature mean/std, ignoring negative (missing) entries."""
    all_static = np.array([p['extended_static'] for p in PTdict], dtype=np.float32)
    means = np.zeros(_N_STATIC, dtype=np.float32)
    stds  = np.ones(_N_STATIC,  dtype=np.float32)
    for d in range(_N_STATIC):
        vals = all_static[:, d]
        obs  = vals[vals >= 0]
        if obs.size > 0:
            means[d] = float(obs.mean())
            stds[d]  = float(obs.std()) + _EPS
    return means, stds


# ---------------------------------------------------------------------------
# P12Dataset  (GRU-D / hourly-binned format)
# ---------------------------------------------------------------------------

class P12Dataset(Dataset):
    """Returns ``(tensor, label)`` where tensor has shape ``(3, 36, 49)``.

    The three channels are normalised X, binary mask M, and time-delta D (hours)
    produced by binning the irregular time series into 49 one-hour slots.
    """

    def __init__(self, root: str) -> None:
        super().__init__()
        PTdict   = np.load(os.path.join(root, "PTdict_list.npy"),  allow_pickle=True)
        outcomes = np.load(os.path.join(root, "arr_outcomes.npy"), allow_pickle=True)

        self.targets = outcomes[:, -1].astype(np.int64)

        feat_means, feat_stds = _feature_stats(PTdict)

        self._data: list[np.ndarray] = []
        for p in PTdict:
            arr    = p['arr'].astype(np.float32)     # (215, 36)
            time   = p['time'].squeeze(1).astype(np.float32)  # (215,) minutes
            length = int(p['length'])

            # Normalise observed values; leave zeros (missing) as-is
            mask_obs  = arr != 0
            arr_norm  = np.where(mask_obs, (arr - feat_means) / feat_stds, 0.0)

            grud = self._hourly_bin(arr_norm, time, length)  # (3, 36, 49)
            self._data.append(grud)

        self._labels = self.targets

    # ------------------------------------------------------------------

    @staticmethod
    def _hourly_bin(arr: np.ndarray, time: np.ndarray, length: int) -> np.ndarray:
        """Convert an irregular time series to a fixed (3, F, 49) GRU-D tensor."""
        X     = np.zeros((_N_FEATURES, _N_BINS), dtype=np.float32)
        M     = np.zeros((_N_FEATURES, _N_BINS), dtype=np.float32)
        D     = np.zeros((_N_FEATURES, _N_BINS), dtype=np.float32)

        x_last = np.zeros(_N_FEATURES, dtype=np.float32)
        t_last = np.zeros(_N_FEATURES, dtype=np.float32)  # bin index of last obs

        for h in range(_N_BINS):
            t_start = h * 60.0
            t_end   = (h + 1) * 60.0

            in_bin  = (time[:length] >= t_start) & (time[:length] < t_end)
            indices = np.where(in_bin)[0]

            if indices.size > 0:
                last_row  = arr[indices[-1]]         # (F,) — last obs in this bin
                observed  = (last_row != 0)
                X[:, h]   = np.where(observed, last_row, x_last)
                M[:, h]   = observed.astype(np.float32)
                x_last    = np.where(observed, last_row, x_last)
                t_last    = np.where(observed, float(h), t_last)
            else:
                X[:, h] = x_last
                M[:, h] = 0.0

            D[:, h] = h - t_last  # hours since last observation

        return np.stack([X, M, D], axis=0)  # (3, 36, 49)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._data[idx])          # (3, 36, 49)
        y = int(self._labels[idx])
        return x, y


# ---------------------------------------------------------------------------
# P12TransformerDataset  (raw padded format for Transformer models)
# ---------------------------------------------------------------------------

class P12TransformerDataset(Dataset):
    """Returns ``((ts, static, times, length), label)`` per sample.

    * ts      : ``(215, 36)`` float32 — normalised time-series values (0 where missing)
    * static  : ``(9,)``      float32 — normalised static/demographic features
    * times   : ``(215,)``    float32 — timestamps in minutes (0-padded)
    * length  : scalar int             — actual sequence length
    * label   : scalar int             — mortality (0 = survived, 1 = died)
    """

    def __init__(self, root: str) -> None:
        super().__init__()
        PTdict   = np.load(os.path.join(root, "PTdict_list.npy"),  allow_pickle=True)
        outcomes = np.load(os.path.join(root, "arr_outcomes.npy"), allow_pickle=True)

        self.targets = outcomes[:, -1].astype(np.int64)

        feat_means, feat_stds = _feature_stats(PTdict)
        stat_means, stat_stds = _static_stats(PTdict)

        self._ts:      list[np.ndarray] = []
        self._static:  list[np.ndarray] = []
        self._times:   list[np.ndarray] = []
        self._lengths: list[int]        = []
        self._labels                    = self.targets

        for p in PTdict:
            arr    = p['arr'].astype(np.float32)                   # (215, 36)
            time   = p['time'].squeeze(1).astype(np.float32)       # (215,)
            length = int(p['length'])
            static = np.array(p['extended_static'], dtype=np.float32)  # (9,)

            # Normalise time-series (only observed entries)
            mask_obs = arr != 0
            arr_norm = np.where(mask_obs, (arr - feat_means) / feat_stds, 0.0)

            # Normalise static; replace -1 (missing) with 0 after normalisation
            static_norm = np.where(
                static >= 0,
                (static - stat_means) / stat_stds,
                0.0,
            ).astype(np.float32)

            self._ts.append(arr_norm.astype(np.float32))
            self._static.append(static_norm)
            self._times.append(time)
            self._lengths.append(length)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ts)

    def __getitem__(self, idx: int):
        ts      = torch.from_numpy(self._ts[idx])        # (215, 36)
        static  = torch.from_numpy(self._static[idx])    # (9,)
        times   = torch.from_numpy(self._times[idx])     # (215,)
        length  = torch.tensor(self._lengths[idx], dtype=torch.long)
        label   = int(self._labels[idx])
        return (ts, static, times, length), label
