"""
PyTorch Dataset for the MIMIC-III in-hospital-mortality task (SAND format).

The data is produced by ``preprocess_mimic3_sand.py`` from the STraTS
preprocessing pipeline (first 24h of each ICU stay, hourly-binned):

  X.npy            (N, 24, 3V) float32 — concat([values, obs_mask, delta], -1)
                                         with V = 129 time-series variables;
                                         values are mean-filled + z-normalised
  demo.npy         (N, 2)      float32 — z-normalised Age, Gender
  targets.npy      (N,)        int64   — in-hospital mortality (~11% positive)
  subject_ids.npy  (N,)        int64   — MIMIC-III SUBJECT_ID (patient id),
                                         optional; used for patient-grouped splits

Each sample is returned as ``((ts, demo), label)``:

  ts    : (24, 3V) float32 — hourly time-series tensor
  demo  : (2,)     float32 — static features
  label : scalar int       — mortality (0 = survived, 1 = died)

The class exposes a ``targets`` attribute (numpy int array) for stratified
splitting, matching the P12/ECG5000 datasets, and a ``groups`` attribute
(patient ids, or ``None`` if ``subject_ids.npy`` is absent) so TimeSeriesDM
can keep all stays of a patient in the same train/val/test split.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MIMIC3SandDataset(Dataset):
    """MIMIC-III mortality, dense hourly SAND format: ``((ts, demo), label)``."""

    def __init__(self, root: str) -> None:
        super().__init__()
        self._X = np.load(os.path.join(root, "X.npy"))
        self._demo = np.load(os.path.join(root, "demo.npy"))
        self.targets = np.load(os.path.join(root, "targets.npy")).astype(np.int64)
        assert len(self._X) == len(self._demo) == len(self.targets)
        self._labels = self.targets

        # Patient ids (SUBJECT_ID) per stay for patient-grouped splitting.
        # Optional: older data dirs may not carry it.
        groups_path = os.path.join(root, "subject_ids.npy")
        if os.path.exists(groups_path):
            self.groups = np.load(groups_path).astype(np.int64)
            assert len(self.groups) == len(self.targets)
        else:
            self.groups = None

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        ts = torch.from_numpy(self._X[idx])      # (24, 3V)
        demo = torch.from_numpy(self._demo[idx])  # (2,)
        label = int(self._labels[idx])
        return (ts, demo), label
