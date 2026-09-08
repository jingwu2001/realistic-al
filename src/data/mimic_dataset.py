"""
PyTorch Dataset for the MIMIC-III in-hospital-mortality task (SAND format).

Produced by ``preprocess_mimic3_sand.py`` (first 24h of each ICU stay,
hourly-binned), which pre-splits the data into fixed train/test objects for two
cohorts. Selected CIFAR-style via constructor args::

    MIMIC3SandDataset(root, train=True,  single_stay_only=False)  # all, pool+val
    MIMIC3SandDataset(root, train=False, single_stay_only=False)  # all, fixed test
    MIMIC3SandDataset(root, train=True,  single_stay_only=True)   # single-stay, pool+val
    ...

On-disk layout under ``root`` (= ``$DATA_ROOT/mimic3_sand``)::

    all/{train,test}/{X,demo,targets,subject_ids}.npy
    single_stay/{train,test}/{X,demo,targets,subject_ids}.npy

  X.npy            (n, 24, 3V) float32 — concat([values, obs_mask, delta], -1);
                                         values mean-filled + z-normalised using
                                         that cohort's TRAIN statistics
  demo.npy         (n, 2)      float32 — z-normalised Age, Gender (train stats)
  targets.npy      (n,)        int64   — in-hospital mortality (~11-12% positive)
  subject_ids.npy  (n,)        int64   — MIMIC-III SUBJECT_ID (patient id)

Each sample is returned as ``((ts, demo), label)``:

  ts    : (24, 3V) float32 — hourly time-series tensor
  demo  : (2,)     float32 — static features
  label : scalar int       — mortality (0 = survived, 1 = died)

Exposes ``targets`` (numpy int array) and ``groups`` (patient ids) so
TimeSeriesDM can carve a patient-grouped validation split from the train object.
The test object is fixed across seeds -- the whole point of the pre-split.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MIMIC3SandDataset(Dataset):
    """MIMIC-III mortality, dense hourly SAND format: ``((ts, demo), label)``.

    Args:
        root: base directory (``$DATA_ROOT/mimic3_sand``).
        train: ``True`` -> the train object (pool+val universe); ``False`` ->
            the fixed test object.
        single_stay_only: ``False`` -> ``all`` cohort (44,812 stays);
            ``True`` -> ``single_stay`` cohort (26,609 stays, one per patient).
    """

    def __init__(
        self, root: str, train: bool = True, single_stay_only: bool = False
    ) -> None:
        super().__init__()
        cohort = "single_stay" if single_stay_only else "all"
        split = "train" if train else "test"
        obj_dir = os.path.join(root, cohort, split)
        if not os.path.isdir(obj_dir):
            raise FileNotFoundError(
                f"MIMIC3Sand object not found: {obj_dir}. Regenerate with "
                f"preprocess_mimic3_sand.py (it now writes "
                f"{{all,single_stay}}/{{train,test}}/ objects)."
            )
        self.root = root
        self.train = train
        self.single_stay_only = single_stay_only

        self._X = np.load(os.path.join(obj_dir, "X.npy"))
        self._demo = np.load(os.path.join(obj_dir, "demo.npy"))
        self.targets = np.load(os.path.join(obj_dir, "targets.npy")).astype(np.int64)
        self.groups = np.load(os.path.join(obj_dir, "subject_ids.npy")).astype(np.int64)
        assert len(self._X) == len(self._demo) == len(self.targets) == len(self.groups)
        self._labels = self.targets

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        ts = torch.from_numpy(self._X[idx])       # (24, 3V)
        demo = torch.from_numpy(self._demo[idx])  # (2,)
        label = int(self._labels[idx])
        return (ts, demo), label
