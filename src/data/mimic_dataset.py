"""
PyTorch Datasets for MIMIC-III per-stay prediction tasks (SAND input format).

Produced by ``preprocess_mimic3_sand.py`` (first 24h of each ICU stay,
hourly-binned), which pre-splits the data into fixed train/test objects for two
cohorts and stores *both* per-stay labels side by side. Two task-specific
dataset classes read the same four objects and each pick only the label they
need:

  MIMIC3ICUDataset  -> in-hospital mortality (binary; the STraTS/SAND task)
  MIMIC3LOSDataset  -> ICU length of stay, bucketed into classes

Both are selected CIFAR-style via constructor args::

    MIMIC3ICUDataset(root, train=True,  single_stay_only=False)  # all, pool+val
    MIMIC3ICUDataset(root, train=False, single_stay_only=False)  # all, fixed test
    MIMIC3LOSDataset(root, train=True,  single_stay_only=True)   # single-stay, pool+val
    ...

On-disk layout under ``root`` (= ``$DATA_ROOT/mimic3_sand``)::

    all/{train,test}/{X,demo,mortality,los,subject_ids}.npy
    single_stay/{train,test}/{X,demo,mortality,los,subject_ids}.npy

  X.npy            (n, 24, 3V) float32 — concat([values, obs_mask, delta], -1);
                                         values mean-filled + z-normalised using
                                         that cohort's TRAIN statistics
  demo.npy         (n, 2)      float32 — z-normalised Age, Gender (train stats)
  mortality.npy    (n,)        int64   — in-hospital mortality (~11-12% positive)
  los.npy          (n,)        float32 — total ICU length of stay in days
  subject_ids.npy  (n,)        int64   — MIMIC-III SUBJECT_ID (patient id)

Each sample is returned as ``((ts, demo), label)``:

  ts    : (24, 3V) float32 — hourly time-series tensor
  demo  : (2,)     float32 — static features
  label : scalar int       — task label (mortality class / LOS bucket)

Both classes expose ``targets`` (numpy int array) and ``groups`` (patient ids)
so TimeSeriesDM can carve a patient-grouped validation split from the train
object. The test object is fixed across seeds -- the whole point of the
pre-split.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Harutyunyan et al. MIMIC-III benchmark length-of-stay buckets are 10 (days):
# [0,1), [1,2), ..., [7,8), [8,14), [14, inf). Our STraTS cohort keeps only stays
# >= 24h, so LOS >= 1.0 day and the first bucket ([0,1)d) is *structurally*
# empty. We drop it and use the remaining 9 ordinal classes. Linear-weighted
# Cohen's kappa depends only on |i-j|, so this global index shift leaves kappa
# identical to the 10-class scheme, while keeping every class populated (no NaN
# in per-class balanced-accuracy / precision / F1, no undefined macro-AUROC).
#   np.digitize(los, edges) -> class c:  0=[1,2) 1=[2,3) ... 6=[7,8) 7=[8,14) 8=[14,inf)
LOS_BIN_EDGES = [2, 3, 4, 5, 6, 7, 8, 14]
LOS_NUM_CLASSES = len(LOS_BIN_EDGES) + 1  # = 9


class _MIMIC3StayBase(Dataset):
    """Shared loader for the MIMIC-III per-stay objects.

    Loads the inputs (``X``, ``demo``) and patient ids common to every task;
    subclasses load their own label file and set ``self.targets`` /
    ``self._labels``. Not meant to be instantiated directly.

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
                f"MIMIC3 object not found: {obj_dir}. Regenerate with "
                f"preprocess_mimic3_sand.py (it writes "
                f"{{all,single_stay}}/{{train,test}}/ objects)."
            )
        self.root = root
        self.train = train
        self.single_stay_only = single_stay_only
        self._obj_dir = obj_dir

        self._X = np.load(os.path.join(obj_dir, "X.npy"))
        self._demo = np.load(os.path.join(obj_dir, "demo.npy"))
        self.groups = np.load(
            os.path.join(obj_dir, "subject_ids.npy")
        ).astype(np.int64)

    def _load(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self._obj_dir, name))

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        ts = torch.from_numpy(self._X[idx])       # (24, 3V)
        demo = torch.from_numpy(self._demo[idx])  # (2,)
        label = int(self._labels[idx])
        return (ts, demo), label


class MIMIC3ICUDataset(_MIMIC3StayBase):
    """MIMIC-III in-hospital mortality: ``((ts, demo), label)``.

    ``targets`` is the binary mortality label (0 = survived, 1 = died,
    ~11-12% positive). The LOS label in the same object is ignored.
    """

    def __init__(
        self, root: str, train: bool = True, single_stay_only: bool = False
    ) -> None:
        super().__init__(root, train, single_stay_only)
        self.targets = self._load("mortality.npy").astype(np.int64)
        self._labels = self.targets
        assert len(self._X) == len(self._demo) == len(self.targets) == len(self.groups)


class MIMIC3LOSDataset(_MIMIC3StayBase):
    """MIMIC-III length-of-stay prediction: ``((ts, demo), label)``.

    ``targets`` is the total ICU LOS bucketed into ``LOS_NUM_CLASSES`` (=9)
    ordinal classes via ``LOS_BIN_EDGES`` (Harutyunyan's 10 buckets minus the
    structurally-empty [0,1)d bin -- see the module-level note). The raw LOS in
    days is kept on ``self.los_days``. The mortality label in the same object is
    ignored. Headline metric is linear-weighted Cohen's kappa (``{mode}/kappa``).
    """

    num_classes = LOS_NUM_CLASSES
    bin_edges = LOS_BIN_EDGES

    def __init__(
        self, root: str, train: bool = True, single_stay_only: bool = False
    ) -> None:
        super().__init__(root, train, single_stay_only)
        self.los_days = self._load("los.npy").astype(np.float32)
        self.targets = np.digitize(self.los_days, LOS_BIN_EDGES).astype(np.int64)
        self._labels = self.targets
        assert len(self._X) == len(self._demo) == len(self.targets) == len(self.groups)
