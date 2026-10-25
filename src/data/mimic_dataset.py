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
from typing import Optional

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

        if os.path.exists(os.path.join(obj_dir, "X_flat.npy")):
            # whole-stay objects (preprocess_mimic3_benchmark.py --hours full):
            # stays stored unpadded back to back, right-padded with zeros here to
            # this object's longest stay so the default collate can stack them
            self._X = None
            self._X_flat = np.load(os.path.join(obj_dir, "X_flat.npy"))
            self._offsets = np.load(os.path.join(obj_dir, "offsets.npy"))
            self._T = int(np.diff(self._offsets).max())
        else:
            self._X = np.load(os.path.join(obj_dir, "X.npy"))
        self._demo = np.load(os.path.join(obj_dir, "demo.npy"))
        self.groups = np.load(
            os.path.join(obj_dir, "subject_ids.npy")
        ).astype(np.int64)

    def _load(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self._obj_dir, name))

    def __len__(self) -> int:
        return len(self._demo)

    def _ts(self, idx: int) -> torch.Tensor:
        if self._X is not None:
            return torch.from_numpy(self._X[idx])  # (T, C)
        stay = self._X_flat[self._offsets[idx]:self._offsets[idx + 1]]
        ts = torch.zeros(self._T, stay.shape[1])
        ts[: len(stay)] = torch.from_numpy(stay)
        return ts

    def __getitem__(self, idx: int):
        ts = self._ts(idx)                        # (24, 3V)
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
        assert len(self) == len(self.targets) == len(self.groups)


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
        assert len(self) == len(self.targets) == len(self.groups)


# Number of Harutyunyan et al. 2019 phenotype conditions (HCUP CCS 2015 groups
# with use_in_benchmark: True). See preprocess_mimic3_sand.build_phenotype_matrix.
PHENO_NUM_CLASSES = 25

# Canonical names of the 25 benchmark conditions, in the SAME order as the
# columns of phenotype.npy. build_phenotype_matrix orders columns by
# ``sorted(condition_names)`` (alphabetical), so this hard-coded list must stay
# in that exact order; it is asserted against a fresh build in the phenotype
# tests. Used only for human-readable per-label metric names (al/phenotype/*);
# the label semantics come from phenotype.npy, never from this list.
PHENO_CONDITION_NAMES = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]
assert len(PHENO_CONDITION_NAMES) == PHENO_NUM_CLASSES


def pheno_single_task_id(data_cfg) -> Optional[int]:
    """Condition index for single-task phenotyping, ``None`` for multi-task.

    ``data.multitask`` (default true) trains all 25 conditions jointly. With
    ``data.multitask=false`` only column ``data.task_id`` of phenotype.npy
    (0-based, see ``PHENO_CONDITION_NAMES``) is used, and the model has a
    single sigmoid output. Raises if ``task_id`` is missing / out of range.
    """
    if data_cfg.get("multitask", True):
        return None
    task_id = data_cfg.get("task_id", None)
    if task_id is None or not 0 <= int(task_id) < PHENO_NUM_CLASSES:
        raise ValueError(
            f"data.multitask=false needs data.task_id in [0, {PHENO_NUM_CLASSES}) "
            f"(a phenotype.npy column, see PHENO_CONDITION_NAMES); got {task_id!r}"
        )
    return int(task_id)


class MIMIC3PhenotypeDataset(_MIMIC3StayBase):
    """MIMIC-III phenotyping: ``((ts, demo), label)`` with a **multi-label** target.

    ``label`` is a ``(25,)`` float multi-hot vector over the 25 HCUP CCS 2015
    benchmark conditions (Harutyunyan et al. 2019), derived from the admission's
    ICD-9 diagnoses (see ``preprocess_mimic3_sand.build_phenotype_matrix``). This
    is the framework's only multi-label task: 25 independent Bernoulli labels, so
    it uses a sigmoid/BCE head and macro-AUROC (``{mode}/macro_auroc``), NOT the
    single-label softmax/argmax path. ``targets`` is the full ``(n, 25)`` matrix
    (2-D, unlike the scalar-label tasks); ``groups`` is the patient id.

    NB the input is still the first-24h SAND window; the label follows the
    benchmark's condition definition (the full-stay benchmark *input* is not
    reproduced -- see docs/roadmap and the plan).

    Single-task mode (``task_id`` set, from ``data.multitask=false``): only that
    condition's column is kept, so ``targets`` is ``(n, 1)`` and each label is a
    ``(1,)`` float -- one Bernoulli output, trained through the same sigmoid/BCE
    path as the 25-label model (``num_classes`` = 1).
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        single_stay_only: bool = False,
        task_id: Optional[int] = None,
    ) -> None:
        super().__init__(root, train, single_stay_only)
        # (n, 25) multi-hot. Kept as float32 for BCEWithLogitsLoss targets.
        pheno = self._load("phenotype.npy").astype(np.float32)
        assert pheno.ndim == 2 and pheno.shape[1] == PHENO_NUM_CLASSES
        self.task_id = task_id
        # keep 2-D (n, 1) in single-task mode so the multi-label code path applies
        self.targets = pheno if task_id is None else pheno[:, [task_id]]
        self.num_classes = self.targets.shape[1]
        self._labels = self.targets
        assert len(self) == len(self.targets) == len(self.groups)

    def __getitem__(self, idx: int):
        # Override the base scalar-int cast: the label is a (C,) float vector.
        ts = self._ts(idx)                        # (24, 3V)
        demo = torch.from_numpy(self._demo[idx])  # (2,)
        label = torch.from_numpy(self._labels[idx])  # (C,) float32, C = 25 or 1
        return (ts, demo), label
