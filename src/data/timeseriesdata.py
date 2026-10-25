# Adapterd from Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts
from functools import partial
from typing import Optional, Sequence, Union
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.ecg5000_dataset import ECG5000Dataset
from data.mimic_dataset import (
    MIMIC3ICUDataset,
    MIMIC3LOSDataset,
    MIMIC3PhenotypeDataset,
)
from data.p12_dataset import P12Dataset, P12TransformerDataset

from .active import ActiveLearningDataset
from .base_datamodule import BaseDataModule
from .utils import ActiveSubset
from .longtail import create_imbalanced_dataset
from .transformations import get_transform


class TimeSeriesDM(BaseDataModule):
    def __init__(
        self,
        data_root: str,
        val_split: Union[float, int] = 0.2,
        batch_size: int = 64,
        dataset: str = "mnist",
        drop_last: bool = False,
        num_workers: int = 12,
        pin_memory: bool = True,
        shuffle: bool = True,
        min_train: int = 5500,
        active: bool = True,
        random_split: bool = True,
        num_classes: int = 10,
        transform_train: str = "basic",
        transform_test: str = "basic",
        shape: Sequence = [28, 28, 1],
        mean: Sequence = (0,),
        std: Sequence = (1,),
        seed: int = 12345,
        persistent_workers: bool = True,
        imbalance: bool = False,
        imb_type: str = "exp",
        imb_factor: float = 0.02,
        timeout: int = 0,
        val_size: Optional[int] = None,
        balanced_sampling: bool = False,
        balanced_test_val: bool = False, # Forces test set and validation set to be balanced
        single_stay_only: bool = False, # mimic3_sand: keep only patients with exactly one ICU stay
        pheno_task_id: Optional[int] = None, # mimic3_pheno: single condition (None = all 25)
        mimic_dir: str = "mimic3_sand", # mimic3_*: object dir under data_root (mimic3_benchmark/arrays_h48_first = benchmark input)
        drop_classes: Optional[Sequence[int]] = None, # ecg5000: classes removed before the split
    ):
        super().__init__(
            val_split=val_split,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            min_train=min_train,
            active=active,
            random_split=random_split,
            seed=seed,
            persistent_workers=persistent_workers,
            timeout=timeout,
            val_size=val_size,
            balanced_sampling=balanced_sampling,
        )

        self.data_root = data_root
        self.dataset = dataset

        self.num_classes = num_classes

        self.mean = mean
        self.std = std
        self.shape = shape
        # P12 and ECG5000 manage their own normalisation; skip the mean/std
        # shape assertion that is only meaningful for image-style transforms.

        self.train_transforms = get_transform(
            transform_train, self.mean, self.std, self.shape
        )
        self.test_transforms = get_transform(
            transform_test, self.mean, self.std, self.shape
        )
        self.imbalance = imbalance
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.balanced_test_val = balanced_test_val
        self.single_stay_only = single_stay_only
        self.pheno_task_id = pheno_task_id
        self.mimic_dir = mimic_dir

        if drop_classes and self.dataset != "ecg5000":
            raise NotImplementedError(f"data.drop_classes is only implemented for ecg5000, not {self.dataset}")

        ### IMPLEMENTATION ###
        # Novel image datasets can be added here
        if self.dataset == "ecg5000":
            # drop_classes removes those classes from every split (pool, val,
            # test) BEFORE the stratified split; labels are renumbered 0..K-1
            self.dataset_cls = partial(ECG5000Dataset, drop_classes=drop_classes)
        elif self.dataset == "p12":
            self.dataset_cls = P12Dataset
        elif self.dataset == "p12_transformer":
            self.dataset_cls = P12TransformerDataset
        elif self.dataset == "mimic3_sand":
            self.dataset_cls = MIMIC3ICUDataset
        elif self.dataset == "mimic3_los":
            self.dataset_cls = MIMIC3LOSDataset
        elif self.dataset == "mimic3_pheno":
            self.dataset_cls = MIMIC3PhenotypeDataset
        else:
            raise NotImplementedError
        self._setup_datasets()

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def _stratified_3way_split(
        self, labels, n_total, test_size, val_size, balanced_eval: bool = False
    ):
        """Return (test_idx, val_idx, pool_idx) with stratification by class label.

        When balanced_eval=True, test and val receive equal numbers of samples
        per class (limited by the minority class), so eval sets are 50/50.
        The pool receives all remaining samples and may still be imbalanced.

        When balanced_eval=True, reserve the test and train labeled for them first.
        """
        rng = np.random.default_rng(self.seed)
        test_idx, val_idx, pool_idx = [], [], []
        classes = np.unique(labels)
        n_classes = len(classes)

        if balanced_eval:
            counts = np.array([np.sum(labels == c) for c in classes])
            n_test_per_class = max(1, test_size // n_classes)
            n_val_per_class = max(1, val_size // n_classes)
            assert min(counts) >= n_test_per_class + n_val_per_class, \
                f"Not enough samples for balanced val and test sets: min class count={min(counts)}, need {n_test_per_class + n_val_per_class} per class (test_size={test_size}, val_size={val_size})"

        for c in classes:
            c_idx = np.where(labels == c)[0]
            rng.shuffle(c_idx)
            if balanced_eval:
                n_test = n_test_per_class
                n_val = n_val_per_class
            else:
                n_test = max(1, round(len(c_idx) * test_size / n_total))
                n_val = max(1, round(len(c_idx) * val_size / n_total))
                if n_test + n_val >= len(c_idx):
                    n_test = max(1, min(n_test, len(c_idx) - 2))
                    n_val = max(1, min(n_val, len(c_idx) - n_test - 1))
            test_idx.append(c_idx[:n_test])
            val_idx.append(c_idx[n_test:n_test + n_val])
            pool_idx.append(c_idx[n_test + n_val:])
        return (
            np.concatenate(test_idx),
            np.concatenate(val_idx),
            np.concatenate(pool_idx),
        )

    def _grouped_stratified_3way_split(
        self, labels, groups, n_total, test_size, val_size, balanced_eval: bool = False
    ):
        """Patient-grouped variant of :meth:`_stratified_3way_split`.

        Every sample belongs to a group (``groups[i]`` = patient id). All
        samples of a group are assigned to the *same* split, so a patient with
        several ICU stays never leaks across train/val/test. Groups are
        shuffled with ``self.seed``, so different seeds yield different splits.

        Splitting is driven by target positive/negative *sample* counts per
        split rather than by whole groups, so class balance is preserved as
        closely as whole-group assignment allows. Positives can only enter a
        split via a positive patient (who typically also brings a few negative
        stays); negatives are then topped up from negative-only patients. With
        ``balanced_eval`` the targets are 50/50; otherwise they follow the
        dataset's base rate (matching the ungrouped stratified split).

        Because whole patients are assigned atomically, realised sizes and
        ratios approximate the targets.
        """
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.seed)

        # Build one index array per group (sample indices belonging to it).
        order = np.argsort(groups, kind="stable")
        _, start = np.unique(groups[order], return_index=True)
        member_lists = np.split(order, start[1:])
        group_label = np.array([int(labels[m].max()) for m in member_lists])
        # (#positive samples, #negative samples) carried by each group.
        group_pos = np.array([int(labels[m].sum()) for m in member_lists])
        group_neg = np.array([len(m) for m in member_lists]) - group_pos

        pos_total = int((labels == 1).sum())
        neg_total = n_total - pos_total

        def targets(size):
            if balanced_eval:
                half = max(1, size // 2)
                return half, half
            return (
                max(1, round(pos_total * size / n_total)),
                max(1, round(neg_total * size / n_total)),
            )

        t_pos, t_neg = targets(test_size)
        v_pos, v_neg = targets(val_size)

        # Shuffle positive-patient and negative-patient groups independently.
        pos_groups = np.where(group_label == 1)[0]
        neg_groups = np.where(group_label == 0)[0]
        rng.shuffle(pos_groups)
        rng.shuffle(neg_groups)
        p_ptr = n_ptr = 0

        def fill(need_pos, need_neg):
            """Consume shuffled groups until pos/neg sample quotas are met."""
            nonlocal p_ptr, n_ptr
            sel, got_pos, got_neg = [], 0, 0
            # positives only come from positive-patient groups
            while p_ptr < len(pos_groups) and got_pos < need_pos:
                gi = pos_groups[p_ptr]; p_ptr += 1
                sel.append(gi); got_pos += group_pos[gi]; got_neg += group_neg[gi]
            # top up negatives from negative-only patients
            while n_ptr < len(neg_groups) and got_neg < need_neg:
                gi = neg_groups[n_ptr]; n_ptr += 1
                sel.append(gi); got_pos += group_pos[gi]; got_neg += group_neg[gi]
            return sel

        test_sel = fill(t_pos, t_neg)
        val_sel = fill(v_pos, v_neg)
        pool_sel = list(pos_groups[p_ptr:]) + list(neg_groups[n_ptr:])
        assert pool_sel, (
            f"no groups left for the pool after filling test/val "
            f"(test_size={test_size}, val_size={val_size})"
        )

        def to_idx(sel):
            return (np.concatenate([member_lists[g] for g in sel])
                    if sel else np.array([], dtype=int))

        return to_idx(test_sel), to_idx(val_sel), to_idx(pool_sel)

    def _grouped_val_pool_split(self, labels, groups, val_size, balanced_eval=False):
        """Carve a patient-grouped validation set (~``val_size`` samples) from
        the train object; the remainder is the labelled pool. This is a 2-way
        split (test is a fixed, separate on-disk object for mimic3_sand). Seeded
        by ``self.seed`` so the validation split varies per seed while the test
        cohort stays fixed. Degenerates to a plain stratified split when every
        group is a singleton (the single-stay cohort).
        """
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.seed)

        order = np.argsort(groups, kind="stable")
        _, start = np.unique(groups[order], return_index=True)
        member_lists = np.split(order, start[1:])
        group_label = np.array([int(labels[m].max()) for m in member_lists])
        group_pos = np.array([int(labels[m].sum()) for m in member_lists])
        group_neg = np.array([len(m) for m in member_lists]) - group_pos

        n_total = len(labels)
        pos_total = int((labels == 1).sum())
        neg_total = n_total - pos_total
        if balanced_eval:
            v_pos = v_neg = max(1, val_size // 2)
        else:
            v_pos = max(1, round(pos_total * val_size / n_total))
            v_neg = max(1, round(neg_total * val_size / n_total))

        pos_groups = np.where(group_label == 1)[0]
        neg_groups = np.where(group_label == 0)[0]
        rng.shuffle(pos_groups)
        rng.shuffle(neg_groups)

        sel, got_pos, got_neg, p_ptr, n_ptr = [], 0, 0, 0, 0
        while p_ptr < len(pos_groups) and got_pos < v_pos:
            gi = pos_groups[p_ptr]; p_ptr += 1
            sel.append(gi); got_pos += group_pos[gi]; got_neg += group_neg[gi]
        while n_ptr < len(neg_groups) and got_neg < v_neg:
            gi = neg_groups[n_ptr]; n_ptr += 1
            sel.append(gi); got_pos += group_pos[gi]; got_neg += group_neg[gi]
        pool_groups = list(pos_groups[p_ptr:]) + list(neg_groups[n_ptr:])
        assert pool_groups, f"no groups left for the pool (val_size={val_size})"

        def to_idx(s):
            return (np.concatenate([member_lists[g] for g in s])
                    if s else np.array([], dtype=int))

        return np.sort(to_idx(sel)), np.sort(to_idx(pool_groups))

    def _grouped_val_pool_split_multiclass(self, labels, groups, val_size):
        """Multiclass counterpart of ``_grouped_val_pool_split`` for the LOS task.

        The binary version buckets groups into pos/neg (label 0/1) and would drop
        any group whose label is >= 2, so it cannot be used for a multiclass
        target. This version carves a patient-grouped, class-stratified
        validation set (~``val_size`` samples) from the train object, with the
        remainder as the labelled pool. Every group is placed in exactly one of
        val/pool, so no sample is ever dropped. Grouping keeps all of a patient's
        stays on the same side (matters only for the ``all`` cohort; the
        single-stay cohort is all singletons). Seeded by ``self.seed``.
        """
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.seed)

        order = np.argsort(groups, kind="stable")
        _, start = np.unique(groups[order], return_index=True)
        member_lists = np.split(order, start[1:])
        # One representative label per group (max LOS bin) for stratification;
        # multi-stay patients (all cohort) follow their max-bin stratum as a unit.
        group_label = np.array([int(labels[m].max()) for m in member_lists])
        group_size = np.array([len(m) for m in member_lists])

        n_total = len(labels)
        val_frac = val_size / n_total
        sel = []
        for c in np.unique(group_label):
            gs = np.where(group_label == c)[0]
            rng.shuffle(gs)
            target = max(1, round(val_frac * int(group_size[gs].sum())))
            got = 0
            for gi in gs:
                if got >= target:
                    break
                sel.append(int(gi))
                got += int(group_size[gi])
        sel_set = set(sel)
        pool_groups = [g for g in range(len(member_lists)) if g not in sel_set]
        assert pool_groups, f"no groups left for the pool (val_size={val_size})"

        def to_idx(s):
            return (np.concatenate([member_lists[g] for g in s])
                    if s else np.array([], dtype=int))

        return np.sort(to_idx(sel)), np.sort(to_idx(pool_groups))

    def _grouped_val_pool_split_multilabel(self, labels, groups, val_size):
        """Patient-grouped RANDOM val/pool carve for the multi-label phenotyping
        task (``mimic3_pheno``).

        The label is a ``(n, C)`` multi-hot vector, so there is no single class
        to stratify on (the binary and multiclass splitters both collapse it and
        would misbehave). We instead carve ~``val_size`` samples by shuffling
        patient groups (keeping each patient's stays together) and taking groups
        until the val budget is met; the remainder is the labelled pool. Every
        group lands in exactly one side, so no sample is dropped. Seeded by
        ``self.seed`` (val varies per seed; the test object is fixed on disk).
        """
        labels = np.asarray(labels)  # (n, C) — unused for stratification
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.seed)

        order = np.argsort(groups, kind="stable")
        _, start = np.unique(groups[order], return_index=True)
        member_lists = np.split(order, start[1:])
        group_size = np.array([len(m) for m in member_lists])

        gidx = np.arange(len(member_lists))
        rng.shuffle(gidx)
        sel, got = [], 0
        for gi in gidx:
            if got >= val_size:
                break
            sel.append(int(gi))
            got += int(group_size[gi])
        sel_set = set(sel)
        pool_groups = [g for g in range(len(member_lists)) if g not in sel_set]
        assert pool_groups, f"no groups left for the pool (val_size={val_size})"

        def to_idx(s):
            return (np.concatenate([member_lists[g] for g in s])
                    if s else np.array([], dtype=int))

        return np.sort(to_idx(sel)), np.sort(to_idx(pool_groups))

    def _balanced_indices(self, labels):
        """Indices of a class-balanced subset (equal per class, majority
        downsampled), used to build balanced eval sets from a fixed test/val
        object when ``balanced_test_val`` is set."""
        labels = np.asarray(labels)
        rng = np.random.default_rng(self.seed)
        classes, counts = np.unique(labels, return_counts=True)
        k = int(counts.min())
        sel = []
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            sel.append(idx[:k])
        return np.sort(np.concatenate(sel))

    def _cohort_indices(self, labels, groups):
        """Indices (into the full dataset) of the samples to actually use.

        Generic: depends only on ``groups``, so it works for any grouped
        dataset. By default every sample is used. With ``single_stay_only`` only
        samples whose group (patient id) occurs exactly once are kept — i.e.
        patients with a single ICU stay; the rest of the pipeline (split,
        wrapping) then runs on that restricted universe.
        """
        n = len(labels)
        if not self.single_stay_only:
            return np.arange(n)
        if groups is None:
            raise ValueError(
                "single_stay_only requires group ids (e.g. subject_ids); "
                "the loaded dataset exposes none."
            )
        groups = np.asarray(groups)
        uniq, counts = np.unique(groups, return_counts=True)
        singletons = uniq[counts == 1]
        allowed = np.where(np.isin(groups, singletons))[0]
        print(
            f"single_stay_only: keeping {len(allowed)}/{n} stays from "
            f"{len(singletons)} single-stay patients"
        )
        return allowed

    def _setup_datasets(self):
        """Creates the active training dataset and validation and test datasets"""
        # ------------------------------------------------------------------ #
        # mimic3_sand: CIFAR-style fixed pre-split. The dataset ships fixed    #
        # train/test objects per cohort (built by preprocess_mimic3_sand.py),  #
        # so the test cohort is identical across seeds. We only carve the      #
        # validation set from the train object, per seed.                      #
        # ------------------------------------------------------------------ #
        if self.dataset in ("mimic3_sand", "mimic3_los", "mimic3_pheno"):
            # All three tasks share the same on-disk objects (built by
            # preprocess_mimic3_sand.py); only the label file differs, handled by
            # self.dataset_cls (MIMIC3ICUDataset / MIMIC3LOSDataset /
            # MIMIC3PhenotypeDataset).
            _root = os.path.join(self.data_root, self.mimic_dir)
            _obj_kwargs = dict(single_stay_only=self.single_stay_only)
            if self.dataset == "mimic3_pheno":
                _obj_kwargs["task_id"] = self.pheno_task_id

            _train_obj = self.dataset_cls(_root, train=True, **_obj_kwargs)
            _n_train_obj = len(_train_obj.targets)
            _val_size = (self.val_split if isinstance(self.val_split, int)
                         else max(1, round(self.val_split * _n_train_obj)))
            if self.dataset == "mimic3_pheno":
                # multi-label (25-condition): no single class to stratify on ->
                # patient-grouped random carve. Also used in single-task mode:
                # the carve ignores labels, so val/pool match the multi-task
                # runs of the same seed.
                _val_idx, _pool_idx = self._grouped_val_pool_split_multilabel(
                    _train_obj.targets, _train_obj.groups, _val_size,
                )
            elif self.dataset == "mimic3_los":
                # multiclass (9-class LOS): the binary pos/neg split would drop
                # any patient whose LOS bin is >= 2.
                _val_idx, _pool_idx = self._grouped_val_pool_split_multiclass(
                    _train_obj.targets, _train_obj.groups, _val_size,
                )
            else:
                _val_idx, _pool_idx = self._grouped_val_pool_split(
                    _train_obj.targets, _train_obj.groups, _val_size,
                    balanced_eval=self.balanced_test_val,
                )

            # train (labelled pool universe)
            self.train_set = ActiveSubset(
                self.dataset_cls(_root, train=True, **_obj_kwargs), _pool_idx
            )
            if self.active:
                # handles normalisation internally; no transform attribute
                self.train_set = ActiveLearningDataset(self.train_set)

            # validation (carved from the train object, per seed)
            _val_full = self.dataset_cls(_root, train=True, **_obj_kwargs)
            self.val_set = Subset(_val_full, _val_idx)

            # test (fixed on-disk object, identical across seeds)
            _test_full = self.dataset_cls(_root, train=False, **_obj_kwargs)
            _test_idx = (self._balanced_indices(_test_full.targets)
                         if self.balanced_test_val else np.arange(len(_test_full)))
            self.test_set = Subset(_test_full, _test_idx)
            return

        # ------------------------------------------------------------------ #
        # Pre-compute stratified split indices for datasets that don't ship   #
        # with their own train/test files (ECG5000, P12).                     #
        # ------------------------------------------------------------------ #
        if self.dataset == "ecg5000":
            _ecg_test_size = 1000
            _ecg_root_tmp  = os.path.join(self.data_root, "ecg5000")
            _ecg_labels    = self.dataset_cls(root=_ecg_root_tmp, split="all").targets
            _ecg_n_total   = len(_ecg_labels)
            _ecg_val_size  = (self.val_split if isinstance(self.val_split, int)
                              else int(self.val_split * _ecg_n_total))
            assert len(np.unique(_ecg_labels)) == self.num_classes, (
                f"data.num_classes={self.num_classes} but ecg5000 has "
                f"{len(np.unique(_ecg_labels))} classes after data.drop_classes"
            )
            _ecg_test_idx, _ecg_val_idx, _ecg_pool_idx = self._stratified_3way_split(
                _ecg_labels,
                _ecg_n_total,
                _ecg_test_size,
                _ecg_val_size,
                balanced_eval=self.balanced_test_val,
            )

        if self.dataset in ("p12", "p12_transformer"):
            _ts_root        = os.path.join(self.data_root, "P12data/processed_data")
            _ts_meta        = self.dataset_cls(root=_ts_root)
            _ts_labels      = _ts_meta.targets
            _ts_groups      = getattr(_ts_meta, "groups", None)

            # Restrict to the desired cohort (e.g. single-stay patients) before
            # splitting. ``allowed`` indexes into the full dataset; the split is
            # drawn on this restricted universe and mapped back afterwards.
            allowed         = self._cohort_indices(_ts_labels, _ts_groups)
            _sub_labels     = _ts_labels[allowed]
            _sub_groups     = None if _ts_groups is None else _ts_groups[allowed]
            _ts_n_total     = len(allowed)
            _ts_test_size   = max(1, round(0.1 * _ts_n_total))
            _ts_val_size    = (self.val_split if isinstance(self.val_split, int)
                               else max(1, round(self.val_split * _ts_n_total)))
            # Grouping only matters when some group carries >1 sample.
            _needs_grouping = (
                _sub_groups is not None
                and np.unique(_sub_groups).size < _sub_groups.size
            )
            if _needs_grouping:
                # patient-level split
                _rel_test, _rel_val, _rel_pool = self._grouped_stratified_3way_split(
                    _sub_labels, _sub_groups, _ts_n_total,
                    _ts_test_size, _ts_val_size,
                    balanced_eval=self.balanced_test_val,
                )
            else:
                _rel_test, _rel_val, _rel_pool = self._stratified_3way_split(
                    _sub_labels, _ts_n_total, _ts_test_size, _ts_val_size,
                    balanced_eval=self.balanced_test_val,
                )
            # Map split-relative indices back to full-dataset indices.
            _ts_test_idx = allowed[_rel_test]
            _ts_val_idx  = allowed[_rel_val]
            _ts_pool_idx = allowed[_rel_pool]

        # ------------------------------------------------------------------ #
        # Build train (pool) set                                               #
        # ------------------------------------------------------------------ #
        if self.dataset == "ecg5000":
            ecg_root = os.path.join(self.data_root, "ecg5000")
            _ecg_full_train = self.dataset_cls(
                root=ecg_root, split="all", transform=self.train_transforms
            )
            self.train_set = ActiveSubset(_ecg_full_train, _ecg_pool_idx)
        elif self.dataset in ("p12", "p12_transformer"):
            _ts_full = self.dataset_cls(root=_ts_root)
            self.train_set = ActiveSubset(_ts_full, _ts_pool_idx)

        if self.imbalance:
            self.train_set = create_imbalanced_dataset(
                self.train_set, imb_type=self.imb_type, imb_factor=self.imb_factor
            )

        if self.active:
            if self.dataset in ("p12", "p12_transformer"):
                # These datasets handle normalisation internally and have no transform attribute.
                self.train_set = ActiveLearningDataset(self.train_set)
            else:
                self.train_set = ActiveLearningDataset(
                    self.train_set, pool_specifics={"transform": self.test_transforms}
                )

        # ------------------------------------------------------------------ #
        # Build val / test sets                                                #
        # ------------------------------------------------------------------ #
        if self.dataset == "ecg5000":
            ecg_root = os.path.join(self.data_root, "ecg5000")
            _ecg_full_eval = self.dataset_cls(
                root=ecg_root, split="all", transform=self.test_transforms
            )
            self.val_set  = Subset(_ecg_full_eval, _ecg_val_idx)
            self.test_set = Subset(_ecg_full_eval, _ecg_test_idx)
        elif self.dataset in ("p12", "p12_transformer"):
            _ts_full_eval = self.dataset_cls(root=_ts_root)
            self.val_set   = Subset(_ts_full_eval, _ts_val_idx)
            self.test_set  = Subset(_ts_full_eval, _ts_test_idx)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_set, mode="test")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, mode="test")


if __name__ == "__main__":
    # Run from the src/ directory as a module to keep imports working:
    #   cd src && python -m data.data
    import os
    data_root = os.getenv("DATA_ROOT")
    assert data_root, "Set the DATA_ROOT environment variable first."

    def _check_dm(name, dm, n_label):
        """Run a standard set of assertions on a freshly built datamodule."""
        print(f"\n{'='*50}\n  Dataset: {name}\n{'='*50}")

        dm.train_set.label_randomly(n_label)

        # Sizes
        print(f"  Pool    : {len(dm.train_set.pool)}")
        print(f"  Labeled : {dm.train_set.n_labelled}")
        print(f"  Val     : {len(dm.val_set)}")
        print(f"  Test    : {len(dm.test_set)}")

        # Dataloaders
        train_loader  = dm.train_dataloader()
        val_loader    = dm.val_dataloader()
        test_loader   = dm.test_dataloader()
        pool_loader   = dm.pool_dataloader()

        def _x_shape(xb):
            if isinstance(xb, (list, tuple)):
                return [tuple(t.shape) if hasattr(t, 'shape') else t for t in xb]
            return tuple(xb.shape)

        xb, yb = next(iter(train_loader))
        print(f"  Train batch x: {_x_shape(xb)}  y: {tuple(yb.shape)}")
        if not isinstance(xb, (list, tuple)):
            assert xb.dtype == torch.float32, "x must be float32"
        assert yb.dtype == torch.int64,   "y must be int64"

        xb, yb = next(iter(val_loader))
        print(f"  Val   batch x: {_x_shape(xb)}  y: {tuple(yb.shape)}")

        xb, yb = next(iter(test_loader))
        print(f"  Test  batch x: {_x_shape(xb)}  y: {tuple(yb.shape)}")

        xb, yb = next(iter(pool_loader))
        print(f"  Pool  batch x: {_x_shape(xb)}  y: {tuple(yb.shape)}")

        # Pool index round-trip
        dm.pool_dataloader(m=20)
        inds = np.arange(10)
        inds_mapped = dm.get_pool_indices(inds)
        assert np.array_equal(inds_mapped, dm.indices[: len(inds_mapped)])
        print(f"  Pool index round-trip: OK")
        print(f"  {name} OK")

    # ------------------------------------------------------------------ #
    # ECG5000                                                              #
    # ------------------------------------------------------------------ #
    dm_ecg = TimeSeriesDM(
        data_root=data_root,
        dataset="ecg5000",
        shape=[1, 140],
        num_classes=5,
        mean=[0.0],
        std=[1.0],
        transform_train="ecg_basic",
        transform_test="basic",
        batch_size=64,
        num_workers=0,
        persistent_workers=False,
    )
    _check_dm("ECG5000", dm_ecg, n_label=200)

    def _dist(label, dm):
        """Print label distribution for pool, val, and test."""
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        pool_targets = dm.train_set._dataset.targets
        for split_name, ys in [
            ("pool", pool_targets),
            ("val",  np.array(dm.val_set.dataset.targets)[dm.val_set.indices]),
            ("test", np.array(dm.test_set.dataset.targets)[dm.test_set.indices]),
        ]:
            classes, counts = np.unique(ys, return_counts=True)
            total = len(ys)
            dist = {int(c): f"{int(n)} ({100*n/total:.1f}%)" for c, n in zip(classes, counts)}
            print(f"  {split_name:4s} (n={total:5d}): {dist}")

    # P12: n=11988, minority class (1)=1707; 10% test = 599/class.
    # val_split=0.1 gives 599/class → safe for balanced splits (598+599=1197 ≤ 1707).
    _P12_KWARGS = dict(
        data_root=data_root,
        dataset="p12",
        shape=[3, 33, 49],
        num_classes=2,
        mean=[0.0],
        std=[1.0],
        transform_train="basic",
        transform_test="basic",
        batch_size=64,
        num_workers=0,
        persistent_workers=False,
        val_split=0.1,
    )
    _P12_BALANCED_KWARGS = _P12_KWARGS

    # ------------------------------------------------------------------ #
    # _check_dm tests for P12 and P12Transformer                          #
    # ------------------------------------------------------------------ #
    dm_p12 = TimeSeriesDM(**_P12_KWARGS)
    _check_dm("P12", dm_p12, n_label=50)

    dm_p12t = TimeSeriesDM(**{**_P12_KWARGS, "dataset": "p12_transformer"})
    _check_dm("P12Transformer", dm_p12t, n_label=50)

    # ------------------------------------------------------------------ #
    # Distribution checks across balanced/imbalanced combos               #
    # ------------------------------------------------------------------ #
    for btv in (True, False):
        kwargs = _P12_BALANCED_KWARGS if btv else _P12_KWARGS
        _dist(
            f"balanced_test_val={btv}  |  no long tail",
            TimeSeriesDM(**kwargs, balanced_test_val=btv),
        )
        for factor in (0.02, 0.01):
            _dist(
                f"balanced_test_val={btv}  |  long tail exp imb_factor={factor}",
                TimeSeriesDM(**kwargs, balanced_test_val=btv,
                             imbalance=True, imb_type="exp", imb_factor=factor),
            )

    print("\nAll checks passed.")

    # ------------------------------------------------------------------ #
    # Launcher smoke-test: label distributions as main.py would log them  #
    # Mirrors exp_p12_transformer_baleval_bal / _imb with imbalance on/off#
    # ------------------------------------------------------------------ #
    def _full_dist(label, dm, n_label, balanced_init):
        import copy
        dm2 = copy.deepcopy(dm)
        dm2.train_set.label_randomly(n_label, balanced=balanced_init)
        ts = dm2.train_set

        def _d(tgts, name):
            tgts = np.asarray(tgts)
            classes, counts = np.unique(tgts, return_counts=True)
            total = len(tgts)
            parts = [f"{int(c)}: {int(n)} ({100*n/total:.1f}%)" for c, n in zip(classes, counts)]
            print(f"  {name:<6}: n={total}  " + "  ".join(parts))

        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        _d(ts.labelled_dataset.targets, "L")
        _d(ts.pool.targets,             "U")
        _d(ts._dataset.targets,         "L+U")
        _d(np.asarray(dm2.val_set.dataset.targets)[dm2.val_set.indices],   "val")
        _d(np.asarray(dm2.test_set.dataset.targets)[dm2.test_set.indices], "test")

    _P12T_BASE = dict(
        data_root=data_root,
        dataset="p12_transformer",
        shape=[215, 36], num_classes=2,
        mean=[0.0], std=[1.0],
        transform_train="basic", transform_test="basic",
        val_split=0.1, balanced_test_val=True,
        num_workers=0, persistent_workers=False,
    )

    for imb, factor in [(False, None), (True, 0.02)]:
        kwargs = dict(**_P12T_BASE)
        if imb:
            kwargs.update(imbalance=True, imb_factor=factor)
        dm_base = TimeSeriesDM(**kwargs)
        for bal_init, active_cfg in [(True, "p12_med_bal"), (False, "p12_med")]:
            _full_dist(
                f"imbalance={imb} imb_factor={factor}  |  active={active_cfg}  balanced_init={bal_init}",
                dm_base, n_label=100, balanced_init=bal_init,
            )

    print("\nLauncher smoke-test passed.")
