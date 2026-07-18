# Adapterd from Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts
from typing import Optional, Sequence, Union
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.ecg5000_dataset import ECG5000Dataset
from data.mimic_dataset import MIMIC3SandDataset
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

        ### IMPLEMENTATION ###
        # Novel image datasets can be added here
        if self.dataset == "ecg5000":
            self.dataset_cls = ECG5000Dataset
        elif self.dataset == "p12":
            self.dataset_cls = P12Dataset
        elif self.dataset == "p12_transformer":
            self.dataset_cls = P12TransformerDataset
        elif self.dataset == "mimic3_sand":
            self.dataset_cls = MIMIC3SandDataset
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

    def _setup_datasets(self):
        """Creates the active training dataset and validation and test datasets"""
        # ------------------------------------------------------------------ #
        # Pre-compute stratified split indices for datasets that don't ship   #
        # with their own train/test files (ECG5000, P12).                     #
        # ------------------------------------------------------------------ #
        if self.dataset == "ecg5000":
            _ecg_test_size = 1000
            _ecg_val_size  = (self.val_split if isinstance(self.val_split, int)
                              else int(self.val_split * 5000))
            _ecg_root_tmp  = os.path.join(self.data_root, "ecg5000")
            _ecg_labels    = self.dataset_cls(root=_ecg_root_tmp, split="all").targets
            _ecg_n_total   = len(_ecg_labels)
            _ecg_test_idx, _ecg_val_idx, _ecg_pool_idx = self._stratified_3way_split(
                _ecg_labels,
                _ecg_n_total,
                _ecg_test_size,
                _ecg_val_size,
                balanced_eval=self.balanced_test_val,
            )

        if self.dataset in ("p12", "p12_transformer", "mimic3_sand"):
            if self.dataset == "mimic3_sand":
                _ts_root    = os.path.join(self.data_root, "mimic3_sand")
            else:
                _ts_root    = os.path.join(self.data_root, "P12data/processed_data")
            _ts_labels      = self.dataset_cls(root=_ts_root).targets
            _ts_n_total     = len(_ts_labels)
            _ts_test_size   = max(1, round(0.1 * _ts_n_total))
            _ts_val_size    = (self.val_split if isinstance(self.val_split, int)
                               else max(1, round(self.val_split * _ts_n_total)))
            _ts_test_idx, _ts_val_idx, _ts_pool_idx = self._stratified_3way_split(
                _ts_labels, _ts_n_total, _ts_test_size, _ts_val_size,
                balanced_eval=self.balanced_test_val,
            )

        # ------------------------------------------------------------------ #
        # Build train (pool) set                                               #
        # ------------------------------------------------------------------ #
        if self.dataset == "ecg5000":
            ecg_root = os.path.join(self.data_root, "ecg5000")
            _ecg_full_train = self.dataset_cls(
                root=ecg_root, split="all", transform=self.train_transforms
            )
            self.train_set = ActiveSubset(_ecg_full_train, _ecg_pool_idx)
        elif self.dataset in ("p12", "p12_transformer", "mimic3_sand"):
            _ts_full = self.dataset_cls(root=_ts_root)
            self.train_set = ActiveSubset(_ts_full, _ts_pool_idx)

        if self.imbalance:
            self.train_set = create_imbalanced_dataset(
                self.train_set, imb_type=self.imb_type, imb_factor=self.imb_factor
            )

        if self.active:
            if self.dataset in ("p12", "p12_transformer", "mimic3_sand"):
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
        elif self.dataset in ("p12", "p12_transformer", "mimic3_sand"):
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
