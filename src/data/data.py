# Adapterd from Pytorch Lighntning Bolts VisionDataModule  : https://github.com/PyTorchLightning/lightning-bolts
from typing import Generator, Optional, Sequence, Union
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from data.mio_dataset import MIOTCDDataset
from data.ecg5000_dataset import ECG5000Dataset
from data.p12_dataset import P12Dataset, P12TransformerDataset

from .active import ActiveLearningDataset
from .base_datamodule import BaseDataModule
from .utils import ActiveSubset
from .longtail import create_imbalanced_dataset
from .skin_dataset import ISIC2016, ISIC2019
from .transformations import get_transform


class TorchVisionDM(BaseDataModule):
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
        timeout: int = 0,
        val_size: Optional[int] = None,
        balanced_sampling: bool = False,
        balanced_test_val: bool = False,
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
        if dataset not in ("ecg5000", "p12", "p12_transformer"):
            if len(shape) == 3:
                assert shape[-1] == len(mean)
                assert shape[-1] == len(std)
            else:
                assert shape[0] == len(mean)
                assert shape[0] == len(std)

        self.train_transforms = get_transform(
            transform_train, self.mean, self.std, self.shape
        )
        self.test_transforms = get_transform(
            transform_test, self.mean, self.std, self.shape
        )
        self.imbalance = imbalance
        self.balanced_test_val = balanced_test_val

        ### IMPLEMENTATION ###
        # Novel image datasets can be added here
        if self.dataset == "mnist":
            self.dataset_cls = MNIST
        elif self.dataset == "cifar10":
            self.dataset_cls = CIFAR10
        elif self.dataset == "cifar100":
            self.dataset_cls = CIFAR100
        elif self.dataset == "fashion_mnist":
            self.dataset_cls = FashionMNIST
        elif self.dataset == "isic2016":
            self.dataset_cls = ISIC2016
        elif self.dataset == "isic2019":
            self.dataset_cls = ISIC2019
        elif self.dataset == "miotcd":
            self.dataset_cls = MIOTCDDataset
        elif self.dataset == "ecg5000":
            self.dataset_cls = ECG5000Dataset
        elif self.dataset == "p12":
            self.dataset_cls = P12Dataset
        elif self.dataset == "p12_transformer":
            self.dataset_cls = P12TransformerDataset
        else:
            raise NotImplementedError
        self._setup_datasets()

        if not self.shuffle:
            raise ValueError("shuffle flag has to be set to true")

    def _stratified_3way_split(self, labels, n_total, test_size, val_size,
                               balanced_eval: bool = False):
        """Return (test_idx, val_idx, pool_idx) with stratification by class label.

        When balanced_eval=True, test and val receive equal numbers of samples
        per class (limited by the minority class), so eval sets are 50/50.
        The pool receives all remaining samples and may still be imbalanced.
        """
        rng = np.random.default_rng(self.seed)
        test_idx, val_idx, pool_idx = [], [], []
        classes = np.unique(labels)
        n_classes = len(classes)

        if balanced_eval:
            n_test_per_class = max(1, test_size // n_classes)
            n_val_per_class  = max(1, val_size  // n_classes)

        for c in classes:
            c_idx = np.where(labels == c)[0]
            rng.shuffle(c_idx)
            if balanced_eval:
                n_test = min(n_test_per_class, len(c_idx) // 3)
                n_val  = min(n_val_per_class,  (len(c_idx) - n_test) // 3)
            else:
                n_test = max(1, round(len(c_idx) * test_size / n_total))
                n_val  = max(1, round(len(c_idx) * val_size  / n_total))
            test_idx.append(c_idx[:n_test])
            val_idx.append( c_idx[n_test:n_test + n_val])
            pool_idx.append(c_idx[n_test + n_val:])
        return (np.concatenate(test_idx),
                np.concatenate(val_idx),
                np.concatenate(pool_idx))

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
                _ecg_labels, _ecg_n_total, _ecg_test_size, _ecg_val_size
            )

        if self.dataset in ("p12", "p12_transformer"):
            _p12_root       = os.path.join(self.data_root, "P12data/processed_data")
            _p12_labels     = self.dataset_cls(root=_p12_root).targets
            _p12_n_total    = len(_p12_labels)
            _p12_test_size  = max(1, round(0.1 * _p12_n_total))
            _p12_val_size   = (self.val_split if isinstance(self.val_split, int)
                               else max(1, round(self.val_split * _p12_n_total)))
            _p12_test_idx, _p12_val_idx, _p12_pool_idx = self._stratified_3way_split(
                _p12_labels, _p12_n_total, _p12_test_size, _p12_val_size,
                balanced_eval=self.balanced_test_val,
            )

        # ------------------------------------------------------------------ #
        # For standard TorchVision datasets, ensure they are downloaded.      #
        # ------------------------------------------------------------------ #
        if self.dataset not in ("ecg5000", "p12", "p12_transformer"):
            try:
                self.dataset_cls(root=self.data_root, download=False)
            except:  # Error is assumed here to stem from data not being present.
                self.dataset_cls(root=self.data_root, download=True)

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
            _p12_full = self.dataset_cls(root=_p12_root)
            self.train_set = ActiveSubset(_p12_full, _p12_pool_idx)
        else:
            self.train_set = self.dataset_cls(
                self.data_root, train=True, transform=self.train_transforms
            )
        if self.dataset not in ("ecg5000", "p12", "p12_transformer"):
            self.train_set = self._split_dataset(self.train_set, train=True)

        if self.imbalance:
            self.train_set = create_imbalanced_dataset(
                self.train_set, imb_type="exp", imb_factor=0.02
            )

        if self.active:
            if self.dataset in ("p12", "p12_transformer"):
                # P12 datasets handle normalisation internally and have no transform attribute.
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
            _p12_full_eval = self.dataset_cls(root=_p12_root)
            self.val_set   = Subset(_p12_full_eval, _p12_val_idx)
            self.test_set  = Subset(_p12_full_eval, _p12_test_idx)
        else:
            self.val_set = self.dataset_cls(
                self.data_root, train=True, transform=self.test_transforms
            )
            self.test_set = self.dataset_cls(
                self.data_root, train=False, transform=self.test_transforms
            )
        if self.dataset not in ("ecg5000", "p12", "p12_transformer"):
            self.val_set = self._split_dataset(self.val_set, train=False)

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

        xb, yb = next(iter(train_loader))
        print(f"  Train batch x: {tuple(xb.shape)}  y: {tuple(yb.shape)}")
        assert xb.dtype == torch.float32, "x must be float32"
        assert yb.dtype == torch.int64,   "y must be int64"

        xb, yb = next(iter(val_loader))
        print(f"  Val   batch x: {tuple(xb.shape)}  y: {tuple(yb.shape)}")

        xb, yb = next(iter(test_loader))
        print(f"  Test  batch x: {tuple(xb.shape)}  y: {tuple(yb.shape)}")

        xb, yb = next(iter(pool_loader))
        print(f"  Pool  batch x: {tuple(xb.shape)}  y: {tuple(yb.shape)}")

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
    dm_ecg = TorchVisionDM(
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

    # ------------------------------------------------------------------ #
    # P12                                                                  #
    # ------------------------------------------------------------------ #
    dm_p12 = TorchVisionDM(
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
    )
    _check_dm("P12", dm_p12, n_label=200)

    print("\nAll checks passed.")
