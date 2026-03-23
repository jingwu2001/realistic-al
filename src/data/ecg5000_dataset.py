"""
PyTorch Dataset for the ECG5000 time series classification benchmark.

The dataset contains 5000 ECG signals of length 140, labelled into 5 classes:
  0 – Normal
  1 – R-on-T Premature Ventricular Contraction
  2 – Premature Ventricular Contraction
  3 – Supra-ventricular Premature or Ectopic Beat
  4 – Unclassified Beat

Pre-requisite: run  `python datasets/download_ecg5000.py`  first.

Transformations for time series – recommended pipeline
-------------------------------------------------------
TRAINING
  1. Z-score normalisation   (per-sample, mean=0, std=1)
  2. Gaussian noise jitter   (σ ≈ 0.01–0.05) – common TS augmentation
  3. Random time-shift       (roll by a few time steps) – optional

TESTING / VALIDATION
  1. Z-score normalisation only (same as training, but no stochastic augmentation)

Why z-score per sample?
  ECG amplitude varies between patients / recordings. Normalising per sample
  makes the representation amplitude-invariant and helps InceptionTime converge.

Why NOT:
  - Temporal flipping: flips the clinical meaning of the ECG waveform.
  - Frequency-domain warping: usually reserved for very large datasets.
  - Image-style augmentations (crop, colour jitter): these are 1-D signals.
"""

import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class ZScoreNormalize:
    """Normalise a single time-series to zero mean and unit variance.

    Operates on a numpy array of shape (C, T) or (T,).
    Falls back to dividing by 1 if std == 0 to avoid NaN.
    """

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True)
        std  = np.where(std == 0, 1.0, std)
        x    = (x - mean) / std
        return torch.tensor(x, dtype=torch.float32)


class AddGaussianNoise:
    """Add zero-mean Gaussian noise with standard deviation `sigma`.

    Operates on a torch.Tensor. Apply AFTER ZScoreNormalize.
    """

    def __init__(self, sigma: float = 0.02):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


class RandomTimeShift:
    """Randomly roll the time series by up to `max_shift` time steps.

    ECG signals are periodic, so a circular shift is a valid augmentation.
    Operates on a torch.Tensor of shape (C, T). Apply AFTER ZScoreNormalize.
    """

    def __init__(self, max_shift: int = 10):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class Compose:
    """Minimal torchvision-style Compose for time-series transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transform(noise_sigma: float = 0.02, max_shift: int = 10) -> Compose:
    """Standard augmentation pipeline for ECG training splits."""
    return Compose([
        ZScoreNormalize(),
        AddGaussianNoise(sigma=noise_sigma),
        RandomTimeShift(max_shift=max_shift),
    ])


def get_eval_transform() -> Compose:
    """Deterministic pipeline for validation / test splits."""
    return Compose([ZScoreNormalize()])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECG5000Dataset(Dataset):
    """ECG5000 time-series classification dataset.

    Expects pre-downloaded .npz files produced by datasets/download_ecg5000.py.

    Args:
        root:      Directory containing the .npz files
                   (default: ``datasets/ecg5000``).
        split:     One of ``'train'``, ``'test'``, or ``'all'``.
        transform: Optional callable applied to each sample (numpy->tensor).
                   If *None*, ZScoreNormalize is applied automatically.

    Attributes:
        data    (np.ndarray): float32 array, shape (N, 1, 140).
        targets (np.ndarray): int array,     shape (N,).
        classes (list[int]):  [0, 1, 2, 3, 4].
    """

    CLASS_NAMES = [
        "Normal",
        "R-on-T PVC",
        "PVC",
        "SP/Ectopic Beat",
        "Unclassified",
    ]
    N_CLASSES = 5

    def __init__(
        self,
        root: str = "/home/jing/Desktop/realistic-al/datasets/ecg5000",
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in ("train", "test", "all"), \
            f"split must be 'train', 'test', or 'all', got '{split}'"

        npz_path = os.path.join(root, f"ecg5000_{split}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Dataset file not found: {npz_path}\n"
                "Run  `python datasets/download_ecg5000.py`  first."
            )

        d = np.load(npz_path)
        self.data    = d["X"].astype(np.float32)   # (N, 1, 140)
        self.targets = d["y"].astype(np.int64)     # (N,)
        self.classes = list(range(self.N_CLASSES))

        # Default transform: z-score only (deterministic, safe for all splits)
        self.transform = transform if transform is not None else get_eval_transform()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x = self.data[index]        # np.ndarray  (1, 140)
        y = int(self.targets[index])
        x = self.transform(x)       # → torch.Tensor (1, 140)
        return x, y


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_ds = ECG5000Dataset(split="train",  transform=get_train_transform())
    test_ds  = ECG5000Dataset(split="test",   transform=get_eval_transform())

    print(f"Train samples : {len(train_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    x, y = train_ds[0]
    print(f"Sample shape  : {x.shape}")   # → torch.Size([1, 140])
    print(f"Sample label  : {y}")
    print(f"x mean={x.mean():.4f}  std={x.std():.4f}")   # ~0, ~1

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    xb, yb = next(iter(loader))
    print(f"Batch shape   : {xb.shape}")  # → torch.Size([64, 1, 140])
