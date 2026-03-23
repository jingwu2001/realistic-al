import argparse
import os

import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.ecg5000_dataset import ECG5000Dataset, get_eval_transform, get_train_transform
from src.models.networks.inception import InceptionBlock


def build_model(n_filters: int, bottleneck_channels: int, use_residual: bool) -> nn.Module:
    """Build the InceptionTime model with given hyperparameters."""
    model = nn.Sequential(
        InceptionBlock(
            in_channels=1,
            n_filters=n_filters,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
        ),
        InceptionBlock(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
        ),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(4 * n_filters, 5),  # 5 classes for ECG5000
    )
    return model


def objective(trial: optuna.Trial, data_root: str, device: torch.device) -> float:
    # 1. Suggest hyperparameters
    epochs = trial.suggest_int("epochs", 5, 50)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    n_filters = trial.suggest_categorical("n_filters", [16, 32, 64])
    bottleneck_channels = trial.suggest_categorical("bottleneck_channels", [16, 32, 64])
    # use_residual = trial.suggest_categorical("use_residual", [True, False])

    # 2. Setup Data (Train/Val split from the train file)
    train_ds_full = ECG5000Dataset(root=data_root, split="train", transform=get_train_transform())
    
    # We will use 80% of the train set for training, 20% for validation during hyperparameter tuning
    train_idx, val_idx = train_test_split(
        range(len(train_ds_full)), test_size=0.2, stratify=train_ds_full.targets, random_state=42
    )

    train_ds = Subset(train_ds_full, train_idx)
    # Use eval transform for validation split
    val_ds_full = ECG5000Dataset(root=data_root, split="train", transform=get_eval_transform())
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Setup Model, Optimizer, Loss
    model = build_model(n_filters, bottleneck_channels, use_residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

        # Report intermediate objective value to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Hyperparameters for InceptionTime on ECG5000")
    parser.add_argument("--data-root", type=str, default="/home/jing/Desktop/realistic-al/datasets/ecg5000", help="Path to ECG5000 .npz files")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of optuna trials")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Optuna study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    study.optimize(
        lambda trial: objective(trial, data_root=args.data_root, device=device),
        n_trials=args.n_trials
    )

    print("\nHyperparameter tuning completed.")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation Accuracy: {study.best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
