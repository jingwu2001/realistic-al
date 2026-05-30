"""
Evaluate saved GRU-D checkpoints from active learning experiments.

Produces per-loop precision, recall, macro F1, AUROC, and AUPRC on the held-out
test set and writes results to `eval_metrics.csv` inside each experiment run directory.

Why this works
--------------
Each AL loop saves two artefacts that are enough to recreate the exact
evaluation performed during training:

  loop-X/checkpoints/epoch=N-step=M.ckpt
      PyTorch-Lightning checkpoint written by ModelCheckpoint (see
      trainer.py:_init_ckpt_callback).  It is a zip archive whose
      "state_dict" key maps directly to BayesianModule.state_dict().

  loop-X/hparams.yaml
      Full Hydra config snapshot written by PL's save_hyperparameters()
      inside BayesianModule.__init__.  It carries every field required to
      rebuild the model (architecture, num_classes, dropout, k …) and the
      datamodule (data_root, seed, val_split, random_split …).

The test split is deterministic: the datamodule factory uses the seed from
the config, so the same indices are held out across every loop of a given run.
We therefore do NOT need to restore data_ckpt.pkl; we only need the test
dataloader.

After loading the checkpoint the forward pass mirrors what _test() /
test_step() do in training:
  - k=10 MC-dropout samples averaged (k is taken from hparams)
  - model in eval() mode so BN/non-MC layers behave correctly
  - metrics computed from the confusion matrix, matching the formula in
    ImbClassMetricCallback.compute_pred_metrics (metrics_callback.py:133)

Usage
-----
    # Evaluate a single run directory
    python src/eval_checkpoints.py \
        experiments/activelearning/p12/active-p12_low/basic_model-gru_d_acq-bandit_ep-50/2026-04-06_15-48-32-261235

    # Evaluate all GRU-D runs under experiments/activelearning/p12/
    python src/eval_checkpoints.py --all-gru

    # Optional flags
    --device cuda:0   (default: cuda:0 if available, else cpu)
    --output-name     name for the output CSV (default: eval_metrics.csv)
"""

import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where the script is called from.
# The project's source lives next to this file.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torchmetrics import AUROC, AveragePrecision, ConfusionMatrix

# These imports work because SRC_DIR is on sys.path
from models.bayesian import BayesianModule
from run_training import get_torchvision_dm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_device(x, device):
    """Move x to device; handles plain tensors and 4-tuples (transformer input)."""
    if isinstance(x, (tuple, list)):
        return type(x)(t.to(device) for t in x)
    return x.to(device)


def find_checkpoint(loop_dir: Path) -> Path:
    """Return the single .ckpt file saved by ModelCheckpoint.

    trainer.py:_init_ckpt_callback saves the best model only (save_last=False
    by default), so there is exactly one .ckpt per loop directory.
    """
    ckpt_dir = loop_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    # Take the lexicographically last one; there is usually only one.
    return ckpts[-1]


def load_model(hparams_path: Path, ckpt_path: Path) -> BayesianModule:
    """Rebuild BayesianModule from hparams and load its weights.

    BayesianModule stores the full Hydra config via save_hyperparameters()
    (abstract_classifier.py / bayesian.py __init__).  OmegaConf.load gives us
    a DictConfig that the constructor accepts directly, recreating the exact
    same architecture that was trained.
    """
    cfg = OmegaConf.load(hparams_path)

    # Instantiate the model with the same architecture used during training.
    model = BayesianModule(cfg)

    # Load weights exactly as trainer.py:_fit does after training completes
    # (trainer.py:178-179):
    #   ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    #   self.model.load_state_dict(ckpt["state_dict"], strict=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, cfg


def build_test_loader(cfg, batch_size: int = 256):
    """Create the configured DataModule and return its test DataLoader.

    get_torchvision_dm (run_training.py:23) is the same factory used by
    main.py:active_loop.  Passing active=False skips the ActiveLearningDataset
    wrapper; we only need the fixed test split.

    The test split is carved out before the train/val split using the seed
    in the config (base_datamodule.py setup()), so it is identical across
    every AL loop of a given run.
    """
    # Override batch size for faster inference (does not affect results).
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["trainer"]["batch_size"] = batch_size
    cfg = OmegaConf.create(cfg)

    dm = get_torchvision_dm(cfg, active_dataset=False)
    dm.prepare_data()
    dm.setup(stage="test")
    return dm.test_dataloader()


def compute_metrics(model: BayesianModule, test_loader, device: torch.device, num_classes: int):
    """Run inference and compute per-class + macro precision, recall, F1, AUROC, AUPRC.

    The inference logic mirrors BayesianModule.test_step / validation_step
    (abstract_classifier.py:177-198):
      - forward(x) uses k=self.k MC samples and averages log-probs
      - argmax over the averaged distribution gives the hard prediction
      - softmax(logprob) gives class probabilities used for AUROC / AUPRC

    Metrics are derived from the confusion matrix using the same formulas as
    ImbClassMetricCallback.compute_pred_metrics (metrics_callback.py:133-173):
      recall_i    = TP_i / (TP_i + FN_i)  =  conf[i,i] / conf[i,:].sum()
      precision_i = TP_i / (TP_i + FP_i)  =  conf[i,i] / conf[:,i].sum()
      f1_i        = 2*TP_i / (2*TP_i + FP_i + FN_i)
                  = 2*conf[i,i] / (conf[i,:].sum() + conf[:,i].sum())
    Macro averages are the unweighted mean across all classes.

    AUROC and AUPRC use macro averaging (unweighted mean over classes), which
    matches the convention used for the other macro-averaged metrics above.
    For binary datasets (num_classes=2, e.g. P12), the torchmetrics 'binary'
    task is used instead, which scores against the positive class (class 1).
    """
    task = "binary" if num_classes == 2 else "multiclass"

    conf_metric  = ConfusionMatrix(task=task if task == "binary" else "multiclass",
                                   num_classes=num_classes, normalize=None).to(device)
    auroc_metric = AUROC(task=task, num_classes=num_classes if task == "multiclass" else None,
                         average="macro" if task == "multiclass" else "micro").to(device)
    auprc_metric = AveragePrecision(task=task,
                                    num_classes=num_classes if task == "multiclass" else None,
                                    average="macro" if task == "multiclass" else "micro").to(device)

    model = model.to(device)
    # eval() disables BN running-stat updates and non-MC dropout; the GRU-D's
    # ConsistentMCDropout remains active because it ignores the training flag
    # (that is its design for Bayesian inference).
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = to_device(x, device), y.to(device)
            # k=None uses model.k (10 by default for GRU-D, from hparams).
            # forward() → mc_nll() averages the k log-softmax outputs
            # (abstract_classifier.py:73-86), then argmax gives the hard pred.
            logprob = model.forward(x, k=None)
            probs   = torch.softmax(logprob, dim=-1)   # (B, C) — needed for AUROC/AUPRC
            preds   = torch.argmax(logprob, dim=-1)

            conf_metric.update(preds, y)
            if task == "binary":
                # torchmetrics binary AUROC/AUPRC expect the positive-class score
                auroc_metric.update(probs[:, 1], y)
                auprc_metric.update(probs[:, 1], y)
            else:
                auroc_metric.update(probs, y)
                auprc_metric.update(probs, y)

    conf = conf_metric.compute().cpu()  # shape: (num_classes, num_classes)

    # Row i = true class i, col j = predicted class j.
    tp = conf.diag()
    row_sums = conf.sum(dim=1)  # true positives + false negatives per class
    col_sums = conf.sum(dim=0)  # true positives + false positives per class

    recall    = tp / row_sums.clamp(min=1)
    precision = tp / col_sums.clamp(min=1)
    f1        = 2 * tp / (row_sums + col_sums).clamp(min=1)

    results = {
        "macro_precision": precision.mean().item(),
        "macro_recall":    recall.mean().item(),
        "macro_f1":        f1.mean().item(),
        "acc":             (tp.sum() / conf.sum()).item(),
        "auroc":           auroc_metric.compute().item(),
        "auprc":           auprc_metric.compute().item(),
    }
    # Also store per-class values for reference
    for cls in range(num_classes):
        results[f"precision_cls{cls}"] = precision[cls].item()
        results[f"recall_cls{cls}"]    = recall[cls].item()
        results[f"f1_cls{cls}"]        = f1[cls].item()

    return results


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate_run(run_dir: Path, device: torch.device, output_name: str = "eval_metrics.csv"):
    """Evaluate all loop checkpoints in a single experiment run directory.

    A run directory is the timestamped folder produced by main.py:active_loop,
    e.g. experiments/activelearning/p12/active-p12_low/
              basic_model-gru_d_acq-bandit_ep-50/2026-04-06_15-48-32-261235/

    Inside it we expect loop-0/, loop-1/, … each containing:
      hparams.yaml   – saved by PL save_hyperparameters() at loop start
      checkpoints/   – best checkpoint saved by ModelCheckpoint
    """
    loop_dirs = sorted(run_dir.glob("loop-*/"), key=lambda p: int(p.name.split("-")[1]))
    if not loop_dirs:
        print(f"  [skip] No loop directories found in {run_dir}")
        return

    rows = []
    for loop_dir in loop_dirs:
        loop_idx = int(loop_dir.name.split("-")[1])
        hparams_path = loop_dir / "hparams.yaml"
        if not hparams_path.exists():
            print(f"  [skip] loop-{loop_idx}: hparams.yaml missing")
            continue

        try:
            ckpt_path = find_checkpoint(loop_dir)
        except FileNotFoundError as e:
            print(f"  [skip] loop-{loop_idx}: {e}")
            continue

        print(f"  loop-{loop_idx}: loading {ckpt_path.name} …", end=" ", flush=True)

        model, cfg = load_model(hparams_path, ckpt_path)

        # The test dataloader is rebuilt every loop.  Because the seed and
        # random_split flag are in hparams.yaml and don't change between loops,
        # each call produces the same test set – but we rebuild it here to be
        # safe and keep the code self-contained.
        test_loader = build_test_loader(cfg)

        num_classes = cfg.data.num_classes
        metrics = compute_metrics(model, test_loader, device, num_classes)
        metrics["loop"] = loop_idx

        print(f"F1={metrics['macro_f1']:.4f}  "
              f"P={metrics['macro_precision']:.4f}  "
              f"R={metrics['macro_recall']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  "
              f"AUPRC={metrics['auprc']:.4f}")
        rows.append(metrics)

        # Free GPU memory between loops
        del model
        torch.cuda.empty_cache()

    if not rows:
        print(f"  [warn] No results collected for {run_dir}")
        return

    df = pd.DataFrame(rows)
    # Put loop column first for readability
    front = ["loop", "macro_precision", "macro_recall", "macro_f1", "acc", "auroc", "auprc"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols].sort_values("loop").reset_index(drop=True)

    out_path = run_dir / output_name
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df


def find_all_transformer_runs(experiments_root: Path):
    """Discover all timestamped run directories under p12_transformer/."""
    run_dirs = []
    for exp_dir in sorted(experiments_root.glob("p12_transformer/**")):
        if not exp_dir.is_dir():
            continue
        for candidate in sorted(exp_dir.iterdir()):
            if candidate.is_dir() and (candidate / ".hydra").exists():
                run_dirs.append(candidate)
    return run_dirs


def find_all_gru_runs(experiments_root: Path):
    """Discover all timestamped run directories that belong to GRU-D experiments.

    Experiment directories follow the naming convention set by Hydra:
      experiments/activelearning/p12/<active_cfg>/<experiment_name>/<timestamp>/
    The experiment_name encodes the model, so we filter for 'gru_d'.
    """
    pattern = "p12/**/basic_model-gru_d*"
    run_dirs = []
    for exp_dir in sorted(experiments_root.glob(pattern)):
        if not exp_dir.is_dir():
            continue
        # Each sub-directory whose name looks like a timestamp is a run.
        for candidate in sorted(exp_dir.iterdir()):
            if candidate.is_dir() and (candidate / ".hydra").exists():
                run_dirs.append(candidate)
    return run_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GRU-D checkpoints: compute precision, recall, F1 per AL loop."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to a single experiment run directory (the timestamped folder).",
    )
    group.add_argument(
        "--all-gru",
        action="store_true",
        help="Scan experiments/activelearning/p12/ and evaluate all GRU-D runs.",
    )
    group.add_argument(
        "--all-transformer",
        action="store_true",
        help="Scan experiments/activelearning/p12_transformer/ and evaluate all transformer runs.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=SRC_DIR.parent / "experiments" / "activelearning",
        help="Root of the experiments tree (default: <repo>/experiments/activelearning).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for inference (default: cuda:0 or cpu).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="eval_metrics.csv",
        help="Filename for the output CSV written inside each run directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if args.all_gru:
        run_dirs = find_all_gru_runs(args.experiments_root)
        if not run_dirs:
            print(f"No GRU-D run directories found under {args.experiments_root}")
            sys.exit(1)
        print(f"Found {len(run_dirs)} GRU-D run(s).")
    elif args.all_transformer:
        run_dirs = find_all_transformer_runs(args.experiments_root)
        if not run_dirs:
            print(f"No transformer run directories found under {args.experiments_root}")
            sys.exit(1)
        print(f"Found {len(run_dirs)} transformer run(s).")
    else:
        run_dirs = [args.run_dir.resolve()]

    all_results = []
    for run_dir in run_dirs:
        print(f"\n=== {run_dir} ===")
        df = evaluate_run(run_dir, device=device, output_name=args.output_name)
        if df is not None:
            df["run"] = run_dir.name
            df["experiment"] = run_dir.parent.name
            all_results.append(df)

    # Optionally combine all runs into one summary CSV in the experiments root.
    if len(all_results) > 1:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = args.experiments_root / "gru_eval_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nCombined summary → {summary_path}")
