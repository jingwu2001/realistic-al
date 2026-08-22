"""Recreate the pre-wandb CIFAR10-Imb runs shown in the March-2026 figure on
wandb, and log their AUBC summaries.

These runs predate the wandb integration (``utils.wandb_utils`` landed
2026-07-11), so there is nothing on wandb to backfill: this CREATES the runs
from the on-disk Hydra run dirs (``.hydra/config.yaml``, per-loop
``metrics.csv``, ``stored.npz``, ``test_metrics.csv``).

Faithful to the figure ``cifar10 - active-cifar10_low - CIFAR10Imb``: it logs
exactly the 13 methods in that plot's legend, and for each method selects the
*same* seed dirs the plot does — ``analysis/plot_simple._select_seed_dirs``
returns the last 3 run dirs by timestamp. (For ``bandit rbf-1.0`` / ``rbf-10.0``
that resolves to the March re-runs, not the February ones — matching the image.)

Every recreated run gets the ``old_run`` tag.

Timestamp caveat
----------------
wandb stamps each run's ``created_at`` at import time (now); the SDK cannot
backdate it. The true run time is preserved instead as config
(``orig_experiment_id`` / ``orig_start_time``), in the session tag, and is
recoverable from the run name. The AL curves use ``al/n_labelled`` as their
x-axis, so they render identically to the originals regardless of wall-clock.

Usage (from the repo root, in the ``real-al`` env):
    python analysis/import_old_runs_wandb.py --dry-run     # parse + plan only
    python analysis/import_old_runs_wandb.py               # create the runs
    python analysis/import_old_runs_wandb.py --overwrite    # replace existing
"""
import argparse
import datetime
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.wandb_utils import (  # noqa: E402
    _read_loop_metrics,
    aubc_summary,
    make_group,
    make_run_name,
    make_tags,
    method_variant,
)

# The 13 legend entries of the figure, as unique substrings of their folder
# names. Anchored to the "drop-0.5" sweep prefix so the co-located "drop-0"
# baseline folders (present in the balanced cifar10 regime) are never matched;
# the trailing "_ep" disambiguates gamma-1.0 from gamma-10.0. NOT included: the
# gamma-0.0009765625 (2^-10) variants (April runs, not in the plot) nor the
# plain no-kernel vendi/bandit.
_SWEEP_PREFIX = "drop-0.5_aug-cifar_randaugmentMC_"
TARGET_SUBSTRINGS = [
    _SWEEP_PREFIX + s
    for s in [
        "acq-vendi_norm-minmax_kernel-rbf_gamma-10.0_ep",
        "acq-vendi_norm-minmax_kernel-rbf_gamma-1.0_ep",
        "acq-vendi_norm-minmax_kernel-rbf_gamma-0.1_ep",
        "acq-vendi_norm-minmax_kernel-cosine_gamma-None_ep",
        "acq-bandit_norm-minmax_kernel-rbf_gamma-10.0_ep",
        "acq-bandit_norm-minmax_kernel-rbf_gamma-1.0_ep",
        "acq-bandit_norm-minmax_kernel-rbf_gamma-0.1_ep",
        "acq-bandit_norm-minmax_kernel-cosine_gamma-None_ep",
        "acq-badge_ep",
        "acq-bald_ep",
        "acq-entropy_ep",
        "acq-random_ep",
        "acq-kcentergreedy_ep",
    ]
]

TS_FMT = "%Y-%m-%d_%H-%M-%S-%f"

# Shared holder so the `now` OmegaConf resolver can resolve ${now:...} in each
# run's saved config to that run's real start time (from its dir name).
_CURRENT_TS = {"dt": None}


def _now_resolver(pattern="%Y-%m-%d_%H-%M-%S-%f"):
    dt = _CURRENT_TS["dt"]
    return dt.strftime(pattern) if dt is not None else "unknown"


OmegaConf.register_new_resolver("now", _now_resolver, replace=True)


def _is_finished(run_dir):
    """A run the figure would actually render: stored.npz plus a test_metrics.csv
    with >=8 AL rounds (mirrors plot_simple._load_seed_df's not-finished filter).
    Aborted/crashed dirs (no metrics) are excluded, exactly as the plot excludes
    them."""
    if not os.path.exists(os.path.join(run_dir, "stored.npz")):
        return False
    csv = os.path.join(run_dir, "test_metrics.csv")
    if not os.path.exists(csv):
        return False
    try:
        return len(pd.read_csv(csv)) >= 8
    except Exception:
        return False


def _select_seed_dirs(exp_dir, before=None):
    """The last 3 *finished* run dirs by timestamp, restricted to runs started
    strictly before ``before`` (a datetime, or None for no cutoff).

    plot_simple takes the last-3 by name and then drops unfinished ones; here we
    filter to finished first so a stray aborted dir sitting in the window can't
    shrink a method to fewer seeds than it actually completed. The ``before``
    cutoff mirrors plot_simple's ``skip_after``: these are the pre-experiment
    runs, so duplicate-seed folders (bandit rbf-1.0) resolve to the February
    runs, not the March re-runs."""
    dirs = sorted((d for d in os.scandir(exp_dir) if d.is_dir()), key=lambda d: d.name)
    finished = []
    for d in dirs:
        try:
            t = datetime.datetime.strptime(d.name, TS_FMT)
        except ValueError:
            continue  # skip .hydra and other non-timestamp dirs
        if before is not None and t >= before:
            continue
        if _is_finished(d.path):
            finished.append(d.path)
    return finished[-3:]


def _commit_at(dt):
    """Best-effort short git commit that was HEAD at datetime ``dt``."""
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "-1", "--before", dt.strftime("%Y-%m-%d %H:%M:%S"), "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            return out[:8]
    except Exception:
        pass
    return "preint"


def _load_cfg(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    # Keys the wandb helpers read that predate this config snapshot.
    if "is_test" not in cfg.trainer:
        cfg.trainer.is_test = False
    return cfg


def _hydra_choices(run_dir):
    h = OmegaConf.load(os.path.join(run_dir, ".hydra", "hydra.yaml"))
    ch = h.hydra.runtime.choices
    active_name = ch.get("active", "")
    data_name = ch.get("data", None) or None
    return active_name, data_name


def _resolved_config(cfg, data_name, active_name):
    try:
        config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        config = OmegaConf.to_container(cfg, resolve=False)
    config.update(
        {
            "query_name": cfg.query.name,
            "model_name": cfg.model.name,
            "data_name": data_name or cfg.data.name,
            "active_name": active_name,
            "seed": cfg.trainer.seed,
            "num_labelled": cfg.active.num_labelled,
            "acq_size": cfg.active.acq_size,
            "num_iter": cfg.active.num_iter,
        }
    )
    return config


def _iter_history(run_dir):
    """Yield (al_iter, n_labelled, loop_metrics) per AL round, in order."""
    stored = np.load(os.path.join(run_dir, "stored.npz"), allow_pickle=True)
    num_samples = list(stored["num_samples"])
    loop_dirs = sorted(
        (d.path for d in os.scandir(run_dir) if d.is_dir() and d.name.startswith("loop-")),
        key=lambda p: int(os.path.basename(p).split("-")[1]),
    )
    n = min(len(num_samples), len(loop_dirs))
    for i in range(n):
        yield i, int(num_samples[i]), _read_loop_metrics(loop_dirs[i])


def _aubc(run_dir):
    stored = np.load(os.path.join(run_dir, "stored.npz"), allow_pickle=True)
    num_samples = stored["num_samples"]
    metrics_df = pd.read_csv(os.path.join(run_dir, "test_metrics.csv"), index_col=0)
    return aubc_summary(metrics_df, num_samples)


def find_target_dirs(regime_dir):
    """Map each of the 13 target substrings to its folder under regime_dir."""
    folders = [d.path for d in os.scandir(regime_dir) if d.is_dir()]
    out = []
    for sub in TARGET_SUBSTRINGS:
        matches = [f for f in folders if sub in os.path.basename(f)]
        if not matches:
            print(f"  WARNING: no folder matches '{sub}'")
            continue
        if len(matches) > 1:
            print(f"  WARNING: {len(matches)} folders match '{sub}': {matches}")
        out.append((sub, matches[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default="jingwusince01-national-taiwan-uni")
    ap.add_argument("--project", default="realistic-al")
    ap.add_argument(
        "--regime-dir",
        default="experiments/activelearning/cifar10_imb/active-cifar10_low",
        help="the active-cifar10_low regime dir shown in the figure",
    )
    ap.add_argument("--dry-run", action="store_true", help="parse + plan, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="delete + recreate runs that already exist")
    ap.add_argument(
        "--before", default="2026-03-01_00-00-00-000000",
        help="only log runs started strictly before this (%%Y-%%m-%%d_%%H-%%M-%%S-%%f); "
             "'none' to disable the cutoff",
    )
    args = ap.parse_args()

    before = None if args.before.lower() == "none" else datetime.datetime.strptime(args.before, TS_FMT)
    regime_dir = os.path.abspath(args.regime_dir)
    print(f"Regime: {regime_dir}")
    print(f"Cutoff: runs started before {before}" if before else "Cutoff: none")
    targets = find_target_dirs(regime_dir)
    print(f"Matched {len(targets)}/{len(TARGET_SUBSTRINGS)} methods\n")

    api = None
    if not args.dry_run:
        import wandb
        api = wandb.Api()

    total = created = skipped = 0
    for sub, exp_dir in targets:
        seed_dirs = _select_seed_dirs(exp_dir, before=before)
        print(f"### {os.path.basename(exp_dir)}")
        print(f"    selected {len(seed_dirs)} seed dir(s) (last-3 before cutoff):")
        for run_dir in seed_dirs:
            total += 1
            ts_name = os.path.basename(run_dir)
            dt = datetime.datetime.strptime(ts_name, TS_FMT)
            _CURRENT_TS["dt"] = dt

            cfg = _load_cfg(run_dir)
            active_name, data_name = _hydra_choices(run_dir)
            data_name = data_name or cfg.data.name

            run_name = make_run_name(cfg, data_name)
            group = make_group(cfg, active_name, data_name)
            session_tag = f"{dt:%Y%m%d-%H%M%S}_{_commit_at(dt)}"
            tags = make_tags(cfg, active_name, session_tag, data_name) + ["old_run"]

            aubc = _aubc(run_dir)
            rel = os.path.relpath(run_dir, os.getcwd())
            run_id = "old-" + hashlib.md5(rel.encode()).hexdigest()[:10]

            print(f"      - {ts_name}  seed={cfg.trainer.seed}")
            print(f"          name : {run_name}")
            print(f"          id   : {run_id}   tags: {tags}")
            print(f"          aubc : {aubc}")

            if args.dry_run:
                continue

            full_id = f"{args.entity}/{args.project}/{run_id}"
            try:
                existing = api.run(full_id)
            except Exception:
                existing = None
            if existing is not None:
                if args.overwrite:
                    print("          (exists -> deleting for overwrite)")
                    existing.delete()
                else:
                    print("          (exists -> skip; use --overwrite to replace)")
                    skipped += 1
                    continue

            config = _resolved_config(cfg, data_name, active_name)
            config.update(
                {
                    "orig_experiment_id": ts_name,
                    "orig_start_time": dt.isoformat(),
                    "imported_from": rel,
                }
            )

            run = wandb.init(
                entity=args.entity,
                project=args.project,
                id=run_id,
                name=run_name,
                group=group,
                tags=tags,
                notes=f"Imported from on-disk run {rel} (original start {dt.isoformat()}).",
                config=config,
                resume="never",
                reinit=True,
            )
            run.define_metric("al/n_labelled")
            run.define_metric("al/*", step_metric="al/n_labelled")
            for i, n_labelled, loop_metrics in _iter_history(run_dir):
                log_dict = {"al_iter": i, "al/n_labelled": n_labelled}
                log_dict.update(loop_metrics)
                run.log(log_dict)
            if aubc:
                run.summary.update(aubc)
            run.finish()
            created += 1
        print()

    print(f"Done. {total} runs planned; created={created}, skipped={skipped}"
          + (" (dry-run: nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
