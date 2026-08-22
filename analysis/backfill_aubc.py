"""Backfill area-under-budget-curve (AUBC) summary metrics onto existing
wandb runs.

For every run in the project, pulls the per-iteration history
(``al/n_labelled`` vs every ``al/test_*`` metric), computes the normalized
trapezoidal area (see ``utils.wandb_utils.compute_aubc``) and writes it into
the run summary as ``al_summary/aubc_<metric>`` — the same keys new runs log
at the end of ``active_loop``. Afterwards the runs table can be sorted or
grouped by AUBC directly.

Usage (from the repo root):
    python analysis/backfill_aubc.py --dry-run          # show, write nothing
    python analysis/backfill_aubc.py                    # write summaries
    python analysis/backfill_aubc.py --search mimic3    # only matching names
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.wandb_utils import compute_aubc  # noqa: E402

import wandb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default="jingwusince01-national-taiwan-uni/realistic-al",
        help="entity/project to backfill",
    )
    parser.add_argument("--search", default="", help="only runs whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="print AUBCs without writing")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many processed runs (0 = all)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="recompute runs that already have al_summary/aubc_* keys",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.project, order="-created_at")

    processed = skipped = 0
    for run in runs:
        if args.search and args.search not in run.name:
            continue
        if run.state == "running":
            continue
        if not args.overwrite and any(k.startswith("al_summary/aubc_") for k in run.summary.keys()):
            skipped += 1
            continue

        hist = run.history(samples=10000, pandas=True)
        if hist is None or hist.empty or "al/n_labelled" not in hist.columns:
            skipped += 1
            continue
        metric_cols = [c for c in hist.columns if c.startswith("al/test_")]
        if not metric_cols:
            skipped += 1
            continue

        x = hist["al/n_labelled"].values
        summary = {}
        for col in metric_cols:
            v = compute_aubc(x, hist[col].values)
            if np.isfinite(v):
                summary[f"al_summary/aubc_{col.replace('al/', '')}"] = round(v, 6)
        if not summary:
            skipped += 1
            continue

        short = {k.replace("al_summary/aubc_test_", ""): v for k, v in summary.items()}
        print(f"{'DRY ' if args.dry_run else ''}{run.name} [{run.state}]: {short}")
        if not args.dry_run:
            run.summary.update(summary)
            run.update()
        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"done: {processed} runs {'previewed' if args.dry_run else 'updated'}, {skipped} skipped")


if __name__ == "__main__":
    main()
