"""Freeze a local, ground-truth snapshot of every wandb run in the project.

wandb run history is append-only: once a script ``wr.log()``s into an existing
run (e.g. ``misc/phenotype_by_task.py`` resuming with ``resume="must"``), there
is no in-place undo. This tool captures a complete local copy of the current
state *before* such a change, so:

  * analysis/plots stay reproducible from local files regardless of the cloud,
  * you can diff old-vs-new to see exactly what a backfill changed,
  * a corrupted run can be rebuilt (delete + re-log) from the frozen values.

For every run it writes the full *unsampled* history (``scan_history``) plus the
config, summary, tags and metadata, as plain CSV/JSON (no parquet engine needed).
Re-running skips runs already captured, so an interrupted snapshot resumes cleanly.

Usage (from the repo root, real-al env, .env sourced)::

    conda run -n real-al python analysis/snapshot_wandb.py
    conda run -n real-al python analysis/snapshot_wandb.py --search mimic3_pheno
    conda run -n real-al python analysis/snapshot_wandb.py --out analysis/wandb_snapshots/pre-pheno-backfill
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import traceback

import pandas as pd
import wandb

DEFAULT_PROJECT = "jingwusince01-national-taiwan-uni/realistic-al"


def _git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .strip()
            .decode()
        )
    except Exception:
        return "nogit"


def _jsonable(obj):
    """Config/summary hold nested dicts, numpy scalars, wandb objects -> stringify safely."""
    return json.loads(json.dumps(obj, default=str))


def snapshot(project, out_dir, search="", overwrite=False):
    hist_dir = os.path.join(out_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(project, order="-created_at")
    total = len(runs)
    print(f"snapshotting {total} runs from {project} -> {out_dir}")

    index_rows, captured, skipped, failed = [], 0, 0, []
    for i, run in enumerate(runs):
        if search and search not in run.name:
            continue
        hist_path = os.path.join(hist_dir, f"{run.id}.csv")

        # Always record the index row (cheap: config/summary are already loaded).
        index_rows.append(
            {
                "id": run.id,
                "name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": str(run.created_at),
                "tags": json.dumps(list(run.tags)),
                "config": json.dumps(_jsonable(dict(run.config))),
                "summary": json.dumps(_jsonable(dict(run.summary))),
                "url": run.url,
            }
        )

        if os.path.exists(hist_path) and not overwrite:
            skipped += 1
            continue

        try:
            rows = list(run.scan_history())  # full, unsampled history
            df = pd.DataFrame(rows)
            df.to_csv(hist_path, index=False)
            captured += 1
        except Exception as e:
            failed.append((run.id, run.name, repr(e)))
            print(f"  !! failed {run.name} ({run.id}): {e}", file=sys.stderr)
            traceback.print_exc()

        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f"  [{i + 1}/{total}] captured={captured} skipped={skipped} failed={len(failed)}")

    # Index of all runs (config + summary snapshot in one file).
    pd.DataFrame(index_rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    manifest = {
        "project": project,
        "search": search or None,
        "snapshot_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "n_runs_indexed": len(index_rows),
        "n_history_captured": captured,
        "n_history_skipped": skipped,
        "n_failed": len(failed),
        "failed": [{"id": i, "name": n, "error": e} for i, n, e in failed],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\ndone: indexed {len(index_rows)} runs, "
        f"captured {captured} histories, skipped {skipped}, failed {len(failed)}"
    )
    print(f"manifest: {os.path.join(out_dir, 'manifest.json')}")
    if failed:
        print("re-run the same command to retry only the failed/missing histories.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="entity/project to snapshot")
    ap.add_argument("--search", default="", help="only runs whose name contains this substring")
    ap.add_argument(
        "--out",
        default=None,
        help="output dir (default: analysis/wandb_snapshots/<UTC timestamp>)",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-download histories already saved")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "wandb_snapshots",
        dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S"),
    )
    snapshot(args.project, out_dir, search=args.search, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
