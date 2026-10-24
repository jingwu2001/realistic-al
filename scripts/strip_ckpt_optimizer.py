"""
Drop optimizer state from Lightning checkpoints to free disk space.

The AL loop never resumes training from a checkpoint, and every offline
analysis loads only ``state_dict``, so ``optimizer_states`` (Adam: 2x the
weights, SGD momentum: 1x) is dead weight. Each file is rewritten to a temp file
next to it, reloaded and checked tensor-by-tensor against the original
state_dict, then atomically renamed over it with the original mtime kept.
Already-stripped files are skipped, so the script can be re-run or resumed.

Kept gentle on a shared machine: each file is fsynced (no dirty-page backlog)
and dropped from the page cache afterwards. Run it at idle I/O priority:
  nice -n 19 ionice -c3 python scripts/strip_ckpt_optimizer.py experiments

Usage:
  python scripts/strip_ckpt_optimizer.py experiments/activelearning/mimic3_sand [--dry-run] [-j 1] [--sleep 0.2]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

MIN_AGE_S = 30 * 60  # skip files written recently (a run may still be writing)
SLEEP_S = 0.0


def _init_worker(sleep_s):
    global SLEEP_S
    SLEEP_S = sleep_s
    torch.set_num_threads(1)


def _drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def strip(path: str):
    p = Path(path)
    st = p.stat()
    if time.time() - st.st_mtime < MIN_AGE_S:
        return path, "recent", st.st_size, st.st_size
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not ckpt.get("optimizer_states"):
        _drop_cache(p)
        return path, "skip", st.st_size, st.st_size
    ckpt["optimizer_states"] = []
    tmp = p.with_name(p.name + ".strip.tmp")
    with open(tmp, "wb") as f:
        torch.save(ckpt, f)
        f.flush()
        os.fsync(f.fileno())
    new = torch.load(tmp, map_location="cpu", weights_only=False)["state_dict"]
    old = ckpt["state_dict"]
    if list(new) != list(old) or not all(torch.equal(new[k], old[k]) for k in old):
        tmp.unlink()
        return path, "MISMATCH", st.st_size, st.st_size
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp, p)
    _drop_cache(p)
    time.sleep(SLEEP_S)
    return path, "ok", st.st_size, p.stat().st_size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0, help="pause (s) after each rewritten file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(str(f) for r in args.roots for f in r.rglob("*.ckpt"))
    total = sum(os.path.getsize(f) for f in paths)
    print(f"{len(paths)} checkpoints, {total / 1e9:.1f} GB", flush=True)
    if args.dry_run:
        sys.exit(0)

    counts, saved = {}, 0
    with Pool(args.jobs, initializer=_init_worker, initargs=(args.sleep,)) as pool:
        for i, (path, status, before, after) in enumerate(
            pool.imap_unordered(strip, paths, chunksize=4), 1
        ):
            counts[status] = counts.get(status, 0) + 1
            saved += before - after
            if status == "MISMATCH":
                print(f"MISMATCH (left unchanged): {path}", flush=True)
            if i % 200 == 0 or i == len(paths):
                print(f"[{i}/{len(paths)}] {counts} freed {saved / 1e9:.1f} GB", flush=True)
    print(f"done: {counts}, freed {saved / 1e9:.1f} GB")
