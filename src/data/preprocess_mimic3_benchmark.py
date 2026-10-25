"""
Convert a mimic3-benchmarks task (Harutyunyan et al. 2019) into the fixed
{all,single_stay}/{train,test}/*.npy objects that TimeSeriesDM reads -- the
counterpart of preprocess_mimic3_sand.py for the benchmark's own input.

Tasks (built by $DATA_ROOT/mimic3_benchmark/run_pipeline.sh +
create_in_hospital_mortality):
  phenotyping            25 CCS labels -> phenotype.npy   (data.name mimic3_pheno)
  in-hospital-mortality  first 48 h of stays >= 48 h -> mortality.npy (data.name mimic3_sand)
Each reads <bench>/<task>/{train,test}/listfile.csv + per-stay timeseries CSVs,
and <bench>/root/{train,test}/{SUBJECT}/episode{N}.csv (Icustay, Age, Gender).

Hourly grid = the benchmark's own Discretizer (mimic3models/preprocessing.py),
with the settings of its models: 1 h bins, last value in a bin, forward fill
(else the variable's normal value), observation masks, one-hot categorical
channels -> 76 channels per hour (59 values + 17 masks). Continuous value
channels are z-normalised with each cohort's train statistics over real hours;
masks and one-hots are left as is.

Window (--hours): phenotyping uses the whole stay in the benchmark.
  --hours N     keep the first (--align first) or last N hours; shorter stays
                are RIGHT-padded with zeros (values and masks), as the
                benchmark's pad_zeros -> X.npy (N, T, 76)
  --hours full  keep every hour; stays are stored unpadded back to back ->
                X_flat.npy (sum of lengths, 76) + offsets.npy (N + 1), padded
                per sample at load time (MIMIC3 datasets)
A real hour is never all-zero (the one-hot channels are always filled), so
all-zero rows mark padding; model=lstm packs on them like Keras Masking().
lengths.npy holds the number of real hours per stay. in-hospital-mortality is
always the benchmark's 48 h.

demo = (Age, Gender) from episode{N}.csv, z-normalised on train like
preprocess_mimic3_sand.py; the benchmark's own models do not use it (run with
model.use_static=false to match them).

Cohorts: all = every stay; single_stay = patients with exactly one stay in this
task's cohort (train+test) -- the same definition as preprocess_mimic3_sand.py.
The train/test split is the benchmark's fixed patient split (no patient overlap).

Output layout (under <out>/):
  {all,single_stay}/{train,test}/{X | X_flat+offsets, demo, <label>, subject_ids,
                                  icustay_ids, lengths}.npy
  meta.json   (channels, T, D, normalisation stats, split sizes, options)

Usage:
  python src/data/preprocess_mimic3_benchmark.py --hours 48     -> arrays_h48_first
  python src/data/preprocess_mimic3_benchmark.py --hours full   -> arrays_full_first
  python src/data/preprocess_mimic3_benchmark.py --task in-hospital-mortality
                                                                -> arrays_ihm_h48_first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

BENCH_CODE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "mimic3-benchmarks"
)
sys.path.insert(0, BENCH_CODE)
from mimic3models.preprocessing import Discretizer  # noqa: E402

STATIC_VARIS = ["Age", "Gender"]
IHM_HOURS = 48  # mimic3benchmark/scripts/create_in_hospital_mortality.py n_hours
_DISCRETIZER = None


def _discretizer() -> Discretizer:
    # settings of mimic3models/{phenotyping,in_hospital_mortality}/main.py
    global _DISCRETIZER
    if _DISCRETIZER is None:
        _DISCRETIZER = Discretizer(
            timestep=1.0, store_masks=True, impute_strategy="previous", start_time="zero"
        )
    return _DISCRETIZER


def _read_timeseries(path: str):
    # as mimic3benchmark.readers.PhenotypingReader._read_timeseries
    with open(path) as f:
        header = f.readline().strip().split(",")
        assert header[0] == "Hours"
        rows = [np.array(line.strip().split(",")) for line in f]
    return np.stack(rows), header


def discretize_stay(args):
    """(ts_path, period_length, hours | None, align) -> (L, 76) float32, unpadded."""
    ts_path, period_length, hours, align = args
    X, header = _read_timeseries(ts_path)
    data, _ = _discretizer().transform(X, header=header, end=period_length)
    if hours is not None:
        data = data[:hours] if align == "first" else data[-hours:]
    return data.astype(np.float32)


def channel_header() -> list:
    d = _discretizer()
    header = []
    for ch in d._id_to_channel:
        if d._is_categorical_channel[ch]:
            header += [f"{ch}->{v}" for v in d._possible_values[ch]]
        else:
            header.append(ch)
    return header + [f"mask->{ch}" for ch in d._id_to_channel]


def load_split(bench: str, task: str, split: str, hours, align: str, jobs: int) -> dict:
    lf = pd.read_csv(os.path.join(bench, task, split, "listfile.csv"))
    subject = lf.stay.str.split("_").str[0].astype(np.int64)
    episode = lf.stay.str.extract(r"episode(\d+)_")[0]
    if task == "phenotyping":
        label_names = list(lf.columns[2:])
        labels = lf[label_names].to_numpy(dtype=np.int8)
        period = lf.period_length
    else:
        label_names = ["in-hospital mortality"]
        labels = lf.y_true.to_numpy(dtype=np.int64)
        period = pd.Series(float(IHM_HOURS), index=lf.index)

    demo, icustay = [], []
    for sid, ep in zip(subject, episode):
        e = pd.read_csv(os.path.join(bench, "root", split, str(sid), f"episode{ep}.csv"))
        demo.append(e.loc[0, STATIC_VARIS].to_numpy(dtype=np.float32))
        icustay.append(int(e.loc[0, "Icustay"]))

    tasks = [(os.path.join(bench, task, split, s), t, hours, align) for s, t in zip(lf.stay, period)]
    with Pool(jobs) as pool:
        stays = pool.map(discretize_stay, tasks, chunksize=64)
    return {
        "stays": stays,
        "demo": np.stack(demo),
        "labels": labels,
        "subject_ids": subject.to_numpy(),
        "icustay_ids": np.array(icustay, dtype=np.int64),
        "label_names": label_names,
    }


def normalise_cohort(stays, demo, train_idx, cont):
    """z-normalise continuous value channels and demo using TRAIN statistics
    over real hours. stays: list of unpadded (L, C). Returns (stays, demo, stats)."""
    tr = np.concatenate([stays[i][:, cont] for i in train_idx])
    means = tr.mean(axis=0, dtype=np.float64)
    stds = tr.std(axis=0, dtype=np.float64)
    stds[stds == 0] = 1.0
    out = []
    for s in stays:
        s = s.copy()
        s[:, cont] = (s[:, cont] - means) / stds
        out.append(s.astype(np.float32))
    dmeans = demo[train_idx].mean(axis=0, dtype=np.float64)
    dstds = demo[train_idx].std(axis=0, dtype=np.float64)
    dstds[dstds == 0] = 1.0
    stats = {"channel_mean": means.tolist(), "channel_std": stds.tolist(),
             "demo_mean": dmeans.tolist(), "demo_std": dstds.tolist()}
    return out, ((demo - dmeans) / dstds).astype(np.float32), stats


def save_stays(out_dir: str, stays, hours) -> None:
    """Fixed window -> right-padded X.npy; full stay -> X_flat.npy + offsets.npy."""
    lengths = np.array([len(s) for s in stays], dtype=np.int64)
    np.save(os.path.join(out_dir, "lengths.npy"), lengths)
    if hours is None:
        np.save(os.path.join(out_dir, "X_flat.npy"), np.concatenate(stays))
        np.save(os.path.join(out_dir, "offsets.npy"), np.r_[0, np.cumsum(lengths)].astype(np.int64))
    else:
        X = np.zeros((len(stays), hours, stays[0].shape[1]), dtype=np.float32)
        for i, s in enumerate(stays):
            X[i, : len(s)] = s
        np.save(os.path.join(out_dir, "X.npy"), X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data_root = os.environ.get("DATA_ROOT", "datasets")
    parser.add_argument("--bench", default=os.path.join(data_root, "mimic3_benchmark"),
                        help="mimic3-benchmarks output dir (task dirs and root/)")
    parser.add_argument("--task", choices=["phenotyping", "in-hospital-mortality"], default="phenotyping")
    parser.add_argument("--hours", default="48",
                        help="window length in hours, or 'full' for the whole stay (phenotyping only)")
    parser.add_argument("--align", choices=["first", "last"], default="first",
                        help="keep the first or the last --hours hours of each stay")
    parser.add_argument("--out", default=None, help="output dir (default <bench>/arrays_...)")
    parser.add_argument("-j", "--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.task == "in-hospital-mortality":
        hours, label_file, name = IHM_HOURS, "mortality.npy", f"arrays_ihm_h{IHM_HOURS}_first"
        args.align = "first"
    else:
        hours = None if args.hours == "full" else int(args.hours)
        label_file = "phenotype.npy"
        name = f"arrays_{'full' if hours is None else f'h{hours}'}_{args.align}"
    out = args.out or os.path.join(args.bench, name)

    header = channel_header()
    cont = np.array([i for i, h in enumerate(header) if "->" not in h])
    splits = {}
    for split in ("train", "test"):
        splits[split] = load_split(args.bench, args.task, split, hours, args.align, args.jobs)
        print(f"{split}: {len(splits[split]['stays'])} stays discretized", flush=True)
    assert splits["train"]["label_names"] == splits["test"]["label_names"]

    keys = ("demo", "labels", "subject_ids", "icustay_ids")
    full = {k: np.concatenate([splits["train"][k], splits["test"][k]]) for k in keys}
    stays = splits["train"]["stays"] + splits["test"]["stays"]
    is_test = np.r_[np.zeros(len(splits["train"]["stays"]), bool), np.ones(len(splits["test"]["stays"]), bool)]
    assert not set(full["subject_ids"][is_test]) & set(full["subject_ids"][~is_test])
    lengths = np.array([len(s) for s in stays])

    uniq, counts = np.unique(full["subject_ids"], return_counts=True)
    cohorts = {
        "all": np.arange(len(is_test)),
        "single_stay": np.where(np.isin(full["subject_ids"], uniq[counts == 1]))[0],
    }
    meta_cohorts = {}
    for cname, cidx in cohorts.items():
        c = {k: v[cidx] for k, v in full.items()}
        c_test = is_test[cidx]
        c_stays, c["demo"], stats = normalise_cohort(
            [stays[i] for i in cidx], c["demo"], np.where(~c_test)[0], cont
        )
        for split, rel in (("train", ~c_test), ("test", c_test)):
            d = os.path.join(out, cname, split)
            os.makedirs(d, exist_ok=True)
            save_stays(d, [c_stays[i] for i in np.where(rel)[0]], hours)
            np.save(os.path.join(d, label_file), c["labels"][rel])
            for k in ("demo", "subject_ids", "icustay_ids"):
                np.save(os.path.join(d, f"{k}.npy"), c[k][rel])
        c_len = lengths[cidx]
        meta_cohorts[cname] = {
            "n_train": int((~c_test).sum()), "n_test": int(c_test.sum()),
            "n_patients": int(np.unique(c["subject_ids"]).size),
            "train_label_rate": np.atleast_1d(c["labels"][~c_test].mean(0)).round(4).tolist(),
            "max_length": int(c_len.max()),
            "frac_truncated": None if hours is None else float((c_len == hours).mean()),
            **stats,
        }
        print(f"[{cname}] train {(~c_test).sum()} / test {c_test.sum()} stays, max length {c_len.max()} h")

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "source": f"mimic3-benchmarks {args.task} (Harutyunyan et al. 2019)",
            "task": args.task, "label_file": label_file, "label_names": splits["train"]["label_names"],
            "T": int(lengths.max()) if hours is None else hours, "window": "full" if hours is None else hours,
            "align": args.align,
            "storage": "X_flat.npy + offsets.npy (unpadded)" if hours is None else "X.npy (right-padded with zeros)",
            "C": len(header), "channels": header, "continuous_channels": cont.tolist(),
            "D": len(STATIC_VARIS), "static_varis": STATIC_VARIS,
            "discretizer": "timestep 1h, last value per bin, impute previous (else normal value), masks",
            "split_source": "mimic3-benchmarks testset.csv (patient-level)",
            "cohorts": meta_cohorts,
        }, f, indent=2)
    print(f"Saved 4 objects (all|single_stay x train|test) -> {out}")


if __name__ == "__main__":
    main()
