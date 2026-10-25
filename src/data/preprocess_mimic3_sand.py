"""
One-time converter: STraTS-preprocessed MIMIC-III -> dense SAND-format arrays,
pre-split into fixed train/test objects (CIFAR-style ``train=True/False``).

Reads the pickle produced by STraTS' preprocess_mimic_iii_large.py
(a 5-tuple: event dataframe, outcomes dataframe, train/val/test id arrays)
and replicates the STraTS Dataset pipeline for model_type='sand'
(dataset.py in the STraTS repo). It also joins the ICU length-of-stay (LOS)
label back from raw MIMIC-III (``ICUSTAYS.csv.gz``, keyed on ``ICUSTAY_ID`` =
STraTS ``ts_id``), so each object carries *both* per-stay labels:

  mortality.npy  in-hospital mortality (binary, the STraTS/SAND task)
  los.npy        total ICU length of stay in days (float; for LOS prediction)

The two labels live side by side in the same four objects; the dataset classes
(``MIMIC3ICUDataset`` / ``MIMIC3LOSDataset`` in mimic_dataset.py) each read only
the label they need. LOS is stored raw (days); binning into classes is a
LOS-dataset concern (Harutyunyan's 10 buckets). Note the STraTS cohort is
filtered to stays >= 24h, so LOS >= 1.0 day and the first bucket ([0,1)d) is
empty.

Pipeline steps (STraTS model_type='sand' branch):

  1. keep events in the first 24h (0 <= minute <= 1440); fix Age>200 -> 91.4
  2. split off static variables (Age, Gender) -> demo matrix (N, 2)
  3. hourly-bin the remaining variables: minute -> max(1, ceil(m/60)) - 1,
     giving T = 24 intervals
  4. build dense (N, T, V) arrays of last-observation values + observation mask
  5. delta channel: time since last observation (in bins), normalised by T
  6. mean-fill unobserved values, then z-normalise per variable
  7. X = concat([values, obs, delta], axis=-1) -> (N, T, 3V)

Fixed split: we reuse STraTS's own patient-grouped split (the train/val/test id
arrays baked in the pickle) so the test cohort is identical across seeds and to
the STraTS paper. STraTS's train+val become one ``train`` object (TimeSeriesDM
carves the per-seed validation set from it); STraTS's test becomes the fixed
``test`` object. This removes the per-seed test resampling that dominated
cross-seed AUPRC variance, mirroring CIFAR's fixed ``train=False`` test.

Two cohorts are emitted (selectable at load time via ``single_stay_only``):
  all         - every ICU stay (44,812)
  single_stay - only patients with exactly one ICU stay (26,609)
The single-stay cohort filters STraTS's split to single-stay patients; because
STraTS splits by patient, the 64/16/20 ratio is preserved and no patient leaks.

Normalisation / mean-fill statistics are computed over each cohort's ``train``
object only (train+val), not the test set -- the CIFAR convention (fixed
constants from the training set), which also avoids test leakage.

Each object also carries a 25-condition multi-hot phenotype label
(``phenotype.npy``, Harutyunyan et al. 2019 / mimic3-benchmarks HCUP CCS 2015
groups) joined back from raw MIMIC-III ``DIAGNOSES_ICD.csv`` via
``ts_id (ICUSTAY_ID) -> HADM_ID (from oc) -> ICD9_CODE -> CCS group``, plus the
per-row ``icustay_ids.npy`` (provenance). Read by ``MIMIC3PhenotypeDataset``.

Output layout (written under <out>/):
  all/train/{X,demo,mortality,los,phenotype,subject_ids,icustay_ids}.npy
  all/test/{X,demo,mortality,los,phenotype,subject_ids,icustay_ids}.npy
  single_stay/train/{...}.npy
  single_stay/test/{...}.npy
  meta.json   (variable names, static names, T, V, per-cohort split sizes,
               phenotype_conditions)

Window (--hours, default 24 = the STraTS setting and the output above): the
same stays, the same 129 channels (the variables seen in the first 24 h, in the
same order) and the same labels, with the first --hours hours of each stay.
STraTS stays last >= 24 h, so the 24 h tensor is never padded; with a longer
window, hours after a stay ends (ICUSTAYS LOS, or its last event if later) are
all-zero rows (values, mask and delta), which model=sand/lstm treat as padding.
Fill / normalisation statistics then use real hours only; delta stays
normalised by T, as in STraTS. Default output for --hours H != 24:
$DATA_ROOT/mimic3_sand_hH.

--hours full keeps every hour of each stay, unpadded: X_flat.npy (sum of stay
lengths, 3V) + offsets.npy (N + 1) + lengths.npy per object, padded per sample at
load time by the MIMIC3 datasets (model=lstm only; SAND needs a fixed T). delta
is in days (/24, the 24 h tensor's scale), so the mask and delta channels of each
stay's first 24 h equal mimic3_sand exactly -- checked at the end of the build.
Default output: $DATA_ROOT/mimic3_sand_full.

Usage:
  conda run -n real-al python src/data/preprocess_mimic3_sand.py \
      --strats-pkl /home/jing/Desktop/STraTS/data/processed/mimic_iii.pkl \
      --icustays /mnt/physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz \
      --out $DATA_ROOT/mimic3_sand
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml

STATIC_VARIS = ["Age", "Gender"]

# Default location of the HCUP CCS 2015 definitions shipped with the vendored
# Harutyunyan benchmark (third_party/mimic3-benchmarks). 25 groups carry
# use_in_benchmark: True -- the phenotyping label set.
DEFAULT_PHENO_DEFS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "third_party",
    "mimic3-benchmarks",
    "mimic3benchmark",
    "resources",
    "hcup_ccs_2015_definitions.yaml",
)

# Harutyunyan et al. MIMIC-III benchmark length-of-stay buckets (days), kept
# here purely as the canonical reference recorded in meta.json. los.npy stores
# raw LOS in days; the ACTUAL binning happens in MIMIC3LOSDataset, which drops
# the structurally-empty [0,1)d bin (cohort >= 24h) and uses 9 classes.
LOS_BIN_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 14]


def normalise_cohort(values_raw, obs, demo_raw, train_idx, real=None):
    """Mean-fill + z-normalise ``values`` and z-normalise ``demo`` using TRAIN
    statistics only, applied to the whole cohort. Returns (values, demo).

    ``real`` (N, T) bool marks hours inside the stay (None = all, the 24 h
    tensor); padded hours are left out of the statistics and set to 0."""
    tr = train_idx
    V = values_raw.shape[-1]
    # mean-fill: per-variable observed mean over train stays
    obs_counts = np.maximum(obs[tr].sum(axis=(0, 1)), 1)
    fill_means = (values_raw[tr] * obs[tr]).sum(axis=(0, 1)) / obs_counts
    values = values_raw * obs + (1 - obs) * fill_means.reshape((1, 1, V))
    # z-normalise values (train mean/std, after fill)
    if real is None:
        means = values[tr].mean(axis=(0, 1), keepdims=True)
        stds = values[tr].std(axis=(0, 1), keepdims=True)
    else:
        real_tr = values[tr][real[tr]]                 # (n real train hours, V)
        means = real_tr.mean(axis=0, dtype=np.float64).reshape((1, 1, V))
        stds = real_tr.std(axis=0, dtype=np.float64).reshape((1, 1, V))
        del real_tr
    stds = (stds == 0) * 1 + (stds > 0) * stds
    values = ((values - means) / stds).astype(np.float32)
    if real is not None:
        values[~real] = 0.0
    # z-normalise demo (train mean/std)
    dmeans = demo_raw[tr].mean(axis=0, keepdims=True)
    dstds = demo_raw[tr].std(axis=0, keepdims=True)
    dstds = (dstds == 0) + (dstds > 0) * dstds
    demo = ((demo_raw - dmeans) / dstds).astype(np.float32)
    return values, demo


def normalise_flat(values_raw, obs, demo_raw, train_rows, train_idx):
    """normalise_cohort for unpadded rows (every row is a real hour): mean-fill
    + z-normalise values on TRAIN rows, z-normalise demo on TRAIN stays."""
    tr = train_rows
    obs_counts = np.maximum(obs[tr].sum(axis=0), 1)
    fill_means = (values_raw[tr] * obs[tr]).sum(axis=0) / obs_counts
    values = values_raw * obs + (1 - obs) * fill_means[None, :]
    means = values[tr].mean(axis=0, dtype=np.float64)
    stds = values[tr].std(axis=0, dtype=np.float64)
    stds = (stds == 0) * 1 + (stds > 0) * stds
    values = ((values - means) / stds).astype(np.float32)
    dmeans = demo_raw[train_idx].mean(axis=0, keepdims=True)
    dstds = demo_raw[train_idx].std(axis=0, keepdims=True)
    dstds = (dstds == 0) + (dstds > 0) * dstds
    return values, ((demo_raw - dmeans) / dstds).astype(np.float32)


def save_full_stay(args, values_raw, obs, delta, offsets, n_real, demo_raw, y, los,
                   subject_ids, phenotype, icustay_ids, is_test, variables, condition_names,
                   N, n_patients, V, D):
    """Per cohort: normalise on train rows, save train/test as whole-stay objects."""
    uniq_sid, counts = np.unique(subject_ids, return_counts=True)
    cohorts = {
        "all": np.arange(N),
        "single_stay": np.where(np.isin(subject_ids, uniq_sid[counts == 1]))[0],
    }
    meta_splits = {}
    for cohort_name, cidx in cohorts.items():
        rows, lens = stay_rows(offsets, cidx)
        c_offsets = np.r_[0, np.cumsum(lens)]
        c_is_test = is_test[cidx]
        c_values, c_demo = normalise_flat(
            values_raw[rows], obs[rows], demo_raw[cidx],
            np.repeat(~c_is_test, lens), np.where(~c_is_test)[0],
        )
        c_obs, c_delta = obs[rows], delta[rows]
        for split_name, rel in (("train", np.where(~c_is_test)[0]), ("test", np.where(c_is_test)[0])):
            r, l = stay_rows(c_offsets, rel)
            g = cidx[rel]
            save_ragged_object(
                os.path.join(args.out, cohort_name, split_name),
                (c_values[r], c_obs[r], c_delta[r]), l, c_demo[rel], y[g], los[g],
                subject_ids[g], phenotype[g], icustay_ids[g],
            )
        del c_values, c_obs, c_delta
        meta_splits[cohort_name] = {
            "n_train": int((~c_is_test).sum()), "n_test": int(c_is_test.sum()),
            "n_patients": int(np.unique(subject_ids[cidx]).size),
            "hours_total": int(lens.sum()), "max_length": int(lens.max()),
        }
        print(f"[{cohort_name}] train {(~c_is_test).sum()} / test {c_is_test.sum()} stays, "
              f"{lens.sum()} hours, longest {lens.max()} h")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "T": int(n_real.max()), "V": V, "D": D, "N": N, "n_patients": n_patients,
            "window_hours": "full",
            "storage": "X_flat.npy + offsets.npy (unpadded whole stays)",
            "variables": list(map(str, variables)), "static_varis": STATIC_VARIS,
            "channel_layout": "concat([values, obs_mask, delta_days], axis=-1)",
            "los_unit": "days", "phenotype_conditions": condition_names,
            "split_source": "strats (train+val vs test)", "splits": meta_splits,
        }, f, indent=2)
    print(f"Saved 4 objects (all|single_stay x train|test) -> {args.out}")
    ref = os.path.join(os.environ.get("DATA_ROOT", "."), "mimic3_sand")
    if os.path.isdir(ref):
        check_first_24h(args.out, ref)


def ragged_hourly(ts_ind, hour, var_ind, value, n_real, V):
    """Whole-stay hourly arrays without padding: stay i occupies rows
    offsets[i]:offsets[i + 1]. Returns (values_raw, obs, delta_hours, offsets),
    delta in hours since the last observation (STraTS recursion, reset per stay)."""
    offsets = np.r_[0, np.cumsum(n_real)].astype(np.int64)
    rows = offsets[ts_ind] + hour
    values_raw = np.zeros((offsets[-1], V), dtype=np.float32)
    obs = np.zeros((offsets[-1], V), dtype=np.float32)
    values_raw[rows, var_ind] = value  # last in hour always comes last upstream
    obs[rows, var_ind] = 1.0
    delta = np.zeros_like(obs)
    starts = offsets[:-1]
    delta[starts] = obs[starts]
    for t in range(1, int(n_real.max())):
        idx = starts[n_real > t] + t
        delta[idx] = (1 - obs[idx]) * (1 + delta[idx - 1])
    return values_raw, obs, delta, offsets


def stay_rows(offsets, idx):
    """Row indices of stays ``idx`` in a ragged array, and their lengths."""
    lens = offsets[idx + 1] - offsets[idx]
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(offsets[idx], lens) + within, lens


def save_ragged_object(out_dir, X_parts, lens, demo, mortality, los, sid, phenotype, icustay_ids):
    """Whole-stay object: X_flat = concat(X_parts, axis=-1) + offsets + lengths."""
    os.makedirs(out_dir, exist_ok=True)
    X = np.lib.format.open_memmap(
        os.path.join(out_dir, "X_flat.npy"), mode="w+", dtype=np.float32,
        shape=(X_parts[0].shape[0], sum(p.shape[-1] for p in X_parts)),
    )
    c = 0
    for part in X_parts:
        X[:, c:c + part.shape[-1]] = part
        c += part.shape[-1]
    X.flush()
    del X
    np.save(os.path.join(out_dir, "offsets.npy"), np.r_[0, np.cumsum(lens)].astype(np.int64))
    np.save(os.path.join(out_dir, "lengths.npy"), lens.astype(np.int64))
    np.save(os.path.join(out_dir, "demo.npy"), demo.astype(np.float32))
    np.save(os.path.join(out_dir, "mortality.npy"), mortality.astype(np.int64))
    np.save(os.path.join(out_dir, "los.npy"), los.astype(np.float32))
    np.save(os.path.join(out_dir, "subject_ids.npy"), sid.astype(np.int64))
    np.save(os.path.join(out_dir, "phenotype.npy"), phenotype.astype(np.int8))
    np.save(os.path.join(out_dir, "icustay_ids.npy"), icustay_ids.astype(np.int64))


def check_first_24h(out, ref):
    """Mask + delta channels of each stay's first 24 h must equal the 24 h tensor."""
    for cohort in ("all", "single_stay"):
        for split in ("train", "test"):
            d, r = os.path.join(out, cohort, split), os.path.join(ref, cohort, split)
            X24 = np.load(os.path.join(r, "X.npy"), mmap_mode="r")
            flat = np.load(os.path.join(d, "X_flat.npy"), mmap_mode="r")
            offsets = np.load(os.path.join(d, "offsets.npy"))
            V = X24.shape[-1] // 3
            assert len(offsets) - 1 == len(X24), (cohort, split)
            rows, _ = stay_rows(offsets, np.arange(len(X24)))
            first = rows[(rows - np.repeat(offsets[:-1], np.diff(offsets))) < 24]
            got = flat[first][:, V:].reshape(len(X24), 24, 2 * V)
            assert np.array_equal(got, X24[:, :, V:]), f"first 24 h differ: {cohort}/{split}"
    print(f"check: mask + delta of the first 24 h equal {ref}")


def _add_hcup_ccs_2015_groups(diagnoses, definitions):
    """Map ``ICD9_CODE`` -> HCUP CCS 2015 group name + benchmark flag.

    Replicated verbatim from the vendored Harutyunyan benchmark
    (``third_party/mimic3-benchmarks/mimic3benchmark/preprocessing.py``
    ``add_hcup_ccs_2015_groups``) so we don't have to import that package
    (its ``__init__`` eagerly pulls the whole benchmark chain). ``definitions``
    is the parsed ``hcup_ccs_2015_definitions.yaml`` dict.
    """
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]["codes"]:
            def_map[code] = (dx, definitions[dx]["use_in_benchmark"])
    diagnoses["HCUP_CCS_2015"] = diagnoses.ICD9_CODE.apply(
        lambda c: def_map[c][0] if c in def_map else None
    )
    diagnoses["USE_IN_BENCHMARK"] = diagnoses.ICD9_CODE.apply(
        lambda c: int(def_map[c][1]) if c in def_map else None
    )
    return diagnoses


def build_phenotype_matrix(oc, diagnoses_csv, definitions_path, all_ids):
    """Build a ``(N, 25)`` multi-hot phenotype label matrix in ``all_ids`` order.

    Label definition follows Harutyunyan et al. 2019 (mimic3-benchmarks): the 25
    HCUP CCS 2015 groups with ``use_in_benchmark: True``. Join path uses keys
    already in hand at preprocess time::

        ts_id (== ICUSTAY_ID) --HADM_ID (from oc)--> ICD9_CODE (DIAGNOSES_ICD)
            --> HCUP CCS 2015 group --> 25 benchmark conditions

    All ICU stays in an admission inherit that admission's diagnosis codes
    (matches the benchmark's ``filter_diagnoses_on_stays``). Stays whose
    admission has none of the 25 conditions get a valid all-zero row.

    Returns ``(phenotype (N, 25) int8, condition_names list[str])`` where
    ``condition_names`` is the canonical alphabetical order (matches the
    benchmark listfile header) and rows are in ``all_ids`` order (1:1 with X).
    """
    definitions = yaml.safe_load(open(definitions_path))
    condition_names = sorted(
        [k for k, v in definitions.items() if v.get("use_in_benchmark")]
    )
    assert (
        len(condition_names) == 25
    ), f"expected 25 benchmark conditions, got {len(condition_names)}"

    # ICD9_CODE MUST be read as str: leading zeros are significant and match the
    # YAML code strings (e.g. "0031"). DIAGNOSES_ICD ships here uncompressed.
    dx = pd.read_csv(
        diagnoses_csv, usecols=["HADM_ID", "ICD9_CODE"], dtype={"ICD9_CODE": str}
    ).dropna(subset=["HADM_ID", "ICD9_CODE"])
    dx["HADM_ID"] = dx["HADM_ID"].astype(np.int64)

    # ts_id (ICUSTAY_ID) <-> HADM_ID bridge from the STraTS outcomes frame.
    bridge = oc[["ts_id", "HADM_ID"]].drop_duplicates().copy()
    bridge["HADM_ID"] = bridge["HADM_ID"].astype(np.int64)
    dx = dx.merge(bridge, on="HADM_ID", how="inner").rename(
        columns={"ts_id": "ICUSTAY_ID"}
    )

    dx = _add_hcup_ccs_2015_groups(dx, definitions)
    dx = dx.loc[dx.USE_IN_BENCHMARK > 0, ["ICUSTAY_ID", "HCUP_CCS_2015"]].drop_duplicates()
    dx["VALUE"] = 1
    mat = dx.pivot(index="ICUSTAY_ID", columns="HCUP_CCS_2015", values="VALUE")
    # Guarantee all 25 columns (canonical order) and all N rows (all_ids order).
    mat = mat.reindex(columns=condition_names).reindex(index=all_ids)
    phenotype = mat.fillna(0).astype(np.int8).to_numpy()
    return phenotype, condition_names


def save_object(out_dir, X_parts, demo, mortality, los, sid, phenotype, icustay_ids):
    """X = concat(X_parts, axis=-1), written through a memmap (no full copy in RAM)."""
    os.makedirs(out_dir, exist_ok=True)
    n, T = X_parts[0].shape[:2]
    X = np.lib.format.open_memmap(
        os.path.join(out_dir, "X.npy"), mode="w+", dtype=np.float32,
        shape=(n, T, sum(p.shape[-1] for p in X_parts)),
    )
    c = 0
    for part in X_parts:
        X[:, :, c:c + part.shape[-1]] = part
        c += part.shape[-1]
    X.flush()
    del X
    np.save(os.path.join(out_dir, "demo.npy"), demo.astype(np.float32))
    np.save(os.path.join(out_dir, "mortality.npy"), mortality.astype(np.int64))
    np.save(os.path.join(out_dir, "los.npy"), los.astype(np.float32))
    np.save(os.path.join(out_dir, "subject_ids.npy"), sid.astype(np.int64))
    # 25-condition multi-hot phenotype labels + per-row ICUSTAY_ID (provenance).
    np.save(os.path.join(out_dir, "phenotype.npy"), phenotype.astype(np.int8))
    np.save(os.path.join(out_dir, "icustay_ids.npy"), icustay_ids.astype(np.int64))
    # Remove the pre-LOS filename if this dir was written by an older run, so a
    # stale mortality-only label can't be mistaken for the current one.
    stale = os.path.join(out_dir, "targets.npy")
    if os.path.exists(stale):
        os.remove(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strats-pkl",
        default="/home/jing/Desktop/STraTS/data/processed/mimic_iii.pkl",
        help="Path to STraTS-preprocessed mimic_iii.pkl",
    )
    parser.add_argument(
        "--icustays",
        default="/mnt/physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz",
        help="Path to raw MIMIC-III ICUSTAYS.csv.gz (source of the LOS label)",
    )
    parser.add_argument(
        "--diagnoses",
        default="/mnt/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv",
        help="Path to raw MIMIC-III DIAGNOSES_ICD.csv (source of phenotype "
        "labels; note it ships uncompressed .csv here, not .csv.gz)",
    )
    parser.add_argument(
        "--phenotype-definitions",
        default=DEFAULT_PHENO_DEFS,
        help="HCUP CCS 2015 definitions YAML (25 benchmark phenotype conditions)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: $DATA_ROOT/mimic3_sand, _hH for --hours H != 24)",
    )
    parser.add_argument(
        "--hours", default="24",
        help="window: the first HOURS hours of each stay (24 = the STraTS setting), "
        "or 'full' for the whole stay (unpadded)",
    )
    args = parser.parse_args()
    H = None if args.hours == "full" else int(args.hours)
    if args.out is None:
        name = "mimic3_sand" if H == 24 else f"mimic3_sand_{'full' if H is None else f'h{H}'}"
        args.out = os.path.join(os.environ.get("DATA_ROOT", "."), name)

    print(f"Loading {args.strats_pkl} ...")
    data, oc, train_ids, val_ids, test_ids = pickle.load(open(args.strats_pkl, "rb"))

    # --- 1. first-H-hours filter + age fix (STraTS dataset.py, mimic_iii branch: H = 24) ---
    data = data.loc[(data.minute >= 0) & ((data.minute <= H * 60) if H else True)]
    data.loc[(data.variable == "Age") & (data.value > 200), "value"] = 91.4

    # STraTS train+val become the train object; STraTS test is the fixed test.
    all_ids_from_strats_split = np.concatenate((train_ids, val_ids, test_ids))
    assert np.isin(all_ids_from_strats_split, data.ts_id.unique()).all()
    ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(all_ids_from_strats_split)}
    data = data.loc[data.ts_id.isin(all_ids_from_strats_split)].copy()

    data["ts_ind"] = data["ts_id"].map(ts_id_to_ind)
    N = len(all_ids_from_strats_split) # Number of stays
    is_test = np.isin(all_ids_from_strats_split, test_ids)  # per-stay, in ts_ind order

    oc = oc.loc[oc.ts_id.isin(all_ids_from_strats_split)].copy()
    oc["ts_ind"] = oc["ts_id"].map(ts_id_to_ind)
    oc = oc.sort_values(by="ts_ind")
    y = np.array(oc["in_hospital_mortality"], dtype=np.int64)
    subject_ids = np.array(oc["SUBJECT_ID"], dtype=np.int64)
    n_patients = int(np.unique(subject_ids).size)
    print(f"N = {N}, positive rate = {y.mean():.4f}, patients = {n_patients}")

    # --- LOS label: total ICU length of stay (days) from raw MIMIC-III. ---
    # ts_id == ICUSTAY_ID, so map each stay to ICUSTAYS.LOS in ts_ind order
    # (all_ids_from_strats_split), the same ordering as y / subject_ids above.
    print(f"Joining LOS from {args.icustays} ...")
    icu = pd.read_csv(args.icustays, usecols=["ICUSTAY_ID", "LOS"])
    los_map = dict(zip(icu.ICUSTAY_ID.to_numpy(), icu.LOS.to_numpy()))
    missing = [int(t) for t in all_ids_from_strats_split if t not in los_map]
    assert not missing, f"{len(missing)} ts_ids missing from ICUSTAYS LOS map"
    los = np.array([los_map[t] for t in all_ids_from_strats_split], dtype=np.float32)
    assert not np.isnan(los).any(), "NaN LOS values (missing OUTTIME?)"
    print(
        f"LOS days: min {los.min():.2f}, median {np.median(los):.2f}, "
        f"max {los.max():.2f}"
    )

    # --- Phenotype label: 25-condition multi-hot (Harutyunyan/mimic3-benchmarks). ---
    # oc carries ts_id (== ICUSTAY_ID) and HADM_ID; diagnoses are keyed on
    # HADM_ID. Built in all_ids_from_strats_split order, so rows align 1:1 with
    # y / subject_ids / los / X below.
    print(f"Building phenotype labels from {args.diagnoses} ...")
    phenotype, condition_names = build_phenotype_matrix(
        oc, args.diagnoses, args.phenotype_definitions, all_ids_from_strats_split
    )
    icustay_ids = all_ids_from_strats_split.astype(np.int64)
    assert phenotype.shape == (N, 25)
    print(
        f"Phenotype {phenotype.shape}: mean {phenotype.sum(1).mean():.2f} "
        f"conditions/stay, {(phenotype.sum(1) == 0).mean():.3f} all-negative. "
        f"Per-condition prevalence:"
    )
    for name, rate in zip(condition_names, phenotype.mean(0)):
        print(f"    {rate:5.3f}  {name}")

    # --- 2. static variables -> demo (N, 2), RAW (z-norm applied per cohort) ---
    static_mask = data.variable.isin(STATIC_VARIS)
    static_data = data.loc[static_mask]
    data = data.loc[~static_mask]
    static_var_to_ind = {v: i for i, v in enumerate(STATIC_VARIS)}
    D = len(STATIC_VARIS)
    demo_raw = np.zeros((N, D), dtype=np.float32)
    # Turn dataframe into an array
    for row in static_data.itertuples():
        demo_raw[row.ts_ind, static_var_to_ind[row.variable]] = row.value

    # --- 3. hourly binning ---
    # channels = the variables seen in the first 24 h, in their 24 h order, so
    # every window has the same 129 channels (a longer window must not add or
    # reorder any)
    variables = data.loc[data.minute <= 24 * 60].variable.unique()
    data = data.loc[data.variable.isin(variables)]
    var_to_ind = {v: i for i, v in enumerate(variables)}
    V = len(variables)
    data["minute"] = data["minute"].apply(lambda x: max(1, int(np.ceil(x / 60))) - 1)
    if H is None:
        # --- 4+5 (whole stay): unpadded rows; stay = LOS hours, or to its last event ---
        last_hour = data.groupby("ts_ind").minute.max().reindex(range(N), fill_value=-1).to_numpy()
        n_real = np.maximum(np.ceil(los * 24 - 1e-3).astype(np.int64), last_hour + 1)
        values_raw, obs, delta, offsets = ragged_hourly(
            data["ts_ind"].to_numpy(), data["minute"].to_numpy(),
            data["variable"].map(var_to_ind).to_numpy(), data["value"].to_numpy(dtype=np.float32),
            n_real, V,
        )
        del data
        delta /= 24  # hours -> days: the 24 h tensor's scale (no fixed T here)
        T = int(n_real.max())
        print(f"V = {V} variables, whole stays: {offsets[-1]} hours, longest {T} h")
        save_full_stay(args, values_raw, obs, delta, offsets, n_real, demo_raw, y, los,
                       subject_ids, phenotype, icustay_ids, is_test, variables, condition_names,
                       N, n_patients, V, D)
        return
    T = H
    assert int(data.minute.max()) + 1 <= T
    print(f"V = {V} variables, T = {T} intervals")

    # --- 4. dense values / obs (RAW; fill+norm applied per cohort) ---
    values_raw = np.zeros((N, T, V), dtype=np.float32)
    obs = np.zeros((N, T, V), dtype=np.float32)
    values_raw[data["ts_ind"], data["minute"], data["variable"].map(var_to_ind)] = \
        data["value"].to_numpy(dtype=np.float32) # from upstream last in hour always comes last
    obs[data["ts_ind"], data["minute"], data["variable"].map(var_to_ind)] = 1.0

    # --- 5. delta: time (in bins) since last observation, normalised by T ---
    delta = np.zeros((N, T, V), dtype=np.float32)
    delta[:, 0, :] = obs[:, 0, :]
    for t in range(1, T):
        delta[:, t, :] = (1 - obs[:, t, :]) * (1 + delta[:, t - 1, :])
    delta = delta / T

    # --- padding: hours after the stay ends (LOS, or its last event if later) ---
    last_obs = np.where(obs.any(axis=2), np.arange(T)[None, :], -1).max(axis=1)
    n_real = np.minimum(T, np.maximum(np.ceil(los * 24 - 1e-3).astype(np.int64), last_obs + 1))
    if (n_real < T).any():
        real = np.arange(T)[None, :] < n_real[:, None]      # (N, T)
        delta[~real] = 0.0
        print(f"padding: {(n_real < T).mean():.3f} of stays shorter than {T} h")
    else:
        real = None  # the 24 h tensor: every stay lasts >= 24 h
    # --- 6+7. per cohort: normalise on train, noem, save train/test ---
    uniq_sid, counts = np.unique(subject_ids, return_counts=True)
    single_pat = uniq_sid[counts == 1]
    cohorts = {
        "all": np.arange(N),
        "single_stay": np.where(np.isin(subject_ids, single_pat))[0],
    }

    meta_splits = {}
    for cohort_name, cidx in cohorts.items():
        c_labels, c_groups, c_los = y[cidx], subject_ids[cidx], los[cidx]
        c_pheno, c_icustay = phenotype[cidx], icustay_ids[cidx]
        c_is_test = is_test[cidx]
        rel_train = np.where(~c_is_test)[0]  # STraTS train+val
        rel_test = np.where(c_is_test)[0]    # STraTS test

        c_values, c_demo = normalise_cohort(
            values_raw[cidx], obs[cidx], demo_raw[cidx], rel_train,
            real=None if real is None else real[cidx],
        )
        c_obs, c_delta = obs[cidx], delta[cidx]

        for split_name, rel in (("train", rel_train), ("test", rel_test)):
            save_object(
                os.path.join(args.out, cohort_name, split_name),
                (c_values[rel], c_obs[rel], c_delta[rel]), c_demo[rel], c_labels[rel],
                c_los[rel], c_groups[rel], c_pheno[rel], c_icustay[rel],
            )
        del c_values, c_obs, c_delta
        meta_splits[cohort_name] = {
            "n_train": int(len(rel_train)),
            "n_test": int(len(rel_test)),
            "n_patients": int(np.unique(c_groups).size),
            "train_pos_rate": float(c_labels[rel_train].mean()),
            "test_pos_rate": float(c_labels[rel_test].mean()),
            "train_los_median": float(np.median(c_los[rel_train])),
            "test_los_median": float(np.median(c_los[rel_test])),
        }
        print(
            f"[{cohort_name}] train {len(rel_train)} (pos {c_labels[rel_train].mean():.4f}) "
            f"/ test {len(rel_test)} (pos {c_labels[rel_test].mean():.4f})"
        )

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "T": T, "V": V, "D": D, "N": N, "n_patients": n_patients,
                "window_hours": H,
                "frac_padded": float((n_real < T).mean()),
                "variables": list(map(str, variables)),
                "static_varis": STATIC_VARIS,
                "channel_layout": "concat([values, obs_mask, delta], axis=-1)",
                "positive_rate": float(y.mean()),
                "los_unit": "days",
                "los_bin_edges": LOS_BIN_EDGES,
                "los_num_classes": len(LOS_BIN_EDGES) + 1,
                "phenotype_conditions": condition_names,
                "n_conditions": len(condition_names),
                "phenotype_source": "harutyunyan hcup_ccs_2015 (25 use_in_benchmark groups)",
                "split_source": "strats (train+val vs test)",
                "splits": meta_splits,
            },
            f,
            indent=2,
        )
    print(f"Saved 4 objects (all|single_stay x train|test) -> {args.out}")


if __name__ == "__main__":
    main()
