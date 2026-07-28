"""
One-time converter: STraTS-preprocessed MIMIC-III -> dense SAND-format arrays.

Reads the pickle produced by STraTS' preprocess_mimic_iii_large.py
(a 5-tuple: event dataframe, outcomes dataframe, train/val/test id arrays)
and replicates the STraTS Dataset pipeline for model_type='sand'
(dataset.py in the STraTS repo):

  1. keep events in the first 24h (0 <= minute <= 1440); fix Age>200 -> 91.4
  2. split off static variables (Age, Gender) -> demo matrix (N, 2)
  3. hourly-bin the remaining variables: minute -> max(1, ceil(m/60)) - 1,
     giving T = 24 intervals
  4. build dense (N, T, V) arrays of last-observation values + observation mask
  5. delta channel: time since last observation (in bins), normalised by T
  6. mean-fill unobserved values, then z-normalise per variable
  7. X = concat([values, obs, delta], axis=-1) -> (N, T, 3V)

Deviation from STraTS: normalisation / fill statistics are computed over the
FULL dataset instead of a fixed train split, because the AL framework
(TimeSeriesDM) draws its own stratified train/val/test splits per seed.
This matches how the P12 datasets in this repo are normalised.

The STraTS train/val/test ids baked into the pickle are *not* reused; they were
drawn once with a fixed seed. Instead we emit the patient id (SUBJECT_ID) of
every stay so TimeSeriesDM can draw a *patient-grouped* split per seed — all
ICU stays of a patient land in the same split, avoiding train/test leakage
(a patient may have several stays, and STraTS reference splits at the patient
level; see preprocess_mimic_iii_large.py).

Output (written to <out>/):
  X.npy            (N, 24, 3V) float32
  demo.npy         (N, 2)      float32   z-normalised Age, Gender
  targets.npy      (N,)        int64     in-hospital mortality
  subject_ids.npy  (N,)        int64     MIMIC-III SUBJECT_ID (patient id)
  meta.json        variable names, static names, T, V

Usage:
  conda run -n real-al python src/data/preprocess_mimic3_sand.py \
      --strats-pkl /home/jing/Desktop/STraTS/data/processed/mimic_iii.pkl \
      --out $DATA_ROOT/mimic3_sand
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

STATIC_VARIS = ["Age", "Gender"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strats-pkl",
        default="/home/jing/Desktop/STraTS/data/processed/mimic_iii.pkl",
        help="Path to STraTS-preprocessed mimic_iii.pkl",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.environ.get("DATA_ROOT", "."), "mimic3_sand"),
        help="Output directory (default: $DATA_ROOT/mimic3_sand)",
    )
    args = parser.parse_args()

    print(f"Loading {args.strats_pkl} ...")
    data, oc, train_ids, val_ids, test_ids = pickle.load(open(args.strats_pkl, "rb"))

    # --- 1. first-24h filter + age fix (STraTS dataset.py, mimic_iii branch) ---
    data = data.loc[(data.minute >= 0) & (data.minute <= 24 * 60)]
    data.loc[(data.variable == "Age") & (data.value > 200), "value"] = 91.4

    # Full dataset: all ids that still have events after filtering.
    sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
    curr_ids = data.ts_id.unique()
    sup_ts_ids = sup_ts_ids[np.isin(sup_ts_ids, curr_ids)]
    ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(sup_ts_ids)}
    data = data.loc[data.ts_id.isin(sup_ts_ids)].copy()
    data["ts_ind"] = data["ts_id"].map(ts_id_to_ind)
    N = len(sup_ts_ids)

    oc = oc.loc[oc.ts_id.isin(sup_ts_ids)].copy()
    oc["ts_ind"] = oc["ts_id"].map(ts_id_to_ind)
    oc = oc.sort_values(by="ts_ind")
    y = np.array(oc["in_hospital_mortality"], dtype=np.int64)
    # Patient id per stay (aligned to ts_ind order), used for patient-grouped
    # train/val/test splitting downstream.
    subject_ids = np.array(oc["SUBJECT_ID"], dtype=np.int64)
    n_patients = int(np.unique(subject_ids).size)
    print(f"N = {N}, positive rate = {y.mean():.4f}, patients = {n_patients}")

    # --- 2. static variables -> demo (N, 2) ---
    static_ii = data.variable.isin(STATIC_VARIS)
    static_data = data.loc[static_ii]
    data = data.loc[~static_ii]
    static_var_to_ind = {v: i for i, v in enumerate(STATIC_VARIS)}
    D = len(STATIC_VARIS)
    demo = np.zeros((N, D), dtype=np.float32)
    for row in static_data.itertuples():
        demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    demo_means = demo.mean(axis=0, keepdims=True)
    demo_stds = demo.std(axis=0, keepdims=True)
    demo_stds = (demo_stds == 0) + (demo_stds > 0) * demo_stds
    demo = ((demo - demo_means) / demo_stds).astype(np.float32)

    # --- 3. hourly binning ---
    variables = data.variable.unique()
    var_to_ind = {v: i for i, v in enumerate(variables)}
    V = len(variables)
    data["minute"] = data["minute"].apply(lambda x: max(1, int(np.ceil(x / 60))) - 1)
    T = int(data.minute.max()) + 1
    print(f"V = {V} variables, T = {T} intervals")

    # --- 4. dense values / obs ---
    values = np.zeros((N, T, V), dtype=np.float32)
    obs = np.zeros((N, T, V), dtype=np.float32)
    ts_ind_arr = data["ts_ind"].to_numpy()
    tstep_arr = data["minute"].to_numpy()
    vind_arr = data["variable"].map(var_to_ind).to_numpy()
    val_arr = data["value"].to_numpy(dtype=np.float32)
    values[ts_ind_arr, tstep_arr, vind_arr] = val_arr
    obs[ts_ind_arr, tstep_arr, vind_arr] = 1.0

    # --- 5. delta: time (in bins) since last observation, normalised by T ---
    delta = np.zeros((N, T, V), dtype=np.float32)
    delta[:, 0, :] = obs[:, 0, :]
    for t in range(1, T):
        delta[:, t, :] = (1 - obs[:, t, :]) * (1 + delta[:, t - 1, :])
    delta = delta / T

    # --- 6. mean-fill + z-normalise (full-data statistics) ---
    obs_counts = obs.sum(axis=(0, 1))
    obs_counts = np.maximum(obs_counts, 1)
    means = (values * obs).sum(axis=(0, 1)) / obs_counts
    values = values * obs + (1 - obs) * means.reshape((1, 1, V))
    means = values.mean(axis=(0, 1), keepdims=True)
    stds = values.std(axis=(0, 1), keepdims=True)
    stds = (stds == 0) * 1 + (stds > 0) * stds
    values = ((values - means) / stds).astype(np.float32)

    # --- 7. save ---
    X = np.concatenate((values, obs, delta), axis=-1).astype(np.float32)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X.npy"), X)
    np.save(os.path.join(args.out, "demo.npy"), demo)
    np.save(os.path.join(args.out, "targets.npy"), y)
    np.save(os.path.join(args.out, "subject_ids.npy"), subject_ids)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "T": T,
                "V": V,
                "D": D,
                "N": N,
                "n_patients": n_patients,
                "variables": list(map(str, variables)),
                "static_varis": STATIC_VARIS,
                "channel_layout": "concat([values, obs_mask, delta], axis=-1)",
                "positive_rate": float(y.mean()),
            },
            f,
            indent=2,
        )
    print(
        f"Saved X {X.shape}, demo {demo.shape}, targets {y.shape}, "
        f"subject_ids {subject_ids.shape} ({n_patients} patients) -> {args.out}"
    )


if __name__ == "__main__":
    main()
