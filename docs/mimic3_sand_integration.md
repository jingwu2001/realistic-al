# MIMIC-III + SAND Integration (2026-07-13)

Summary of the session that added the MIMIC-III in-hospital-mortality task with
the SAND model to the AL framework, ported from Jing's STraTS fork at
`~/Desktop/STraTS`. Companion docs: [handoff.md](handoff.md) (track 3),
[codebase_guide.md](codebase_guide.md) §2.4/§3.

## The task

- **Data**: MIMIC-III ICU stays, first 24h of events, in-hospital mortality.
  N = 44,812, ~11.5% positive. 129 time-series variables hourly-binned into
  T = 24 intervals; 2 static features (Age, Gender).
- **Sample format**: `((ts, demo), label)` with `ts: (24, 387)` =
  `concat([values, obs_mask, delta], -1)` (mean-filled + z-normalised) and
  `demo: (2,)`.
- **Model**: SAND / SAnD (Song et al., AAAI 2018 "Attend and Diagnose"),
  a causal-band-masked transformer with dense interpolation over time.

## How the pieces work

End-to-end flow: STraTS pickle → offline preprocessing → `.npy` files →
`MIMIC3SandDataset` → `TimeSeriesDM` (split + AL wrapping) → dataloaders →
`BayesianSANDModel` / query strategies.

### 1. Offline preprocessing (`src/data/preprocess_mimic3_sand.py`)

Input is the pickle written by STraTS's `preprocess_mimic_iii_large.py`: a
5-tuple of an **event dataframe** (`ts_id, minute, variable, value` — one row
per observation), an **outcomes dataframe** (`ts_id, in_hospital_mortality`),
and train/val/test id arrays (**reused** as the fixed split — see step 6/7).
Steps, replicating STraTS's `model_type='sand'` branch:

1. Keep events with `0 <= minute <= 1440` (first 24h); fix `Age > 200 → 91.4`
   (MIMIC's anonymised ages).
2. Split off the static variables (Age, Gender) into a `(N, 2)` demo matrix;
   z-normalise it. The remaining 129 variables form the time series.
3. Hourly binning: `bin = max(1, ceil(minute/60)) - 1` → bins 0..23, T = 24.
4. Dense `(N, 24, 129)` arrays: `values` (last observation in a bin wins) and
   binary `obs` mask.
5. `delta` = bins since the last observation (STraTS recursion
   `δ_t = (1-o_t)(1+δ_{t-1})`, reset to 0 when observed), divided by T.
6. **Fixed split, reusing STraTS's ids.** STraTS's `train`+`val` ids become one
   `train` object, its `test` ids the fixed `test` object. Two cohorts are
   emitted: `all` (44,812 stays) and `single_stay` (26,609 — patients with
   exactly one ICU stay; the split is filtered to them, which preserves the
   64/16/20 ratio since STraTS splits by patient). Mean-fill + per-variable
   z-normalisation stats are computed over **each cohort's `train` object only**
   (train+val, not test) — the CIFAR convention, and it fixes the old full-data
   leakage. The demo matrix is z-normalised the same way.
7. `X = concat([values, obs, delta], axis=-1)` → `(n, 24, 387)` float32, saved
   per cohort/split as `X/demo/targets/subject_ids.npy` under
   `$DATA_ROOT/mimic3_sand/{all,single_stay}/{train,test}/`, plus a shared
   `meta.json` (variable names, dims, channel layout, per-cohort split sizes).

### 2. Dataset class (`src/data/mimic_dataset.py`)

`MIMIC3SandDataset(root, train=True, single_stay_only=False)` selects one of
the four pre-split objects (CIFAR-style): `train`/`test` × `all`/`single_stay`.
It loads that object's `.npy` files and serves `((ts, demo), label)` with
`ts: (24, 387)` float32, `demo: (2,)` float32, `label` int. Normalisation is
baked in at preprocess time, so there is no transform pipeline (the
`transform_train/test: basic` keys are inert, as for P12). The `targets`
attribute (numpy int64) feeds `label_balanced` and the per-seed validation
carve; `groups` (patient ids from `subject_ids.npy`) drives the patient-grouped
val carve. The test object is fixed across seeds — the point of the pre-split.

### 3. Datamodule (`src/data/timeseriesdata.py::TimeSeriesDM`)

`run_training.py::get_torchvision_dm` picks `TimeSeriesDM` because
`mimic3_sand` is in its `time_series_datasets` set; the dataset name comes from
`cfg.data.name`. Inside `_setup_datasets`:

- **Split** — mimic3_sand has its own branch (separate from P12/ECG5000). The
  **test object is fixed** (STraTS's test cohort, identical across seeds); only
  the **validation set is carved per seed** from the `train` object via
  `_grouped_val_pool_split` (patient-grouped, seeded by `trainer.seed`), the
  remainder being the AL pool. `val_split` (0.1) is the val fraction *of the
  train object*. `balanced_test_val` balances the eval sets by downsampling the
  majority class (`_balanced_indices`). This removes the old per-seed test
  resampling — the dominant cross-seed AUPRC variance source (seeds used to
  evaluate on ~90%-disjoint test cohorts). See
  [mimic3_sand_strats_comparison.md](mimic3_sand_strats_comparison.md). P12 and
  ECG5000 still use the runtime `_stratified_3way_split` / grouped 3-way split.
- **Consequence of balanced eval**: MIMIC has only ~5,150 positives, and the
  balanced eval sets consume 4,480 of them, so the remaining pool of 35,852 is
  ~1.9% positive — far more imbalanced than the natural 11.5%. (Same mechanism
  as P12 baleval, where the pool drops from ~14% to ~5% positive.) The natural
  config `mimic3_sand.yaml` keeps ~11.5% everywhere instead.
- **Wrapping** — the pool indices become an `ActiveSubset` (a `Subset` that
  still exposes `.targets`), wrapped in `ActiveLearningDataset`: a boolean
  `labelled` mask over the pool; `label()` takes **pool-relative** indices
  (they shift after every acquisition — never cache them); `.labelled_set` /
  `.pool` views drive the labelled and pool dataloaders. Because MIMIC/P12
  normalise internally, the wrapper is built **without** `pool_specifics`
  (no train→test transform swap on the pool, unlike the image datasets).
- **Eval sets** — val is a `Subset` of the `train` object (via the carved val
  indices); test is a `Subset` of the separate, smaller `test` object.
- **Loaders** — the train loader oversamples the labelled set to
  `active.min_train` (100) samples per epoch via `RandomFixedLengthSampler`
  (or `WeightedRandomSampler` if `data.balanced_sampling=true`), so "an epoch"
  is constant-size regardless of how many labels exist yet.

### 4. Training + querying (per AL iteration)

`main.py::active_loop` seeds the initial labels (`active.balanced: True` in
`mimic3_med_bal` uses `label_balanced`, i.e. equal per class; `mimic3_med`
labels randomly → ~1.9% positive under baleval), then for each of `num_iter`
iterations trains a **fresh** model and queries:

- `trainer.py::ActiveTrainingLoop` builds `BayesianModule` (Lightning) around
  the registry network `sand` → `BayesianSANDModel`. Checkpointing monitors
  **`val/auroc`** (added for `mimic3_sand`, same as p12), and
  `ImbClassMetricCallback` logs per-class/balanced metrics.
- Batches keep x as the `(ts, demo)` tuple end-to-end: Lightning moves tuple
  elements to device; `BayesianSANDModel.forward(x, k)` overrides
  `BayesianModule.forward` to encode the tuple once into the 832-dim feature
  vector, replicate it k× for MC-dropout, and return `(B, k, 2)` logits.
  `model.weighted_loss: True` weighs the NLL by inverse class frequency of the
  current labelled set (sklearn "balanced" convention).
- Uncertainty queries (bald/entropy/…) consume the `(B, k, C)` logits;
  diversity queries (vendi/gvendi/badge/kcentergreedy/bandit arms) call
  `get_features` → the same 832-dim penultimate vector. All query code moves
  tuple inputs element-wise to the device. Selected loader indices are mapped
  back through `dm.get_pool_indices` before `train_set.label()`.

## New files

| File | What it is |
|---|---|
| `src/data/preprocess_mimic3_sand.py` | One-time converter: STraTS `mimic_iii.pkl` → four pre-split objects `$DATA_ROOT/mimic3_sand/{all,single_stay}/{train,test}/{X,demo,targets,subject_ids}.npy` + shared `meta.json`. Reuses STraTS's patient-grouped split (train+val vs test); train-only normalisation per cohort. |
| `src/data/mimic_dataset.py` | `MIMIC3SandDataset(root, train, single_stay_only)` — CIFAR-style selector of the four objects; exposes `.targets` and `.groups` (patient ids) for the per-seed val carve. |
| `src/models/networks/bayesian_sand.py` | `BayesianSANDModel` + ported SAND blocks (`MultiHeadAttention`, conv `FeedForward`, `TransformerBlock`, `DenseInterpolation`). Deterministic backbone; `ConsistentMCDropout` + linear head only (repo convention). Feature dim = `hid_dim*M + hid_dim` = 832. |
| `src/models/networks/sand.py` | Registry entry `sand` (filename-keyed, per repo pattern). |
| `src/config/model/sand.yaml` | Defaults follow the STraTS MIMIC run: hid_dim 64, 4 layers, 4 heads, r 24, M 12, dropout 0.2, lr 5e-4. |
| `src/config/data/mimic3_sand.yaml`, `mimic3_sand_baleval.yaml` | Data configs (natural vs balanced 50/50 test+val). |
| `src/config/active/mimic3_low.yaml`, `mimic3_med.yaml`, `mimic3_med_bal.yaml` | AL settings mirroring the p12 counterparts (med: 100 initial / acq 100 / 10 iters). |
| `launchers/exp_mimic3_sand_basic.py` | Baseline queries: random, entropy, bald, badge, kcentergreedy × 2 data settings (baleval + natural) × 3 seeds = 30 runs (batchbald/variationratios commented out — batchbald prohibitive on the ~35k pool without `active.m`). |
| `launchers/exp_mimic3_sand_kernels.py` | vendi/gvendi/bandit × kernels {rbf, cosine, linear} × normalizations {minmax, l2, zscore, none} × 2 data settings × 3 seeds = 216 runs. All three queries read the shared `query.vendi.*` block — gvendi is a config alias of vendi with `vendi.use_grad: true` (gradient embeddings), and bandit inherits the block — so a single set of sweep keys drives all of them. Gamma pinned to 1.0 (median heuristic buggy — roadmap P3). |

## Modified files

- `src/data/timeseriesdata.py` — `mimic3_sand` branch in `TimeSeriesDM`; the
  P12 split/setup block generalised (`_p12_*` → `_ts_*`) with a per-dataset
  root (`P12data/processed_data` vs `mimic3_sand`).
- `src/run_training.py` — `mimic3_sand` added to `time_series_datasets`.
- `src/trainer.py` — checkpoint monitor `val/auroc` + `ImbClassMetricCallback`
  for `mimic3_sand` (same as p12).
- `src/models/networks/__init__.py` — imports for `bayesian_sand` / `sand`.
- `docs/handoff.md`, `docs/codebase_guide.md` — task documented.

## Decisions & deviations (and why)

1. **Train-only normalisation stats** (per cohort, computed at preprocess time
   over STraTS's fixed train+val): matches the CIFAR convention and STraTS, and
   removes the earlier full-dataset leakage. (P12 still uses its full-data
   stats; the same fix could be applied there.)
2. **Attention-dropout bug fixed in the port**: STraTS's `modeling_sand.py`
   calls `F.dropout(A, p)` without the `training` flag, so attention dropout
   stayed active at eval. The port passes `self.training` — required for the
   deterministic-backbone MC-dropout scheme.
3. **Head**: STraTS's 1-logit BCE head replaced by an `n_classes` softmax head
   to match the framework's loss/metric plumbing (`weighted_loss` via config).
4. **`query.bandit.num_classes=2`** set in the kernel launcher: the bandit's
   class-distribution context normalises by `log(num_classes)` and
   `bandit.yaml` hard-codes 10. **Latent issue: existing P12 bandit runs never
   overrode this** (used log(10) instead of log(2)) — rescales one context
   feature; consider fixing P12 launchers before cross-dataset comparisons.
5. **Two launchers, not one**: the launcher grid is a cartesian product and
   `joint_iteration` can't scope a sweep to a subset of queries — one combined
   launcher would re-run the plain methods 12× redundantly and pollute their
   run names with kernel/norm fields. Matches the CIFAR precedent
   (`exp_cifar10_basic.py` vs `exp_cifar10_basic_tune_kernel.py`).
6. **Launcher gotcha**: `joint_iteration=[]` crashes `launcher.py` (indexes
   `[0]`); use `None` when there is nothing to zip.

## Verification performed

- Unit checks: forward → `(B, k, 2)`, `get_features` → 832-dim, registry entry
  resolves, dataset loads (float32; a numpy int64 type-promotion originally
  upcast X.npy to float64/3.2GB — fixed, now 1.6GB).
- Hydra dry runs (`trainer.dry_run=true`): natural split (val/test n=4481,
  11.5% pos) and baleval (2240/2240 per class, pool 35,802).
- Two 2-iteration end-to-end AL smoke runs on GPU: `query=random` (natural) and
  `query=vendi` (baleval) — full train → query → label loop, exit 0.
- P12 regression check after the `TimeSeriesDM` refactor: identical split
  sizes/batches as before.
- Both launchers `--debug`-printed (15 / 108 commands, combos + names correct);
  one gvendi (`linear`/`none`) and one bandit (`rbf`/`minmax`) command composed
  and dry-ran end-to-end.

## How to run

```bash
source .env && conda activate real-al

# single run
python src/main.py data=mimic3_sand_baleval model=sand active=mimic3_med query=vendi optim=adam

# grids (always --debug first)
python launchers/exp_mimic3_sand_basic.py --debug     # 30 runs (baleval + natural split)
python launchers/exp_mimic3_sand_kernels.py --debug   # 216 runs; chunk with --num_start/--num_end
```

Data is regenerable from the STraTS pickle:

```bash
python src/data/preprocess_mimic3_sand.py \
  --strats-pkl ~/Desktop/STraTS/data/processed/mimic_iii.pkl --out $DATA_ROOT/mimic3_sand
```

## Open items

- Done for MIMIC: train-only normalisation via the fixed pre-split. P12 still
  uses full-data stats — apply the same fix there if needed.
- Optional: set `query.bandit.num_classes=2` in the P12 launchers.
