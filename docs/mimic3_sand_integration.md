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
and train/val/test id arrays (used only to enumerate stays; the AL framework
re-splits later). Steps, replicating STraTS's `model_type='sand'` branch:

1. Keep events with `0 <= minute <= 1440` (first 24h); fix `Age > 200 → 91.4`
   (MIMIC's anonymised ages).
2. Split off the static variables (Age, Gender) into a `(N, 2)` demo matrix;
   z-normalise it. The remaining 129 variables form the time series.
3. Hourly binning: `bin = max(1, ceil(minute/60)) - 1` → bins 0..23, T = 24.
4. Dense `(N, 24, 129)` arrays: `values` (last observation in a bin wins) and
   binary `obs` mask.
5. `delta` = bins since the last observation (STraTS recursion
   `δ_t = (1-o_t)(1+δ_{t-1})`, reset to 0 when observed), divided by T.
6. Mean-fill unobserved values with the per-variable mean of observed entries,
   then z-normalise each variable over all `(N, T)` cells. Stats come from the
   **full dataset** — see deviation 1 below.
7. `X = concat([values, obs, delta], axis=-1)` → `(44812, 24, 387)` float32,
   saved with `demo.npy`, `targets.npy`, and `meta.json` (variable names, dims,
   channel layout) to `$DATA_ROOT/mimic3_sand/`.

### 2. Dataset class (`src/data/mimic_dataset.py`)

`MIMIC3SandDataset(root)` loads the three `.npy` files into RAM (~1.7 GB) and
serves `((ts, demo), label)` with `ts: (24, 387)` float32, `demo: (2,)`
float32, `label` int. Normalisation is baked in at preprocess time, so there is
no transform pipeline (the `transform_train/test: basic` keys in the data
config are inert for this dataset, as for P12). The `targets` attribute (numpy
int64) is what the datamodule's stratified splitter and `label_balanced` read.

### 3. Datamodule (`src/data/timeseriesdata.py::TimeSeriesDM`)

`run_training.py::get_torchvision_dm` picks `TimeSeriesDM` because
`mimic3_sand` is in its `time_series_datasets` set; the dataset name comes from
`cfg.data.name`. Inside `_setup_datasets`:

- **Split** — `_stratified_3way_split`, seeded by `trainer.seed` (a different
  split every seed): test = 10% of N, val = `val_split` fraction (0.1), both
  stratified per class; the remainder is the AL pool. With
  `balanced_test_val: true` (the `_baleval` data config) test and val instead
  get equal per-class counts: 2,240 + 2,240 per class.
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
- **Eval sets** — a second full dataset instance is built and val/test are
  `Subset`s of it (so the process holds two copies of X, ~3.4 GB total).
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
| `src/data/preprocess_mimic3_sand.py` | One-time converter: STraTS `mimic_iii.pkl` → `$DATA_ROOT/mimic3_sand/{X,demo,targets}.npy` + `meta.json`. Replicates STraTS's SAND pipeline (24h filter, Age>200 fix, hourly binning, delta channel, mean-fill, z-norm). |
| `src/data/mimic_dataset.py` | `MIMIC3SandDataset` — loads the `.npy` files, exposes `.targets` for stratified splitting (P12 pattern). |
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

1. **Full-data normalisation stats** (not STraTS's train-split stats): the AL
   framework re-splits stratified per seed, so no fixed train split exists at
   preprocess time. This matches the existing P12 convention but *not* CIFAR
   (fixed published train-set constants) or STraTS. Mild feature-scale leakage;
   the principled fix would be pool-only stats computed inside the datamodule
   after the split, applied to P12 too.
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

- Optional: pool-only normalisation inside the datamodule (fixes the mild
  leakage for both MIMIC and P12).
- Optional: set `query.bandit.num_classes=2` in the P12 launchers.
