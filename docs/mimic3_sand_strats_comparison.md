# MIMIC-III SAND: STraTS comparison & fixed-split experiment

Purpose: understand why the MIMIC-III in-hospital-mortality active-learning runs
plateau, and how they compare to the STraTS SAND full-data reference. This doc
records the cohort sizes, the reference numbers, the variance analysis, and the
planned move to a CIFAR-style fixed test split. Companion docs:
[mimic3_sand_integration.md](mimic3_sand_integration.md) (data pipeline),
[mimic.md](mimic.md) (upstream STraTS preprocessing).

## Cohort sizes

Data: STraTS `mimic_iii.pkl` → `preprocess_mimic3_sand.py` → `$DATA_ROOT/mimic3_sand/`.
Task: in-hospital mortality from the first 24h of an adult ICU stay.

| Cohort | ICU stays | Patients (SUBJECT_ID) | Admissions (HADM_ID) | Mortality |
|---|---|---|---|---|
| **All patients** | 44,812 | 33,395 | 42,095 | 11.49% (5,150 pos) |
| **Single-stay only** | 26,609 | 26,609 | 26,609 | 12.18% (3,240 pos) |

- Hierarchy: 33,395 patients → 42,095 admissions → 44,812 ICU stays.
- Stays per patient: mean 1.34, max 36; 6,786 patients (20%) have >1 stay.
- Single-stay-only (config `data.single_stay_only=true`) keeps 59.4% of stays and
  is slightly sicker (12.18% vs 11.49%). It is a *runtime* cohort filter; `dedup`
  (one label-preserving stay per patient) is a different ~33k-stay cohort.

## STraTS SAND reference (the numbers to beat)

From `~/Desktop/STraTS/sand.md` Part 5 (test split, mean over 10 folds; STraTS uses
a **fixed** 64/16/20 patient split — train 28,790 / val 7,144 / test 8,878 — and
only varies the training window + init). NB: STraTS SAND has the attention bug that
this repo's port fixes, so it is an approximate reference.

| train_frac | n train | AUROC | AUPRC |
|---|---|---|---|
| 0.1 | 2,879 | 0.8432 | 0.4594 |
| 0.2 | 5,758 | 0.8611 | 0.4988 |
| 0.3 | 8,637 | 0.8660 | 0.5124 |
| 0.4 | 11,516 | 0.8694 | 0.5181 |
| 0.5 | 14,395 | 0.8734 | 0.5274 |

## Findings so far (2026-09-06/07)

- **The AL runs plateau early.** With `mimic3_med` (start 100, step 100, 10 rounds),
  test AUPRC rises fast then flattens ~0.43 by 1,000 labels; AUROC ~0.81. Far below
  the STraTS full-data ceiling (0.53 AUPRC / 0.87 AUROC).
- **The plateau is mostly a sample-size effect.** A random-acquisition sweep
  (`sweep_exp_mimic3_sand_basic.py`, mimicking the SAND recipe: batch 16, grad-clip
  0.3, AdamW-equiv wd=0, constant lr 5e-4, weighted loss) climbing to 10,000 labels
  reaches ~0.50 mean AUPRC (best seed 0.52–0.53, matching the paper), still rising at
  10k — so more labels keep helping; the curve tracks STraTS closely at matched budget.
- **Large cross-seed std, especially AUPRC.** At 10k labels: AUPRC 0.476 vs 0.522
  across two seeds (spread ~0.046); AUROC spread only ~0.024. Causes:
  1. AUPRC is intrinsically high-variance under imbalance (~515 test positives; var ~1/n_pos).
  2. **Each seed re-draws the test/val split** — `TimeSeriesDM` seeds the grouped split
     with `trainer.seed`, so the 3 seeds evaluate on ~90%-disjoint test cohorts
     (pairwise Jaccard ≈ 0.05). STraTS instead fixed its test set. This is the main
     framework-specific variance driver.
  3. Patient-grouped splits reduce effective N; small per-seed val set adds
     checkpoint-selection noise.
- **CIFAR-10 does not have this problem**: torchvision ships a fixed 10k test
  (`train=False`), so only val + labeled subset + init vary per seed. Any
  `TimeSeriesDM` dataset (MIMIC, P12, ecg5000) inherits the re-split issue; the
  torchvision image datasets do not.

## Implemented: CIFAR-style fixed test split (2026-09-07)

MIMIC now behaves like CIFAR — a fixed test cohort across all seeds, with only
val + labeled pool + init varying:

1. `preprocess_mimic3_sand.py` pre-splits and writes four objects:
   `mimic3_sand/{all,single_stay}/{train,test}/`.
2. `MIMIC3SandDataset(root, train, single_stay_only)` selects one, like
   `CIFAR10(train=...)`.
3. `TimeSeriesDM` (mimic3_sand branch): `test` object is fixed;
   `_grouped_val_pool_split` carves val from the `train` object per seed, rest =
   pool. `balanced_test_val` downsamples the eval sets to 50/50.

Decisions made:
- **Split source:** reuse STraTS's exact patient-grouped split (train+val → the
  `train` object; test → the fixed `test` object). Test is thus STraTS's exact
  8,878-stay cohort (all) — directly comparable to the paper. The single-stay
  cohort filters that split (17,012+4,276 train / 5,321 test); the 64/16/20 ratio
  survives filtering because STraTS splits by patient.
- **Cohorts:** both `all` (44,812) and `single_stay` (26,609) are emitted;
  `data.single_stay_only` picks one at load time.
- **Normalization:** computed over each cohort's `train` object (train+val) only,
  baked in — the CIFAR convention; removes the old full-data leakage. The
  seed-dependent pool was rejected as a stats source.

Verified: test set identical across seeds (all: 8,878; single_stay: 5,321), val
varies per seed, baleval balances to 50/50, and a 1-epoch end-to-end run trains
and evaluates cleanly.

## LOS label distribution (2026-09-10)

Length-of-stay task (`mimic3_los`, config `single_stay_only: false` → **all**
cohort). Labels are the total ICU LOS bucketed by `MIMIC3LOSDataset` via
`np.digitize(los_days, LOS_BIN_EDGES)`, `LOS_BIN_EDGES = [2,3,4,5,6,7,8,14]` →
**9 ordinal classes** (Harutyunyan's 10 buckets minus the structurally-empty
`[0,1)d` bin, cohort ≥ 24h). See [../src/data/mimic_dataset.py](../src/data/mimic_dataset.py) lines 53–62, 150–151.

Splits are the same fixed CIFAR-style objects used for mortality: the `train`
object is pool+val (AL carves val per seed), the `test` object is fixed across
seeds. "All" = train + test. Counts read directly from
`$DATA_ROOT/mimic3_sand/{cohort}/{split}/los.npy`.

### `all` cohort (n = 44,812; train 35,934 / test 8,878)

| class | LOS (days) | All | train (pool+val) | test |
|---|---|---|---|---|
| 0 | [1,2)   | 16,045 (35.81%) | 12,836 (35.72%) | 3,209 (36.15%) |
| 1 | [2,3)   |  9,329 (20.82%) |  7,461 (20.76%) | 1,868 (21.04%) |
| 2 | [3,4)   |  5,224 (11.66%) |  4,179 (11.63%) | 1,045 (11.77%) |
| 3 | [4,5)   |  3,228 ( 7.20%) |  2,557 ( 7.12%) |   671 ( 7.56%) |
| 4 | [5,6)   |  2,095 ( 4.68%) |  1,690 ( 4.70%) |   405 ( 4.56%) |
| 5 | [6,7)   |  1,479 ( 3.30%) |  1,188 ( 3.31%) |   291 ( 3.28%) |
| 6 | [7,8)   |  1,104 ( 2.46%) |    911 ( 2.54%) |   193 ( 2.17%) |
| 7 | [8,14)  |  3,451 ( 7.70%) |  2,818 ( 7.84%) |   633 ( 7.13%) |
| 8 | [14,∞)  |  2,857 ( 6.38%) |  2,294 ( 6.38%) |   563 ( 6.34%) |

Median LOS: all 2.65 d, train 2.66 d, test 2.63 d. Effective #classes
(exp-entropy of the ALL distribution): **6.39**.

### `single_stay` cohort (n = 26,609; train 21,288 / test 5,321)

| class | LOS (days) | All | train (pool+val) | test |
|---|---|---|---|---|
| 0 | [1,2)   | 10,276 (38.62%) | 8,225 (38.64%) | 2,051 (38.55%) |
| 1 | [2,3)   |  5,517 (20.73%) | 4,393 (20.64%) | 1,124 (21.12%) |
| 2 | [3,4)   |  2,987 (11.23%) | 2,368 (11.12%) |   619 (11.63%) |
| 3 | [4,5)   |  1,812 ( 6.81%) | 1,427 ( 6.70%) |   385 ( 7.24%) |
| 4 | [5,6)   |  1,155 ( 4.34%) |   926 ( 4.35%) |   229 ( 4.30%) |
| 5 | [6,7)   |    822 ( 3.09%) |   642 ( 3.02%) |   180 ( 3.38%) |
| 6 | [7,8)   |    621 ( 2.33%) |   515 ( 2.42%) |   106 ( 1.99%) |
| 7 | [8,14)  |  1,885 ( 7.08%) | 1,551 ( 7.29%) |   334 ( 6.28%) |
| 8 | [14,∞)  |  1,534 ( 5.76%) | 1,241 ( 5.83%) |   293 ( 5.51%) |

Median LOS: all 2.42 d, train 2.42 d, test 2.44 d. Effective #classes: **6.08**.

Notes:
- Heavily head-skewed: class 0 (`[1,2)d`) alone is ~36% (all) / ~39%
  (single_stay); the mid classes 4–6 are each < 5%; the tail class 8 (`[14,∞)`)
  reabsorbs long stays to ~6%. This is why the headline metric is **linear-weighted
  Cohen's kappa** rather than accuracy, and why every class stays populated.
- Distributions are near-identical across train/test (the STraTS patient-grouped
  split is not stratified on LOS but is large enough to be balanced), so the fixed
  test split is representative.
- single_stay is slightly shorter-stay (more mass in `[1,2)`, less in the tail),
  consistent with it being a distinct, marginally healthier cohort.


