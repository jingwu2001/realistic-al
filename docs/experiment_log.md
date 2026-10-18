# Experiment Log

Chronological record of experiment runs launched (newest first). For the
standing inventory of settings see [experiment_settings.md](experiment_settings.md);
for the MIMIC/STraTS analysis see
[mimic3_sand_strats_comparison.md](mimic3_sand_strats_comparison.md).

---

## 2026-09-08 — MIMIC-III SAND: length-of-stay learning curves, both cohorts

Launched the MIMIC-III **LOS** learning-curve sweep via
[launchers/sweep_exp_mimic3_los_basic.py](../launchers/sweep_exp_mimic3_los_basic.py)
(commit `6310cdfc`, branch `open-source`). Mirror of the 2026-09-07 mortality
sweep, but `data=mimic3_los` — 9-class ordinal length-of-stay instead of binary
mortality.

**Grid — 126 runs:**
- **Methods (7):** random, badge, kcentergreedy, bald, entropy, vendi-q2, vendi-q2-sequential
- **Budgets (3), (start, step) with num_iter=10:** (100,100)→100..1000, (500,500)→500..5000, (1000,1000)→1000..10000
- **Cohorts (2):** `data.single_stay_only` = False (all, 44,812) / True (single-stay, 26,609)
- **Seeds (3):** 12345, 12346, 12347

**Task/metric:** 9-class ordinal LOS (Harutyunyan CustomBins minus the empty
`[0,1)d` bin; effective class count ~6.4). Headline = linear-weighted Cohen's
kappa (`test/kappa`, AUBC as `al_summary/aubc_test_kappa`); checkpoints selected
on `val/kappa`. Same STraTS SAND recipe as the mortality sweep (lr 5e-4, dropout
0.2, weighted loss, batch 16, clip 0.3, adam wd=0, constant lr, 50 ep, fp32,
deterministic; vendi block pinned rbf/gamma=dim/zscore).

**Where the results are:**
- wandb: project `realistic-al`, `config.data_name = mimic3_los`,
  `config.active_name = mimic3_med`. Run names collide across budgets/cohorts —
  disambiguate by `config.acq_size` and `config.data.single_stay_only`; x-axis
  `al/n_labelled`.
- on disk: `experiments/activelearning/mimic3_los/active-mimic3_med/20260908-6310cdfc/`

**Purpose:** LOS is the imbalanced-multiclass regime where diversity acquisition
should separate from baselines (unlike binary mortality). See
[experiment_settings.md](experiment_settings.md#mimic-iii--length-of-stay-sand).

---

## 2026-09-07 — MIMIC-III SAND: random+baseline learning curves, both cohorts

Launched the combined MIMIC-III learning-curve sweep via
[launchers/sweep_exp_mimic3_sand_basic.py](../launchers/sweep_exp_mimic3_sand_basic.py)
(commit `39cf72cc`, branch `open-source`).

**Grid — 126 runs** (`random` runs first):
- **Methods (7):** random, badge, kcentergreedy, bald, entropy, vendi-q2, vendi-q2-sequential
- **Budgets (3), (start, step) with num_iter=10:** (100,100)→100..1000, (500,500)→500..5000, (1000,1000)→1000..10000
- **Cohorts (2):** `data.single_stay_only` = False (all, 44,812 stays) and True (single-stay, 26,609)
- **Seeds (3):** 12345, 12346, 12347

**Data / split:** `data=mimic3_sand`, bal eval OFF (natural ~11.5% positive). Uses
the new CIFAR-style fixed split — test cohort is STraTS's own split, identical
across seeds; the seed only moves the val carve, labeled pool, and init.

**Recipe (mimics STraTS SAND):** model=sand, lr 5e-4, dropout 0.2, weighted loss,
batch_size 16, gradient_clip_val 0.3, optim=adam + weight_decay=0 (≡ AdamW wd=0),
lr_scheduler=null (constant lr), 50 epochs, fp32, deterministic. Vendi block pinned
(rbf / gamma="dim" / zscore; sequential batch_size=1).

**Where the results are:**
- wandb: project `realistic-al`, group `mimic3_sand/mimic3_med/random`. Run names
  collide (`mimic3_sand/random/seed-*`) across budgets/cohorts — disambiguate by the
  config fields `acq_size` (budget), `data.single_stay_only` (cohort), `seed`; x-axis
  `al/n_labelled`.
- on disk: `experiments/activelearning/mimic3_sand/active-mimic3_med/20260907-39cf72cc/`

**Purpose:** learning curves for both cohorts to compare against the STraTS SAND
full-data reference (does AL performance keep climbing with more labels, and does the
single-stay cohort behave differently). See
[mimic3_sand_strats_comparison.md](mimic3_sand_strats_comparison.md).

## 2026-09-15 - MIMIC-Phenotype
launched full 3 runs for baselines + vendi-q2 sweep
sequential not yet. Also q=1 not yet


also launchced `q=1.0+minmax` for `acq_size=1000`
Completed all vendi-q2 settings sweep (meaningful kernel + rbf-1.0) and all baselines x 3 seeds

## 2026-09-16
started 12346 & 12347 for all (`exp_cifar10imb_basic_tune_kernel.py`)
baselines + 
vendi-q2, rbf-med x (minmax, none, l2)
(seed 12345 completed and these vendi settings are top performers) 

## 2026-09-21
started 12346 & 12347 for all baselines
(`exp_cifar10imb_basic_tune_kernel.py`)

cifar-10-imb-med
q2
  [rbf] * [med, dim] * [min-max, z-score, l2, none] * [feat, grad]
  [cos] * [min-max, z-score, none]                  * [feat, grad]

q1
  rbf 1.0 min-max [feat, grad]
  rbf med l2 grad
  rbf dim [min-max l2] grad
  rbf cos grad

## 2026-09-21
launched vendi secular with just 2 iterations

## 2026-09-28
launched launchers/sweep_exp_mimic3_pheno_single_task.py
(single-/multi-stay x (100, 500) * 3 seeds)


## 2026-10-09
launched launchers/sweep_exp_mimic3_pheno_single_task_lstm_demo2.py
ended before oct 11 4:33pm
lstm, demographic (or static) dim=2, feature dimension 256+2=258
task 8/11, acq_size 500/100

## 2026-10-11 5:16pm
launched launchers/sweep_exp_mimic3_pheno_single_task_lstm_demo2_tune_vendi.py
z-score normalization to avoid static features taking up too much distance

## 2026-10-13 11:44pm
launched launchers/exp_ecg5000_basic_tune_kernel.py
ecg5000 with kernel

