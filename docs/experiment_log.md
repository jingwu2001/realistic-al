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

## 2026-10-15 to 2026-10-18
completed the full 4-class ecg5000 sweep (InceptionTime, same recipe as exp_ecg5000_basic.py)
grid: 
  ecg5000_low (25 -> 250) / ecg5000_med (50 -> 500) 
  x active.balanced True/False x 6 seeds (12345-12350)
methods: 
  random, entropy, bald, badge, kcentergreedy
  + {vendi-q2, bandit (vendi-q2 arm)} x {feature, gradient} x 7 kernels
    kernels: rbf-median x {l2, minmax, zscore, none}, rbf-dim-zscore, cosine x {zscore, minmax}
132 cells x 6 seeds = 792 runs, all finished by oct 18 7:43am
- oct 15: resumed the balanced-init sweep, crashed on disk full
- oct 17 12:41am-5:17am: launchers/run_ecg5000_4c_pending.sh (balanced rerun, missing seeds, feature bandit random init, grad med)
- oct 18 2:09am-7:43am: launchers/run_ecg5000_4c_fill_grid.sh (feature bandit balanced init, grad low)
folders ecg5000_4c/active-*/2026101{4,5,7,8}-239a94ed7 (same commit, pool across folders)
extra: vendi-q2 rbf-1.0-l2 feature (dropped config) has only 3 seeds with balanced init


## 2026-10-19 1:11am
launched launchers/sweep_exp_mimic3_pheno_single_task_lstm_nodemo_baselines.py
ended oct 19 ~7:30pm, 60/60 runs done
lstm, time series only (model.use_static=False, no Age/Gender), feature dimension 256
random init only (active.balanced=False)
random, bald, badge, entropy, kcentergreedy x task 8/11 x (100,100)/(500,500) x 3 seeds
folder 20261019-2bb88f3d2, token _demo-False_bal-False; wandb mimic3_pheno_tNN_lstm_nodemo

## 2026-10-19 11:40pm
launched launchers/sweep_exp_mimic3_pheno_single_task_lstm_nodemo_tune_vendi.py --jobs 4
same recipe as the nodemo baselines; 240 runs
{bandit (vendi-q2 arm), vendi-q2} x 10 kernel configs x task 8/11 x (100,100)/(500,500) x 3 seeds
feature: rbf-median x {l2, minmax, zscore, none}, rbf-dim-zscore, cosine x {zscore, none}
gradient: rbf-median-l2, cosine x {zscore, none}
oct 20 10pm: 199/240 done, no errors; expected to end ~oct 21 1:30am

## 2026-10-20 11:05pm
queued launchers/run_mimic3_pheno25_lstm_nodemo.sh 1053662 (starts when the single-task nodemo tune_vendi launcher exits, ~oct 21 1:30am)
lstm, 25 conditions (data.multitask=True), time series only (model.use_static=False), random init only (active.balanced=False)
1. sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py: random, bald, badge, entropy, kcentergreedy x (100,100)/(500,500) x 3 seeds = 30 runs
2. sweep_exp_mimic3_pheno_lstm_nodemo_tune_vendi.py: {bandit (vendi-q2 arm), vendi-q2} x 10 kernel configs x (100,100)/(500,500) x 3 seeds = 120 runs
   same 10 kernel configs as the single-task nodemo sweep (7 feature + 3 gradient), not re-probed on 25 labels
--jobs 4; wandb mimic3_pheno_lstm_nodemo; chain log experiments/activelearning/mimic3_pheno25_lstm_nodemo_chain_20261020.log
smoke test before launch (bandit, grad rbf-median-l2, 2 rounds x 2 epochs): ok, peak RAM 7.5 GB, < 1 GB VRAM
