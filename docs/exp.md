# Experiment grids

Finished runs (`aubc.json` / `stored.npz`) as of 2026-10-24, read from each
run's `loop-0/hparams.yaml`. Every run has 10 AL rounds with start = step.

Notation:
- **baselines** = random, entropy, BALD, BADGE, k-center greedy
- vendi kernel = `embedding:kernel-gamma-normalization`; `feat` = penultimate
  features, `grad` = last-layer gradient embedding (BADGE-style)
- **vendi** = Vendi score q=1, **vendi-q2** = q=2, **bandit** = LinUCB with a
  BALD arm and a vendi-q2 arm (alpha 1.0), **bandit-q1** = older bandit with a
  vendi q=1 arm
- **MIMIC-10** kernel set = feat: rbf-median-{l2, minmax, zscore, none},
  rbf-dim-zscore, cosine-{zscore, none}; grad: rbf-median-l2,
  cosine-{zscore, none}
- **ECG-7** kernel set = rbf-median-{l2, minmax, zscore, none},
  rbf-dim-zscore, cosine-{zscore, minmax}
- static dim = width of the Age/Gender embedding concatenated to the features
  (0 = time series only)

## MIMIC-III

Fixed everywhere: random initial set (`active.balanced=False`), SAND with the
original STraTS attention (now `model=sand_legacy`; was `model=sand`). Folders under
`experiments/activelearning/<data>/active-mimic3_med/`.

| Model | Task | Static dim | Single-stay | Budgets (start) | Seeds | Methods | Runs | Folders |
|---|---|---|---|---|---|---|---|---|
| LSTM | Phenotyping single-task (t8, t11) | 256 (old, linear) | True | 100, 500 | 3 | baselines; vendi-q2 feat:rbf-median-l2 (+ vendi feat:rbf-median-l2, t8 @100) | 75 | `20261006-3bb8f11fd` |
| | Phenotyping single-task (t8, t11) | 2 | True | 100, 500 | 3 | baselines; vendi-q2 feat:rbf-median-{l2, zscore} | 88 | `20261009-fc59d4043`, `20261011-1fee7b2f0` |
| | Phenotyping single-task (t8, t11) | 0 | True | 100, 500 | 3 | baselines; vendi-q2 × MIMIC-10; bandit × MIMIC-10 | 300 | `20261019-2bb88f3d2` |
| | Phenotyping multi-task (25 labels) | 0 | True | 100, 500 | 3 | baselines; vendi-q2 × MIMIC-10; bandit feat:rbf-dim-zscore, feat:rbf-median-l2 (+ grad:rbf-median-l2, 1 seed @100) | 103 | `20261021-54f0b2b6c`, `20261022-54f0b2b6c` |
| SAND | Phenotyping single-task (t8, t11) | 64 | True, False | 100, 500 | 3 | baselines; vendi + vendi-q2 feat:rbf-median-l2 (+ grad:rbf-median-l2, t8 single-stay=False @100) | 168 | `20260929-491df56c`, `20260930-491df56c`, `20261001-491df56c` |
| | Phenotyping single-task (t8, t11) | 0 | True | 100, 500 | 3 | baselines; vendi + vendi-q2 feat:rbf-median-l2 | 83 | `20261007-9d6a06939`, `20261007-3bb8f11fd` |
| | Phenotyping multi-task (25 labels) | 64 | False | 250, 1000 | 3 (some 1000 configs: 1–2) | baselines; vendi-q2 × {feat, grad} × rbf-{median-l2, median-minmax, median-zscore, median-none, dim-zscore, 1.0-minmax} | 87 | `20260912-3ad63434` … `20260916-3ad63434` |
| | Mortality | 64 | True, False | 100, 500, 1000 | 3 | baselines; vendi-q2 + vendi-q2-sequential feat:rbf-dim-zscore | 135 | `20260906-39cf72cc`, `20260907-39cf72cc` |
| | Mortality (Jul–Aug, older) | 64 | not set | 100 | 2–4 | baselines + BatchBALD, variation ratios, gvendi; vendi × 9 feat kernels (rbf-{median, dim}-{l2, minmax, zscore, none}, cosine); bandit-q1 × same 9 | 165 | `20260719-592e2f89` … `20260803-89844322` |
| | Length of stay | 64 | True, False | 100, 500, 1000 | 3 | baselines; vendi-q2 + vendi-q2-sequential feat:rbf-dim-zscore | 126 | `20260908-6310cdfc` |

Not run yet: LSTM multi-task with static 2 or 256, SAND multi-task with
static 0, SAND with static 2 (launcher exists), SAND with real attention (`model=sand`, `_attnfix`),
any MIMIC balanced initial set.

## ECG5000

Model: InceptionTime (`n_filters=64`, 2 blocks) only. No static input.
Folders under `experiments/activelearning/<data>/active-ecg5000_{low,med}/`.

| Model | Task | Initial set | Budgets | Seeds | Methods | Runs | Folders |
|---|---|---|---|---|---|---|---|
| InceptionTime | 4-class (`ecg5000_4c`, class 4 dropped) | random, balanced (5 per class) | low 25→250, med 50→500 | 6 | baselines; vendi-q2 × {feat, grad} × ECG-7 (+ feat:rbf-1.0-l2, 3 seeds with balanced init); bandit × {feat, grad} × ECG-7; extra bandit-q1 feat:rbf-median-{l2, minmax} @low | 828 | `20261014-239a94ed7` … `20261018-239a94ed7` |
| | 5-class (`ecg5000`, older) | random | 10→100, 25→250, 50→500, 250→2500 | varies (6–13) | baselines; vendi feat:rbf-1.0-minmax; bandit-q1 feat:rbf-1.0-minmax | 252 | March folders (no wave dir), `20261013-80d4aeae6` |

The 4-class grid is a full cross: 2 budgets × 2 initial sets × 33 methods
(5 baselines + 2 × 2 × 7) × 6 seeds = 792 runs, plus the extras.
