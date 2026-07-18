# Sweep Run Tables (one seed)

One row per run, in **launch order for a single seed** (`trainer.seed=12345`).
With the default 3 seeds every row runs 3x: in the `sweep_exp_*_query` launchers
the 3 seeds occupy consecutive launcher indices (row r -> launches 3r-2..3r);
in the MIMIC launchers seeds nest as outer blocks instead — when slicing with
`--num_start/--num_end`, confirm indices with `--debug` first. Generated
2026-07-16 from the launcher grids (verified against `--debug` output) —
regenerate with `python launchers/gen_sweep_run_tables.py` after editing a
launcher.

Column notes: **emb** `feat` = vendi (feature embeddings), `grad` = gvendi
(`vendi.use_grad=true`, BADGE-style last-layer gradients). **gamma** applies to
rbf only (`dim` = 1/(2D), `median` = 1/(2·median²)). Baselines show `-` for the
kernel columns (inert placeholders).


| dataset | launcher(s) | runs/seed |
|---|---|---|
| CIFAR-10 (+imb) | `sweep_exp_cifar10_query.py` | 26 |
| CIFAR-100 | `sweep_exp_cifar100_query.py` | 25 |
| P12 | `sweep_exp_p12_transformer_query.py` | 50 |
| MIMIC-III | `sweep_exp_mimic3_sand_basic.py` + `sweep_exp_mimic3_sand_kernels.py` | 50 |
| **total** | | **151** |

## CIFAR-10 / CIFAR-10-imb — 26 runs

vendi rows: ~1.2 h (low) / ~4.8 h (med) / ~0.75 h (imb-low) / ~1.5-2 h (imb-med); baselines ~35-50 min (batchbald: est. 1-1.5 h low / 3-6 h med).

| # | data | active | query | emb | kernel | norm | gamma |
|---|---|---|---|---|---|---|---|
| 1 | cifar10_imb | cifar10_med | random | - | - | - | - |
| 2 | cifar10_imb | cifar10_med | entropy | - | - | - | - |
| 3 | cifar10_imb | cifar10_med | kcentergreedy | - | - | - | - |
| 4 | cifar10_imb | cifar10_med | bald | - | - | - | - |
| 5 | cifar10_imb | cifar10_med | badge | - | - | - | - |
| 6 | cifar10_imb | cifar10_med | variationratios | - | - | - | - |
| 7 | cifar10_imb | cifar10_med | gvendi | grad | rbf | minmax | 1.0 |
| 8 | cifar10_imb | cifar10_med | vendi | feat | rbf | minmax | 1.0 |
| 9 | cifar10_imb | cifar10_med | gvendi | grad | cosine | none | - |
| 10 | cifar10_imb | cifar10_med | gvendi | grad | rbf | l2 | dim |
| 11 | cifar10_imb | cifar10_med | gvendi | grad | rbf | l2 | median |
| 12 | cifar10_imb | cifar10_med | gvendi | grad | rbf | minmax | dim |
| 13 | cifar10_imb | cifar10_med | gvendi | grad | rbf | minmax | median |
| 14 | cifar10_imb | cifar10_med | gvendi | grad | rbf | zscore | dim |
| 15 | cifar10_imb | cifar10_med | gvendi | grad | rbf | zscore | median |
| 16 | cifar10_imb | cifar10_med | gvendi | grad | rbf | none | dim |
| 17 | cifar10_imb | cifar10_med | gvendi | grad | rbf | none | median |
| 18 | cifar10_imb | cifar10_med | vendi | feat | cosine | none | - |
| 19 | cifar10_imb | cifar10_med | vendi | feat | rbf | l2 | dim |
| 20 | cifar10_imb | cifar10_med | vendi | feat | rbf | l2 | median |
| 21 | cifar10_imb | cifar10_med | vendi | feat | rbf | minmax | dim |
| 22 | cifar10_imb | cifar10_med | vendi | feat | rbf | minmax | median |
| 23 | cifar10_imb | cifar10_med | vendi | feat | rbf | zscore | dim |
| 24 | cifar10_imb | cifar10_med | vendi | feat | rbf | zscore | median |
| 25 | cifar10_imb | cifar10_med | vendi | feat | rbf | none | dim |
| 26 | cifar10_imb | cifar10_med | vendi | feat | rbf | none | median |

## CIFAR-100 — 25 runs

vendi rows: long query phases at L>=500 (~7 min/round full pool); batchbald-med/high are the heaviest rows (hours per round).

| # | active | query | emb | kernel | norm | gamma |
|---|---|---|---|---|---|---|
| 1 | cifar100_low | random | - | - | - | - |
| 2 | cifar100_low | entropy | - | - | - | - |
| 3 | cifar100_low | kcentergreedy | - | - | - | - |
| 4 | cifar100_low | bald | - | - | - | - |
| 5 | cifar100_low | badge | - | - | - | - |
| 6 | cifar100_low | batchbald | - | - | - | - |
| 7 | cifar100_low | variationratios | - | - | - | - |
| 8 | cifar100_low | vendi | feat | rbf | l2 | dim |
| 9 | cifar100_low | vendi | feat | rbf | l2 | median |
| 10 | cifar100_low | vendi | feat | rbf | minmax | dim |
| 11 | cifar100_low | vendi | feat | rbf | minmax | median |
| 12 | cifar100_low | vendi | feat | rbf | zscore | dim |
| 13 | cifar100_low | vendi | feat | rbf | zscore | median |
| 14 | cifar100_low | vendi | feat | rbf | none | dim |
| 15 | cifar100_low | vendi | feat | rbf | none | median |
| 16 | cifar100_low | vendi | feat | cosine | none | - |
| 17 | cifar100_low | gvendi | grad | rbf | l2 | dim |
| 18 | cifar100_low | gvendi | grad | rbf | l2 | median |
| 19 | cifar100_low | gvendi | grad | rbf | minmax | dim |
| 20 | cifar100_low | gvendi | grad | rbf | minmax | median |
| 21 | cifar100_low | gvendi | grad | rbf | zscore | dim |
| 22 | cifar100_low | gvendi | grad | rbf | zscore | median |
| 23 | cifar100_low | gvendi | grad | rbf | none | dim |
| 24 | cifar100_low | gvendi | grad | rbf | none | median |
| 25 | cifar100_low | gvendi | grad | cosine | none | - |

## P12 (transformer) — 50 runs

Cheap: ~7.6k pool, binary task; vendi rows a few minutes of query per round.

| # | eval split | query | emb | kernel | norm | gamma |
|---|---|---|---|---|---|---|
| 1 | p12_transformer_baleval | random | - | - | - | - |
| 2 | p12_transformer | random | - | - | - | - |
| 3 | p12_transformer_baleval | entropy | - | - | - | - |
| 4 | p12_transformer | entropy | - | - | - | - |
| 5 | p12_transformer_baleval | kcentergreedy | - | - | - | - |
| 6 | p12_transformer | kcentergreedy | - | - | - | - |
| 7 | p12_transformer_baleval | bald | - | - | - | - |
| 8 | p12_transformer | bald | - | - | - | - |
| 9 | p12_transformer_baleval | badge | - | - | - | - |
| 10 | p12_transformer | badge | - | - | - | - |
| 11 | p12_transformer_baleval | batchbald | - | - | - | - |
| 12 | p12_transformer | batchbald | - | - | - | - |
| 13 | p12_transformer_baleval | variationratios | - | - | - | - |
| 14 | p12_transformer | variationratios | - | - | - | - |
| 15 | p12_transformer_baleval | gvendi | grad | rbf | l2 | dim |
| 16 | p12_transformer | gvendi | grad | rbf | l2 | dim |
| 17 | p12_transformer_baleval | gvendi | grad | rbf | l2 | median |
| 18 | p12_transformer | gvendi | grad | rbf | l2 | median |
| 19 | p12_transformer_baleval | gvendi | grad | rbf | minmax | dim |
| 20 | p12_transformer | gvendi | grad | rbf | minmax | dim |
| 21 | p12_transformer_baleval | gvendi | grad | rbf | minmax | median |
| 22 | p12_transformer | gvendi | grad | rbf | minmax | median |
| 23 | p12_transformer_baleval | gvendi | grad | rbf | zscore | dim |
| 24 | p12_transformer | gvendi | grad | rbf | zscore | dim |
| 25 | p12_transformer_baleval | gvendi | grad | rbf | zscore | median |
| 26 | p12_transformer | gvendi | grad | rbf | zscore | median |
| 27 | p12_transformer_baleval | gvendi | grad | rbf | none | dim |
| 28 | p12_transformer | gvendi | grad | rbf | none | dim |
| 29 | p12_transformer_baleval | gvendi | grad | rbf | none | median |
| 30 | p12_transformer | gvendi | grad | rbf | none | median |
| 31 | p12_transformer_baleval | gvendi | grad | cosine | none | - |
| 32 | p12_transformer | gvendi | grad | cosine | none | - |
| 33 | p12_transformer_baleval | vendi | feat | rbf | l2 | dim |
| 34 | p12_transformer | vendi | feat | rbf | l2 | dim |
| 35 | p12_transformer_baleval | vendi | feat | rbf | l2 | median |
| 36 | p12_transformer | vendi | feat | rbf | l2 | median |
| 37 | p12_transformer_baleval | vendi | feat | rbf | minmax | dim |
| 38 | p12_transformer | vendi | feat | rbf | minmax | dim |
| 39 | p12_transformer_baleval | vendi | feat | rbf | minmax | median |
| 40 | p12_transformer | vendi | feat | rbf | minmax | median |
| 41 | p12_transformer_baleval | vendi | feat | rbf | zscore | dim |
| 42 | p12_transformer | vendi | feat | rbf | zscore | dim |
| 43 | p12_transformer_baleval | vendi | feat | rbf | zscore | median |
| 44 | p12_transformer | vendi | feat | rbf | zscore | median |
| 45 | p12_transformer_baleval | vendi | feat | rbf | none | dim |
| 46 | p12_transformer | vendi | feat | rbf | none | dim |
| 47 | p12_transformer_baleval | vendi | feat | rbf | none | median |
| 48 | p12_transformer | vendi | feat | rbf | none | median |
| 49 | p12_transformer_baleval | vendi | feat | cosine | none | - |
| 50 | p12_transformer | vendi | feat | cosine | none | - |

## MIMIC-III (SAND) — 50 runs

Two launchers: baselines (`_basic`, rows 1-14) then vendi grid (`_kernels`, rows 15-70); run numbers restart per launcher for `--num_start/--num_end` (basic: 1-14, kernels: 1-56 at one seed). batchbald: est. hours/run (float64 greedy over ~30k pool).

| # | eval split | query | emb | kernel | norm | gamma |
|---|---|---|---|---|---|---|
| 1 | mimic3_sand_baleval | random | - | - | - | - |
| 2 | mimic3_sand | random | - | - | - | - |
| 3 | mimic3_sand_baleval | entropy | - | - | - | - |
| 4 | mimic3_sand | entropy | - | - | - | - |
| 5 | mimic3_sand_baleval | kcentergreedy | - | - | - | - |
| 6 | mimic3_sand | kcentergreedy | - | - | - | - |
| 7 | mimic3_sand_baleval | bald | - | - | - | - |
| 8 | mimic3_sand | bald | - | - | - | - |
| 9 | mimic3_sand_baleval | badge | - | - | - | - |
| 10 | mimic3_sand | badge | - | - | - | - |
| 11 | mimic3_sand_baleval | variationratios | - | - | - | - |
| 12 | mimic3_sand | variationratios | - | - | - | - |
| 13 | mimic3_sand_baleval | batchbald | - | - | - | - |
| 14 | mimic3_sand | batchbald | - | - | - | - |
| 15 | mimic3_sand_baleval | gvendi | grad | rbf | l2 | dim |
| 16 | mimic3_sand_baleval | gvendi | grad | rbf | l2 | median |
| 17 | mimic3_sand_baleval | gvendi | grad | rbf | minmax | dim |
| 18 | mimic3_sand_baleval | gvendi | grad | rbf | minmax | median |
| 19 | mimic3_sand_baleval | gvendi | grad | rbf | zscore | dim |
| 20 | mimic3_sand_baleval | gvendi | grad | rbf | zscore | median |
| 21 | mimic3_sand_baleval | gvendi | grad | rbf | none | dim |
| 22 | mimic3_sand_baleval | gvendi | grad | rbf | none | median |
| 23 | mimic3_sand_baleval | gvendi | grad | cosine | none | - |
| 24 | mimic3_sand | gvendi | grad | rbf | l2 | dim |
| 25 | mimic3_sand | gvendi | grad | rbf | l2 | median |
| 26 | mimic3_sand | gvendi | grad | rbf | minmax | dim |
| 27 | mimic3_sand | gvendi | grad | rbf | minmax | median |
| 28 | mimic3_sand | gvendi | grad | rbf | zscore | dim |
| 29 | mimic3_sand | gvendi | grad | rbf | zscore | median |
| 30 | mimic3_sand | gvendi | grad | rbf | none | dim |
| 31 | mimic3_sand | gvendi | grad | rbf | none | median |
| 32 | mimic3_sand | gvendi | grad | cosine | none | - |
| 33 | mimic3_sand_baleval | vendi | feat | rbf | l2 | dim |
| 34 | mimic3_sand_baleval | vendi | feat | rbf | l2 | median |
| 35 | mimic3_sand_baleval | vendi | feat | rbf | minmax | dim |
| 36 | mimic3_sand_baleval | vendi | feat | rbf | minmax | median |
| 37 | mimic3_sand_baleval | vendi | feat | rbf | zscore | dim |
| 38 | mimic3_sand_baleval | vendi | feat | rbf | zscore | median |
| 39 | mimic3_sand_baleval | vendi | feat | rbf | none | dim |
| 40 | mimic3_sand_baleval | vendi | feat | rbf | none | median |
| 41 | mimic3_sand_baleval | vendi | feat | cosine | none | - |
| 42 | mimic3_sand | vendi | feat | rbf | l2 | dim |
| 43 | mimic3_sand | vendi | feat | rbf | l2 | median |
| 44 | mimic3_sand | vendi | feat | rbf | minmax | dim |
| 45 | mimic3_sand | vendi | feat | rbf | minmax | median |
| 46 | mimic3_sand | vendi | feat | rbf | zscore | dim |
| 47 | mimic3_sand | vendi | feat | rbf | zscore | median |
| 48 | mimic3_sand | vendi | feat | rbf | none | dim |
| 49 | mimic3_sand | vendi | feat | rbf | none | median |
| 50 | mimic3_sand | vendi | feat | cosine | none | - |
