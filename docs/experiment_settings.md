# Experiment Settings Overview

Inventory of active-learning experiment settings that have actually been run,
organized by **dataset → imbalance → active regime (query/acq size)**. For each
setting: the query strategies attempted (from the on-disk run folders under
`experiments/activelearning/`) and whether that setting is tracked on **Weights
& Biases** (`jingwusince01-national-taiwan-uni/realistic-al`, 787 non-test runs
as of 2026-09-12).

> **How to read this**
> - "Active regime" = the `src/config/active/*.yaml` used. `num_labelled` =
>   initial labelled pool, `acq_size` = per-round budget, all with `num_iter=10`
>   acquisition rounds.
> - "On wandb" is determined from the live wandb project (grouped by
>   `data` + `active`), not just from disk. Disk folders are the superset of
>   *attempts*; some disk folders are smoke tests that never made it to wandb.
> - Query-strategy names as they appear in run-dir names. Note the naming
>   evolution: the diversity method is `vendi` (feature embeddings) / `gvendi`
>   (last-layer gradients, `vendi.use_grad=true`); older CIFAR runs wrote the
>   same idea as `vendi` / `bandit` with `norm`/`kernel`/`gamma` suffixes.

---

## How to filter runs on wandb

Project **`realistic-al`** (entity `jingwusince01-national-taiwan-uni`). Every run
is stamped by [`src/utils/wandb_utils.py`](../src/utils/wandb_utils.py) with a
group, a run name, tags, and the **full resolved Hydra config**, so any
experiment can be sliced three ways.

**Identity scheme**
- **Group:** `{data_name}/{active_name}/{method-variant}` —
  e.g. `cifar10_imb/cifar10_med/vendi-feat-rbf-minmax-q1.0`
- **Run name:** `{data_name}/{method-variant}/seed-{seed}` —
  e.g. `cifar10_imb/vendi-feat-rbf-minmax-q1.0/seed-12345`
- **Method-variant tag:** plain methods (`random`, `bald`, `badge`, `entropy`,
  `kcentergreedy`, `batchbald`, `variationratios`) use their short name as-is;
  the diversity methods (`vendi`, `gvendi`, `vendi-q2`, `bandit`) append
  `-{emb}-{kernel}-{norm}-q{q}` (and `-a{alpha}`/`-s{quality}` when set), where
  `emb` is `feat` (features) or `grad` (last-layer gradients), and an `rbf`
  kernel folds gamma in (`rbfdim`, `rbf1.0`, `rbfmed`).
- **Tags:** `data_name`, `active_name`, `query_name` (short, e.g. `vendi-q2`), a
  session tag `{YYYYMMDD}-{HHMMSS}_{git-commit}` (one per launch), plus
  per-hyperparameter tags for the diversity methods (`kernel:rbf`, `norm:zscore`,
  `emb:grad`, `q:2`, `alpha:…`, `quality:…`).
- **Config fields** (best for exact filtering in the Runs table):
  `config.data_name`, `config.active_name`, `config.query_name`,
  `config.acq_size`, `config.num_labelled`, `config.seed`, `config.model_name`,
  plus the entire nested config (`config.data.single_stay_only`,
  `config.query.vendi.kernel`, …).

**Recipe for one experiment** — set these in the Runs-table filter panel:
```
config.data_name   = <data config>    # imbalance / eval-split axis
config.active_name = <active regime>   # init/acq budget
config.query_name  = <method>          # optional: pin one strategy
```
Add `config.seed`, `config.acq_size`, `config.data.single_stay_only`, or a
`kernel:…`/`norm:…` tag to narrow further. **X-axis for AL curves:**
`al/n_labelled` (labels acquired); `al_iter` is the round index. AUBC summaries
live under `al_summary/aubc_test_*`.

> Naming caveats:
> - **`data_name` is the Hydra *data-config choice*, not `cfg.data.name`.** This
>   is what separates `cifar10` vs `cifar10_imb`, `p12_transformer` vs
>   `p12_transformer_baleval`, and `mimic3_sand` vs `mimic3_sand_baleval` —
>   variants that share the same `cfg.data.name`.
> - The **MIMIC learning-curve sweeps vary `acq_size` and cohort *inside* one
>   `active_name` (`mimic3_med`)**, so run names collide across budgets/cohorts —
>   disambiguate with `config.acq_size` and `config.data.single_stay_only`.
> - Imported pre-wandb runs carry the tag **`old_run`**; smoke/wiring tests carry
>   **`test`** / **`wandb_test`** — exclude those to see only real runs.

---

## Summary table

Seed pool is `{12345, 12346, 12347}` (filter one with `config.seed`). The
**Seeds** column is the *max* distinct seeds reached by any method in that
setting — **per-method counts vary a lot**, so see the full
[Seeds per query method](#seeds-per-query-method) breakdown below. (Counts are
from a live wandb pull of 787 non-test runs; disk-only settings are noted.)

| Dataset | Imbalance | Active regime | init / acq | Model | Seeds | On wandb? |
|---|---|---|---|---|---|---|
| CIFAR-10 | balanced | `cifar10_low` | 50 / 50 | ResNet | 3 | ✅ yes (imported `old_run`) |
| CIFAR-10 | balanced | `cifar10_med` | 250 / 250 | ResNet | 1–3 | ✅ yes (mostly single-seed; only random/vendi at 3) |
| CIFAR-10 | balanced | `cifar10_high` | 1000 / 1000 | ResNet | 1–3 | ⚠️ partial (only vendi at 3; rest 1–2) |
| CIFAR-10 | **imbalanced** | `cifar10_low` | 50 / 50 | ResNet + wloss | 3 | ✅ yes (imported `old_run` only) |
| CIFAR-10 | **imbalanced** | `cifar10_med` | 250 / 250 | ResNet + wloss | **1** | ✅ yes (HP sweep — all methods single-seed 12345) |
| CIFAR-10 | **imbalanced** | `cifar10_high` | 1000 / 1000 | ResNet + wloss | — | ❌ smoke only (`ep-3`, disk); not on wandb |
| ECG5000 | native (imbal.) | `ecg5000_10` | 10 / 10 | InceptionTime | 3 | ❌ **no** (disk only) |
| ECG5000 | native (imbal.) | `ecg5000_low` | 25 / 25 | InceptionTime | 3 | ❌ **no** (disk only) |
| ECG5000 | native (imbal.) | `ecg5000_med` | 50 / 50 | InceptionTime | 3 | ❌ **no** (disk only) |
| ECG5000 | native (imbal.) | `ecg5000_high` | 250 / 250 | InceptionTime | 3 | ❌ **no** (disk only) |
| P12 (PhysioNet) | native (imbal.) | `p12_med` | 100 / 100 | Transformer2 | 3 | ✅ yes (standard eval) |
| P12 (PhysioNet) | balanced pool | `p12_med_bal` | 100 / 100 | Transformer2 | 3 | ✅ yes (standard eval) |
| P12 — `baleval` | native / bal pool | `p12_med` / `p12_med_bal` | 100 / 100 | Transformer2 | 3 | ❌ **disk only** (not on wandb) |
| MIMIC-III (mortality) | native (imbal.) | `mimic3_med` | 100 / 100 | SAND | 1–3 | ✅ yes (baselines 3; old vendi/gvendi/bandit 1) |
| MIMIC-III (mortality) — `baleval` | native (imbal.) | `mimic3_med` | 100 / 100 | SAND | 1–3 | ✅ yes (balanced eval) |
| MIMIC-III (**LOS**, 9-class) | native (imbal.) | `mimic3_med` | 100/100, 500/500, 1000/1000 | SAND | 3 | ✅ yes (learning-curve sweep, launched 2026-09-08) |

**Corrections from the live pull:** P12 `baleval` is **disk-only** (0 runs on
wandb — earlier notes claiming otherwise were wrong); MIMIC-III `baleval` (balanced
test) *is* on wandb. ECG5000 has no wandb coverage at all. The CIFAR-10 `*_med`
sweeps are largely **single-seed** hyperparameter exploration, not 3-seed
replicates.

---

## Seeds per query method

One row per **data / active-regime / query-method** combination, with the number
of **distinct seeds** actually present. Numbers below are from a live wandb pull
(787 non-test runs); `old_run` = imported pre-wandb runs. Seed pool is always a
subset of `{12345, 12346, 12347}` — a count of `1` means only seed `12345` was
run (typically a single-seed HP sweep), not a missing seed to be filled unless
you want replicates.

### On wandb

**`cifar10`** (balanced)

| Active regime | Query method | Seeds | Notes |
|---|---|---|---|
| `cifar10_low` | random | 3 | imported `old_run` |
| `cifar10_low` | entropy | 3 | imported `old_run` |
| `cifar10_low` | bald | 3 | imported `old_run` |
| `cifar10_low` | badge | 3 | imported `old_run` |
| `cifar10_low` | kcentergreedy | 3 | imported `old_run` |
| `cifar10_low` | vendi | 3 | imported `old_run` |
| `cifar10_low` | bandit | 3 | imported `old_run` |
| `cifar10_med` | random | 3 |  |
| `cifar10_med` | entropy | 1 |  |
| `cifar10_med` | bald | 1 |  |
| `cifar10_med` | badge | 1 |  |
| `cifar10_med` | kcentergreedy | 1 |  |
| `cifar10_med` | batchbald | 1 |  |
| `cifar10_med` | variationratios | 1 |  |
| `cifar10_med` | vendi | 3 |  |
| `cifar10_med` | gvendi | 1 |  |
| `cifar10_high` | random | 2 | incomplete |
| `cifar10_high` | entropy | 1 | incomplete |
| `cifar10_high` | bald | 1 | incomplete |
| `cifar10_high` | badge | 1 | incomplete |
| `cifar10_high` | kcentergreedy | 1 | incomplete |
| `cifar10_high` | vendi | 3 |  |

**`cifar10_imb`** (imbalanced, rho 0.02) — the `cifar10_med` regime is a
single-seed HP sweep (kernel/norm/gamma/q variants all at seed 12345).

| Active regime | Query method | Seeds | Notes |
|---|---|---|---|
| `cifar10_low` | random | 3 | imported `old_run` |
| `cifar10_low` | entropy | 3 | imported `old_run` |
| `cifar10_low` | bald | 3 | imported `old_run` |
| `cifar10_low` | badge | 3 | imported `old_run` |
| `cifar10_low` | kcentergreedy | 3 | imported `old_run` |
| `cifar10_low` | vendi | 3 | imported `old_run` |
| `cifar10_low` | bandit | 3 | imported `old_run` |
| `cifar10_med` | random | 1 |  |
| `cifar10_med` | entropy | 1 |  |
| `cifar10_med` | bald | 1 |  |
| `cifar10_med` | badge | 1 |  |
| `cifar10_med` | kcentergreedy | 1 |  |
| `cifar10_med` | variationratios | 1 |  |
| `cifar10_med` | vendi | 1 |  |
| `cifar10_med` | gvendi | 1 |  |
| `cifar10_med` | vendi-q2 | 1 |  |
| `cifar10_med` | vendi-q2-sequential | 1 |  |

**`p12_transformer`** (standard eval) — `p12_med` = natural pool, `p12_med_bal` =
balanced initial pool.

| Active regime | Query method | Seeds | Notes |
|---|---|---|---|
| `p12_med` | random | 3 |  |
| `p12_med` | entropy | 3 |  |
| `p12_med` | bald | 3 |  |
| `p12_med` | badge | 3 |  |
| `p12_med` | kcentergreedy | 3 |  |
| `p12_med` | vendi | 3 |  |
| `p12_med` | bandit | 3 |  |
| `p12_med_bal` | random | 3 |  |
| `p12_med_bal` | entropy | 3 |  |
| `p12_med_bal` | bald | 3 |  |
| `p12_med_bal` | badge | 3 |  |
| `p12_med_bal` | kcentergreedy | 3 |  |
| `p12_med_bal` | vendi | 3 |  |
| `p12_med_bal` | bandit | 3 |  |

**`mimic3_sand`** (mortality, standard eval) and **`mimic3_sand_baleval`**
(balanced eval). Seeds are per method; for the recent learning-curve methods
(random / badge / kcentergreedy / bald / entropy / vendi-q2[-sequential]) each of
the 3 seeds spans **3 budgets × 2 cohorts** — pin `config.acq_size` +
`config.data.single_stay_only` to isolate one curve. The old diversity grid
(`vendi` / `gvendi` / `bandit`) is single-seed, single-budget exploration.

| Eval split | Query method | Seeds | Notes |
|---|---|---|---|
| `mimic3_sand` | random | 3 |  |
| `mimic3_sand` | entropy | 3 |  |
| `mimic3_sand` | bald | 3 |  |
| `mimic3_sand` | badge | 3 |  |
| `mimic3_sand` | kcentergreedy | 3 |  |
| `mimic3_sand` | batchbald | 3 |  |
| `mimic3_sand` | variationratios | 3 |  |
| `mimic3_sand` | vendi | 1 | old grid |
| `mimic3_sand` | gvendi | 1 | old grid |
| `mimic3_sand` | bandit | 1 | old grid |
| `mimic3_sand` | vendi-q2 | 3 |  |
| `mimic3_sand` | vendi-q2-sequential | 3 |  |
| `mimic3_sand_baleval` | random | 3 |  |
| `mimic3_sand_baleval` | entropy | 3 |  |
| `mimic3_sand_baleval` | bald | 3 |  |
| `mimic3_sand_baleval` | badge | 3 |  |
| `mimic3_sand_baleval` | kcentergreedy | 3 |  |
| `mimic3_sand_baleval` | batchbald | 3 |  |
| `mimic3_sand_baleval` | variationratios | 3 |  |
| `mimic3_sand_baleval` | vendi | 1 | old grid |
| `mimic3_sand_baleval` | gvendi | 1 | old grid |
| `mimic3_sand_baleval` | bandit | 1 | old grid |

**`mimic3_los`** (9-class LOS, standard eval) — each seed spans 3 budgets × 2
cohorts (as above).

| Active regime | Query method | Seeds | Notes |
|---|---|---|---|
| `mimic3_med` | random | 3 |  |
| `mimic3_med` | entropy | 3 |  |
| `mimic3_med` | bald | 3 |  |
| `mimic3_med` | badge | 3 |  |
| `mimic3_med` | kcentergreedy | 3 |  |
| `mimic3_med` | vendi-q2 | 3 |  |
| `mimic3_med` | vendi-q2-sequential | 3 |  |

### Disk only (not on wandb)

Uniform on disk: **3 seeds** (`12345`, `12346`, `12347`) for every listed method.

- **ECG5000** — regimes `ecg5000_{10,low,med,high}`, methods {random, entropy,
  bald, badge, kcentergreedy, vendi, bandit}. Not used for reporting (memory:
  "Main Datasets").
- **P12 `baleval`** (`p12_transformer_baleval`) — regimes `p12_med`, `p12_med_bal`,
  methods {random, entropy, bald, badge, kcentergreedy, vendi, bandit}.
- **CIFAR-10-imb `cifar10_high`** — `ep-3` smoke only; not a real experiment.

---

## CIFAR-10 (balanced)

Model: ResNet (`basic`), `randaugmentMC` augmentation, dropout 0 or 0.5, `ep-200`.
Disk: `experiments/activelearning/cifar10/`.

**wandb filter:** `config.data_name = cifar10`, `config.active_name` = the row
(`cifar10_low`/`cifar10_med`/`cifar10_high`), `config.query_name` = a method.
Imported curves are tagged `old_run`.

| Active regime | init / acq | Query strategies run | On wandb? |
|---|---|---|---|
| `cifar10_low` | 50 / 50 | random, entropy, bald, badge, kcentergreedy, batchbald, variationratios, vendi, gvendi, bandit (+ vendi/bandit kernel variants: rbf γ∈{0.1,1,10,median}, cosine) | ✅ yes — 55 runs (mix of native + imported `old_run`) |
| `cifar10_med` | 250 / 250 | random, entropy, bald, badge, kcentergreedy, batchbald, variationratios, vendi, gvendi, bandit (+ kernel variants) | ✅ yes — 18 runs |
| `cifar10_high` | 1000 / 1000 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ⚠️ partial — 9 runs, 1 crashed, not a full seed×method grid |

---

## CIFAR-10 — imbalanced (`cifar10_imb`, weighted loss)

Model: ResNet + weighted loss (`wloss-True`), `randaugmentMC`, dropout 0/0.5, `ep-200`.
Disk: `experiments/activelearning/cifar10_imb/`. Imbalance ratio `rho = 0.02`
(`src/config/data/cifar10_imb.yaml`). This is the regime where the diversity
methods (vendi/gvendi) are expected to separate from baselines.

**wandb filter:** `config.data_name = cifar10_imb`, `config.active_name` = the row
(`cifar10_low`/`cifar10_med`/`cifar10_high`), `config.query_name` = a method
(`vendi-q2`, `bald`, `badge`, …). For diversity variants add tag filters such as
`kernel:rbf` / `norm:zscore` / `emb:grad`, or `config.query.vendi.q`. Imported
curves are tagged `old_run`.

| Active regime | init / acq | Query strategies run | On wandb? |
|---|---|---|---|
| `cifar10_low` | 50 / 50 | random, entropy, bald, badge, kcentergreedy, batchbald, variationratios, vendi, bandit (+ vendi/bandit kernel variants incl. rbf γ=1/1024) | ✅ yes — 38 runs, **all tagged `old_run`** (imported, pre-wandb) |
| `cifar10_med` | 250 / 250 | random, entropy, bald, badge, kcentergreedy, variationratios, vendi, gvendi, **vendi-l2, vendi-minmax, vendi-q2, vendi-q2-sequential** (+ kernel/norm/gamma grids, `grad-{True,False}`, `bs-{1,10,50}`) | ✅ yes — ~110 runs; includes the recent `vendi-q2` (22) and `vendi-q2-sequential` (62) sweeps (Aug–Sep 2026) |
| `cifar10_high` | 1000 / 1000 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ⚠️ smoke only — disk folders are `ep-3` smoke runs; 6 runs on wandb, no real training budget |

---

## ECG5000  ❌ not on wandb

Model: InceptionTime (`basic`), `ep-17`. Native class imbalance (5 classes).
Disk: `experiments/activelearning/ecg5000/`. **None of these settings are on
wandb** — they exist only as local run folders.

> Note: per project convention (memory: "Main Datasets"), ECG5000 is not used
> for verification/reporting; CIFAR-10 / P12 are the reporting datasets.

**wandb filter:** none — nothing was logged. Results exist only under
`experiments/activelearning/ecg5000/` on disk.

| Active regime | init / acq | Query strategies run | On wandb? |
|---|---|---|---|
| `ecg5000_10` | 10 / 10 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ❌ no |
| `ecg5000_low` | 25 / 25 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ❌ no |
| `ecg5000_med` | 50 / 50 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ❌ no |
| `ecg5000_high` | 250 / 250 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ❌ no |

---

## P12 (PhysioNet 2012, transformer)  ✅ on wandb

Model: Transformer2 (`transformer2`), dropout 0.5, `ep-50`, binary in-hospital
mortality. Disk: `experiments/activelearning/p12_transformer/` and
`p12_transformer_baleval/`.

Two independent axes beyond the query strategy:
- **Initial pool**: `p12_med` (natural class ratio) vs `p12_med_bal` (balanced
  initial labelled pool). Both 100 / 100.
- **Eval split**: standard test set (`p12_transformer`, **on wandb**) vs
  **balanced** test set (`p12_transformer_baleval`, **disk only** — a live pull
  found 0 baleval P12 runs on wandb).

**wandb filter:** `config.data_name = p12_transformer` (only the standard split is
on wandb); `config.active_name` = the **initial pool** (`p12_med` = natural,
`p12_med_bal` = balanced); `config.query_name` = a method.

| Active regime | init / acq | Query strategies run | On wandb? |
|---|---|---|---|
| `p12_med` | 100 / 100 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ✅ yes — 7 methods × 3 seeds (standard eval) |
| `p12_med_bal` | 100 / 100 | random, entropy, bald, badge, kcentergreedy, vendi, bandit | ✅ yes — 7 methods × 3 seeds (standard eval) |

> The `p12_transformer_baleval` (balanced-test) variant exists **only on disk**
> (`experiments/activelearning/p12_transformer_baleval/`, 3 seeds/method) — it was
> never logged to wandb, despite an earlier note here saying otherwise.

---

## MIMIC-III — mortality (SAND)  ✅ on wandb

Model: SAND (`sand`), dropout 0.2, `ep-50`, weighted loss (`wl-True`), binary
in-hospital mortality, STraTS fixed split. Disk:
`experiments/activelearning/mimic3_sand/` and `mimic3_sand_baleval/`.

Only one active regime was run: `mimic3_med` (100 / 100). Two eval splits:
standard (`mimic3_sand`) and balanced test (`mimic3_sand_baleval`).

| Eval split | init / acq | Query strategies run | On wandb? |
|---|---|---|---|
| `mimic3_sand` (standard) | 100 / 100 | random, entropy, bald, badge, kcentergreedy, batchbald, variationratios, vendi, gvendi, bandit (+ full vendi/gvendi norm × kernel × gamma grid: norm∈{none,l2,minmax,zscore}, kernel∈{rbf,cosine}, gamma∈{dim,median}) | ✅ yes — ~76 runs (some `failed`/`crashed`) |
| `mimic3_sand_baleval` (balanced test) | 100 / 100 | same as above | ✅ yes — ~78 runs |

**wandb filter:** `config.data_name` = the eval split (`mimic3_sand` = standard,
`mimic3_sand_baleval` = balanced test), `config.active_name = mimic3_med`,
`config.query_name` = a method. For the diversity grid add `kernel:{rbf,cosine}`
/ `norm:{none,l2,minmax,zscore}` / `emb:{feat,grad}` tags (or the
`config.query.vendi.*` fields). Exclude `failed`/`crashed` runs by state.

> Per memory ("Method Separation Regime"): the binary MIMIC mortality task does
> not separate methods — a known negative result, not a coverage gap.

---

## MIMIC-III — length-of-stay (SAND)  ✅ on wandb

Model: SAND (`sand`), dropout 0.2, `ep-50`, weighted loss, **9-class ordinal LOS**
(Harutyunyan CustomBins minus the empty `[0,1)d` bin; cohort ≥ 24h; effective
class count ~6.4). Headline metric is linear-weighted Cohen's **kappa**
(`test/kappa`, AUBC-summarised as `al_summary/aubc_test_kappa`; checkpoints
selected on `val/kappa`). Launched 2026-09-08 (commit `6310cdfc`) via
[launchers/sweep_exp_mimic3_los_basic.py](../launchers/sweep_exp_mimic3_los_basic.py).
Disk: `experiments/activelearning/mimic3_los/`.

This is the imbalanced-multiclass regime (unlike binary mortality) where
diversity acquisition is expected to separate. Learning-curve sweep — a single
`active_name` (`mimic3_med`) but three `(start, step)` budgets and both cohorts:

| Axis | Values |
|---|---|
| Methods (7) | random, badge, kcentergreedy, bald, entropy, vendi-q2, vendi-q2-sequential |
| Budgets (3), `(start, step)`, `num_iter=10` | (100,100)→100..1000, (500,500)→500..5000, (1000,1000)→1000..10000 |
| Cohort — `data.single_stay_only` | False (all, 44,812 stays) / True (single-stay, 26,609) |
| Seeds (3) | 12345, 12346, 12347 |

7 methods × 3 budgets × 2 cohorts × 3 seeds = **126 runs**. Recipe mirrors the
STraTS SAND mortality sweep (lr 5e-4, batch 16, clip 0.3, adam wd=0, constant lr,
fp32, deterministic; vendi block pinned rbf/gamma=dim/zscore).

**wandb filter:** `config.data_name = mimic3_los`, `config.active_name = mimic3_med`,
`config.query_name` = a method. Because budget and cohort vary *inside* one
`active_name`, run names collide — pin `config.acq_size` (100/500/1000) and
`config.data.single_stay_only` (True/False) to isolate a single curve.

---

## Not counted here

`experiments/activelearning/test/`, `test_bandit_smoke/`, and the `ep-3` CIFAR
folders are smoke / debug runs, not real experiment settings. Config exists for
several regimes never run (e.g. `cifar100_*`, `isic19_*`, `miotcd_*`,
`mnist_batchbald*`) — no run folders on disk and nothing on wandb.
