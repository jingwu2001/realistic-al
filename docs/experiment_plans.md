# Experiment Plans

Forward-looking plan for the CIFAR Vendi experiments. **Part 1** records the
Vendi settings already run (so the gaps are explicit); **Part 2** is the plan for
what to run next.

**Scope:** CIFAR-10 (balanced) and CIFAR-10-imb (`rho=0.02`), regimes
`cifar10_low` (50/50) and `cifar10_med` (250/250), all `ep-200`, ResNet
(+ weighted loss for imb). Vendi coverage for the other datasets (P12, MIMIC)
lives in [experiment_settings.md](experiment_settings.md); this doc is CIFAR-only.

Seed pool is `{12345, 12346, 12347}`; "1 seed" = `12345` only.

---

## Part 1 — Vendi settings already run

### The Vendi grid axes

Every Vendi-family run is a point in this space (run-dir / wandb naming in
parentheses):

- **Embedding** (`query.vendi.use_grad`): features (`vendi`, "feat") or
  last-layer gradients (`gvendi`, "grad").
- **Rényi order q** (`query.vendi.q`): `q=1` (Shannon VendiScore, method name
  `vendi`/`gvendi`) or `q=2` (closed-form, method name `vendi-q2`).
- **Selection**: batch / one-shot top-k (default) or greedy **sequential**
  (`vendi-q2-sequential`, sub-batch `bs ∈ {1, 10, 50}`).
- **Normalization** (`query.vendi.normalization`): `none`, `l2`, `minmax`, `zscore`.
- **Kernel** (`query.vendi.kernel`): `cosine`, `linear`, `rbf`.
- **RBF gamma** (`query.vendi.gamma`): heuristics `dim` / `median`; early CIFAR
  runs also used fixed `{0.1, 1.0, 10.0, 1/1024}`. (`cosine`/`linear` take no gamma;
  under `l2` normalization `cosine`≡`linear`, so those pairs were skipped.)

### Coverage matrix

| Setting | q1 feat (`vendi`) | q1 grad (`gvendi`) | q2 feat | q2 grad | q2-sequential | Seeds |
|---|---|---|---|---|---|---|
| CIFAR-10 `low` | ✅ full norm×kernel | ✅ full norm×kernel | ❌ | ❌ | ❌ | 3 |
| CIFAR-10 `med` | ✅ full norm×kernel | ✅ full norm×kernel | ❌ | ❌ | ❌ | 3 |
| CIFAR-10-imb `low` | ⚠️ plain only (+partial grid) | ❌ | ❌ | ❌ | ❌ | 3 plain / 1 grid |
| CIFAR-10-imb `med` | ✅ norm×kernel grid | ✅ norm×kernel grid | ✅ 11 combos | ✅ 11 combos | ✅ 11 × bs{1,10,50} | **1** |

"full norm×kernel" = `{none, l2, minmax, zscore} × {cosine, linear, rbf-dim,
rbf-median}` (l2 only with rbf) = 14 combos per embedding, 28 Vendi-family
variants per setting.

### Per-setting detail

**CIFAR-10 balanced — `low` and `med`** (July 2026 sweep, commit `aef44a17`,
launched 2026-07-16, **3 seeds**)
- q=1 only, both embeddings (`vendi` feat + `gvendi` grad).
- Full norm×kernel grid: `none`/`minmax`/`zscore` × `{cosine, linear, rbf-dim,
  rbf-median}`, plus `l2 × {rbf-dim, rbf-median}`.
- Also present (legacy/exploratory, single-variant): imported `old_run` plain
  `vendi` and `bandit` with fixed rbf gammas `{0.1, 1.0, 10.0}` / cosine.
- **Not run:** any q=2, any sequential.

**CIFAR-10-imb — `low`** (imbalanced)
- Only the imported `old_run` plain `vendi` (default config) at 3 seeds is a real
  result; the July grid attempt on disk is a 1-seed fragment (`l2-rbf-{dim,median}`,
  `minmax-rbf-dim` only).
- **Not run:** `gvendi`, the full q1 grid at 3 seeds, any q=2, any sequential.

**CIFAR-10-imb — `med`** (the main HP-exploration setting, **all 1 seed = 12345**)
- **q=1**, both embeddings: norm×kernel grid (rbf `dim`/`median` + `cosine`;
  `minmax` also carries legacy fixed gammas `{1/1024, 0.1, 1.0, 10.0}`). No
  `linear` kernel here.
- **q=2**, both embeddings (`grad-{False,True}`, Aug 2026, commit `dd52e17b`,
  launched 2026-08-27): 11 norm×kernel combos each = `l2×{rbf-dim,rbf-median}`,
  `{minmax,none,zscore}×{cosine,rbf-dim,rbf-median}`. (~22 runs)
- **q=2-sequential**, both embeddings (Sep 2026, commit `dd52e17b`, launched
  2026-09-03): same 11 combos × `bs ∈ {1, 10, 50}`. (~62 runs)

### Key caveats

- **Balanced (low/med) = 3 seeds; all imb = 1 seed (12345).** The imb med grid is
  single-seed HP exploration, not replicated results.
- **q=2 and sequential exist only for imb `med`.** Balanced settings and imb `low`
  are q=1 only.
- Imported pre-wandb runs carry the `old_run` tag; exclude `test`/`wandb_test`.
- wandb filter per setting: `config.data_name` (`cifar10` / `cifar10_imb`) +
  `config.active_name` (`cifar10_low` / `cifar10_med`), then `config.query_name`
  and tags `kernel:…` / `norm:…` / `emb:…` / `q:…`. See
  [experiment_settings.md](experiment_settings.md#how-to-filter-runs-on-wandb).

---

## Part 2 — To run next

> Scaffold below lists the coverage gaps that fall out of Part 1. Prioritize /
> prune to taste — these are candidates, not commitments.

### Observed gaps

- **Seed replication for imb `med`** — the entire q1/q2/sequential grid is
  seed 12345 only. Pick the winning variant(s) and run seeds 12346/12347 for
  3-seed error bars.
- **imb `low` diversity coverage** — no `gvendi`, no full q1 grid at 3 seeds, no
  q2/sequential. This is the strongest imbalance regime; likely the most valuable
  gap to fill.
- **q=2 / sequential on balanced** — never run on CIFAR-10 balanced `low`/`med`
  (if a balanced q1-vs-q2 comparison is wanted).

### Plan

_TODO (yours): which variants × settings × seeds, priority order, launcher +
budget._

| Setting | Method / variant | Seeds to add | Priority | Launcher |
|---|---|---|---|---|
|  |  |  |  |  |
