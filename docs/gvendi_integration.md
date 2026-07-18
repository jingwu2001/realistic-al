# gvendi Integration + Query Sweep Launchers (2026-07-13)

> **Update 2026-07-16 — vendi/gvendi merged via `use_grad`.** The separate
> `_get_gvendi` implementation and `query.gvendi.*` config block are gone.
> The vendi block now carries `use_grad` (+ `grad_embedding_type`,
> `labeled_true_labels`); `query=gvendi` is a pure config alias
> (`gvendi.yaml` inherits `vendi.yaml` and sets `vendi.use_grad: true`).
> A shared helper `query_diversity.get_vendi_embeddings(vendi_cfg, …)`
> selects features vs gradients and is used by both `_get_vendi` and the
> **bandit's diversity arm** — so `query=bandit ++query.vendi.use_grad=true`
> runs the bandit with gradient embeddings (wandb tag `bandit-grad-…`).
> Override gradient options via `++query.vendi.*` everywhere; the
> `query.gvendi.*` keys below are historical.
>
> The sweep launchers were subsequently re-specced (same date): baselines are
> now {random, entropy, bald, badge, batchbald, variationratios} (kcentergreedy
> and bandit dropped from the sweeps), and vendi runs the FULL
> use_grad × {rbf,linear,cosine} × {l2,minmax,zscore,none} grid — including the
> three mathematically identical cells (cosine,l2) == (cosine,none) ==
> (linear,l2). Run counts: cifar10 180, cifar100 270, p12 90, mimic basic 30,
> mimic kernels 144, test 8. The launcher-detail section below describes the
> earlier deduplicated grid.

Summary of the changes from the gvendi implementation session: the new
gradient-Vendi query method, the shared kernel-machinery changes it required,
the three query-sweep launchers, and everything found/fixed along the way.
Design source: [g_vendi.md](g_vendi.md) pseudo-code +
[gradient_implementation_plan.md](gradient_implementation_plan.md) (roadmap P2).

## What was built

1. **`query=gvendi`** — Vendi-score acquisition on BADGE-style last-layer
   gradient embeddings instead of penultimate features. Reuses the entire
   vendi pipeline (normalization, kernels, bordered-matrix eigendecomposition),
   so it supports the same `normalization` (l2/minmax/zscore/none) and
   `kernel` (rbf/cosine/linear) options.
2. **Linear kernel** for vendi, gvendi, and bandit (was rbf/cosine only).
3. **Query sweep launchers** for CIFAR-10, CIFAR-100, and P12: all baseline
   methods + vendi/gvendi/bandit over a kernel × normalization grid.

## The gvendi method

`src/query/query_diversity.py::_get_gvendi`:

- Embedding of sample x: flattened last-layer gradient
  g_x = (p_x − e_{y_x}) h_xᵀ, computed by the (extended) BADGE helper
  `get_grad_embedding`. **Labeled set uses ground-truth labels** (config
  `labeled_true_labels: true`), the **pool uses argmax pseudo-labels**
  (BADGE convention) — per the g_vendi.md pseudo-code.
- **Raw gradient norms are always returned** in `extra_info`, computed
  *before* normalization (under l2 they would degenerate to 1):
  `grad_norm_acq` / `grad_norm_else` ([max, min, median], same format as the
  score stats) and per-sample `grad_norms_acquired`. For the "linear"
  embedding ‖g_x‖_F = ‖p_x − e_y‖·‖h_x‖ — this is the s(x) quality signal
  for the future qVS work.
- Then identical to `_get_vendi`: `normalize_features` →
  `vendi_from_features` → top-`acq_size` by score.
- Config: [src/config/query/gvendi.yaml](../src/config/query/gvendi.yaml).
  Extra keys vs vendi: `grad_embedding_type` (`linear` = C·Z gradient,
  `bias` = C-dim error vector — the "error-vector norm only" ablation) and
  `labeled_true_labels`.
- wandb naming: `_gvendi_variant` in `src/utils/wandb_utils.py`, e.g.
  `gvendi-grad-lin-l2-q1.0` (emb tag becomes `gradbias` for
  `grad_embedding_type: bias`).

**Deliberate design choice:** gradients are *materialized* (N × C·Z), not the
factorized K^err ⊙ K^feat kernel from the plan §1.4. minmax/zscore normalize
per gradient *dimension*, so supporting all normalizations requires the
materialized embedding anyway; the factorization remains an acceleration for
the cosine/linear + {l2, none} subset (open item).

## Shared machinery changes

All in `src/query/query_diversity.py` unless noted:

- `vendi_from_features(vendi_cfg, …)` now takes the **method sub-config**
  (`cfg.query.vendi` or `cfg.query.gvendi`) instead of the full cfg; call
  sites in `_get_vendi` and `query_bandit.py` updated.
- `compute_kernel_matrix` gained `'linear'` (plain `x @ y.T`).
- Linear kernel correctness in the Vendi computation (its diagonal is ‖x‖²,
  not 1): new `kernel_self_similarity(kernel, x)` fills the bordered matrix's
  candidate diagonal entry, and the exact eigenvalue path now normalizes by
  the **trace** (`ev / ev.sum()`) instead of `/(L+1)` — bit-identical for
  unit-diagonal kernels (rbf/cosine), correct for linear.
- `get_grad_embedding` gained `use_true_labels` (default False = BADGE
  behavior unchanged) and now stores **detached** embeddings (previously a
  latent autograd-graph leak for non-`small_head` models).
- `query_bandit.py::calculate_vendi_score` routes through
  `compute_kernel_matrix` + trace normalization → bandit supports the linear
  kernel too (identical values for rbf/cosine, verified numerically).
- Small fixes: `NameError` in `vendi_from_features`' unsupported-gamma error
  message; two stale mocks in `tests/test_bandit_features.py` (fakes predated
  the `(scores, eig_time)` return and the 3-arg `calculate_vendi_score`) —
  these 6 tests were failing before this session.

## Sweep launchers

`launchers/exp_cifar10_query_sweep.py`, `exp_cifar100_query_sweep.py`,
`exp_p12_transformer_query_sweep.py`. Hyperparameters mirror each dataset's
`exp_*_basic` launcher (CIFAR-100 keeps its dropout-only-where-needed
convention; P12 uses the baleval + weighted-loss setup with `p12_med_bal`).

- Baselines (run once each): random, entropy, kcentergreedy, bald, badge.
  batchbald/variationratios deliberately excluded (repo baseline convention).
- vendi / gvendi / bandit × 10 kernel-normalization combos — redundant pairs
  dropped per plan §3.2: rbf×{l2,minmax,zscore,none} +
  {cosine,linear}×{minmax,zscore,none} (l2+cosine ≡ none+cosine,
  l2+linear ≡ cosine). γ fixed at 1.0 (rbf only).
- Run counts: 210 (CIFAR-10: 35 rows × 2 active × 3 seeds),
  315 (CIFAR-100: × 3 active), 105 (P12).
- Both `query.vendi.*` and `query.gvendi.*` keys are passed with identical
  aligned values (`++` overrides), so one row spec drives vendi, bandit
  (reads `query.vendi.*`) and gvendi (reads `query.gvendi.*`); the keys are
  inert for baselines.

Each file carries a local `QuerySweepLauncher(ExperimentLauncher)` subclass
(`launcher.py` deliberately untouched) handling two base-class limitations:

1. **Product explosion**: `BaseLauncher.parse_product` materializes the full
   cartesian product before filtering by `joint_iteration` — ~35⁶ combos with
   these aligned lists. The subclass zips each joint group into one compound
   axis first (same run list, instant).
2. **Naming**: the base class doesn't know gvendi, and its stripping of
   `_norm-/_kernel-/_gamma-` name parts for non-vendi queries is **dead code**
   (it edits the naming convention after `.`-removal, so the dotted patterns
   never match — pre-existing bug, worked around, not fixed in launcher.py).
   The subclass strips on the *formatted* name: baselines lose the inert
   kernel parts, cosine/linear lose the rbf-only gamma part.

Usage: `python launchers/exp_cifar10_query_sweep.py --debug` to print the
commands, drop `--debug` to launch, `--num_start/--num_end` to slice.

## Verification

- `tests/test_gvendi.py` (17 tests): all kernel × normalization combos, grad
  norms invariant to normalization, **l2+linear ≡ cosine exact score match**,
  true- vs pseudo-label gradients, `kernel_self_similarity`. Full suite:
  27 passed.
- End-to-end short AL loops (train → gvendi query → label → finalize, checked
  `extra_info_0.npz` contents): **P12 transformer** (linear+l2),
  **CIFAR-10** (rbf+minmax, full ~49.7k pool), **CIFAR-100** (cosine+zscore,
  `active.m=10000`). Launchers verified via `--debug` (counts + name checks).

### Findings

- **rbf + γ=1.0 saturates on gradient embeddings**: on CIFAR-10 (d=5120) all
  Vendi scores ≈ L+1 with a near-tied ranking. cosine/linear are the
  informative kernels for gvendi until a sensible γ is chosen (the median
  heuristic is itself inverted — roadmap P3).
- **CIFAR-100 timing**: L=500 → 501×501 eigendecompositions per candidate;
  ~88 s eig time for a 10k pool (RTX 5090), ≈7 min/round full pool.

## Not implemented (open items)

- **qVS (quality-weighted vendi/gvendi)** — plan §2. The s(x) signal (raw
  grad norms) is logged but unused in the score.
- **d×d eigendecomposition for the linear kernel** — only pays off when
  d < L+1; for gvendi d = C·Z ≫ L+1, so the bordered matrix is already the
  cheap side. The real win is the rank-1/secular-equation update (roadmap P4).
- **Factorized gradient kernel** (plan §1.4) and Fisher/multi-layer variants.
- Baselines batchbald/variationratios in the sweeps, γ sweep for rbf.
