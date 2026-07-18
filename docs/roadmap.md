# Roadmap — What Needs to Be Done (as of 2026-07-11)

Prioritized work items with step-by-step instructions. Each item states *why*,
*exactly what to change*, and *how to verify*. Work top-down unless Jing says
otherwise. When an item is finished, record the outcome (including negative
results) in the log at the bottom.

Priorities: **P0** unblocks everything · **P1** current experimental focus ·
**P2** the research core · **P3** correctness debt · **P4/P5** performance &
method extensions · **P6** polish.

---

## P0 — Commit the in-flight working tree

The working tree carries ~2 months of uncommitted changes mixed with cleanup.
Until this is committed, every wandb `session_tag` mislabels what actually ran.

Inventory (verify with `git status` — it may have moved on):

1. **Real fixes (commit first, one logical commit each):**
   - `src/main.py` + `src/utils/wandb_utils.py` — `_subset_targets` now falls
     back to `subset.targets` when the split is a raw dataset (CIFAR test set)
     rather than a `torch Subset`. Fixes a crash when logging label distributions
     on torchvision datasets.
   - `src/utils/wandb_utils.py` — run name now prefixed with `cfg.data.name`.
   - `launchers/launcher.py` — escape commas in `--notes` for Hydra.
   - `requirements.txt` — added `wandb`.
   - `analysis/plot_simple.py` — large refactor: structured folder-name parsing,
     family palettes + kernel/gamma markers, per-method gamma filters,
     P12 eval metrics (`EVAL_METRICS`), nested `commit-*` dir discovery.
2. **Cleanup deletions (separate commit):** `Raindrop` submodule,
   `check_experiments.ipynb`, `nohup.out`, `docs/assets/al_loop.png`,
   `experiments/activelearning/test/*` Hydra debris. For the submodule, remove
   properly: `git rm Raindrop` and drop its `.gitmodules` entry.
3. **Doc moves (separate commit):** `docs/handoff.md` was rewritten;
   `docs/integration.md` and `docs/p12_dataset.md` deleted (content absorbed —
   see docs/README.md). Consider `git mv transformer_p12_data_processing.md docs/`.
4. **Untracked**: `analysis/plots_simple/*` outputs and `20260408_Nano5_slide.pdf`
   should stay untracked (add `analysis/plots_simple/` to `.gitignore`).

Verify after committing: `git status` clean except intentional untracked;
rung 1–2 of the playbook verification ladder still pass.

---

## P1 — P12 imbalance experiment campaign (current focus)

**Why**: the last three commits built P12 launchers with different imbalance
settings; the runs (or their analysis) are the immediate next step. The research
question: *how do query methods rank on a clinically imbalanced task (P12
mortality, ~14% positive), and does the answer change with the imbalance-handling
strategy (weighted loss vs balanced sampling) and with an artificially long-tailed
pool?*

**Setup that already exists** (don't rebuild):

- Data: `data=p12_transformer_baleval` — balanced 50/50 test+val
  (`balanced_test_val: true`), natural pool. `data.imbalance=True, imb_factor=0.02`
  additionally long-tails the pool.
- Budget: `active=p12_med` (random 100 initial) or `p12_med_bal` (50/50 initial),
  `acq_size=100`, 10 iterations, `min_train=100`.
- Model: `model=transformer2` (`BayesianTransformerModel2`, dropout_p 0.5, k=10);
  checkpoint monitor is `val/auroc` (`src/trainer.py`).
- Launchers (7 query methods × 3 seeds × imbalance {off, on@0.02} = 42 runs each):
  - `launchers/exp_p12_transformer_baleval_weightedloss.py` — imbalanced batches,
    `model.weighted_loss=True`, `active=p12_med`.
  - `launchers/exp_p12_transformer_baleval_balancedsampling.py` —
    `data.balanced_sampling=True` (WeightedRandomSampler), `active=p12_med_bal`.

**Steps**:

1. `--debug` both launchers; eyeball the 84 commands (folder names must carry
   `bs-`/`wl-`/`imb-`/`imbf-` fields — they do via `naming_conv`).
2. Run one grid slot end-to-end first: `--num_start 0 --num_end 1 --test`.
   Confirm in wandb: `al/val_auroc` rises, `init_label_dist/*` matches the
   intended setting (L balanced for `p12_med_bal`, imbalanced for `p12_med`).
3. Launch both grids (locally or `--bsub`). GPU-hours are modest (50 epochs,
   small transformer); expect the vendi/bandit runs to dominate query time.
4. Plot per setting (playbook §4), headline metric **AUROC/AUPRC vs #labels**,
   not accuracy (balanced eval makes accuracy readable, but AUROC is the monitor):

   ```bash
   python analysis/plot_simple.py --prefix p12_wl  --dataset p12_transformer_baleval \
     --regime active-p12_med     --dir-filter wl-True
   python analysis/plot_simple.py --prefix p12_bs  --dataset p12_transformer_baleval \
     --regime active-p12_med_bal --dir-filter bs-True
   ```

5. Analysis writeup (new `docs/results_p12.md`): per setting, method ranking by
   AUBC of AUROC; whether vendi/bandit beat uncertainty baselines when the pool
   is long-tailed (diversity should matter more there); acquired-class
   composition over rounds (`stored.npz::added_labels` — does vendi acquire more
   minority samples than entropy?).

**Acceptance**: all 84 runs present in wandb with correct group/tags; plots +
`docs/results_p12.md` with the ranking table and 2–3 sentence takeaways per setting.

---

## P2 — Gradient-kernel + qVS vendi (the research core)

**Why**: current vendi scores pure feature diversity — no uncertainty signal, and
top-k selection is batch-redundant. [ideas.md](ideas.md) Ideas 1–3 fix the
representation; [gradient_implementation_plan.md](gradient_implementation_plan.md)
is the agreed implementation plan (greedy selection deliberately deferred).
This item translates that plan into concrete edits.

**Order of work** (from the plan, §4): factorized gradient kernel → qVS → sweeps
→ Fisher/multi-layer only if justified.

### 2a. Factorized gradient kernel

Files: `src/query/query_diversity.py`, `src/config/query/vendi.yaml`.

1. Config: add to `vendi.yaml`:
   ```yaml
   embedding: feat      # feat | grad          (fisher later)
   alpha: null          # null=pure; else blend alpha*K_grad + (1-alpha)*K_feat
   ```
2. New helper `get_embeddings_and_probs(model, loader)`: like `get_embeddings`
   (keep the tuple-input dispatch!) but also runs
   `probs = softmax(model.model.classifier(features), dim=1)` per batch.
   `model.model.classifier` is guaranteed by the same assertion used in
   `get_grad_embedding` (line ~172).
3. In `vendi_from_features`, when `embedding == "grad"`:
   - `a = (p - onehot(argmax p))`, row-L2-normalized → `K_err = A_x @ A_y.T`.
   - cosine: `K_grad = K_err ⊙ K_feat_cos`. RBF: expand
     ‖g_x−g_y‖² = 2 − 2·K_err⊙K_lin (unit-normalized factors) and exponentiate —
     see plan §1.4; keep everything float64 like the existing path.
   - `alpha` blend: `K = alpha*K_grad + (1-alpha)*K_feat` (always PSD).
   Implementation detail: the bordered-matrix loop only consumes `K_LL` and
   `K_UL`, so the *only* change is how those two matrices are produced — factor
   the kernel construction out of `vendi_from_features` into a function returning
   `(K_LL, K_UL)` and branch on `embedding` there.
4. **Wandb naming**: nothing to do — `_vendi_variant` in
   `src/utils/wandb_utils.py` already picks up `embedding`/`alpha`/`quality`.
5. **Tests** (this is the make-or-break step — add to `tests/test_vendi.py`):
   - *Exactness*: small MLP + random data; materialize BADGE gradients with the
     existing `get_grad_embedding(..., "linear")`; assert
     `cos(g_x,g_y) == K_err⊙K_feat` entrywise to ≤1e-5.
   - *PSD & unit diagonal* of `K_grad` on random inputs.
   - *α=0 recovers feat*: identical scores to `embedding: feat`.

### 2b. qVS quality weighting

1. Config: `quality: null  # null | gradnorm | errnorm | entropy | bald`,
   optional `quality_gamma: 1.0` (exponent on s).
2. Quality scores from the same forward pass:
   `s_gradnorm = ‖p−e_ŷ‖·‖h‖` (compute **before** feature normalization — L2
   normalization destroys ‖h‖), `s_errnorm = ‖p−e_ŷ‖`, `s_entropy` from probs;
   `bald` reuses `query_uncertainty` machinery (needs `model(x, agg=False)`).
3. Scoring rule = plan §2.2 **option 1** (labeled points get s=1):
   `score(u) = s(u)^γ · VS_q(L ∪ {u})` — i.e. multiply the existing per-candidate
   Vendi scores by `s(u)^γ` before the top-k. One line after the eig loop.
4. Log the two factors separately: add `al/query/quality_*` and keep raw vendi
   stats, so the balance is diagnosable (improve_logging.md §1.3 leftover).
5. Test: with `quality: gradnorm` and constant features, ranking must follow s;
   with constant s, ranking must equal pure vendi.

### 2c. Sweeps

Pattern a launcher on `launchers/exp_cifar10imb_basic_tune_kernel.py`; extend
`naming_conv` with `emb-{query.vendi.embedding}_a-{query.vendi.alpha}_s-{query.vendi.quality}`.
Axes (plan §3): kernel {cos, rbf} × embedding {feat, grad} × quality {none,
gradnorm, entropy} × q {1.0}, 3 seeds, CIFAR-10 first (`active=cifar10_low`),
then `cifar10_imb`, then P12. Mind the interaction table in plan §3.2 (e.g. L2
norm + cosine = no-op) to avoid wasted runs.

**Acceptance**: tests green; timing not worse than feat-vendi by >20%
(`timing.csv::eig_time_s` unchanged, only kernel construction adds work);
sweep plots show whether grad/quality variants beat feat-vendi and BADGE/BALD.

---

## P3 — Vendi correctness debt (fix before big sweeps)

All in `src/query/query_diversity.py`. These change scores, so land them
*before* P2's sweeps, in a tagged commit.

1. **Median heuristic is inverted** — **FIXED 2026-07-16**: gamma resolution
   now lives in `query_diversity.resolve_gamma` (shared by vendi/gvendi and
   the bandit), implementing `sigma = median(d); gamma = 1/(2*sigma**2)` plus
   a new `"dim"` heuristic (`gamma = 1/(2*D)`); unit-tested in
   `tests/test_gvendi.py::test_resolve_gamma`. Note: any pre-fix
   `gamma=median` runs used the inverted value — don't mix them with post-fix
   results.
2. **`approx: true` path normalizes a truncated spectrum** (~line 489): LOBPCG
   keeps top (L+1)/3 eigenvalues and renormalizes them to sum 1, whereas the
   exact path divides by the full trace (L+1). The truncated renormalization
   systematically inflates normalized eigenvalues and changes the entropy — this
   is why the degenerate-spectrum warning exists. Either divide by (L+1) and
   accept a defective distribution, or leave to P4 (better approximations) and
   document `approx: true` as unreliable. Don't use `approx: true` for results runs.
3. **Latent `NameError`** — **FIXED 2026-07-13** (subsumed by the
   `resolve_gamma` refactor above).
4. **Linear kernel missing** — **FIXED 2026-07-13**: `compute_kernel_matrix`
   supports linear; non-unit diagonal handled via `kernel_self_similarity` +
   trace-normalized eigenvalues (see gvendi_integration.md).
5. Cosmetic: `calculate_extra_info` key `"else"`/typo `unacquried` — renaming
   changes wandb keys (`al/query/unacquired_*` already maps them); leave keys,
   fix internals only if touching the file anyway.
6. **Test gap**: neither `tests/test_vendi.py` nor anything else pins down a
   *known-answer* Vendi score (e.g. K = I ⇒ VS = n; rank-1 K ⇒ VS = 1). Add
   these two closed-form cases — they catch normalization regressions from items
   1–2 automatically.

---

## P4 — Fast Vendi: integrate the `vendi-approx` findings

**Why**: exact scoring is O(U·L³) eigendecompositions per round; it already
needs batching on CIFAR and will not scale to larger L or pools (ideas.md §0, §5).

`vendi-approx/` (separate git repo, own `plan.md`) benchmarks approximation
schemes: stochastic SVD, Cauchy interlacing + secular equation (rank-1 update of
a pre-diagonalized K_LL — the right tool for the bordered-matrix structure),
Nyström, and combinations, measuring time, eigenvalue error, Vendi error, and
**rank correlation** of the induced acquisition ranking.

**Steps**:

1. Finish/read out the study (`vendi-approx/results.json`): pick the method with
   Spearman ≥0.99 vs exact at the biggest speedup.
2. Port the winner into `src/query/query_diversity.py` behind
   `approx: exact | secular | nystrom | lobpcg` (replacing today's boolean).
   The secular path: eigendecompose `K_LL` **once per round** (O(L³)), then each
   candidate is a symmetric rank-1 border — its spectrum solves the secular
   equation in O(L²); batch over candidates on GPU.
3. Acceptance test: on a real checkpoint's features (not random data), acquisition
   set overlap with exact ≥ 95% at acq_size 50/100; `eig_time_s` speedup logged.
4. Longer-term (only if L grows past ~2k): the dual/covariance formulation of
   ideas.md §5 — explicit feature maps from P2 make it available.

---

## P5 — Greedy batch selection (gated — do not start before P2+P4 land)

ideas.md Idea 4: top-k vendi acquires near-duplicates; greedy marginal-gain
selection with the (submodular) log-Vendi objective fixes it, and P4's secular
machinery makes it affordable (one re-diagonalization per greedy step). The gate
(per gradient_implementation_plan.md): timings first. Baselines to implement
alongside for ablation: score-then-diversify (top-4k shortlist → greedy inside)
and vendi-weighted k-means++. Headline experiment: redundancy gap vs acq_size
∈ {64, 256, 1024} — plus batch-internal Vendi of the acquired batch as the
mechanism metric (also an improve_logging.md leftover).

---

## P6 — Polish & robustness (background tasks, pick up opportunistically)

1. **Bandit context features doc**: `query/bandit.py` (LinUCB) + 6-D contexts in
   `config/query/bandit.yaml` (`mean_bald_*`/`vendi_*`, `cls_dist`, `t`, `bias`)
   are tested (`tests/test_bandit_features.py`) but undocumented — add a short
   section to codebase_guide.md §3.4 once the design settles.
2. **Reward design**: bandit reward = raw val-metric gain between rounds; it
   shrinks as learning saturates, biasing late rounds toward exploration
   (`alpha`-term dominates). Consider gain normalized by round or by random-arm
   baseline. Experiment, don't assume.
3. **eval_checkpoints.py** and the FixMatch branch are unexercised recently —
   before relying on them, smoke-test; they may have drifted (e.g. new tuple-input
   datasets are *not* handled in `models/fixmatch.py`).
4. **CI**: a GitHub Action running playbook rungs 1–2 (pytest + dry_run on
   cifar10/p12 configs, CPU, `use_wandb=false`) would catch most breakage. Needs a
   slim CPU env (the pinned torch 1.12 has CPU wheels) and a tiny data fixture —
   MNIST download is small enough; P12 needs a synthetic fixture.
5. **`datasets/` hygiene**: `datasets/__MACOSX`, duplicate `p12`/`P12data` roots —
   document which is live (`P12data/processed_data`) or clean up.

---

## Decision log / outcomes

Append entries here as items complete: date, item, what happened, links to
wandb groups or plots. Include negative results.

- *(empty — nothing completed since this roadmap was written, 2026-07-11)*
