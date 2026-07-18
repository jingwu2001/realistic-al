# Docs Index

Documentation for working on this repo (a fork of the realistic-al AL benchmark,
extended with the **vendi** query method, a LinUCB **bandit**, and **P12/time-series**
support). Written so a fresh session — human or model — can get productive fast.

## Reading order for a new session

1. **[handoff.md](handoff.md)** — current state of the project: what exists, what's
   in flight, what's uncommitted. Always start here.
2. **[agent_playbook.md](agent_playbook.md)** — *how* to do anything here: environment,
   verification ladder, launchers, plotting, wandb conventions, gotchas. Read before
   running or changing code.
3. **[roadmap.md](roadmap.md)** — *what* to do next: prioritized work items with
   detailed step-by-step instructions and acceptance criteria.
4. **[codebase_guide.md](codebase_guide.md)** — full architecture walkthrough
   (data/model/query layers, config groups, the AL loop). Read the section relevant
   to your task; the Quick Reference table at the end maps "I want to…" → files.

## Research direction docs

| Doc | What it is |
|---|---|
| [ideas.md](ideas.md) | Research ideas for vendi v2: factorized gradient kernels, qVS quality weighting, greedy submodular batches, dual/covariance formulation. The "why" behind the roadmap. |
| [gradient_implementation_plan.md](gradient_implementation_plan.md) | Concrete implementation plan for the gradient-kernel + qVS work (Ideas 1–3, 5). The roadmap's P2 item executes this. |
| [improve_logging.md](improve_logging.md) | Logging cleanup plan. **Mostly implemented** — `src/utils/wandb_utils.py` exists and does §1.1–1.4 and §2. Remaining items (batch-internal Vendi metric, quality/diversity factor logging) land with the gradient work. |

## Reference / theory docs

| Doc | What it is |
|---|---|
| [active_learning_methods.md](active_learning_methods.md) | Survey of AL objectives & estimators (BALD, BatchBALD, EPIG, BADGE, BAIT, EGL, …) with a unified notation. Use when comparing methods or writing paper text. |
| [information_theory_primer.md](information_theory_primer.md) | Entropy/MI primer keyed to Kirsch & Gal (arXiv:2208.00549). Background for the methods doc. |
| [../transformer_p12_data_processing.md](../transformer_p12_data_processing.md) | Step-by-step data flow of the P12 transformer pipeline (raw files → batches → `BayesianTransformerModel2` forward). Lives at repo root; candidate to move here. |
| [mimic3_sand_integration.md](mimic3_sand_integration.md) | MIMIC-III mortality task + SAND model integration (2026-07-13): files added, port decisions/deviations, launchers, verification, open items. |
| [gvendi_integration.md](gvendi_integration.md) | gvendi (gradient-Vendi) query method + query sweep launchers (2026-07-13): design decisions, shared kernel-machinery changes, verification findings, open items. Executes the first part of roadmap P2. |
| [sweep_run_tables.md](sweep_run_tables.md) | One-row-per-run tables (single seed) for the query sweeps on CIFAR-10(-imb)/CIFAR-100/P12/MIMIC, in launch order with per-block cost anchors — the pruning checklist. Generated from the launcher grids (2026-07-16). |

## Removed docs (recoverable from git history)

`integration.md` (upstream's "add your own method/dataset" guide) and
`p12_dataset.md` (P12 statistics) are deleted in the working tree but exist at
`HEAD` — recover with `git show HEAD:docs/integration.md` if needed. Their key
content is absorbed into codebase_guide.md and transformer_p12_data_processing.md.
