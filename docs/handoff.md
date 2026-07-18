# Handoff — Context for Future Conversations

Start here, then see **[docs/README.md](README.md)** for the full docs index,
**[agent_playbook.md](agent_playbook.md)** for how to run/verify anything, and
**[roadmap.md](roadmap.md)** for the prioritized work plan.

## Project

Jing is developing an active learning (AL) method called **vendi**
(Vendi-score-based diversity acquisition) and is building a more sophisticated
version of it (gradient kernels + quality weighting — see
[gradient_implementation_plan.md](gradient_implementation_plan.md)). The code is
a fork of the "realistic-al" benchmark framework (PyTorch Lightning + Hydra).

Two experimental tracks:

1. **CIFAR-10** (balanced `cifar10` + long-tail `cifar10_imb`) — the original
   focus; kernel/gamma tuning launchers exist (`exp_cifar10*_tune_kernel.py`).
2. **P12 (PhysioNet 2012 ICU mortality, time-series, ~14% positive)** — the
   current focus per recent commits: `TimeSeriesDM`, transformer/GRU-D models,
   and launchers comparing imbalance-handling strategies (weighted loss vs
   balanced sampling, natural vs long-tailed pool). Running these is roadmap P1.
3. **MIMIC-III (in-hospital mortality, time-series, ~11.5% positive, N=44812)**
   — added 2026-07-13 with the **SAND** model (Song et al., AAAI 2018), both
   ported from Jing's STraTS fork at `~/Desktop/STraTS`. Data:
   `data/preprocess_mimic3_sand.py` converts STraTS's `mimic_iii.pkl` →
   `$DATA_ROOT/mimic3_sand/` (dense hourly (24, 387) = 129 vars × values/mask/
   delta + Age/Gender static; full-data normalisation stats, P12 convention).
   Run: `data=mimic3_sand` (or `mimic3_sand_baleval`) `model=sand`
   `active=mimic3_med[_bal]|mimic3_low`. Model lives in
   `models/networks/bayesian_sand.py` (backbone deterministic, MC-dropout
   head only; STraTS's always-on attention dropout bug fixed in the port).

A full architecture walkthrough lives in **[codebase_guide.md](codebase_guide.md)**;
below is the minimal map.

## Repo map (src/)

- `main.py` — AL entry point (Hydra, `config/config.yaml` defaults). Runs `active_loop`: for each of `num_iter` iterations → train fresh model → query pool → label → log to wandb (`al/*` metrics) + `stored.npz`/`test_metrics.csv`/`timing.csv`.
- `run_training.py` — plain training entry; also hosts `get_torchvision_dm()` (datamodule factory; despite the name it also builds `TimeSeriesDM` for ecg5000/p12) and `label_active_dm()` (initial label seeding).
- `trainer.py::ActiveTrainingLoop` — one AL iteration: builds `BayesianModule`, pl.Trainer, fits, reloads best-val checkpoint (monitor: `val/auroc` for p12/isic2016, `val/w_acc` for miotcd/ecg5000/isic2019, else `val/acc`), `.active_callback()` runs the `QuerySampler`.
- Data layer: `data/active.py::ActiveLearningDataset` (boolean labelled mask; `.pool`/`.labelled_set` use test transforms; `label()` takes **pool-relative** indices) ← wrapped by `data/data.py::TorchVisionDM` (images) or `data/timeseriesdata.py::TimeSeriesDM` (ecg5000/p12/p12_transformer, stratified 3-way split, optional `balanced_test_val`) ← both extend `data/base_datamodule.py::BaseDataModule` (train/val split, `min_train` oversampling, `pool_dataloader`/`labeled_dataloader`, `get_pool_indices`, optional `balanced_sampling`).
- Model layer: `models/abstract_classifier.py::AbstractClassifier` (pl.LightningModule; MC-dropout forward `k` samples, `get_features` → penultimate N×D features used by all diversity queries, optional `weighted_loss`) → `models/bayesian.py::BayesianModule` → networks in `models/networks/` (registry keyed on **filename**; CIFAR: `bayesian_resnet`; P12: `transformer`/`transformer2` (shared `bayesian_transformer.py`), `gru_d`; only the classifier head is Bayesian via `ConsistentMCDropout`).
- Query layer: `query/query.py::QuerySampler` dispatches by `cfg.query.name` → `query_uncertainty.py` (bald/entropy/variationratios/batchbald/random) or `query_diversity.py` (kcentergreedy/badge/**vendi**). All query code handles tuple inputs (P12 batches). `query/bandit.py` + `query_bandit.py` = LinUCB bandit with 6-D contexts (`config/query/bandit.yaml`), reward = val-metric gain, handled in `main.py`.
- `utils/wandb_utils.py` — all wandb wiring: run name `{data}/{method-variant}/seed-{seed}`, method-variant tags encode vendi/bandit hyperparameters (and already anticipate `embedding`/`alpha`/`quality` for the gradient work).
- FixMatch semi-supervised branch: `main_fixmatch.py`, `trainer_fix.py`, `models/fixmatch.py`, `data/sem_sl.py` (not recently exercised).

## Vendi method (current implementation)

`query/query_diversity.py::_get_vendi` + `vendi_from_features`:

1. Embed labelled (L) and pool (U) with `get_features`; normalize (`minmax` default; also l2/zscore/none).
2. Kernel matrices $K_{LL}$, $K_{UL}$ (RBF with $\gamma$, or cosine).
3. For each pool point: bordered $(L+1)\times(L+1)$ matrix, batched `eigvalsh` (or LOBPCG top-$(L+1)/3$ if `approx: true` — unreliable, see roadmap P3), eigenvalues normalized, Rényi entropy of order $q$, score $= \exp(H_q)$ (Vendi score).
4. Acquire top `acq_size` scores; `extra_info` logs score stats + `eig_time_s`, surfaced as `al/query/*` in wandb.

Config: `config/query/vendi.yaml` (`kernel: rbf`, `gamma: 1.0` or `median`, `q: 1.0`, `normalization: minmax`, `batch_size: 64`, `approx: false`). **Known issues** (median-heuristic gamma inverted, approx-path normalization, missing linear kernel) are documented with fixes in [roadmap.md](roadmap.md) P3.

`vendi-approx/` (separate embedded git repo) benchmarks eigenvalue-approximation schemes (secular equation, Nyström, …) to speed this up — roadmap P4.

## Typical runs

```bash
source .env && conda activate real-al

# CIFAR-10
python src/main.py data=cifar10 model=resnet active=cifar10_low query=vendi ++trainer.seed=12345
# active/cifar10_low: 50 initial balanced labels, acq_size 50, 10 iterations
# data=cifar10_imb: long-tail (exp, rho=0.02) variant

# P12 transformer
python src/main.py data=p12_transformer_baleval model=transformer2 active=p12_med_bal \
  query=vendi optim=adam ++model.dropout_p=0.5 ++trainer.max_epochs=50

# Grids go through launchers (always --debug first):
python launchers/exp_p12_transformer_baleval_weightedloss.py --debug
```

Needs env vars `DATA_ROOT`/`EXPERIMENT_ROOT` (`source .env`); wandb project
`realistic-al` (`++trainer.use_wandb=false` to disable; `trainer.dry_run=true`
for config/data checks). Full command/flag reference: [agent_playbook.md](agent_playbook.md) §1–2.

## Gotchas (verified in code)

- Pool indices shift after every `label()` call — never cache them across iterations.
- Model retrains from scratch each iteration; best-val checkpoint reloaded before querying.
- `min_train` oversampling (5500 CIFAR / 100 P12) means "epochs" are constant-size regardless of label count.
- CIFAR resnet config default `dropout_p: 0` → the k MC samples are identical; uncertainty methods need `dropout_p>0` (P12 configs use 0.5); vendi doesn't care.
- `cfg.active.num_labelled` is mutated (incremented) during the run.
- Query scoring uses test transforms (pool_specifics swap) and `model.eval()`; P12 datasets normalize internally and are wrapped without pool_specifics.
- Old pinned PL API (`pl.Trainer(gpus=…)`) — don't modernize.

## State as of 2026-07-11

- **Working tree is dirty** with ~2 months of uncommitted work: small fixes
  (label-dist logging on raw test sets, launcher notes escaping, run-name data
  prefix, `wandb` in requirements), a large `analysis/plot_simple.py` refactor
  (P12 metrics, kernel/gamma filtering), and cleanup deletions (Raindrop
  submodule, old docs, notebook, Hydra debris). **Roadmap P0 = commit this.**
- P12 imbalance-comparison launchers are ready; the runs/analysis are the next
  experimental step (**roadmap P1**).
- Gradient-Vendi is implemented as `query=gvendi` (2026-07-13): BADGE-style
  last-layer gradient embeddings through the shared vendi machinery — all
  normalizations, kernels rbf/cosine/linear (linear needed a proper kernel
  diagonal + trace-normalized eigenvalues), raw pre-normalization gradient
  norms always in `extra_info` (the future qVS quality signal). Tests in
  `tests/test_gvendi.py`; verified end-to-end on P12 transformer (linear+l2),
  CIFAR-10 (rbf+minmax), and CIFAR-100 (cosine+zscore, `active.m=10000`).
  Caveat for sweeps: rbf with `gamma: 1.0` saturates on the high-dim gradient
  embeddings (CIFAR-10: all scores ≈ L+1, near-tied ranking) — prefer cosine/
  linear or a much smaller gamma. qVS and the factorized-kernel acceleration
  remain (**roadmap P2**; plan in gradient_implementation_plan.md).
  Sweep launchers: `exp_{cifar10,cifar100,p12_transformer}_query_sweep.py`
  (baselines + vendi/gvendi/bandit × kernel×normalization grid; bandit's
  `calculate_vendi_score` now supports the linear kernel too).
  2026-07-16: vendi/gvendi merged — `query.vendi.use_grad` flips the embedding
  source (gvendi = config alias), and since bandit inherits the vendi block,
  `query=bandit ++query.vendi.use_grad=true` gives the bandit's diversity arm
  gradient embeddings. `query.gvendi.*` keys no longer exist.
  Full change summary: [gvendi_integration.md](gvendi_integration.md).
- MIMIC-III + SAND task added (2026-07-13, uncommitted): see track 3 above.
  Verified: dry run + 2-iteration AL smoke runs with `query=random` and
  `query=vendi`. Preprocessed data sits in `$DATA_ROOT/mimic3_sand/`
  (regenerable from `~/Desktop/STraTS/data/processed/mimic_iii.pkl`).
- improve_logging.md is essentially done (`utils/wandb_utils.py`); leftovers are
  folded into roadmap P2/P5.
- Tests: `tests/test_vendi.py`, `tests/test_bandit_features.py`,
  `src/test/test_chunked_pdist.py` — run via
  `conda run -n real-al python -m pytest tests/ -v`.
