# Agent Playbook — Working Effectively in This Repo

Operating manual for future sessions (Claude or otherwise). Follow this and you
will match the quality bar of past sessions: every claim verified in code, every
change validated with the verification ladder below, results reproducible from
launchers, and wandb runs that are filterable months later.

## 0. Session bootstrap

1. `source .env` — sets `DATA_ROOT` (=`datasets/`) and `EXPERIMENT_ROOT` (=`experiments/`).
   Nothing runs without these.
2. Use the **`real-al`** conda env for everything: `conda run -n real-al <cmd>` or
   `conda activate real-al`. (Sibling envs `real-al-2/-test/-time/…` exist; `real-al`
   is canonical per `CLAUDE.md`.)
3. Read [handoff.md](handoff.md) for current state, then the section of
   [codebase_guide.md](codebase_guide.md) relevant to your task.
4. Check `git status` — this repo often carries meaningful uncommitted work.
   Never assume HEAD == working tree.

Key pins: Python deps in `requirements.txt` (torch 1.12.0, torchvision 0.13,
torchmetrics 0.9.2, an older PyTorch Lightning). The PL API is the **old** one:
`pl.Trainer(gpus=…)`, `TQDMProgressBar`, `LightningDataModule` without
`setup()` laziness. Don't "modernize" Trainer calls — they'll break against the
pinned version.

## 1. The verification ladder

Run these in order after any change; each rung is cheap and catches a different class of bug.
All commands from the **repo root** unless noted.

| Rung | Command | Catches | Time |
|---|---|---|---|
| 1. Unit tests | `conda run -n real-al python -m pytest tests/ -v` | vendi scoring, bandit context features | seconds |
| 2. Config/data dry run | `conda run -n real-al python src/main.py data=cifar10 model=resnet active=cifar10_low query=vendi trainer.dry_run=true` | Hydra config errors, datamodule construction, label seeding. Logs L/U/val/test label distributions and exits before training. No wandb. | ~1 min |
| 3. Datamodule self-check | `cd src && conda run -n real-al python -m data.timeseriesdata` | ECG5000/P12 splits, batch shapes/dtypes, pool index round-trip, balanced/imbalanced distributions | ~2 min |
| 4. End-to-end smoke | `conda run -n real-al python launchers/smoke_test.py` (CIFAR-10, random query, 1 epoch, 1 AL iter, acq 2) — or run any real launcher with `--test` | full train→query→label loop, checkpointing, wandb wiring | few min |

Definition of done for a code change: rungs 1–2 pass always; rung 4 whenever the
change touches the loop, a model, or a query strategy. If you changed
query/vendi logic, also confirm one real acquisition produces sensible
`extra_info` (score max ≥ median ≥ min, nonzero `eig_time_s`).

To keep experiments out of the real results tree while testing, either use
`--dry_run` on a launcher (reroutes `trainer.experiments_root=/tmp/dry_run`) or
override `trainer.experiment_name=test`. Disable wandb with
`++trainer.use_wandb=false`.

## 2. Running experiments (launchers)

Launchers in `launchers/` don't train; they expand a grid and shell out to
`python src/main.py <hydra overrides>` per combination.

Anatomy (see `launchers/exp_p12_transformer_baleval_weightedloss.py` as the model):

- `config_dict` — Hydra **config-group** choices (`model=`, `query=`, `data=`, `active=`, `optim=`). Lists are swept (cross product).
- `hparam_dict` — `++key=value` overrides. Lists are swept.
- `joint_iteration` — lists of keys that iterate **together** (zip, not product), e.g. `["model.dropout_p", "query"]` pairs each query with its dropout.
- `naming_conv` — template for `trainer.experiment_name`; **this becomes the results folder path and is parsed by the plotting script**. Keep the `key-{value}` underscore convention (`acq-vendi`, `kernel-rbf`, `gamma-0.1`, `drop-0.5`…) or plots will silently skip your runs.

CLI flags (all launchers, via `ExperimentLauncher.add_argparse_args`):

| Flag | Effect |
|---|---|
| `-d` / `--debug` | print the generated commands, run nothing. **Always do this first.** |
| `--test` | quick smoke variant: 2 AL iterations, 3 epochs; tags wandb run `test` |
| `--dry-run` | passes `trainer.dry_run` behavior: label distributions only, results to `/tmp/dry_run` |
| `--num_start N --num_end M` | run only grid slots N..M (for splitting across machines) |
| `--notes "…"` | attached as wandb notes (commas are escaped for Hydra) |
| `-b` / `--bsub` | prefix commands with `~/run_active.sh` for the LSF cluster |

Direct single runs are fine too:

```bash
conda run -n real-al python src/main.py data=cifar10 model=resnet \
  active=cifar10_low query=vendi ++trainer.seed=12345 ++model.dropout_p=0
```

Hydra notes: config groups use `key=value`; ad-hoc/override keys use `++key=value`;
values containing commas must escape them (`\,`) or Hydra splits them.

## 3. Where results land, and how to read them

Each run writes to
`$EXPERIMENT_ROOT/activelearning/{experiment_name}/{experiment_id}/` where
`experiment_name` is the launcher's `naming_conv` path and `experiment_id` is a
timestamp (`2026-07-11_18-04-…`). Contents:

- `stored.npz` — val/test accuracy per AL iteration, acquired labels, requested pool indices, bandit arms (−1 when not bandit).
- `test_metrics.csv` — last-epoch test metrics per loop (row = AL iteration). For imbalanced datasets includes `test/auroc`, `test/auprc`, `test/f1_cls1`, `test/precision_cls1`, `test/recall_cls1`, `test/w_acc` from `ImbClassMetricCallback`.
- `timing.csv` — per-iteration `train_time_s`, `query_time_s`, `eig_time_s`.
- `extra_info_{i}.npz` — query score diagnostics per iteration (vendi/bandit).
- `loop-{i}/` — per-iteration TensorBoard + `metrics.csv` + checkpoints; `data_ckpt` = labelled-mask checkpoint.
- `.hydra/config.yaml` — the fully resolved config (ground truth for what ran).

**wandb** (project `realistic-al`): run name `{data}/{method-variant}/seed-{seed}`,
group `{data}/{active}/{method-variant}`, where method-variant encodes vendi/bandit
hyperparameters (e.g. `vendi-feat-rbf-minmax-q1.0`) — built in
`src/utils/wandb_utils.py::method_variant`. Per-iteration metrics live under
`al/*` keyed on `al_iter`; query-score diagnostics under `al/query/*`; initial
label distributions in `init_label_dist/*` summary keys. The full resolved config
is in `wandb.config`, so any hyperparameter is filterable. If you add a new swept
hyperparameter to vendi, extend `_vendi_variant` so run names disambiguate
(`embedding`/`alpha`/`quality` keys are already anticipated there).

## 4. Plotting / analysis

`analysis/plot_simple.py` walks `$EXPERIMENT_ROOT/activelearning/{dataset}/{regime}/…`,
parses experiment folder names (`acq-`, `kernel-`, `gamma-`, `drop-` fields), takes
the **last 3 seed dirs** per experiment (time-window filterable), and plots AL
curves (accuracy + AUROC/AUPRC/F1/precision/recall for the imbalanced datasets)
with AUBC values. Output goes to `analysis/plots_simple/{timestamp}/`.

```bash
conda run -n real-al python analysis/plot_simple.py \
  --prefix p12_wl --dataset p12_transformer_baleval --regime active-p12_med \
  --query-methods vendi bandit random bald \
  --dir-filter wl-True --note "weighted loss, imbalanced pool"
```

Useful flags: `--allowed-gammas` / `--acq-gammas acq:gamma` (filter kernel sweeps),
`--kernel-methods` (which methods may carry kernel variants), `--skip-before/--skip-after`
(timestamp windows), `--distinct-colors` (for kernel/gamma sweep plots),
`--acq-exclude-paths acq:substring` (drop runs from a bad commit). Method → key
mapping and palette are at the top of the file (`QUERYMETHODS`, `PALETTE`).

If a method is missing from a plot: check (1) its `acq-` key is in `QUERYMETHODS`,
(2) the folder isn't caught by `SKIP_DIR`, (3) it has ≥1 seed dir with `test_metrics.csv`.

## 5. Extending the framework (recipes)

The Quick Reference table in [codebase_guide.md](codebase_guide.md) §4 maps tasks
to files. Condensed:

- **New query strategy**: add to `NAMES` + dispatch in `src/query/query_diversity.py`
  (feature-based) or `query_uncertainty.py` (probability-based); add
  `src/config/query/<name>.yaml` with `name: <name>`. Diversity methods receive
  `(cfg, model, labeled_dataloader, unlabeled_dataloader, acq_size)` and return
  pool-loader-relative indices (+ scores, + optional `extra_info` dict).
- **New model**: one file per model name in `src/models/networks/` (the registry
  keys on the *filename*), `@register_model` on a builder, shared logic in a
  separate non-registered module (pattern: `bayesian_transformer.py` shared +
  `transformer.py`/`transformer2.py` thin wrappers); add `src/config/model/<name>.yaml`.
  Keep only the classifier head Bayesian (ConsistentMCDropout + Linear) for
  consistency with ResNet/GRU-D/Transformer.
- **New dataset**: image-style → `TorchVisionDM` (`src/data/data.py`); time-series →
  `TimeSeriesDM` (`src/data/timeseriesdata.py`, has the stratified 3-way split
  machinery); add `src/config/data/<name>.yaml` and, if budgets differ, a
  `src/config/active/<name>_*.yaml`. Datasets whose batches are tuples/lists
  (e.g. P12 transformer) are already handled by the query layer's
  `isinstance(x, (list, tuple))` input dispatch — keep that convention.
- **Change what "best model" means**: per-dataset monitor metric in
  `src/trainer.py::_init_ckpt_callback` (P12/ISIC2016 → `val/auroc`,
  miotcd/ecg5000/isic2019 → `val/w_acc`, else `val/acc`).

## 6. Gotchas (verified in code — re-verify before relying on them)

Carry-overs from the AL loop (details in codebase_guide.md §Gotchas):

- `label()` takes **pool-relative** indices; the pool re-indexes after every
  acquisition. Never cache pool indices across iterations.
- Model retrains **from scratch** each AL iteration; best-val checkpoint reloaded before querying.
- `min_train` oversampling (5500 for CIFAR, 100 for P12) makes "epochs" constant-size.
- `dropout_p: 0` ⇒ all k MC samples identical ⇒ BALD/entropy degenerate. Vendi
  (features only) is unaffected. P12 configs use `dropout_p: 0.5`, CIFAR resnet default is 0.
- `cfg.active.num_labelled` is mutated (incremented) inside `active_loop`.
- Pool/labelled loaders use test transforms; the train loader uses train augments.

Operational:

- **Run everything from the repo root** — launchers use relative `src/main.py`;
  the src-internal imports assume `src/` on `sys.path` (tests do `sys.path.append("src")`).
- Hydra changes the working directory to the run dir at `@hydra.main` entry;
  relative writes in `active_loop` (e.g. `timing.csv`) land there, not in the repo.
- The wandb run is created **once per experiment** in `main()`; per-iteration
  `WandbLogger(experiment=wandb_run)` reuses it — don't call `wandb.init` elsewhere.
- `trainer.dry_run=true` skips wandb + training; `trainer.use_wandb=false` trains without wandb.
- Old PL pin: `pl.Trainer(gpus=…)`, `weights_only=False` torch.load — keep as is.
- `vendi-approx/` is a **separate git repo** (embedded); `Raindrop` was a
  submodule, currently deleted in the working tree. Don't `git add -A` blindly
  from the root; review what you're staging.
- Some datasets under `datasets/` are gitignored/local; P12 processed data lives
  at `datasets/P12data/processed_data/` (11 988 samples).

## 7. Quality bar for research code here

- Before designing anything vendi-related, read [ideas.md](ideas.md) and
  [gradient_implementation_plan.md](gradient_implementation_plan.md) — decisions
  (e.g. "greedy is out of scope until timings exist") are already made there.
- Math claims get verified numerically in a unit test before they ship (e.g. the
  gradient-kernel factorization must be checked against materialized BADGE
  gradients, bit-for-bit up to float tolerance).
- Every new hyperparameter: config default + wandb name/tag support + a note in
  the relevant doc. A sweep that can't be told apart in wandb is a wasted sweep.
- Never change scoring code silently between sweep batches — results become
  incomparable. Tag runs (session tag already includes the git commit) and commit
  before launching big sweeps.
- Report negative results in the docs (append to roadmap.md item log) — "tried X,
  no gain, because Y" saves the next session a week.
