# Codebase Guide — Active Learning Framework (`src/`)

This codebase is a fork/extension of the **"realistic-al"** active learning benchmark framework, built on **PyTorch Lightning + Hydra**. It runs pool-based active learning: train a model on a small labelled set, score the unlabelled pool with a query strategy, label the top samples, repeat. Your method (**vendi**) is one of the query strategies, implemented in `query/query_diversity.py`.

The walkthrough below uses the CIFAR-10 experiment as the running example, e.g.:

```bash
python main.py data=cifar10 model=resnet active=cifar10_low query=vendi
```

---

## 1. The Full Pipeline (CIFAR-10 example)

### 1.1 Entry point and configuration

`main.py` is the entry point for AL experiments. It's decorated with `@hydra.main(config_path="./config", config_name="config")`, so the run is fully described by a composed Hydra config with these groups:

| Config group | Example file | What it controls |
|---|---|---|
| `trainer` | `config/trainer/base.yaml` | seed, epochs (200 max / 50 min), batch size 64, GPU count, wandb, paths (`DATA_ROOT`, `EXPERIMENT_ROOT` env vars) |
| `data` | `config/data/cifar10.yaml` | dataset name, shape `[32,32,3]`, 10 classes, `val_split: 5000`, normalization mean/std, transform names (`cifar_basic` train / `basic` test) |
| `model` | `config/model/resnet.yaml` | `bayesian_resnet`, base `resnet18`, lr 0.1, wd 0.005, `dropout_p`, `k: 50` (MC samples) |
| `optim` | `config/optim/sgd_cosine.yaml` etc. | optimizer + LR scheduler choice |
| `active` | `config/active/cifar10_low.yaml` | AL budget: `num_labelled: 50` (initial), `acq_size: 50`, `num_iter: 10`, `balanced: True`, `min_train: 5500`, `m` (pool subsampling, Null = full pool) |
| `query` | `config/query/vendi.yaml` | strategy name + hyperparams (vendi: `kernel: rbf`, `gamma`, `q`, `normalization: minmax`, `batch_size: 64`, `approx`) |

So `active=cifar10_low query=vendi` means: start with 50 balanced labels, do 10 AL iterations acquiring 50 samples each with the vendi strategy (ending at 550 labels).

### 1.2 `main()` — setup

1. `setup_logger()`, print config, `utils.set_seed(cfg.trainer.seed)`.
2. If `query.name == "bandit"`, build a `BanditManager` (contextual bandit that switches between uncertainty and diversity arms).
3. Unless `dry_run`, initialize a **wandb run** with group `{data}/{active}/{query}` and per-iteration metrics under the `al/*` namespace keyed on `al_iter`.
4. Call `active_loop(...)`.

### 1.3 `active_loop()` — the AL outer loop

```
datamodule = get_torchvision_dm(cfg)          # builds TorchVisionDM (see §2)
label_active_dm(cfg, num_labelled, balanced, datamodule)   # seed labels
for i in range(num_iter):
    loop = ActiveTrainingLoop(cfg, count=i, datamodule=datamodule, ...)
    loop.main()                                # train from scratch + test
    active_store = loop.active_callback()      # run query strategy on pool
    datamodule.train_set.label(active_store.requests)   # "oracle" labels them
    cfg.active.num_labelled += acq_size
```

Details per iteration `i`:

1. **Fresh model every iteration.** `ActiveTrainingLoop` (in `trainer.py`) deep-copies the datamodule and constructs a brand-new `BayesianModule` — there is no warm-starting across AL iterations.
2. **`loop.main()`**: saves a checkpoint of the labelled mask (`data_ckpt`), builds a `pl.Trainer`, fits, reloads the *best* checkpoint (selected by `val/acc` for CIFAR-10), then runs test. Logs go to `{experiment_root}/{experiment_name}/{experiment_id}/loop-{i}/` (TensorBoard + `metrics.csv` + wandb).
3. **Bandit reward bookkeeping** (only when `query=bandit`): reward for the arm picked in iteration `i-1` is the val-accuracy gain between iterations.
4. **Pool-exhaustion guard**: if the remaining pool is smaller than `acq_size`, stop early.
5. **Query**: `loop.active_callback()` builds a `QuerySampler` (or `BanditQuerySampler`) and calls `active_callback(datamodule)`, which returns an `ActiveStore` dataclass (`requests` = pool indices to label, `labels`, `n_labelled`, val/test accuracy of the current model, `extra_info` — for vendi this carries score stats and `eig_time_s`).
6. **Labelling**: `datamodule.train_set.label(requests)` flips the boolean mask — the "oracle" is just the ground-truth CIFAR-10 labels already present in the dataset.
7. **Logging**: train/query/eig timings and per-loop metrics logged to wandb; label distributions of L, U, val, test are logged at the start.

After the loop, results are persisted in the Hydra run dir: `timing.csv`, `test_metrics.csv` (last-epoch test metrics per loop), `stored.npz` (val/test acc curves, acquired labels, requested indices, bandit arms), and `extra_info_{i}.npz` per iteration.

### 1.4 One training run, start to finish — a walkthrough of `ActiveTrainingLoop`

This section traces a single AL iteration's training run chronologically through the code. The frame is iteration `i` of `active_loop` (`main.py`):

```python
training_loop = ActiveTrainingLoop(cfg, count=i, datamodule=datamodule,
                                   base_dir=os.getcwd(), wandb_run=wandb_run, ...)
with Timer() as train_timer:
    training_loop.main()
```

Two calls: the constructor builds every component; `main()` executes them. (`base_dir` = `os.getcwd()` is the Hydra run dir — Hydra chdirs into `{experiments_root}/{experiment_name}/{experiment_id}`, so `loop.log_dir` = that dir + `/loop-{i}`, the same place the loggers write to.)

#### Step 0 — `ActiveTrainingLoop.__init__` (`trainer.py`): build everything up front

The constructor runs these in order:

1. `self.datamodule = deepcopy(datamodule)` — the run trains on a **snapshot** of the datamodule. No data is loaded or split here: `TorchVisionDM` already built all datasets in *its* `__init__` (via `_setup_datasets()`: download if missing → 45k/5k train/val split → optional imbalancing → wrap in `ActiveLearningDataset`), back when `active_loop` called `get_torchvision_dm(cfg)`. The deep copy carries those datasets plus the current labelled mask; `label()` later mutates only the original.
2. `self._init_model()` → **model initialization**. One line: `self.model = BayesianModule(self.cfg)`. Its `__init__` (`models/bayesian.py`) does:
   - `self.save_hyperparameters(config)` — the full Hydra config becomes `self.hparams` (and is embedded in every checkpoint).
   - `self.model = build_model(config, num_classes, data_shape)` (`models/networks/build.py`) — looks up `cfg.model.name` (e.g. `bayesian_resnet`) in the registry and constructs the network with **freshly random weights**. A brand-new model every AL iteration — no warm start.
   - `self.k = cfg.model.k` — the MC-sample count used at eval time (e.g. 50).
   - Optionally `self.load_from_ssl_checkpoint()` (if `model.load_pretrained`) and `self.init_ema_model(config.model.use_ema)` (frozen deep copy of the network + an `EMAWeightUpdate` hook).
   - If `model.weighted_loss`: installs a *placeholder* uniform-weight `NLLLoss` (real weights are computed in step 4, once the labelled set is known).
   - Note what does **not** exist yet: no optimizer (Lightning creates it inside `fit`, step 5b) and no metrics (created in `setup_data_params`, step 4).
3. `self.ckpt_callback = self._init_ckpt_callback()` — a `pl.callbacks.ModelCheckpoint` with `dirpath=loop-{i}/checkpoints`. The monitored metric is chosen by dataset name: `val/acc` (max) by default, `val/auroc` for `isic2016`/`p12`, `val/w_acc` for `isic2019`/`miotcd`/`ecg5000`; `train/acc` if there is no val loader.
4. `self.callbacks = self._init_callbacks()` — `LearningRateMonitor`, the checkpoint callback from step 3, `EarlyStopping("val/acc")` *only if* `trainer.early_stop` (default False), dataset-specific metric callbacks (`ImbClassMetricCallback` adds `val/w_acc` & co. for the imbalanced datasets; `ISIC2016MetricCallback`), and a `TQDMProgressBar`.
5. `self.loggers = self._init_loggers()` — three Lightning loggers that later receive every `self.log(...)` call:
   - `TensorBoardLogger(save_dir=experiments_root, name=experiment_name/experiment_id, version="loop-{i}")`
   - `CSVLogger(...)` with the same name/version → writes `loop-{i}/metrics.csv`
   - if wandb is on, `WandbLogger(experiment=self.wandb_run)` — it **reuses** the run created in `main()` (no new `wandb.init`), so the wandb step counter keeps increasing across AL iterations instead of resetting.

#### `main()` — five calls in order

```python
def main(self):
    self._setup_log_struct()                                   # step 1
    self.datamodule.train_set.save_checkpoint(self.data_ckpt_path)  # step 2
    self._init_trainer()                                       # step 3
    self._fit()                                                # steps 4–6
    if self.trainer.interrupted: return                        # (active_loop also aborts)
    if self.cfg.trainer.run_test:
        self._test()                                           # step 7
```

#### Step 1 — `_setup_log_struct()`: create the loop dir, write `meta.json`

Creates `loop-{i}/` if needed and saves `meta.json` (timestamp + git commit/diff state via `log_git`). This is the first write of the run — everything else lands in the same directory.

#### Step 2 — `train_set.save_checkpoint(data_ckpt)`: persist the labelled mask

`ActiveLearningDataset.save_checkpoint` dumps the boolean labelled mask to `loop-{i}/data_ckpt`, so it is always reconstructible which samples were labelled when this model was trained.

#### Step 3 — `_init_trainer()`: build the `pl.Trainer`

Constructed directly from `cfg.trainer`: `max_epochs: 200`, `min_epochs: 50`, `check_val_every_n_epoch: 1`, `precision: 32`, `gradient_clip_val: 0`, `benchmark=True` unless `deterministic`, and — crucially — `logger=self.loggers` and `callbacks=self.callbacks` from step 0, which is how the checkpointing and logging behavior gets wired into the fit.

#### Step 4 — `_fit()`, before Lightning takes over

```python
datamodule = self.model.wrap_dm(self.datamodule)  # identity for BayesianModule
self.model.setup_data_params(datamodule)          # manual pre-fit hook
self.trainer.fit(model=self.model, datamodule=datamodule)
```

- `wrap_dm` is a no-op here (FixMatch overrides it to swap in the semi-supervised dual loaders).
- `setup_data_params(dm)` (`abstract_classifier.py`) is called *manually* before `fit` because two things must be known first:
  - It calls `dm.train_dataloader()` once just to take `len(...)` → `self.train_iters_per_epoch`. The cosine scheduler (created later inside fit) needs this to size its warmup/decay in steps.
  - If `model.weighted_loss`: reads the labelled targets from `dm.train_set.targets`, computes sklearn-style balanced class weights, and replaces `self.loss_fct` with a weighted `NLLLoss`.
  - Lazily creates the torchmetrics now that `num_classes` is known: `Accuracy` ×3 (train/val/test) and `MulticlassRecall(average="macro")` (= balanced test accuracy).

#### Step 5 — inside `trainer.fit`: what Lightning calls, in order

**(a) Data loading.** Lightning requests the loaders from the datamodule at fit start. `TorchVisionDM` just forwards to `BaseDataModule.get_dataloader`:

- `train_dataloader()` → `get_dataloader(self.train_set, mode="train")`. `self.train_set` is the `ActiveLearningDataset`, whose `__getitem__`/`__len__` expose **only the labelled samples, with train augmentations** (`cifar_basic`: random crop + flip + normalize). Sampler choice: if the labelled set is smaller than `min_train` (5500), a `RandomFixedLengthSampler` oversamples it to 5500 draws per epoch (constant gradient-step count regardless of label budget), or a class-balanced `WeightedRandomSampler(num_samples=min_train)` when `balanced_sampling` is on. Common settings: `batch_size: 64`, `num_workers: 16`, persistent workers, `seed_worker` for reproducible augmentations.
- `val_dataloader()` → `get_dataloader(self.val_set, mode="test")` — the 5k val split with **test transforms**, sequential, no `drop_last`.

**(b) Optimizer construction.** Lightning now calls `model.configure_optimizers()` (`abstract_classifier.py`): SGD (momentum/nesterov) or Adam over parameter groups built by `exclude_from_wt_decay` (bn/bias excluded from weight decay if `exclude_bn_bias`; optional frozen encoder or separate encoder/head LRs for `finetune`). For `cosine_decay`, the `LambdaLR` warmup+cosine schedule is sized as `warmup_epochs × train_iters_per_epoch` / `max_epochs × train_iters_per_epoch` — the numbers recorded in step 4 — and steps **per optimizer step**, not per epoch.

**(c) `on_fit_start`.** Logs the hparams to TensorBoard with `val/acc`/`test/acc` metric placeholders (this is what makes the TB hparams tab work). Lightning then runs its standard 2-batch validation sanity check before epoch 0.

**(d) The epoch loop** (up to `max_epochs: 200`; with `early_stop: False` it always runs the full 200). Each epoch:

1. `on_train_epoch_start` — resets `acc_train`; puts the frozen encoder and the EMA model (if any) in eval mode.
2. **Training batches.** For every batch `(x, y)` from the train loader, Lightning calls `BayesianModule.training_step(batch, batch_idx)`, which is just `self.step(batch, k=1)`:
   - `step` calls `self.forward(x, k=1)` → `select_forward_model()` returns `self.model` (we're training) → one stochastic forward pass (a single MC-dropout sample; with `dropout_p: 0` an ordinary forward) → `mc_nll` reduces to a plain `log_softmax` since `k=1`.
   - `loss = self.loss_fct(logprob, y)` — NLL on the log-probs.
   - `self.log("train/loss", loss)` (per step → hits TB, CSV, wandb) and `acc_train.update(preds, y)`.
   - On the very first batch of epoch 0, `visualize_inputs` writes an 8×8 grid of the (augmented) input images to TensorBoard under `train/data`.
   - Lightning then does backward + optimizer step, the cosine scheduler steps, `on_train_batch_end` updates the EMA weights (if EMA is on), and `LearningRateMonitor` records the LR.
3. **Validation** (`check_val_every_n_epoch: 1` → every epoch): `on_validation_epoch_start` resets `acc_val`, then per batch `AbstractClassifier.validation_step` calls `self.step(batch)` with `k=None` → **defaults to the full `self.k`** (e.g. 50):
   - `forward(x, k=50, agg=True)`: `select_forward_model()` now returns the EMA model if one exists, else the raw model; the nn.Module-level `BayesianModule` runs the deterministic body once, replicates features ×k, runs the MC-dropout head → `B × 50 × C` logits; `mc_nll` aggregates them to marginal log-probs ($\log\frac1k\sum_j\mathrm{softmax}(z_j)$ via logsumexp). Dropout stays active in eval by design (`ConsistentMCDropout`).
   - Logs `val/loss` (epoch-aggregated) and, in `on_validation_epoch_end`, `val/acc`.
4. **After validation, the callbacks fire:** `ModelCheckpoint` compares the fresh `val/acc` (or `val/auroc`/`val/w_acc`) against the best so far and, if improved, writes `loop-{i}/checkpoints/epoch=...-step=....ckpt` (only the best is kept; `save_last: False`). `EarlyStopping` checks its patience if enabled. `CSVLogger` appends the epoch's row to `metrics.csv`; the same scalars stream to TensorBoard and wandb.
5. `on_train_epoch_end` logs `train/acc` (the epoch-accumulated metric).

#### Step 6 — back in `_fit()`: reload the best checkpoint

```python
if not fast_dev_run and cfg.trainer.load_best_ckpt:   # default True
    best_path = self.ckpt_callback.best_model_path
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    self.model.load_state_dict(ckpt["state_dict"], strict=True)
```

The in-memory model is overwritten with the **best-val-metric epoch's weights** — so everything downstream (the test pass in step 7 *and* the query in `active_callback`) uses the best model, not the last epoch's. `_fit` ends with `gc.collect()` + `torch.cuda.empty_cache()`.

#### Step 7 — `_test()`: `trainer.test(model, datamodule)`

Lightning requests `test_dataloader()` (held-out test set, test transforms, sequential) and runs one pass of `AbstractClassifier.test_step` per batch — identical mechanics to validation (full `k` MC samples, `mc_nll` aggregation, EMA model if present) with two additions: `bacc_test` (macro recall = balanced accuracy) and the dataset-specific callback metrics (`ImbClassMetricCallback` → `test/w_acc` etc.). `on_test_epoch_end` logs `test/acc` and `test/bacc`. These land as the **final rows of `metrics.csv`** — which is exactly what `active_loop` later harvests: it takes each loop's last `test/*` row into `test_metrics.csv`.

#### Step 8 — after `main()` returns: per-iteration bookkeeping in `active_loop`

Back in `main.py`, with the trained loop object in hand:

- the `Timer` around `loop.main()` yields `train_time_s`;
- the best val score is read from `loop.ckpt_callback.best_model_score` (this is the bandit reward signal);
- `loop.log_save_dict()` writes `save_dict.json` if anything was collected;
- after the query, `log_al_iteration(...)` (`utils/wandb_utils.py`) re-reads `loop-{i}/metrics.csv` and logs **one summary point** to wandb under the `al/*` namespace keyed on `al_iter`: `val/*` columns aggregated as *max over epochs* (≈ the best checkpoint), `train/*`/`test/*` as the *last* non-NaN value, plus `al/n_labelled`, timings (`al/train_time_s`, `al/query_time_s`, `al/eig_time_s`), and query-score diagnostics under `al/query/*`.

#### Recap — what gets logged where

Everything for iteration `i` lives in `{experiments_root}/{experiment_name}/{experiment_id}/loop-{i}/`:

| Artifact | Written in step | Contents |
|---|---|---|
| `meta.json` | 1 | timestamp + git commit/status |
| `data_ckpt` | 2 | labelled-mask state at the start of the iteration |
| `checkpoints/*.ckpt` | 5.4 | best model by the monitored val metric (weights + hparams) |
| TensorBoard events | 5 throughout | all `self.log` scalars, hparams table, input-image grids, LR curve |
| `metrics.csv` | 5.4 / 7 | every logged scalar per epoch; test rows appended last; re-read for wandb `al/*` and `test_metrics.csv` |
| `save_dict.json` | 8 | optional extra run info |
| wandb (live) | 5 throughout | same scalars as TB, on the shared run with a continuous step counter |
| wandb (`al/*`) | 8 | one aggregated point per AL iteration, keyed on `al_iter` |

### 1.5 Other entry points

- `run_training.py` — single supervised training run (no AL loop). Also hosts `get_torchvision_dm()` and `label_active_dm()`, which `main.py` imports.
- `main_fixmatch.py` / `run_training_fixmatch.py` / `trainer_fix.py` — same structure but semi-supervised: `FixMatchTrainingLoop` swaps the model for `FixMatch` (`models/fixmatch.py`), which via `wrap_dm()` gives the datamodule a concatenated labelled+unlabelled dataloader (`data/sem_sl.py`, `ConcatDataloader`).
- `eval_checkpoints.py` — post-hoc evaluation of saved checkpoints.

---

## 2. The Data Layer

There are four layers of abstraction, from bottom to top:

```
torchvision CIFAR10                      (raw dataset, has .targets)
  └─ ActiveSubset                        (data/utils.py — Subset + .targets/.transform passthrough)
       └─ ActiveLearningDataset          (data/active.py — labelled/pool bookkeeping)
            └─ TorchVisionDM             (data/data.py — LightningDataModule)
                 extends BaseDataModule  (data/base_datamodule.py — loaders, splits)
```

### 2.1 `data/active.py` — `ActiveLearningDataset`

Adapted from BaAL. Wraps the training dataset with a **boolean mask `labelled`** over all indices ("oracle" indices). Key API:

- `__getitem__`/`__len__`: the dataset *behaves as the labelled set* — Lightning's train loader iterates only labelled samples (with train-time augmentations).
- `pool` (property): deep-copies the underlying dataset, applies `pool_specifics` (for CIFAR-10 this **replaces train transforms with test transforms** — important: query scoring sees un-augmented images), and returns an `ActiveSubset` of the unlabelled indices.
- `labelled_set` (property): same but for labelled indices — also with test transforms. This is what vendi/coreset/badge use as the "labelled" side.
- `label(index)`: **index is relative to the pool**, translated internally to oracle indices (`_pool_to_oracle_index`). This is why acquiring changes pool indexing every iteration.
- `label_randomly(n)` / `label_balanced(n_per_class, num_classes)`: initial seeding.
- `state_dict`/`save_checkpoint`: persist the mask per AL iteration (saved as `data_ckpt` by the trainer).

### 2.2 `data/base_datamodule.py` — `BaseDataModule`

A `pl.LightningDataModule` base class holding split and dataloader logic:

- `_split_dataset()` / `_get_splits()`: train/val split. For CIFAR-10, `val_split: 5000` → 45 000 train (becomes the AL pool universe) + 5 000 val, split randomly with the run seed. Optional `val_size` sub-selects a smaller, class-balanced validation set.
- `get_dataloader(dataset, mode="train")`: the important trick is **`min_train` (5500)**. If the labelled set is smaller than `min_train`, it uses `RandomFixedLengthSampler` to oversample it to 5500 samples per "epoch" — so an epoch has a constant number of gradient steps even with 50 labels. Optional `balanced_sampling` swaps in a `WeightedRandomSampler` (class-balanced) for imbalanced setups.
- `pool_dataloader(batch_size, m)`: loader over `train_set.pool` (no shuffle, test transforms). If `m` is set, a random subset of the pool of size `m` is scored instead of the full pool; it records `self.indices` so that…
- `get_pool_indices(inds)`: …maps loader-relative indices back to pool indices.
- `labeled_dataloader()`: loader over `train_set.labelled_set` (test transforms, no shuffle) — used by diversity queries.

### 2.3 `data/data.py` — `TorchVisionDM`

The concrete datamodule for image datasets (MNIST, **CIFAR-10**, CIFAR-100, FashionMNIST, ISIC, MIO-TCD). Its `_setup_datasets()`:

1. Downloads CIFAR-10 if needed.
2. `train_set` = CIFAR10(train=True, transform=**train_transforms**) → `_split_dataset(train=True)` → 45k `ActiveSubset`.
3. If `imbalance: True` (the `cifar10_imb` config): `create_imbalanced_dataset()` (`data/longtail.py`) subsamples classes with an exponential profile (`rho: 0.02` → rarest class has 2% of the largest).
4. If `active: True`: wraps in `ActiveLearningDataset(train_set, pool_specifics={"transform": test_transforms})`.
5. `val_set` = a second CIFAR10(train=True) instance with **test transforms**, split with the same seed → the complementary 5k. `test_set` = CIFAR10(train=False).

Transforms come from `data/transformations.py::get_transform` by name (`cifar_basic` = random crop + horizontal flip + normalize; `basic` = to-tensor + normalize; also randaugment variants for FixMatch in `data/utils.py`).

### 2.4 Other data files

- `data/utils.py`: `ActiveSubset` (Subset that exposes `.targets` and forwards `.transform` to the base dataset), `RandomFixedLengthSampler`, `ConcatDataloader` + FixMatch transforms, `seed_worker`.
- `data/longtail.py`: imbalanced (long-tail) dataset construction.
- `data/sem_sl.py`: builds the FixMatch (labelled, unlabelled-weak/strong) dual loaders.
- `data/timeseriesdata.py`, `ecg5000_dataset.py`, `p12_dataset.py`, `mimic_dataset.py`, `skin_dataset.py`, `mio_dataset.py`: non-CIFAR datasets; `TimeSeriesDM` is selected in `get_torchvision_dm` for `ecg5000`/`p12`/`mimic3_sand`. MIMIC-III (`mimic3_sand`, in-hospital mortality, ~11.5% positive, N=44812) is produced from the STraTS repo's preprocessed pickle by `data/preprocess_mimic3_sand.py` → `$DATA_ROOT/mimic3_sand/` (X: (N, 24, 387) hourly-binned values/mask/delta, demo: Age+Gender, meta.json has the variable list).

### 2.5 Who calls what (data flow in one AL iteration)

```
get_torchvision_dm(cfg)         → TorchVisionDM (train_set = ActiveLearningDataset)
label_active_dm(...)            → train_set.label_balanced / label_randomly
ActiveTrainingLoop._fit()       → dm.train_dataloader() → labelled samples, train augments,
                                  oversampled to min_train
                                  dm.val_dataloader()   → 5k val, test transforms
QuerySampler.query_samples()    → dm.pool_dataloader()    (pool, test transforms)
                                  dm.labeled_dataloader() (labelled, test transforms)
                                  → strategy returns loader indices
                                  → dm.get_pool_indices() maps back to pool indices
active_loop                     → train_set.label(pool_indices)  (mask update)
```

---

## 3. The Model Layer — `AbstractClassifier` and Friends

Two things are confusingly both named "Bayesian": a Lightning module (`models/bayesian.py::BayesianModule`) and a plain nn.Module base (`models/bayesian_module.py::BayesianModule`). They live at different levels:

```
pl.LightningModule
  └─ AbstractClassifier            (models/abstract_classifier.py)  — training/eval/optim logic
       ├─ BayesianModule           (models/bayesian.py)             — the standard AL classifier
       └─ FixMatch                 (models/fixmatch.py)             — semi-supervised variant

nn.Module
  └─ BayesianModule (nn.Module version, models/bayesian_module.py)  — MC-sampling forward machinery
       └─ ResNet                   (models/networks/bayesian_resnet.py) — actual network
          (+ bayesian_wide_resnet, bayesian_mnist, vgg, mlp, transformer, gru_d, sand, …)
```

### 3.1 `AbstractClassifier` (Lightning)

Carries everything except the network and training step:

- **`forward(x, k, agg, ema)`**: the central inference API. Runs the network with `k` MC-dropout samples → logits of shape $B \times k \times C$. If `agg=True`, `mc_nll()` converts to marginal log-probabilities: $\log \frac{1}{k}\sum_j \mathrm{softmax}(z_j)$ via logsumexp. If `agg=False` you get raw per-sample logits — this is what uncertainty queries (BALD, entropy, BatchBALD) consume.
- **`step(batch, k)`**: computes NLL loss on the aggregated log-probs; used by `validation_step`/`test_step` (with the full `k` from the config, e.g. 50) and by subclasses' `training_step` (with `k=1`).
- **`get_features(x)`**: penultimate-layer features from the underlying network, flattened to $N \times D$. **This is the hook all diversity strategies — including vendi — rely on.**
- **EMA support** (`init_ema_model`, `select_forward_model`, `on_train_batch_end`): optionally keeps an exponential-moving-average copy of the network; eval uses EMA when present (mainly for FixMatch).
- **Metrics/logging**: torchmetrics `Accuracy` for train/val/test + balanced accuracy (macro recall) for test; metrics are lazily created in `setup_data_params()` once `num_classes` is known.
- **`setup_data_params(dm)`**: called by the trainer before fit; records `train_iters_per_epoch` (needed by the cosine scheduler) and optionally builds a class-weighted `NLLLoss` (`weighted_loss` for imbalanced data).
- **`configure_optimizers()`**: SGD/Adam with weight-decay exclusion for bn/bias, cosine-warmup / step schedulers, optional frozen encoder or finetune mode (different LRs for encoder vs. head — used with SSL-pretrained checkpoints via `load_from_ssl_checkpoint`).
- **`wrap_dm(dm)`**: identity here; overridden by FixMatch to inject the semi-supervised dataloaders.
- `training_step` is abstract — subclasses provide it.

### 3.2 `BayesianModule` (Lightning, `models/bayesian.py`)

The concrete classifier used in every standard AL run (instantiated in `ActiveTrainingLoop._init_model`). Tiny: `save_hyperparameters(config)` (this is why `self.hparams.model...` works everywhere), builds the network with `build_model(config)`, sets `self.k = cfg.model.k`, optionally loads an SSL checkpoint / EMA, and defines `training_step` = supervised step with `k=1` + accuracy logging.

### 3.3 `BayesianModule` (nn.Module, `models/bayesian_module.py`) and the networks

Implements consistent MC-dropout mechanics (from BatchBALD):

- `forward(x, k)` = `det_forward_impl(x)` (deterministic CNN body, run **once**) → replicate features $k$ times ($B \to B{\cdot}k$) → `mc_forward_impl` (stochastic head with dropout) → reshape to $B \times k \times C$. This makes 50 MC samples cheap: only the head is run k times.
- `ConsistentMCDropout(2D)`: dropout that samples one mask per MC index and reuses it across batches during eval — required for BatchBALD's joint entropies.

`models/networks/bayesian_resnet.py::ResNet` is the CIFAR-10 network (`name: bayesian_resnet`, registered via `models/networks/registry.py`; `build.py::build_model` looks it up by `cfg.model.name`): a torchvision resnet18 with CIFAR stem (3×3 conv, no maxpool), `fc = Identity`, plus a separate `classifier` head. `det_forward_impl` = resnet body, `mc_forward_impl` = classifier, `get_features` = resnet body output (512-d). With `dropout_p: 0` (default CIFAR config) the model is effectively deterministic and the $k$ samples are identical — uncertainty methods only become "Bayesian" if you set `dropout_p > 0`.

### 3.4 How the model plugs into the query layer

`QuerySampler.ranking_step` (`query/query.py`) dispatches on `cfg.query.name`:

- **Uncertainty family** (`query/query_uncertainty.py`, names: `bald`, `entropy`, `variationratios`, `batchbald`, `random`): calls `model(x, agg=False)` on each pool batch → $B \times k \times C$ logits → entropy/BALD/etc. score → top-`acq_size` (BatchBALD instead does greedy joint-MI on the log-probs).
- **Diversity family** (`query/query_diversity.py`, names: `kcentergreedy`, `badge`, `vendi`): uses `model.get_features` on pool and labelled loaders.
  - **vendi (your method)**: embeds L and U with `get_features`, normalizes features (`minmax`/`l2`/`zscore`), builds kernel matrices $K_{LL}$ and $K_{UL}$ (RBF or cosine). For each pool candidate $u$ it forms the $(L{+}1)\times(L{+}1)$ bordered kernel matrix, eigendecomposes it in batches (`torch.linalg.eigvalsh`, or LOBPCG top-$k$ if `approx`), computes the Rényi entropy of order $q$ of the normalized eigenvalues, and scores $u$ by the Vendi score $\exp(H_q)$. Top `acq_size` scores are acquired; `extra_info` records score stats and eigendecomposition time.
- **Bandit** (`query/query_bandit.py` + `query/bandit.py`): LinUCB contextual bandit choosing between an uncertainty arm and a vendi/diversity arm each iteration, with context features like mean BALD, vendi scores, class-distribution entropy; rewarded by val-acc gain in `main.py`.

The model is set to `.eval()` before querying (`QuerySampler.__init__`) — dropout stochasticity during scoring comes only from `ConsistentMCDropout`'s masks, not regular dropout.

---

## 4. Quick Reference

| I want to… | Look at |
|---|---|
| Add a new query strategy | `query/query_diversity.py` or `query_uncertainty.py` (`NAMES` + dispatch), plus a `config/query/*.yaml` |
| Change vendi scoring | `query/query_diversity.py::_get_vendi`, `vendi_from_features`; hyperparams in `config/query/vendi.yaml` |
| Add a dataset | `data/data.py::TorchVisionDM.__init__` (dataset_cls map), `config/data/*.yaml` |
| Add a network | `models/networks/` + `@register_model`, `config/model/*.yaml` |
| Change AL budget | `config/active/*.yaml` |
| Change what "best model" means | `trainer.py::_init_ckpt_callback` (monitor metric per dataset) |
| Understand outputs of a run | Hydra run dir: `stored.npz`, `test_metrics.csv`, `timing.csv`, `loop-{i}/metrics.csv` |

### Gotchas

- `label()` takes **pool-relative indices**, and the pool re-indexes after every acquisition — never cache pool indices across iterations.
- Pool and labelled loaders use **test transforms** (via `pool_specifics`), the train loader uses train augmentations.
- `min_train: 5500` means epochs are oversampled — "epoch" counts don't correspond to passes over the labelled data.
- `cfg.active.num_labelled` is mutated inside `active_loop` (incremented each iteration) — it's a running counter, not a constant.
- Model is retrained **from scratch** each AL iteration and the best-val checkpoint is reloaded before querying.
- With `dropout_p: 0`, BALD/entropy degenerate (all MC samples identical); vendi is unaffected since it only uses features.
