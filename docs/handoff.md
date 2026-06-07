# Launcher and Datamodule Handoff

## Launcher Execution

The launcher files in `launchers/` do not train models directly. They build Hydra command-line arguments and then run the actual experiment script as a subprocess.

For both examples:

- `launchers/exp_cifar10imb_basic.py`
- `launchers/exp_p12_transformer.py`

the experiment script is:

```bash
src/main.py
```

The path comes from:

```python
path_to_ex_file = "src/main.py"
```

At runtime, each launcher does roughly this:

1. Adds launcher-only CLI args with `ExperimentLauncher.add_argparse_args(parser)`.
2. Parses args such as `--debug`, `--bsub`, `--test`, `--num_start`, `--num_end`, and `--notes`.
3. Applies launcher-level changes with `ExperimentLauncher.modify_params_for_args(...)`.
4. Constructs an `ExperimentLauncher`.
5. Calls `launcher.launch_runs()`.

`ExperimentLauncher.launch_runs()` expands the grid from `config_dict` and `hparam_dict`, applies `joint_iteration`, generates experiment names, and runs commands like:

```bash
python /path/to/repo/src/main.py \
  model=resnet query=random data=cifar10_imb active=cifar10_low optim=sgd_cosine \
  ++trainer.seed=12345 ++trainer.max_epochs=200 \
  ++trainer.experiment_name=cifar10_imb/active-cifar10_low/...
```

With `--bsub`, the command prefix changes from `python` to:

```bash
~/run_active.sh python
```

With `--debug`, the launcher prints commands but does not run them.

## `src/main.py` Entry Point

The subprocess enters `src/main.py` through Hydra:

```python
@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
```

Hydra combines the base config with launcher-provided selections such as:

```bash
model=resnet
query=bald
data=cifar10_imb
active=cifar10_low
optim=sgd_cosine
```

and override args such as:

```bash
++trainer.seed=12345
++model.dropout_p=0.5
++trainer.max_epochs=200
```

Then `main(cfg)` sets up logging, prints config, seeds randomness, optionally creates a `BanditManager`, optionally starts W&B, and calls:

```python
active_loop(...)
```

## Single Experiment Before `ActiveTrainingLoop`

Before an `ActiveTrainingLoop` is constructed, `active_loop()` performs two important data steps:

```python
datamodule = get_active_dm_from_config(cfg)
label_active_dm(cfg, num_labelled, balanced, datamodule)
```

In normal launcher runs, `get_active_dm_from_config` is `get_torchvision_dm()` from `src/run_training.py`.

So the flow is:

```text
Hydra cfg
  -> get_torchvision_dm(cfg)
  -> TorchVisionDM(...)
  -> BaseDataModule.__init__(...)
  -> TorchVisionDM._setup_datasets()
  -> label_active_dm(...)
  -> ActiveTrainingLoop(...)
```

By the time `ActiveTrainingLoop` receives the datamodule, the datasets have already been built, split, optionally imbalanced, wrapped for active learning, and initially labeled.

## `BaseDataModule`

`BaseDataModule` lives in `src/data/base_datamodule.py` and inherits from PyTorch Lightning:

```python
class BaseDataModule(pl.LightningDataModule):
```

It is the generic datamodule base for this codebase. It does not choose CIFAR, P12, or any concrete dataset class. Instead, it provides shared behavior:

- stores loader settings such as `batch_size`, `num_workers`, `pin_memory`, `shuffle`, `persistent_workers`, and `timeout`
- stores split settings such as `val_split`, `random_split`, `seed`, and `val_size`
- stores active-learning settings such as `active`, `min_train`, and `balanced_sampling`
- implements `_split_dataset()`
- implements `_get_splits()`
- implements `get_dataloader()`
- implements `pool_dataloader()`
- implements `labeled_dataloader()`
- implements `get_pool_indices()`

The central assumption is that, for active learning, `self.train_set` may be an `ActiveLearningDataset`. In that case, the training dataloader only sees labeled samples, while `pool_dataloader()` exposes the unlabeled pool.

Conceptually, `BaseDataModule` is the reusable active-learning datamodule infrastructure. It does not know how to instantiate MNIST, CIFAR, ISIC, ECG5000, or P12. It only assumes that a subclass will eventually populate:

```python
self.train_set
self.val_set
self.test_set
```

Once those attributes exist, `BaseDataModule` knows how to split, sample, and load them.

`_get_splits()` interprets `val_split` in two modes:

- integer: exact number of validation samples
- float: fraction of the dataset to reserve for validation

`_split_dataset()` uses those split sizes. If `random_split=True`, it calls `torch.utils.data.random_split(...)` with `self.seed`, then converts the resulting subsets into `ActiveSubset`s. This conversion matters because this code frequently needs `.targets` and `.transform`; a plain PyTorch `Subset` hides those attributes behind `.dataset`.

If `val_size` is set, `_split_dataset()` further trims the validation subset in a class-balanced way. It reads labels from `dataset_val.targets`, takes the same number of validation samples per class, and has special handling for `isic2019` and `miotcd`.

`get_dataloader()` is the central loader factory. For training, it has two important active-learning behaviors:

- if the current labeled set is smaller than `min_train`, it oversamples to create enough training iterations per epoch
- if `balanced_sampling=True`, it builds a `WeightedRandomSampler` from class counts

This is why early active-learning rounds can train even when only a small seed set has been labeled.

For active acquisition, `pool_dataloader()` returns a loader over:

```python
self.train_set.pool
```

That only works when `self.train_set` is an `ActiveLearningDataset`. If `m` is provided, `pool_dataloader()` first samples a temporary subset of the pool and stores the sampled indices in `self.indices`. Later, `get_pool_indices()` maps query results from this temporary loader back to indices in the full current pool.

`labeled_dataloader()` returns a loader over:

```python
self.train_set.labelled_set
```

This is mainly used by acquisition strategies that need a test-transform view of the labeled samples, for example embedding-distance or CoreSet-style methods.

## `TorchVisionDM`

`TorchVisionDM` lives in `src/data/data.py` and inherits from `BaseDataModule`:

```python
class TorchVisionDM(BaseDataModule):
```

The inheritance relationship is:

```text
pytorch_lightning.LightningDataModule
  -> BaseDataModule
    -> TorchVisionDM
```

Despite the name, `TorchVisionDM` handles more than TorchVision datasets. It selects concrete dataset classes for:

- MNIST
- CIFAR-10
- CIFAR-100
- Fashion-MNIST
- ISIC2016
- ISIC2019
- MIO-TCD
- ECG5000
- P12
- P12 transformer

`TorchVisionDM.__init__()` first calls `BaseDataModule.__init__()` to store the generic settings. It then stores concrete dataset settings such as:

- `data_root`
- `dataset`
- `num_classes`
- `mean`
- `std`
- `shape`
- train/test transforms
- imbalance settings

Then it chooses `self.dataset_cls` and immediately calls:

```python
self._setup_datasets()
```

So dataset construction is eager, not lazy.

The relationship between `data.py` and `base_datamodule.py` is therefore:

```text
BaseDataModule
  owns generic split/dataloader/pool behavior

TorchVisionDM
  chooses the concrete dataset class
  creates train/val/test datasets
  optionally wraps train_set in ActiveLearningDataset
  delegates actual dataloader creation back to BaseDataModule
```

At the end of `data.py`, the Lightning hooks are thin wrappers:

```python
def train_dataloader(self):
    return self.get_dataloader(self.train_set, mode="train")

def val_dataloader(self):
    return self.get_dataloader(self.val_set, mode="test")

def test_dataloader(self):
    return self.get_dataloader(self.test_set, mode="test")
```

Those methods are defined on `TorchVisionDM`, but the real dataloader policy comes from `BaseDataModule.get_dataloader()`.

## What `_setup_datasets()` Builds

`TorchVisionDM._setup_datasets()` creates:

```python
self.train_set
self.val_set
self.test_set
```

For standard image-style datasets such as CIFAR:

1. Ensure the dataset exists locally, downloading if needed.
2. Build the train split with training transforms.
3. Split the original training set into train and validation using `_split_dataset()`.
4. If `imbalance=True`, apply `create_imbalanced_dataset(...)`.
5. If `active=True`, wrap the train set in `ActiveLearningDataset`.
6. Build validation and test sets with test transforms.

For `cifar10_imb`, the config uses:

```yaml
name: cifar10
imbalance: True
val_split: 5000
```

So the concrete dataset is still CIFAR-10. The train split is split into train and validation, the train side is made long-tailed, then it is wrapped for active learning.

For `p12_transformer`, the path is different:

1. Load all P12 labels.
2. Create stratified test, validation, and pool indices with `_stratified_3way_split()`.
3. Build `self.train_set = ActiveSubset(_p12_full, _p12_pool_idx)`.
4. Wrap train set in `ActiveLearningDataset`.
5. Build validation and test as `Subset(_p12_full_eval, indices)`.

P12 does not use the normal TorchVision train/test split path.

## `ActiveLearningDataset`

`ActiveLearningDataset` lives in `src/data/active.py`. It wraps a normal dataset and tracks which samples are labeled.

On construction, it stores the underlying dataset:

```python
self._dataset = dataset
```

and creates a boolean label mask:

```python
self.labelled = np.zeros(len(self._dataset), dtype=bool)
```

Initially, all samples are unlabeled.

Internally, the wrapper keeps two related notions of state:

```python
self.labelled
self._state.labelled_ind
```

`self.labelled` is the full boolean mask over the wrapped dataset. `self._state.labelled_ind` is the cached list of currently labeled underlying indices. Whenever labels change, `init_state()` rebuilds this cache.

Its key behavior is that `__len__()` returns the number of labeled samples:

```python
return self._state.num_label
```

and `__getitem__()` indexes only into labeled samples:

```python
return self._dataset[self._state.labelled_ind[index]]
```

This means that once `self.train_set` is an `ActiveLearningDataset`, the normal training dataloader trains only on the labeled subset.

It also exposes:

- `pool`: a dataset view of the unlabeled samples
- `labelled_set`: a dataset view of the labeled samples, usually with test/pool transforms
- `label(index)`: marks one or more pool-relative samples as labeled
- `label_randomly(n)`: labels random pool samples
- `label_balanced(n_per_class, num_classes)`: labels a class-balanced seed set
- `n_labelled`: number of labeled samples
- `n_unlabelled`: number of unlabeled samples

Important indexing detail: `label(index)` expects indices relative to the current pool, not absolute indices into the underlying dataset. It converts pool indices back into underlying dataset indices internally.

For example, suppose the wrapped dataset has five samples and samples `0` and `3` are already labeled:

```text
underlying indices:  0  1  2  3  4
labelled mask:       T  F  F  T  F
current pool:           0  1     2
underlying pool idx:    1  2     4
```

Calling `label(1)` labels pool index `1`, which corresponds to underlying dataset index `2`. This pool-relative convention is why query code can select from `datamodule.train_set.pool` and pass the selected positions back into `datamodule.train_set.label(...)`.

For non-P12 image datasets, `TorchVisionDM` wraps the active train set like this:

```python
ActiveLearningDataset(
    self.train_set,
    pool_specifics={"transform": self.test_transforms},
)
```

That means pool and labeled-set views use test transforms rather than training augmentation. This matters for acquisition methods that score the unlabeled pool or compare pool embeddings to labeled embeddings.

For P12/P12 transformer, the wrapper is simpler:

```python
ActiveLearningDataset(self.train_set)
```

because those datasets manage normalization internally and do not expose the same image-style transform attribute.

## Initial Labeling

After `TorchVisionDM` is built, `active_loop()` calls `label_active_dm(...)`.

For imbalanced datasets with `balanced=True`, such as `cifar10_imb`, the function first labels `balanced_per_cls` examples per class, then labels any remaining requested samples randomly:

```python
label_balance = cfg.data.num_classes * balanced_per_cls
datamodule.train_set.label_balanced(...)
label_random = num_labelled - label_balance
datamodule.train_set.label_randomly(label_random)
```

For `cifar10_low`, `num_labelled=50`, `num_classes=10`, and `balanced_per_cls=5`, so it labels exactly 5 examples per class.

For `cifar10_med`, `num_labelled=250`, so it labels 50 balanced examples first and 200 random examples after that.

For `p12_med_bal`, the active config has:

```yaml
num_labelled: 100
balanced: True
```

and the data config has:

```yaml
num_classes: 2
```

So it labels `100 // 2 = 50` examples per class.

## State Handed To `ActiveTrainingLoop`

Right before:

```python
training_loop = ActiveTrainingLoop(
    cfg,
    count=i,
    datamodule=datamodule,
    base_dir=os.getcwd(),
    ...
)
```

the datamodule already contains:

```text
datamodule.train_set = ActiveLearningDataset(...)
datamodule.val_set   = validation Dataset/Subset
datamodule.test_set  = test Dataset/Subset
```

and `datamodule.train_set.labelled` already has `True` entries for the initial seed set.

Training uses:

```python
datamodule.train_dataloader()
```

which delegates to:

```python
BaseDataModule.get_dataloader(self.train_set, mode="train")
```

Because `self.train_set.__len__()` returns only the labeled count, training only sees labeled samples.

Acquisition later uses:

```python
datamodule.pool_dataloader()
```

which returns a dataloader over:

```python
self.train_set.pool
```

That is the unlabeled remainder. After acquisition, selected pool-relative indices are passed to:

```python
datamodule.train_set.label(active_store.requests)
```

which grows the labeled set for the next active-learning iteration.
