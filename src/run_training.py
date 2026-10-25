import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import utils
from data.base_datamodule import BaseDataModule
from data.data import TorchVisionDM
from data.mimic_dataset import PHENO_CONDITION_NAMES, pheno_single_task_id
from data.timeseriesdata import TimeSeriesDM
from trainer import ActiveTrainingLoop
from utils import config_utils
from utils.log_utils import setup_logger


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    setup_pheno_task(cfg)
    setup_mimic_input_dims(cfg)
    config_utils.print_config(cfg)
    train(cfg)


def setup_pheno_task(cfg: DictConfig) -> None:
    """mimic3_pheno single-task mode (``data.multitask=false``): train only on
    condition ``data.task_id`` -> one sigmoid output, so ``data.num_classes`` is
    set to 1 here. Call at startup, before anything reads ``data.num_classes``
    (wandb config, model head, datamodule). No-op for other datasets.
    """
    if cfg.data.name != "mimic3_pheno":
        return
    task_id = pheno_single_task_id(cfg.data)
    if task_id is None:
        return
    cfg.data.num_classes = 1
    logger.info(
        f"mimic3_pheno single-task: task_id={task_id} "
        f"({PHENO_CONDITION_NAMES[task_id]}), num_classes=1"
    )


def setup_mimic_input_dims(cfg: DictConfig) -> None:
    """MIMIC tasks: take the model's input size from data.shape = [T, d_inp], so
    one model config serves both the STraTS tensor ([24, 387]) and the
    benchmark arrays (e.g. [48, 76], data=mimic3_pheno_bench). Call at startup,
    before the model is built. No-op for other datasets.
    """
    if not cfg.data.name.startswith("mimic3"):
        return
    T, d_inp = cfg.data.shape
    for key, value in (("T", T), ("d_inp", d_inp)):
        if key in cfg.model and cfg.model[key] != value:
            logger.info(f"model.{key}: {cfg.model[key]} -> {value} (from data.shape)")
            cfg.model[key] = value


def get_torchvision_dm(cfg: DictConfig, active_dataset: bool = True) -> BaseDataModule:
    """Initialize the configured DataModule.

    Args:
        config (DictConfig): Config obtained
        active_dataset (bool, optional): . Defaults to True.

    Returns:
        BaseDataModule: DataModule used for training.
    """
    balanced_sampling = False
    if "balanced_sampling" in cfg.data:
        balanced_sampling = cfg.data.balanced_sampling

    imbalance = None
    if "imbalance" in cfg.data:
        imbalance = cfg.data.imbalance
    imb_type = "exp"
    if "type" in cfg.data:
        imb_type = cfg.data.type
    imb_factor = 0.02
    if "rho" in cfg.data:
        imb_factor = cfg.data.rho
    val_size = None
    if "val_size" in cfg.data:
        val_size = cfg.data.val_size
    balanced_test_val = False
    if "balanced_test_val" in cfg.data:
        balanced_test_val = cfg.data.balanced_test_val
    single_stay_only = False
    if "single_stay_only" in cfg.data:
        single_stay_only = cfg.data.single_stay_only
    pheno_task_id = None
    if cfg.data.name == "mimic3_pheno":
        pheno_task_id = pheno_single_task_id(cfg.data)
    drop_classes = None
    if cfg.data.get("drop_classes", None):
        drop_classes = list(cfg.data.drop_classes)

    time_series_datasets = {"ecg5000", "p12", "p12_transformer", "mimic3_sand", "mimic3_los", "mimic3_pheno"}
    dm_cls = TimeSeriesDM if cfg.data.name in time_series_datasets else TorchVisionDM

    datamodule = dm_cls(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
        min_train=cfg.active.min_train,
        val_split=cfg.data.val_split,
        random_split=cfg.active.random_split,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
        num_workers=cfg.trainer.num_workers,
        seed=cfg.trainer.seed,
        active=active_dataset,
        persistent_workers=cfg.trainer.persistent_workers,
        imbalance=imbalance,
        imb_type=imb_type,
        imb_factor=imb_factor,
        timeout=cfg.trainer.timeout,
        val_size=val_size,
        balanced_sampling=balanced_sampling,
        balanced_test_val=balanced_test_val,
        single_stay_only=single_stay_only,
        pheno_task_id=pheno_task_id,
        mimic_dir=cfg.data.get("mimic_dir", "mimic3_sand"),
        drop_classes=drop_classes,
    )

    return datamodule


def label_active_dm(
    cfg: DictConfig,
    num_labelled: int,
    balanced: bool,
    datamodule: BaseDataModule,
    balanced_per_cls: int = 5,
):
    """Label the Dataset according to rules.
    Specific for imbalanced datasets (miotcd, isic2019, isic2016, ecg5000 & datamodule.imbalance=True):
    balanced=True labels `balanced_per_cls` samples per class, the rest of the budget randomly.
    mimic3_pheno (multi-hot targets): balanced=True labels num_labelled // 2
    positives and negatives in single-task mode, else num_labelled // C positives
    per condition (rarest first), the rest of the budget randomly.

    Args:
        cfg (DictConfig): config from main
        num_labelled (int): number of samples for anntation
        balanced (bool): label (part of dataset) balanced
        datamodule (BaseDataModule): Datamodule with active train_set
        balanced_per_cls (int, optional): #samples drawn balanced per class for specific cases. Defaults to 5.
    """
    cfg.data.num_classes = cfg.data.num_classes
    if cfg.data.name == "mimic3_pheno" and balanced:
        # the generic branch below would draw only negatives in single-task
        # mode (num_classes=1) and crash on the 25-label multi-hot targets
        if cfg.data.num_classes == 1:
            datamodule.train_set.label_balanced(
                n_per_class=num_labelled // 2, num_classes=2
            )
        else:
            datamodule.train_set.label_balanced_multilabel(
                n_per_label=num_labelled // cfg.data.num_classes
            )
        label_random = num_labelled - datamodule.train_set.n_labelled
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif cfg.data.name in ["isic2019", "miotcd", "isic2016", "ecg5000"] and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif datamodule.imbalance and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)


@logger.catch
def train(cfg: DictConfig):
    """Run standard training.

    Args:
        cfg (DictConfig): config from main
    """
    active_dataset = cfg.active.num_labelled is not None
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    datamodule = get_torchvision_dm(cfg, active_dataset)
    if cfg.active.num_labelled:
        label_active_dm(cfg, cfg.active.num_labelled, cfg.active.balanced, datamodule)

    training_loop = ActiveTrainingLoop(
        cfg, datamodule, active=False, base_dir=os.getcwd()
    )
    training_loop.main()
    training_loop.log_save_dict()


if __name__ == "__main__":
    main()
