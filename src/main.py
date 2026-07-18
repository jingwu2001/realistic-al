import math
import os
import time
from typing import Callable

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

import utils
from data.base_datamodule import BaseDataModule
from run_training import get_torchvision_dm, label_active_dm
from trainer import ActiveTrainingLoop
from query.bandit import BanditManager
from utils import config_utils
from utils.timer import Timer
from utils.log_utils import setup_logger
from utils.wandb_utils import init_wandb, log_al_iteration, log_label_distributions


@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    # Initialize Bandit Manager if the query method is bandit
    bandit_manager = None
    if cfg.query.name == "bandit":
        logger.info("Initializing Bandit Manager")
        # Check if specific bandit params are in config, else use defaults
        bandit_config = cfg.query.bandit
        bandit_alpha = bandit_config.alpha if "bandit" in cfg.query and "alpha" in bandit_config else 0.1
        bandit_context_dim = bandit_config.context_dim if "bandit" in cfg.query and "context_dim" in bandit_config else 4
        bandit_manager = BanditManager(context_dim=bandit_context_dim, alpha=bandit_alpha)

    if cfg.trainer.dry_run:
        logger.info("dry_run=True: skipping wandb init and training")

    wandb_run = init_wandb(cfg)

    if cfg.trainer.get("wandb_test", False):
        # wandb wiring check: the run above carries the real name/group/tags/
        # config; stop before any data loading, training, or acquisition.
        # exit_code=1 marks the run "failed" on wandb so wiring checks are
        # never mistaken for finished results runs (they are also tagged
        # "wandb_test" for bulk deletion).
        logger.info("wandb_test=True: wandb run created; exiting before data/training")
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        return

    exit_code = 0
    try:
        active_loop(
            cfg,
            ActiveTrainingLoop,
            get_torchvision_dm,
            cfg.active.num_labelled,
            cfg.active.balanced,
            cfg.active.acq_size,
            cfg.active.num_iter,
            bandit_manager=bandit_manager,
            wandb_run=wandb_run,
        )
    except Exception:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


@logger.catch
def active_loop(
    cfg: DictConfig,
    ActiveTrainingLoop=ActiveTrainingLoop,
    get_active_dm_from_config: Callable[
        [DictConfig, bool], BaseDataModule
    ] = get_torchvision_dm,
    num_labelled: int = 100,
    balanced: bool = True,
    acq_size: int = 10,
    num_iter: int = 0,
    bandit_manager: BanditManager = None,
    wandb_run=None,
):
    """Perform Active Learning over multiple loops.

    Args:
        cfg (DictConfig): config from main
        ActiveTrainingLoop (Class, optional): Class that defines what happens during the training. Defaults to ActiveTrainingLoop.
        get_active_dm_from_config (Callable, optional): class returning a datamodule usable for active learning. Defaults to get_torchvision_dm.
        num_labelled (int, optional): starting budget for active learning. Defaults to 100.
        balanced (bool, optional): whether starting budget is drawn balanced. Defaults to True.
        acq_size (int, optional): query size in each active learning loop. Defaults to 10.
        num_iter (int, optional): number of active learning loops. Defaults to 0.
        bandit_manager (BanditManager, optional): Manager for bandit query strategy. Defaults to None.
    """
    logger.info("Instantiating Datamodule")
    datamodule = get_active_dm_from_config(cfg)
    label_active_dm(cfg, num_labelled, balanced, datamodule)

    def _label_dist(targets) -> str:
        tgts = np.asarray(targets)
        classes, counts = np.unique(tgts, return_counts=True)
        total = len(tgts)
        parts = [f"{int(c)}: {int(n)} ({100*n/total:.1f}%)" for c, n in zip(classes, counts)]
        return f"n={total}  " + "  ".join(parts)

    def _subset_targets(subset):
        # val_set is a torch Subset (carved from train); test_set is often the
        # raw dataset (e.g. CIFAR10/MNIST) with no .indices/.dataset.
        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            return np.asarray(subset.dataset.targets)[subset.indices]
        return np.asarray(subset.targets)

    ts = datamodule.train_set
    logger.info(
        "Label distribution before training:\n"
        "  L     : {}\n"
        "  U     : {}\n"
        "  L+U   : {}\n"
        "  val   : {}\n"
        "  test  : {}",
        _label_dist(ts.labelled_set.targets),
        _label_dist(ts.pool.targets),
        _label_dist(ts._dataset.targets),
        _label_dist(_subset_targets(datamodule.val_set)),
        _label_dist(_subset_targets(datamodule.test_set)),
    )

    log_label_distributions(wandb_run, datamodule)

    if cfg.trainer.dry_run:
        logger.info("dry_run=True: exiting before training")
        return

    if num_iter == 0:
        num_iter = math.ceil(len(datamodule.train_set) / acq_size)

    active_stores = []
    metric_paths = []
    
    last_val_acc = 0.0
    last_arm = None
    timing_records = []

    for i in range(num_iter):
        logger.info("Start Active Loop {}".format(i))
        # Perform active learning iteration with training and labeling
        training_loop = ActiveTrainingLoop(
            cfg, count=i, datamodule=datamodule, base_dir=os.getcwd(),
            bandit_manager=bandit_manager, wandb_run=wandb_run,
        )
        logger.info("Start Training of Loop {}".format(i))
        with Timer() as train_timer:
            training_loop.main()
        
        # --- Bandit Reward Logic ---
        # Get current validation accuracy
        if training_loop.ckpt_callback.best_model_score:
            current_val_acc = training_loop.ckpt_callback.best_model_score.item()
        else:
            current_val_acc = 0.0
            
        logger.info(f"Loop {i}: Val Acc: {current_val_acc}")
        
        # If we selected an arm last time, update bandit with the gain
        if bandit_manager and last_arm is not None:
            reward = current_val_acc - last_val_acc
            logger.info(f"Bandit Update: Arm {last_arm}, Reward {reward} ({current_val_acc} - {last_val_acc})")
            bandit_manager.update(last_arm, reward)
            
        last_val_acc = current_val_acc

        if training_loop.trainer.interrupted:
            return

        # Save metrics for this completed training loop before attempting the query
        metric_paths.append(training_loop.log_dir)
        training_loop.log_save_dict()

        # Guard: stop early if the pool is too small to fill the next acquisition batch
        pool_size = len(datamodule.train_set.pool)
        if pool_size < acq_size:
            logger.warning(
                f"Pool exhausted after loop {i} ({pool_size} samples remaining, "
                f"acq_size={acq_size}). Stopping early and saving results."
            )
            timing_records.append({"iteration": i, "train_time_s": round(train_timer.elapsed, 4), "query_time_s": float("nan"), "eig_time_s": float("nan")})
            log_al_iteration(
                wandb_run, i, cfg, train_timer, log_dir=training_loop.log_dir
            )
            del training_loop
            break

        logger.info("Start Acquisition of Loop {}".format(i))
        with Timer() as query_timer:
            active_store = training_loop.active_callback()
        eig_time = active_store.extra_info.get("eig_time_s", float("nan")) if active_store.extra_info else float("nan")
        timing_records.append({"iteration": i, "train_time_s": round(train_timer.elapsed, 4), "query_time_s": round(query_timer.elapsed, 4), "eig_time_s": eig_time})
        
        # Capture the arm selected for THIS query (to be rewarded in NEXT loop)
        if bandit_manager:
            if active_store.extra_info and "bandit_arm" in active_store.extra_info:
                last_arm = active_store.extra_info["bandit_arm"]
                logger.info(f"Loop {i}: Query selected arm {last_arm}")
            else:
                last_arm = None
                
        datamodule.train_set.label(active_store.requests)
        active_stores.append(active_store)
        log_al_iteration(
            wandb_run, i, cfg, train_timer,
            query_timer=query_timer, eig_time=eig_time,
            log_dir=training_loop.log_dir, extra_info=active_store.extra_info,
        )
        cfg.active.num_labelled += cfg.active.acq_size
        logger.info("Finalized Loop {}".format(i))
        del training_loop
        time.sleep(1)

    store_path = "."
    if timing_records:
        pd.DataFrame(timing_records).to_csv(os.path.join(store_path, "timing.csv"), index=False)
    metrics_df = []
    for metric_path in metric_paths:
        # laod metrics from csv
        metric_df = pd.read_csv(os.path.join(metric_path, "metrics.csv"))
        # select metrics for test  data
        cols = [col for col in metric_df.columns if "test" in col]
        metric_df = metric_df.loc[:, cols]
        metric_dict = dict(metric_df.iloc[-1])
        metrics_df.append(metric_dict)
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(os.path.join(store_path, "test_metrics.csv"))

    val_accs = np.array([active_store.accuracy_val for active_store in active_stores])
    test_accs = np.array([active_store.accuracy_test for active_store in active_stores])
    num_samples = np.array([active_store.n_labelled for active_store in active_stores])
    add_labels = np.stack(
        [active_store.labels for active_store in active_stores], axis=0
    )
    request_pool = np.array([active_store.requests for active_store in active_stores])

    # Extract bandit arms if available
    bandit_arms = []
    for store in active_stores:
        if store.extra_info and "bandit_arm" in store.extra_info:
            bandit_arms.append(store.extra_info["bandit_arm"])
        else:
            bandit_arms.append(-1)
    bandit_arms = np.array(bandit_arms)

    np.savez(
        os.path.join(store_path, "stored.npz"),
        val_acc=val_accs,
        test_acc=test_accs,
        num_samples=num_samples,
        added_labels=add_labels,
        request_pool=request_pool,
        bandit_arms=bandit_arms,
    )

    for i, store in enumerate(active_stores):
        if store.extra_info is not None:
            np.savez(os.path.join(store_path, f"extra_info_{i}.npz"), **store.extra_info)

    logger.success("Active Loop was finalized")
    # Log bandit history potentially
    if bandit_manager:
        logger.info(f"Final Bandit State: Counts {bandit_manager.counts}, Values {bandit_manager.values}")


if __name__ == "__main__":
    main()
