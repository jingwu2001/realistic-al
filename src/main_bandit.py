import math
import os
import time
from typing import Callable

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import utils
from data.base_datamodule import BaseDataModule
from run_training import get_torchvision_dm, label_active_dm
from src.trainer_bandit import ActiveTrainingLoopBandit
from src.query.bandit import BanditManager
from utils import config_utils
from utils.log_utils import setup_logger

# We reuse utility functions from main.py but implement a custom loop

@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging (Bandit)")
    config_utils.print_config(cfg)
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)
    
    # Initialize Bandit Manager if the query method is bandit
    bandit_manager = None
    if cfg.query.name == "bandit":
        logger.info("Initializing Bandit Manager")
        # Check if specific bandit params are in config, else use defaults
        bandit_alpha = cfg.bandit.alpha if "bandit" in cfg and "alpha" in cfg.bandit else 0.1
        bandit_context_dim = cfg.bandit.context_dim if "bandit" in cfg and "context_dim" in cfg.bandit else 4
        bandit_manager = BanditManager(context_dim=bandit_context_dim, alpha=bandit_alpha)
    
    active_loop_bandit(
        cfg,
        ActiveTrainingLoopBandit,
        get_torchvision_dm,
        cfg.active.num_labelled,
        cfg.active.balanced,
        cfg.active.acq_size,
        cfg.active.num_iter,
        bandit_manager=bandit_manager
    )


@logger.catch
def active_loop_bandit(
    cfg: DictConfig,
    ActiveTrainingLoop=ActiveTrainingLoopBandit,
    get_active_dm_from_config: Callable[
        [DictConfig, bool], BaseDataModule
    ] = get_torchvision_dm,
    num_labelled: int = 100,
    balanced: bool = True,
    acq_size: int = 10,
    num_iter: int = 0,
    bandit_manager: BanditManager = None
):
    logger.info("Instantiating Datamodule")
    datamodule = get_active_dm_from_config(cfg)
    label_active_dm(cfg, num_labelled, balanced, datamodule)

    if num_iter == 0:
        num_iter = math.ceil(len(datamodule.train_set) / acq_size)

    active_stores = []
    metric_paths = []
    
    last_val_acc = 0.0 # Initial roughly 0 (or 1/num_classes but 0 is fine for diff)
    last_arm = None

    for i in range(num_iter):
        logger.info("Start Active Loop {}".format(i))
        # Perform active learning iteration with training and labeling
        training_loop = ActiveTrainingLoop(
            cfg, count=i, datamodule=datamodule, base_dir=os.getcwd(), bandit_manager=bandit_manager
        )
        logger.info("Start Training of Loop {}".format(i))
        training_loop.main()
        
        # --- Bandit Reward Logic ---
        # Get current validation accuracy
        if training_loop.ckpt_callback.best_model_score:
            current_val_acc = training_loop.ckpt_callback.best_model_score.item()
        else:
            current_val_acc = 0.0 # Should not happen if training ran
            
        logger.info(f"Loop {i}: Val Acc: {current_val_acc}")
        
        # If we selected an arm last time, update bandit with the gain
        if bandit_manager and last_arm is not None:
            reward = current_val_acc - last_val_acc
            logger.info(f"Bandit Update: Arm {last_arm}, Reward {reward} ({current_val_acc} - {last_val_acc})")
            bandit_manager.update(last_arm, reward)
            
        last_val_acc = current_val_acc
        
        if training_loop.trainer.interrupted:
            return
            
        logger.info("Start Acquisition of Loop {}".format(i))
        # This calls ranking_step which will use bandit to select arm
        active_store = training_loop.active_callback()
        
        # Capture the arm selected for THIS query (to be rewarded in NEXT loop)
        if bandit_manager:
            if active_store.extra_info and "bandit_arm" in active_store.extra_info:
                last_arm = active_store.extra_info["bandit_arm"]
                logger.info(f"Loop {i}: Query selected arm {last_arm}")
            else:
                last_arm = None
                
        datamodule.train_set.label(active_store.requests)
        active_stores.append(active_store)
        training_loop.log_save_dict()
        cfg.active.num_labelled += cfg.active.acq_size
        logger.info("Finalized Loop {}".format(i))
        metric_paths.append(training_loop.log_dir)
        del training_loop
        time.sleep(1)

    # Final saving logic (same as main.py)
    store_path = "."
    metrics_df = []
    for metric_path in metric_paths:
        try:
            metric_df = pd.read_csv(os.path.join(metric_path, "metrics.csv"))
            cols = [col for col in metric_df.columns if "test" in col]
            metric_df = metric_df.loc[:, cols]
            metric_dict = dict(metric_df.iloc[-1])
            metrics_df.append(metric_dict)
        except Exception as e:
            logger.warning(f"Could not load metrics for {metric_path}: {e}")
            
    if metrics_df:
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv(os.path.join(store_path, "test_metrics.csv"))

    val_accs = np.array([active_store.accuracy_val for active_store in active_stores])
    test_accs = np.array([active_store.accuracy_test for active_store in active_stores])
    num_samples = np.array([active_store.n_labelled for active_store in active_stores])
    add_labels = np.stack(
        [active_store.labels for active_store in active_stores], axis=0
    )
    request_pool = np.array([active_store.requests for active_store in active_stores])

    np.savez(
        os.path.join(store_path, "stored.npz"),
        val_acc=val_accs,
        test_acc=test_accs,
        num_samples=num_samples,
        added_labels=add_labels,
        request_pool=request_pool,
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
