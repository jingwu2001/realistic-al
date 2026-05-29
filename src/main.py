import math
import os
import subprocess
import time
from typing import Callable

import wandb

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
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

    wandb_run = None
    if cfg.trainer.use_wandb:
        hydra_choices = HydraConfig.get().runtime.choices
        active_name = hydra_choices.get("active", "")
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
        _eid = cfg.trainer.experiment_id  # "2026-05-23_18-53-00-411537"
        date_str = _eid[:10].replace("-", "")   # "20260523"
        time_str = _eid[11:19].replace("-", "")  # "185300"
        session_tag = f"{date_str}-{time_str}_{commit}"  # "20260523-185300_f8fe30ab"
        wandb_run = wandb.init(
            project=cfg.trainer.wandb_project,           # e.g. "realistic-al"
            name=f"{cfg.query.name}/seed-{cfg.trainer.seed}",  # e.g. "vendi/seed-12345"
            group=f"{cfg.data.name}/{active_name}/{cfg.query.name}",  # e.g. "cifar10_imb/cifar10_low/vendi"
            tags=[cfg.data.name, active_name, cfg.query.name, session_tag] + (["test"] if cfg.trainer.is_test else []),  # e.g. ["cifar10_imb", "cifar10_low", "vendi", "20260523_f8fe30ab"]
            notes=str(cfg.trainer.wandb_notes) or None,    # e.g. "run on hpc"
            config={
                "query": cfg.query.name,
                "model": cfg.model.name,
                "data": cfg.data.name,
                "active": active_name,
                "seed": cfg.trainer.seed,
                "num_labelled": cfg.active.num_labelled,
                "acq_size": cfg.active.acq_size,
                "num_iter": cfg.active.num_iter,
            },
        )
        wandb_run.define_metric("al/*", step_metric="al_iter")

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


def _read_loop_metrics(log_dir) -> dict:
    """Return al/-prefixed metrics from a loop's metrics.csv, aggregated per AL iteration.

    val/* → max across epochs (best checkpoint value)
    train/* and test/* → last non-NaN value
    """
    try:
        df = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
        result = {}
        for col in df.columns:
            if col in ("epoch", "step"):
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            if col.startswith("val/"):
                val = series.max()
            else:
                val = series.iloc[-1]
            result[f"al/{col.replace('/', '_')}"] = val
        return result
    except Exception:
        pass
    return {}


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
            if wandb_run is not None:
                log_dict = {"al_iter": i, "al/n_labelled": cfg.active.num_labelled, "al/train_time_s": round(train_timer.elapsed, 4)}
                log_dict.update(_read_loop_metrics(training_loop.log_dir))
                wandb_run.log(log_dict)
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
        if wandb_run is not None:
            log_dict = {
                "al_iter": i,
                "al/n_labelled": cfg.active.num_labelled,
                "al/train_time_s": round(train_timer.elapsed, 4),
                "al/query_time_s": round(query_timer.elapsed, 4),
                "al/eig_time_s": eig_time,
            }
            log_dict.update(_read_loop_metrics(training_loop.log_dir))
            wandb_run.log(log_dict)
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
