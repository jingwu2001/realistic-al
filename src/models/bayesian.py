import math
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.networks import build_model

from .abstract_classifier import AbstractClassifier


class BayesianModule(AbstractClassifier):
    def __init__(self, config: DictConfig):
        """Simple Bayesian Neural Network which can be used with BatchBALD.
        For a Non-Bayesian Network set config.dopout_p to 0.

        Args:
            config (DictConfig): all hparams.
        """
        super().__init__(eman=True)
        self.save_hyperparameters(config)
        # Multi-label task (e.g. mimic3_pheno): sigmoid/BCE head + metrics.
        self.multilabel = "multilabel" in config.data and bool(config.data.multilabel)
        self.model = build_model(
            config, num_classes=config.data.num_classes, data_shape=config.data.shape
        )
        self.k = self.hparams.model.k
        if self.hparams.model.load_pretrained:
            self.load_from_ssl_checkpoint()
        self.init_ema_model(use_ema=config.model.use_ema)

        if self.multilabel:
            # Default multi-label loss; per-label pos_weight is filled in by
            # setup_data_params when model.weighted_loss is set.
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            weighted_loss = False
            if "weighted_loss" in self.hparams.model:
                weighted_loss = self.hparams.model.weighted_loss
            if weighted_loss:
                # the weights are overwritten at a later stage.
                self.loss_fct = nn.NLLLoss(weight=torch.ones(self.hparams.data.num_classes))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """Perform training step for the current batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): data from dataloader
            batch_idx (int): batch counter

        Returns:
            dict: out_dict for logging
        """
        mode = "train"
        loss, logprob, preds, y = self.step(batch, k=1)
        self.log(f"{mode}/loss", loss)
        self.acc_train.update(preds, y)
        if batch_idx == 0 and self.current_epoch == 0:
            if isinstance(batch[0], torch.Tensor) and len(batch[0].shape) == 4:
                self.visualize_inputs(batch[0], name=f"{mode}/data")
        return {
            "loss": loss,
            "logprob": logprob,
            "label": y,
        }
