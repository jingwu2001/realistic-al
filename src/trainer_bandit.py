from src.trainer import ActiveTrainingLoop
from src.query.query_bandit import BanditQuerySampler
from omegaconf import DictConfig
from data.data import TorchVisionDM
from typing import Union
import os

class ActiveTrainingLoopBandit(ActiveTrainingLoop):
    def __init__(self, *args, bandit_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bandit_manager = bandit_manager

    def active_callback(self):
        """Execute active learning logic with bandit support.
        Returns the queries to the oracle."""
        self.model = self.model.to(self.device)
        # Use BanditQuerySampler instead of QuerySampler
        query_sampler = BanditQuerySampler(
            self.cfg, self.model, count=self.count, device=self.device,
            bandit_manager=self.bandit_manager
        )
        query_sampler.setup()
        stored = query_sampler.active_callback(self.datamodule)
        return stored
