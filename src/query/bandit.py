import numpy as np
from loguru import logger
from . import utils

class BanditManager:
    """
    A manager that treats diveristy and uncertainty sampling as the two arms of a contextual bandit.
    """
    def __init__(self, context_dim: int = 4, alpha: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.context_dim = context_dim
        
        # Initialize bandit parameters
        self.beta = np.zeros([2, context_dim])[:, :, None] # (2, d, 1)
        I = np.identity(context_dim)
        self.A = np.stack([I, I], axis=0) # (2, d, d)
        
        self.start_state = None
        
        self._tie_break_rng = np.random.default_rng(seed)
        
        self.latest_x = None
        self.latest_arm = None
        
        # For compatibility with logging in main loop if needed, though we use beta/A/alpha mainly
        self.counts = {0: 0, 1: 0} 
        self.values = {0: 0.0, 1: 0.0}

    def calc_arms_scores(self, x):
        """Calculate the score for each arm with estimated coefficients and features X"""
        # x is expected to be (2, d)
        
        # 1. Inverse and Mean Reward
        A_inv = np.linalg.inv(self.A)               # (2, d, d)
        theta = (A_inv @ self.beta).squeeze(-1)     # (2, d, 1) -> (2, d)

        mean_reward = np.sum(theta * x, axis=1) # (2,)

        # 2. Uncertainty Term: alpha * sqrt(x^T * A^-1 * x)
        x_batch = x[:, :, None]
        matrix_variance = x_batch.transpose(0, 2, 1) @ A_inv @ x_batch
        uncertainty = self.alpha * np.sqrt(matrix_variance.flatten())

        return mean_reward + uncertainty

    def select_arm(self, features):
        """
        Selects an arm based on features.
        features: np.ndarray of shape (2, context_dim)
            [0] -> features for Uncertainty Arm
            [1] -> features for Diversity Arm
        Returns: 0 (Uncertainty) or 1 (Diversity)
        """
        self.latest_x = features
        arms_scores = self.calc_arms_scores(features)
        
        logger.info(f"Bandit features: {features}")
        logger.info(f"Bandit beta: {self.beta.squeeze()}")
        logger.info(f"Bandit arm scores: {arms_scores}")
        
        bern = self._tie_break_rng.binomial(n=1, p=0.5)

        if arms_scores[0] > arms_scores[1] or (arms_scores[0] == arms_scores[1] and bern == 0):
            logger.info("Uncertainty used in bandit")
            if arms_scores[0] == arms_scores[1]:
                logger.info("Uncertainty chosen by tie break")
            self.latest_arm = 0
            return 0 # 'bald' / Uncertainty
            
        elif arms_scores[0] < arms_scores[1] or (arms_scores[0] == arms_scores[1] and bern == 1):
            logger.info("Diversity used in bandit")
            if arms_scores[0] == arms_scores[1]:
                logger.info("Diversity chosen by tie break")
            self.latest_arm = 1
            return 1 # 'vendi' / Diversity
            
        return 0 # Default fallback

    def update(self, arm, reward):
        """
        Update the bandit parameters.
        arm: 0 or 1
        reward: float
        """
        logger.info(f"Bandit Update: Arm {arm}, Reward {reward}")
        
        # Map arm name to index if string passed (though main logic should pass int/index if possible, but let's handle it)
        arm_idx = arm
        if isinstance(arm, str):
             if 'bald' in arm or 'uncertainty' in arm: arm_idx = 0
             elif 'vendi' in arm or 'diversity' in arm: arm_idx = 1
        
        if self.latest_x is None:
            logger.warning("Bandit update called but no features stored (latest_x is None). Skipping update.")
            return

        x_arm = self.latest_x[arm_idx][:, None] # (d, 1)
        
        self.A[arm_idx] += x_arm @ x_arm.T 
        self.beta[arm_idx] += x_arm * reward
        
        # Update simple stats for logging
        self.counts[arm_idx] += 1
        self.values[arm_idx] = ((self.values[arm_idx] * (self.counts[arm_idx]-1)) + reward) / self.counts[arm_idx]
