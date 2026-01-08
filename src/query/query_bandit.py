from typing import Optional, Tuple
import numpy as np
import torch
from omegaconf import DictConfig
import torch.nn as nn
from torch.utils.data import DataLoader
from data.base_datamodule import BaseDataModule
from .query import QuerySampler
from .bandit import BanditManager
from . import query_uncertainty
from . import query_diversity

class BanditQuerySampler(QuerySampler):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        count: Optional[int] = None,
        device: str = "cuda:0",
        bandit_manager: Optional[BanditManager] = None
    ):
        super().__init__(cfg, model, count, device)
        self.bandit_manager = bandit_manager

    def ranking_step(
        self, pool_loader: DataLoader, labeled_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if not self.bandit_manager:
            return super().ranking_step(pool_loader, labeled_loader)

        cfg = self.cfg
        model = self.model
        device = self.device
        acq_size = self.acq_size
        
        # --- 1. Compute Uncertainty (BALD) Arm Features ---
        # Temporarily perform BALD sampling
        # We need BALD scores for the whole pool
        # To get BALD scores specifically, we might need to force the function if cfg.query.name is 'bandit'
        # The reference code uses BALD specifically.
        # Let's get BALD function directly
        bald_fct = query_uncertainty._get_bald_fct(model)
        bald_scores_all, _ = query_sampler(
            pool_loader,
            acq_function=bald_fct,
            post_acq_function=lambda x: x,
            acq_size=acz_size,
            device=device
        )
        assert len(bald_scores_all.shape) == 1
        b_queried_values, b_queried_idx = torch.topk(bald_scores_all, k=acq_size, largest=True)
        
        # get bald scores of the pool samples not queried
        mask = torch.ones_like(bald_scores_all)
        mask[b_queried_idx] = False
        b_leftout_values = bald_scores_all[mask]
        
        mean_bald_q = b_leftout_values.numpy().mean()
        mean_bald_rest = b_leftout_values.numpy().mean()


        normalization = cfg.query.vendi.normalization

        feat_labeled = get_embeddings(labeled_dataloader, cpu=False, numpy=False)
        feat_unlabeled = get_embeddings(pool_loader, cpu=False, numpy=False)

        feat_labeled, feat_unlabeled = normalize_features(feat_labeled, feat_unlabeled, normalization)

        scores = vendi_from_features(cfg, feat_labeled, feat_unlabeled)

        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        acq_indices = sorted_indices[:acq_size]

        features_Q = feat_unlabeled[acq_indices]

        features_Q_L = torch.cat([features_Q, features_L], dim=0)

        gamma, q = cfg.query.vendi.gamma, cfg.query.vendi.q
        vendi_Q_L = calculate_vendi_score(features_Q_L, gamma, q)
        vendi_Q = calculate_vendi_score(features_L, gamma, q)

        total_iter = cfg.active.num_iter
        if total_iter == 0: # Avoid division by zero if not set, though main sets it
             total_iter = 1
        t_normalized = (self.count + 1) / total_iter
        
        # [mean_bald(Q), mean_bald(Rest), t, 1]
        features_bald = np.array([mean_bald_q, mean_bald_rest, t_normalized, 1.0])
        
        # [vendi(Q), vendi(L+Q), t, 1]
        features_vendi = np.array([vendi_Q, vendi_Q_L, t_normalized, 1.0])
        
        context_features = np.stack([features_bald, features_vendi]) # (2, 4)
        
        # --- 4. Select Arm ---
        selected_arm = self.bandit_manager.select_arm(context_features)
        
        # --- 5. Return Indices of Selected Arm ---
        if selected_arm == 0: # BALD
            return bald_selected_indices, all_bald_scores[bald_selected_indices], {"bandit_arm": 0}
        else: # Vendi
             # For scores, we can return the element-wise scores computed by Vendi (entropy gain)
             # _get_vendi returns (indices, scores, extra)
            return vendi_indices, vendi_element_scores, {"bandit_arm": 1}


def calculate_vendi_score(feats, gamma=None, q=1.0):
    # Kernel matrix
    K = query_diversity.rbf_kernel(feats, gamma=gamma).to(torch.float64)
    # Normalize by N
    N = K.shape[0]
    K = K / N
    # Eigenvalues
    ev = torch.linalg.eigvalsh(K)
    # Entropy
    entropy = query_diversity.renyi_entropy(ev.unsqueeze(0), q).item()
    return np.exp(entropy)