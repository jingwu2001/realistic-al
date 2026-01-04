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
        
        # Get scores for all pool data
        # We can reuse query_uncertainty.query_sampler logic but returns acq_ind, acq_scores
        # But we need raw scores for mean calculation of un-acquired
        # Let's manually iterate to get all scores
        all_bald_scores = []
        for batch in pool_loader:
             # batch is [x, y]
             scores_batch = query_uncertainty.acq_from_batch(batch, bald_fct, device=device)
             all_bald_scores.append(scores_batch)
        all_bald_scores = np.concatenate(all_bald_scores)
        
        # Select top-k for BALD
        bald_indices_sorted = np.argsort(all_bald_scores)[::-1]
        bald_selected_indices = bald_indices_sorted[:acq_size]
        bald_remaining_indices = bald_indices_sorted[acq_size:]
        
        mean_bald_q = np.mean(all_bald_scores[bald_selected_indices])
        mean_bald_rest = np.mean(all_bald_scores[bald_remaining_indices])
        
        # --- 2. Compute Diversity (Vendi) Arm Features ---
        # We need Vendi scores. Vendi sampling is greedy or iterative.
        # realistic-al/src/query/query_diversity.py _get_vendi returns indices and scores
        # But we need Vendi(Q) and Vendi(L+Q).
        # _get_vendi computes conditional entropy/gain? No, it computes Vendi score directly if configured?
        # Looking at _get_vendi in query_diversity.py, it sorts by score (entropy gain per sample?)
        # Actually _get_vendi implements greedy selection by maximizing marginal Vendi score or similar?
        # Wait, _get_vendi implementation:
        # It calculates K matrix, then iterates U // batch_size.
        # It calculates entropy of adding ONE sample to L? 
        # "K_batch[:, L, :L] = K_UL_batch" -> this suggests it checks Vendi of L + {x}.
        # And it returns indices sorted by this score. 
        # This matches "Vendi(L + {x})".
        # But for the BANDIT feature, we need Vendi(Q) (the whole batch Q) and Vendi(L+Q).
        # So first we identify Q (the top-k produced by Vendi strategy).
        
        # Let's call _get_vendi to identify the set Q that Vendi WOULD select
        # We need to temporarily mock cfg.query.vendi if needed, or assume it's set
        # bandit.yaml doesn't have [vendi] block maybe? User said "The uncertainty and diversity methods should be bald and vendi... already be implemented".
        # We assume cfg includes vendi params.
        
        vendi_indices, vendi_element_scores, _ = query_diversity._get_vendi(
            cfg, model, labeled_loader, pool_loader, acq_size=acq_size
        )
        # vendi_indices are relative to pool (0..N_pool-1)
        
        # Now we need Vendi(Q) and Vendi(L+Q)
        # We need features (embeddings) for this
        # Extract features for L and Q
        # We can't easily cache features inside this method efficiently without refactoring, 
        # but _get_vendi already extracted them. However it returned indices.
        # We have to re-extract or modify _get_vendi to return features (not modifying other files).
        # We must re-extract features.
        
        # Extract features for L
        features_L = torch.tensor([], device=device)
        for inputs, _ in labeled_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                features_batch = model.get_features(inputs)
            features_L = torch.cat((features_L, features_batch), 0)
            
        # Extract features for Pool (to get Q)
        features_P = torch.tensor([], device=device)
        # We need to preserve order to match vendi_indices
        # pool_loader should be deterministic (no shuffle)
        for inputs, _ in pool_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                features_batch = model.get_features(inputs)
            features_P = torch.cat((features_P, features_batch), 0)
            
        # Get Q features
        features_Q_vendi = features_P[vendi_indices]
        
        # Normalize if needed (Vendi usually needs normalized features)
        norm_type = cfg.query.vendi.normalization
        # Normalize L and P together as done in _get_vendi usually, or just normalizing Q is enough?
        # _get_vendi calls normalize_features(feat_labeled, feat_unlabeled)
        # We should replicate this for consistency
        features_L_norm, features_P_norm = query_diversity.normalize_features(features_L, features_P, norm_type)
        features_Q_vendi_norm = features_P_norm[vendi_indices]
        
        # Calculate Vendi(Q)
        # calc_vendi is not in query_diversity top level. It has renyi_entropy and rbf_kernel.
        # utils.calc_vendi in reference code seems to wrap this.
        # We implement calculate_vendi here using rbf_kernel and renyi_entropy
        
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

        gamma = cfg.query.vendi.gamma
        q = cfg.query.vendi.q
        
        vendi_score_Q = calculate_vendi_score(features_Q_vendi_norm, gamma=gamma, q=q)
        
        # Calculate Vendi(L+Q)
        vendi_score_LQ = calculate_vendi_score(torch.cat([features_L_norm, features_Q_vendi_norm], dim=0), gamma=gamma, q=q)
        
        # --- 3. Construct Context Features ---
        # Normalized time t = count / num_iter
        # We need total iterations. cfg.active.num_iter
        total_iter = cfg.active.num_iter
        if total_iter == 0: # Avoid division by zero if not set, though main sets it
             total_iter = 1
        t_normalized = self.count / total_iter
        
        # [mean_bald(Q), mean_bald(Rest), t, 1]
        features_bald = np.array([mean_bald_q, mean_bald_rest, t_normalized, 1.0])
        
        # [vendi(Q), vendi(L+Q), t, 1]
        features_vendi = np.array([vendi_score_Q, vendi_score_LQ, t_normalized, 1.0])
        
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
