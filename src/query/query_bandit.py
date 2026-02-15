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
from models.bayesian_module import ConsistentMCDropout

def set_dropout_p(model: nn.Module, p: float):
    for module in model.modules():
        if isinstance(module, ConsistentMCDropout):
            module.p = p

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

        original_p = cfg.model.dropout_p

        set_dropout_p(model, original_p)
        
        # --- 1. Compute Uncertainty (BALD) Arm Features ---
        # Temporarily perform BALD sampling
        # We need BALD scores for the whole pool
        # To get BALD scores specifically, we might need to force the function if cfg.query.name is 'bandit'
        # The reference code uses BALD specifically.
        # Let's get BALD function directly
        bald_fct = query_uncertainty._get_bald_fct(model)
        def _return_all(scores, size):
            return scores, size

        bald_scores_all, _ = query_uncertainty.query_sampler(
            pool_loader,
            acq_function=bald_fct,
            post_acq_function=_return_all,
            acq_size=acq_size,
            device=device
        )
        assert len(bald_scores_all.shape) == 1
        bald_scores_all_sorted = np.argsort(bald_scores_all)[::-1]
        b_queried_idx = bald_scores_all_sorted[:acq_size]
        b_queried_values = bald_scores_all[b_queried_idx]
        
        # get bald scores of the pool samples not queried
        mask = np.ones_like(bald_scores_all, dtype=bool)
        mask[b_queried_idx] = False
        b_leftout_values = bald_scores_all[mask]
        
        mean_bald_q = b_leftout_values.mean()
        mean_bald_rest = b_leftout_values.mean()


        set_dropout_p(model, 0.0)

        normalization = cfg.query.vendi.normalization

        feat_labeled = query_diversity.get_embeddings(model, labeled_loader, cpu=False, numpy=False)
        feat_unlabeled = query_diversity.get_embeddings(model, pool_loader, cpu=False, numpy=False)

        feat_labeled, feat_unlabeled = query_diversity.normalize_features(feat_labeled, feat_unlabeled, normalization)

        scores = query_diversity.vendi_from_features(cfg, feat_labeled, feat_unlabeled)

        sorted_indices_vendi = np.argsort(scores)[::-1].copy()
        sorted_scores_vendi = scores[sorted_indices_vendi]
        acq_indices_vendi = sorted_indices_vendi[:acq_size]
        acq_vals_vendi = sorted_scores_vendi[:acq_size]

        

        feat_Q = feat_unlabeled[acq_indices_vendi]

        feat_Q_L = torch.cat([feat_Q, feat_labeled], dim=0)

        gamma, q = cfg.query.vendi.gamma, cfg.query.vendi.q
        vendi_Q_L = calculate_vendi_score(feat_Q_L, gamma, q)
        vendi_Q = calculate_vendi_score(feat_Q, gamma, q)
        vendi_L = calculate_vendi_score(feat_labeled, gamma, q)

        set_dropout_p(model, original_p)

        total_iter = cfg.active.num_iter
        if total_iter == 0: # Avoid division by zero if not set, though main sets it
             total_iter = 1
        t_normalized = (self.count + 1) / total_iter
        
        # [mean_bald(Q), mean_bald(Rest), t, 1]
        features_bald = np.array([mean_bald_q, mean_bald_rest, t_normalized, 1.0])
        
        # [vendi(L), vendi(L+Q), t, 1]
        features_vendi = np.array([vendi_L, vendi_Q_L, t_normalized, 1.0])

        # Normalize first two features (separately) so that they sum to 1 if configured
        if cfg.query.bandit.normalize_features:
            num_classes = 10
            features_bald[:2] = features_bald[:2] / np.log(num_classes)
            features_vendi[0] /= feat_labeled.shape[0]
            features_vendi[1] /= (feat_labeled.shape[0] + feat_Q.shape[0])
        
        context_features = np.stack([features_bald, features_vendi]) # (2, 4)

        # --- 4. Select Arm ---
        selected_arm = self.bandit_manager.select_arm(context_features)
        
        # --- 5. Return Indices of Selected Arm ---
        if selected_arm == 0: # BALD
            return b_queried_idx, bald_scores_all[b_queried_idx], {"bandit_arm": 0}
        else: # Vendi
            return acq_indices_vendi, acq_vals_vendi, {"bandit_arm": 1}


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


if __name__ == "__main__":
    import unittest.mock as mock
    from torch.utils.data import TensorDataset

    # Mock configuration
    cfg = DictConfig({
        "model": {"dropout_p": 0.5},
        "query": {
            "vendi": {
                "normalization": "none",
                "gamma": 1.0,
                "q": 1.0,
                "batch_size": 200
            },
            "name": "bandit"
        },
        "active": {"num_iter": 10, "m": 1},
        "training": {"batch_size": 2}
    })

    # Dummy Model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.dropout = ConsistentMCDropout(0.5)
        
        def forward(self, x, agg=False):
            return self.linear(x)

    model = DummyModel()

    # Dummy Data
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    # Dummy BanditManager
    class MockBanditManager:
        def select_arm(self, features):
            print(f"Selecting arm with features: \n{features}")
            return 0 # Select BALD

    bandit_manager = MockBanditManager()

    # Initialize Sampler
    sampler = BanditQuerySampler(cfg, model, count=0, device="cpu", bandit_manager=bandit_manager)
    sampler.acq_size = 5

    # Mock external calls to avoid complex dependencies
    # We patch query_uncertainty and query_diversity in this module's namespace
    
    # NOTE: Since we are running this script directly or as a module, "src.query.query_bandit" import might vary
    # depending on how it's invoked. We assume standard usage.
    
    with mock.patch("src.query.query_bandit.query_uncertainty") as mock_unc, \
         mock.patch("src.query.query_bandit.query_diversity") as mock_div:
        
        # Mock query_sampler to return random scores for BALD
        # Returns: (bald_scores_all, _)
        # bald_scores_all should be (N,)
        mock_unc.query_sampler.return_value = (np.random.rand(20), None)
        mock_unc._get_bald_fct.return_value = lambda x: x

        # Mock get_embeddings for Vendi
        mock_div.get_embeddings.return_value = torch.randn(20, 10)
        
        # Mock normalize_features
        mock_div.normalize_features.return_value = (torch.randn(20, 10), torch.randn(20, 10))

        # Mock vendi_from_features return scores
        mock_div.vendi_from_features.return_value = np.random.rand(20)
        
        # Mock rbf_kernel and renyi_entropy for calculate_vendi_score (called inside ranking_step -> calculate_vendi_score)
        # Actually calculate_vendi_score calls query_diversity.rbf_kernel and renyi_entropy which are imported attributes in query_bandit
        # So mocking mock_div.rbf_kernel works if calculate_vendi_score uses query_diversity.rbf_kernel
        mock_div.rbf_kernel.return_value = torch.randn(5, 5)
        mock_div.renyi_entropy.return_value = torch.tensor(1.0)

        print("--- Testing BanditQuerySampler.ranking_step ---")
        try:
            indices, values, info = sampler.ranking_step(loader, loader)
            print("Selected Indices:", indices)
            print("Selected Values:", values)
            print("Info:", info)
            print("\n*** Test Passed! ***")
        except Exception as e:
            print("\n*** Test Failed! ***")
            print(e)
            import traceback
            traceback.print_exc()