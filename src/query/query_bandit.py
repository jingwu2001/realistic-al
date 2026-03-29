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
from .query_diversity import rbf_kernel, cosine_similarity
from models.bayesian_module import ConsistentMCDropout


def _class_distribution_entropy(labels: np.ndarray, num_classes: int) -> float:
    """Normalized entropy of the labeled-set class distribution.

    Returns a value in [0, 1], where 1 = perfectly balanced and 0 = all samples
    in one class.  Uses log(num_classes) for normalization so the scale is
    always comparable regardless of the number of classes.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    p = counts / counts.sum()
    p_nonzero = p[p > 0]
    ent = -np.sum(p_nonzero * np.log(p_nonzero))
    return float(ent / np.log(num_classes))

def set_dropout_p(model: nn.Module, p: float):
    for module in model.modules():
        if isinstance(module, ConsistentMCDropout):
            module.p = p

def _get_rng_states(device):
    import random
    cpu_rng = torch.random.get_rng_state()
    gpu_rng = None
    if isinstance(device, torch.device) and device.type == 'cuda':
        gpu_rng = torch.cuda.get_rng_state(device)
    elif isinstance(device, str) and 'cuda' in device:
        gpu_rng = torch.cuda.get_rng_state(device)
    np_rng = np.random.get_state()
    py_rng = random.getstate()
    return cpu_rng, gpu_rng, np_rng, py_rng

def _set_rng_states(device, states):
    import random
    cpu_rng, gpu_rng, np_rng, py_rng = states
    torch.random.set_rng_state(cpu_rng)
    if gpu_rng is not None:
        if isinstance(device, torch.device) and device.type == 'cuda':
            torch.cuda.set_rng_state(gpu_rng, device)
        elif isinstance(device, str) and 'cuda' in device:
            torch.cuda.set_rng_state(gpu_rng, device)
    np.random.set_state(np_rng)
    random.setstate(py_rng)

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

        before_state = _get_rng_states(device)

        # ------------------------------------------------------------------ #
        # 1. BALD scores over the pool (with dropout enabled)                 #
        # ------------------------------------------------------------------ #
        set_dropout_p(model, original_p)

        bald_fct = query_uncertainty._get_bald_fct(model)

        def _return_all(scores, size):
            return scores, size

        bald_scores_pool, _ = query_uncertainty.query_sampler(
            pool_loader,
            acq_function=bald_fct,
            post_acq_function=_return_all,
            acq_size=acq_size,
            device=device,
        )
        assert len(bald_scores_pool.shape) == 1

        # Q_u  — top-k by BALD (uncertainty arm)
        sorted_bald = np.argsort(bald_scores_pool)[::-1].copy()   # .copy() → positive strides
        acq_indices_bald = sorted_bald[:acq_size]           # pool indices
        acq_vals_bald    = bald_scores_pool[acq_indices_bald]

        mean_bald_Q_u = bald_scores_pool[acq_indices_bald].mean()   # FIX: was using leftout
        mean_bald_Q_d_placeholder = None   # filled below after Vendi selects Q_d

        # BALD scores over the labeled set (baseline reference)
        bald_scores_L, _ = query_uncertainty.query_sampler(
            labeled_loader,
            acq_function=bald_fct,
            post_acq_function=_return_all,
            acq_size=len(labeled_loader.dataset),
            device=device,
        )
        mean_bald_L = float(bald_scores_L.mean())

        bald_state = _get_rng_states(device)
        _set_rng_states(device, before_state)

        # ------------------------------------------------------------------ #
        # 2. Vendi scores / embeddings (dropout disabled)                     #
        # ------------------------------------------------------------------ #
        set_dropout_p(model, 0.0)

        normalization = cfg.query.vendi.normalization
        feat_labeled   = query_diversity.get_embeddings(model, labeled_loader, cpu=False, numpy=False)
        feat_unlabeled = query_diversity.get_embeddings(model, pool_loader,   cpu=False, numpy=False)
        feat_labeled, feat_unlabeled = query_diversity.normalize_features(
            feat_labeled, feat_unlabeled, normalization
        )

        vendi_scores_pool = query_diversity.vendi_from_features(cfg, feat_labeled, feat_unlabeled)

        # Q_d  — top-k by Vendi (diversity arm)
        sorted_vendi      = np.argsort(vendi_scores_pool)[::-1].copy()
        acq_indices_vendi = sorted_vendi[:acq_size]            # pool indices
        acq_vals_vendi    = vendi_scores_pool[acq_indices_vendi]

        # Features for Vendi arm: vendi(L + Q_u), vendi(L + Q_d), vendi(L)
        feat_Q_u   = feat_unlabeled[acq_indices_bald]
        feat_Q_d   = feat_unlabeled[acq_indices_vendi]

        vendi_L       = calculate_vendi_score(cfg, feat_labeled)
        vendi_Q_u_L   = calculate_vendi_score(cfg, torch.cat([feat_Q_u, feat_labeled], dim=0))
        vendi_Q_d_L   = calculate_vendi_score(cfg, torch.cat([feat_Q_d, feat_labeled], dim=0))

        # Now we can compute mean_bald(Q_d): BALD scores of the diversity-selected samples
        mean_bald_Q_d = float(bald_scores_pool[acq_indices_vendi].mean())

        vendi_state = _get_rng_states(device)

        set_dropout_p(model, original_p)

        # ------------------------------------------------------------------ #
        # 3. Class distribution entropy                                       #
        # ------------------------------------------------------------------ #
        num_classes = cfg.query.bandit.get("num_classes", 10)
        labeled_targets = np.array(labeled_loader.dataset.targets)
        cls_dist = _class_distribution_entropy(labeled_targets, num_classes)

        # ------------------------------------------------------------------ #
        # 4. Assemble 6-D context features                                   #
        # ------------------------------------------------------------------ #
        total_iter = cfg.active.num_iter if cfg.active.num_iter > 0 else 1
        t_normalized = (self.count + 1) / total_iter

        # BALD arm: [mean_bald(Q_u), mean_bald(Q_d), mean_bald(L), cls_dist, t, 1]
        features_bald  = np.array([
            mean_bald_Q_u, mean_bald_Q_d, mean_bald_L,
            cls_dist, t_normalized, 1.0
        ])

        # Vendi arm: [vendi(L+Q_u), vendi(L+Q_d), vendi(L), cls_dist, t, 1]
        features_vendi = np.array([
            vendi_Q_u_L, vendi_Q_d_L, vendi_L,
            cls_dist, t_normalized, 1.0
        ])

        if cfg.query.bandit.normalize_features:
            # BALD features: scale by log(C) so they live in roughly [0, 1]
            features_bald[:3] = features_bald[:3] / np.log(num_classes)
            # Vendi features: scale by set sizes  (same logic as before, extended)
            n_L  = feat_labeled.shape[0]
            n_Q  = acq_size
            features_vendi[0] /= (n_L + n_Q)   # vendi(L+Q_u)
            features_vendi[1] /= (n_L + n_Q)   # vendi(L+Q_d)
            features_vendi[2] /= n_L            # vendi(L)

        context_features = np.stack([features_bald, features_vendi])  # (2, 6)

        # ------------------------------------------------------------------ #
        # 5. Select arm and restore RNG to the end-state of the chosen method #
        # ------------------------------------------------------------------ #
        selected_arm = self.bandit_manager.select_arm(context_features)

        if selected_arm == 0:  # BALD / uncertainty
            _set_rng_states(device, bald_state)
            return acq_indices_bald, acq_vals_bald, {"bandit_arm": 0}
        else:                  # Vendi / diversity
            _set_rng_states(device, vendi_state)
            return acq_indices_vendi, acq_vals_vendi, {"bandit_arm": 1}


def calculate_vendi_score(cfg, feats):
    """
    IMPORTANT: sklearn kernels take numpy array or torch tensor on CPU, not tensor on GPU
    """
    vendi_cfg = cfg.query.vendi
    gamma, q, kernel = vendi_cfg.gamma, vendi_cfg.q, vendi_cfg.kernel

    # Kernel matrix
    if kernel == 'rbf':
        K = rbf_kernel(feats, gamma=gamma)
    elif kernel == 'cosine':
        K = cosine_similarity(feats)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    K = torch.as_tensor(K, dtype=torch.float64)
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
            "name": "bandit",
            "bandit": {
                "normalize_features": False,
                "num_classes": 10,
            }
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