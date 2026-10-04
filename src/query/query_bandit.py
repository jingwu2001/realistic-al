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

def _get_query_state(model: nn.Module, device):
    """Everything a query can change that later rounds depend on: the global
    RNGs (the next round's weight init, sampler and augmentation seeds draw from
    them) and the ConsistentMCDropout masks (reused by the post-query val/test
    evaluation; rebuilding a mask draws from the CUDA RNG)."""
    masks = [
        (module, module.p, module._mask)
        for module in model.modules()
        if isinstance(module, ConsistentMCDropout)
    ]
    return _get_rng_states(device), masks

def _set_query_state(device, state):
    rng_states, masks = state
    _set_rng_states(device, rng_states)
    for module, p, mask in masks:
        module.p, module._mask = p, mask

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

        # state_before -> BALD arm -> state_after_bald -> (context features)
        # -> back to state_before -> vendi arm -> state_after_vendi -> (context)
        # -> choose an arm and restore the state after that arm, so a bandit that
        # keeps picking one arm reproduces that method's run exactly. "State"
        # includes the MC-dropout masks: the gradient embedding turns dropout off
        # and back on inside the vendi arm (disable_mc_dropout), clearing them.
        state_before = _get_query_state(model, device)

        # ------------------------------------------------------------------ #
        # 1. Uncertainty arm == query=bald                                   #
        # ------------------------------------------------------------------ #
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

        # Q_u  — top-k by BALD (uncertainty arm), same ranking as query=bald
        acq_indices_bald = np.argsort(bald_scores_pool)[::-1][:acq_size].copy()   # .copy() → positive strides
        acq_vals_bald    = bald_scores_pool[acq_indices_bald]

        state_after_bald = _get_query_state(model, device)

        mean_bald_Q_u = float(acq_vals_bald.mean())

        # BALD scores over the labeled set (baseline reference), with the same
        # MC-dropout masks as the pool pass
        bald_scores_L, _ = query_uncertainty.query_sampler(
            labeled_loader,
            acq_function=bald_fct,
            post_acq_function=_return_all,
            acq_size=len(labeled_loader.dataset),
            device=device,
        )
        mean_bald_L = float(bald_scores_L.mean())

        # ------------------------------------------------------------------ #
        # 2. Diversity arm == query=vendi / gvendi, or vendi-q2              #
        # ------------------------------------------------------------------ #
        _set_query_state(device, state_before)

        vendi_query = query_diversity._get_vendi_q2 if _use_vendi_q2(cfg) else query_diversity._get_vendi
        embeddings = {}  # the arm's normalized embeddings, reused for the context
        acq_indices_vendi, acq_vals_vendi, _ = vendi_query(
            cfg, model, labeled_loader, pool_loader,
            acq_size=acq_size, embeddings_out=embeddings,
        )
        acq_indices_vendi = np.ascontiguousarray(acq_indices_vendi)

        state_after_vendi = _get_query_state(model, device)

        feat_labeled, feat_unlabeled = embeddings["labeled"], embeddings["unlabeled"]

        # Features for Vendi arm: vendi(L + Q_u), vendi(L + Q_d), vendi(L)
        feat_Q_u   = feat_unlabeled[acq_indices_bald]
        feat_Q_d   = feat_unlabeled[acq_indices_vendi]

        
        # resolve the rbf bandwidth on the same (normalized) embeddings the
        # kernel sees; shared float/"median"/"dim" heuristics with vendi
        gamma = None
        if cfg.query.vendi.kernel == "rbf":
            gamma = query_diversity.resolve_gamma(cfg.query.vendi.gamma, feat_labeled)

        vendi_L       = calculate_vendi_score(cfg, gamma, feat_labeled)
        vendi_Q_u_L   = calculate_vendi_score(cfg, gamma, torch.cat([feat_Q_u, feat_labeled], dim=0))
        vendi_Q_d_L   = calculate_vendi_score(cfg, gamma, torch.cat([feat_Q_d, feat_labeled], dim=0))

        # Now we can compute mean_bald(Q_d): BALD scores of the diversity-selected samples
        mean_bald_Q_d = float(bald_scores_pool[acq_indices_vendi].mean())

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
        # 5. Select arm and restore the end state of the chosen method       #
        # ------------------------------------------------------------------ #
        selected_arm = self.bandit_manager.select_arm(context_features)

        if selected_arm == 0:  # BALD / uncertainty
            _set_query_state(device, state_after_bald)
            return acq_indices_bald, acq_vals_bald, {"bandit_arm": 0}
        else:                  # Vendi / diversity
            _set_query_state(device, state_after_vendi)
            return acq_indices_vendi, acq_vals_vendi, {"bandit_arm": 1}


def _use_vendi_q2(cfg) -> bool:
    """query.bandit.use_vendi_q2: diversity arm runs vendi-q2 instead of vendi."""
    return bool(cfg.query.bandit.get("use_vendi_q2", False))


def calculate_vendi_score(cfg, gamma, feats):
    """
    IMPORTANT: sklearn kernels take numpy array or torch tensor on CPU, not tensor on GPU
    """
    vendi_cfg = cfg.query.vendi
    q, kernel = vendi_cfg.q, vendi_cfg.kernel

    if _use_vendi_q2(cfg):
        # closed-form VS_2 = tr(K)^2 / ||K||_F^2, no eigensolve; float64 before
        # the kernel, matching query_diversity.vendi_q2_from_features
        feats = feats.to(torch.float64)
        K = query_diversity.compute_kernel_matrix(kernel, gamma, feats, feats)
        K = torch.as_tensor(K, dtype=torch.float64)
        return float(torch.diagonal(K).sum() ** 2 / (K ** 2).sum().clamp(min=1e-12))

    # Kernel matrix — shared implementations (rbf/cosine/linear)
    K = query_diversity.compute_kernel_matrix(kernel, gamma, feats, feats)
    K = torch.as_tensor(K, dtype=torch.float64)
    # Eigenvalues, normalized by the trace: identical to the previous /N for
    # unit-diagonal kernels (rbf/cosine), required for linear (diagonal != 1)
    ev = torch.linalg.eigvalsh(K).clamp(min=0)
    ev = ev / ev.sum().clamp(min=1e-12)
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

        # Mock the diversity arm (_get_vendi): fills the normalized embeddings
        # and returns (acq_indices, scores, extra_info)
        def fake_get_vendi(cfg, model, labeled_loader, pool_loader, acq_size, embeddings_out):
            embeddings_out["labeled"], embeddings_out["unlabeled"] = torch.randn(20, 10), torch.randn(20, 10)
            return np.arange(acq_size), np.random.rand(acq_size), {}
        mock_div._get_vendi.side_effect = fake_get_vendi

        # Mock the kernel matrix and renyi_entropy for calculate_vendi_score
        # (calculate_vendi_score uses query_diversity.compute_kernel_matrix)
        mock_div.compute_kernel_matrix.return_value = torch.randn(5, 5)
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
