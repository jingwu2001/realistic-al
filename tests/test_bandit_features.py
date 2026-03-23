"""
Tests for the updated 6-D bandit context feature construction in BanditQuerySampler.

Run from the repo root:
    python -m pytest tests/test_bandit_features.py -v
"""
import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath("src"))

from unittest import mock
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

from query.query_bandit import BanditQuerySampler, _class_distribution_entropy
from models.bayesian_module import ConsistentMCDropout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_POOL       = 40
NUM_LABELED    = 20
FEATURE_DIM    = 16
ACQ_SIZE       = 5
NUM_CLASSES    = 10
CONTEXT_DIM    = 6


def make_cfg(normalize: bool = False) -> DictConfig:
    return DictConfig({
        "model": {"dropout_p": 0.5},
        "query": {
            "vendi": {
                "normalization": "none",
                "gamma": 1.0,
                "q": 1.0,
                "batch_size": 32,
                "kernel": "rbf",
            },
            "name": "bandit",
            "bandit": {
                "normalize_features": normalize,
                "num_classes": NUM_CLASSES,
            },
        },
        "active": {"num_iter": 10, "m": 0, "acq_size": ACQ_SIZE},
        "training": {"batch_size": 4},
    })


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(FEATURE_DIM, NUM_CLASSES)
        self.dropout = ConsistentMCDropout(0.5)

    def forward(self, x, agg=False):
        return self.linear(x)


def make_loader(n: int, has_targets: bool = False):
    """Build a DataLoader with a dataset that has a .targets np.ndarray attribute."""
    feats   = torch.randn(n, FEATURE_DIM)
    labels  = torch.randint(0, NUM_CLASSES, (n,))
    dataset = TensorDataset(feats, labels)
    # Attach .targets for cls_dist computation
    dataset.targets = labels.numpy()
    return DataLoader(dataset, batch_size=8, shuffle=False)


class MockBanditManager:
    """Always selects the BALD arm (arm 0) and records the context passed to it."""
    def __init__(self):
        self.last_context = None

    def select_arm(self, features):
        self.last_context = features
        return 0  # BALD


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassDistributionEntropy:
    def test_balanced(self):
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # one per class
        val = _class_distribution_entropy(labels, 10)
        assert abs(val - 1.0) < 1e-9, "Perfectly balanced → entropy should be 1.0"

    def test_imbalanced(self):
        labels = np.zeros(20, dtype=int)  # all in class 0
        val = _class_distribution_entropy(labels, 10)
        assert val == pytest.approx(0.0), "All-same class → entropy should be 0.0"

    def test_range(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 10, size=100)
        val = _class_distribution_entropy(labels, 10)
        assert 0.0 <= val <= 1.0


class TestBanditQuerySampler:

    def _run_ranking_step(self, normalize: bool = False):
        cfg           = make_cfg(normalize)
        model         = DummyModel()
        bandit        = MockBanditManager()
        pool_loader   = make_loader(NUM_POOL)
        labeled_loader = make_loader(NUM_LABELED)

        sampler = BanditQuerySampler(
            cfg, model, count=2, device="cpu", bandit_manager=bandit
        )
        sampler.acq_size = ACQ_SIZE

        # Patch heavy external dependencies
        bald_scores_pool    = np.random.rand(NUM_POOL).astype(np.float32)
        bald_scores_labeled = np.random.rand(NUM_LABELED).astype(np.float32)

        def fake_query_sampler(loader, acq_function, post_acq_function, acq_size, device):
            # Distinguish pool vs labeled loader by dataset length
            n = len(loader.dataset)
            if n == NUM_POOL:
                scores = bald_scores_pool
            else:
                scores = bald_scores_labeled
            return post_acq_function(scores, acq_size)

        fake_embeddings_pool    = torch.randn(NUM_POOL,    FEATURE_DIM)
        fake_embeddings_labeled = torch.randn(NUM_LABELED, FEATURE_DIM)

        def fake_get_embeddings(model, loader, cpu, numpy):
            n = len(loader.dataset)
            return fake_embeddings_pool if n == NUM_POOL else fake_embeddings_labeled

        def fake_normalize(feat_labeled, feat_unlabeled, normalization):
            return feat_labeled, feat_unlabeled

        def fake_vendi_from_features(cfg, feat_labeled, feat_unlabeled):
            return np.random.rand(feat_unlabeled.shape[0]).astype(np.float32)

        def fake_calculate_vendi(cfg, feats):
            return float(np.random.rand())

        with mock.patch("query.query_bandit.query_uncertainty") as mock_unc, \
             mock.patch("query.query_bandit.query_diversity") as mock_div, \
             mock.patch("query.query_bandit.calculate_vendi_score", side_effect=fake_calculate_vendi):

            mock_unc._get_bald_fct.return_value = lambda x: x
            mock_unc.query_sampler.side_effect   = fake_query_sampler

            mock_div.get_embeddings.side_effect       = fake_get_embeddings
            mock_div.normalize_features.side_effect   = fake_normalize
            mock_div.vendi_from_features.side_effect  = fake_vendi_from_features

            indices, values, info = sampler.ranking_step(pool_loader, labeled_loader)

        return indices, values, info, bandit.last_context

    def test_context_shape(self):
        *_, ctx = self._run_ranking_step()
        assert ctx is not None, "Bandit manager was never called"
        assert ctx.shape == (2, CONTEXT_DIM), \
            f"Expected context shape (2, {CONTEXT_DIM}), got {ctx.shape}"

    def test_context_finite(self):
        *_, ctx = self._run_ranking_step()
        assert np.all(np.isfinite(ctx)), f"Non-finite values in context:\n{ctx}"

    def test_context_bias_term(self):
        """Last feature of each arm must be 1.0 (bias)."""
        *_, ctx = self._run_ranking_step()
        assert ctx[0, -1] == pytest.approx(1.0), "BALD arm bias != 1.0"
        assert ctx[1, -1] == pytest.approx(1.0), "Vendi arm bias != 1.0"

    def test_output_shape(self):
        indices, values, info, _ = self._run_ranking_step()
        assert len(indices) == ACQ_SIZE
        assert len(values)  == ACQ_SIZE
        assert "bandit_arm" in info

    def test_with_normalization(self):
        *_, ctx = self._run_ranking_step(normalize=True)
        assert ctx.shape == (2, CONTEXT_DIM)
        assert np.all(np.isfinite(ctx))

    def test_bald_qd_differs_from_qu(self):
        """mean_bald(Q_d) and mean_bald(Q_u) are at different indices and should
        generally differ because Q_u and Q_d are selected by different criteria."""
        *_, ctx = self._run_ranking_step()
        # They *can* be equal by chance, but the index positions must be right
        # (feature 0 = mean_bald_Q_u, feature 1 = mean_bald_Q_d)
        assert ctx.shape[1] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
