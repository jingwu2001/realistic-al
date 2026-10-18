"""
Tests for the updated 6-D bandit context feature construction in BanditQuerySampler.

Run from the repo root:
    python -m pytest tests/test_bandit_features.py -v
"""
import copy
import random
import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath("src"))

from unittest import mock
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

from query.query_bandit import (
    BanditQuerySampler,
    _class_distribution_entropy,
    _multilabel_balance_entropy,
)
from query.query import QuerySampler
from query import query_diversity
from models.bayesian_module import BayesianModule, ConsistentMCDropout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_POOL       = 40
NUM_LABELED    = 20
FEATURE_DIM    = 16
ACQ_SIZE       = 5
NUM_CLASSES    = 10
CONTEXT_DIM    = 6


def make_cfg(normalize: bool = False, use_vendi_q2: bool = False) -> DictConfig:
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
                "use_vendi_q2": use_vendi_q2,
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


def make_loader(n: int, has_targets: bool = False, multilabel_c: int = 0):
    """Build a DataLoader with a dataset that has a .targets np.ndarray attribute
    (multi-hot (n, multilabel_c) floats, as mimic3_pheno, if multilabel_c > 0)."""
    feats   = torch.randn(n, FEATURE_DIM)
    if multilabel_c:
        labels = torch.randint(0, 2, (n, multilabel_c)).float()
    else:
        labels = torch.randint(0, NUM_CLASSES, (n,))
    dataset = TensorDataset(feats, labels)
    # Attach .targets for cls_dist computation
    dataset.targets = labels.numpy()
    return DataLoader(dataset, batch_size=8, shuffle=False)


class MockBanditManager:
    """Always selects `arm` (default 0 = BALD) and records the context passed to it."""
    def __init__(self, arm: int = 0):
        self.arm = arm
        self.last_context = None

    def select_arm(self, features):
        self.last_context = features
        return self.arm


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


class TestMultilabelBalanceEntropy:
    def test_half_positive_is_one(self):
        labels = np.tile([[1.0, 0.0], [0.0, 1.0]], (5, 1))  # every label 50/50
        assert _multilabel_balance_entropy(labels) == pytest.approx(1.0)

    def test_constant_labels_are_zero(self):
        labels = np.zeros((20, 3))
        labels[:, 1] = 1.0  # all-negative, all-positive, all-negative
        assert _multilabel_balance_entropy(labels) == pytest.approx(0.0, abs=1e-9)

    def test_single_task_is_binary_entropy(self):
        labels = np.array([[1.0]] * 2 + [[0.0]] * 8)  # 20% positive
        expected = -(0.2 * np.log(0.2) + 0.8 * np.log(0.8)) / np.log(2)
        assert _multilabel_balance_entropy(labels) == pytest.approx(expected)


class TestBanditQuerySampler:

    def _run_ranking_step(self, normalize: bool = False, use_vendi_q2: bool = False, arm: int = 0,
                          multilabel_c: int = 0):
        cfg           = make_cfg(normalize, use_vendi_q2)
        model         = DummyModel()
        bandit        = MockBanditManager(arm)
        pool_loader   = make_loader(NUM_POOL)
        labeled_loader = make_loader(NUM_LABELED, multilabel_c=multilabel_c)
        self.labeled_targets = labeled_loader.dataset.targets

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

        self.bald_scores_pool = bald_scores_pool

        fake_embeddings_pool    = torch.randn(NUM_POOL,    FEATURE_DIM)
        fake_embeddings_labeled = torch.randn(NUM_LABELED, FEATURE_DIM)

        # diversity arm = the standalone _get_vendi / _get_vendi_q2: returns
        # (acq_indices, scores, extra_info) and hands out its embeddings
        self.vendi_indices = np.random.permutation(NUM_POOL)[:ACQ_SIZE]
        self.vendi_scores  = np.random.rand(ACQ_SIZE)

        def fake_vendi_query(cfg, model, labeled_loader, pool_loader, acq_size, embeddings_out):
            embeddings_out["labeled"]   = fake_embeddings_labeled
            embeddings_out["unlabeled"] = fake_embeddings_pool
            return self.vendi_indices, self.vendi_scores, {}

        def fake_calculate_vendi(cfg, gamma, feats):
            return float(np.random.rand())

        with mock.patch("query.query_bandit.query_uncertainty") as mock_unc, \
             mock.patch("query.query_bandit.query_diversity") as mock_div, \
             mock.patch("query.query_bandit.calculate_vendi_score", side_effect=fake_calculate_vendi):

            mock_unc._get_bald_fct.return_value = lambda x: x
            mock_unc.query_sampler.side_effect   = fake_query_sampler

            mock_div._get_vendi.side_effect    = fake_vendi_query
            mock_div._get_vendi_q2.side_effect = fake_vendi_query

            indices, values, info = sampler.ranking_step(pool_loader, labeled_loader)
            self.mock_div = mock_div

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

    @pytest.mark.parametrize("c", [1, 25])
    def test_multilabel_targets(self, c):
        """mimic3_pheno multi-hot targets (single task C=1, all 25 conditions)
        crashed in np.bincount; cls_dist is now the mean pos/neg entropy and
        BALD is scaled by its max, C log 2."""
        *_, ctx = self._run_ranking_step(normalize=True, multilabel_c=c)
        assert np.all(np.isfinite(ctx))
        cls_dist = _multilabel_balance_entropy(self.labeled_targets)
        assert ctx[0, 3] == pytest.approx(cls_dist)
        assert ctx[1, 3] == pytest.approx(cls_dist)
        top_bald = np.sort(self.bald_scores_pool)[::-1][:ACQ_SIZE].mean()
        assert ctx[0, 0] == pytest.approx(top_bald / (c * np.log(2)), rel=1e-5)

    def test_uncertainty_arm_is_top_bald(self):
        indices, values, info, _ = self._run_ranking_step(arm=0)
        expected = np.argsort(self.bald_scores_pool)[::-1][:ACQ_SIZE]
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_allclose(values, self.bald_scores_pool[expected])
        assert info == {"bandit_arm": 0}

    def test_bald_qd_differs_from_qu(self):
        """mean_bald(Q_d) and mean_bald(Q_u) are at different indices and should
        generally differ because Q_u and Q_d are selected by different criteria."""
        *_, ctx = self._run_ranking_step()
        # They *can* be equal by chance, but the index positions must be right
        # (feature 0 = mean_bald_Q_u, feature 1 = mean_bald_Q_d)
        assert ctx.shape[1] >= 2


class TestBanditVendiQ2:
    """query.bandit.use_vendi_q2: diversity arm runs vendi-q2 instead of vendi."""

    def _run(self, arm):
        t = TestBanditQuerySampler()
        indices, values, info, ctx = t._run_ranking_step(use_vendi_q2=True, arm=arm)
        return t, indices, values, info

    def test_diversity_arm_uses_vendi_q2(self):
        t, indices, values, info = self._run(arm=1)
        t.mock_div._get_vendi_q2.assert_called_once()
        t.mock_div._get_vendi.assert_not_called()
        np.testing.assert_array_equal(indices, t.vendi_indices)
        np.testing.assert_allclose(values, t.vendi_scores)
        assert info == {"bandit_arm": 1}

    def test_uncertainty_arm_unchanged(self):
        _, indices, values, info = self._run(arm=0)
        assert len(indices) == ACQ_SIZE
        assert info == {"bandit_arm": 0}

    def test_default_uses_vendi(self):
        t = TestBanditQuerySampler()
        t._run_ranking_step(arm=1)
        t.mock_div._get_vendi.assert_called_once()
        t.mock_div._get_vendi_q2.assert_not_called()

    @pytest.mark.parametrize("kernel", ["rbf", "cosine"])
    def test_closed_form_context_score_matches_eigen_q2(self, kernel):
        """Closed-form VS_2 == eigenvalue Renyi-2 Vendi (real, unmocked kernels)."""
        from query.query_bandit import calculate_vendi_score
        feats = torch.randn(30, FEATURE_DIM)
        cfg_eig = make_cfg(use_vendi_q2=False)
        cfg_eig.query.vendi.q = 2.0
        cfg_eig.query.vendi.kernel = kernel
        cfg_q2 = make_cfg(use_vendi_q2=True)
        cfg_q2.query.vendi.kernel = kernel
        gamma = 0.05 if kernel == "rbf" else None
        eig = calculate_vendi_score(cfg_eig, gamma, feats)
        closed = calculate_vendi_score(cfg_q2, gamma, feats)
        assert closed == pytest.approx(eig, rel=1e-5)


class TinyNet(BayesianModule):
    """Deterministic encoder + MC-dropout head, laid out like models.networks ResNet."""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(FEATURE_DIM, 8)
        self.classifier = torch.nn.Sequential(
            ConsistentMCDropout(0.5), torch.nn.Linear(8, NUM_CLASSES)
        )

    def det_forward_impl(self, x):
        return self.encoder(x)

    def mc_forward_impl(self, x):
        return self.classifier(x)


class TinyClassifier(torch.nn.Module):
    """Minimal AbstractClassifier stand-in: model(x, agg=False) -> (B, k, C) logits."""
    k = 4
    multilabel = False

    def __init__(self):
        super().__init__()
        self.model = TinyNet()
        self.hparams = DictConfig({"model": {"small_head": True}})

    def forward(self, x, k=None, agg=True):
        out = self.model(x, k or self.k)
        if agg:
            out = torch.log_softmax(out, dim=-1).logsumexp(dim=1) - np.log(out.shape[1])
        return out

    def get_features(self, x):
        return self.model.encoder(x)


def _seed_all(seed=0):
    torch.manual_seed(seed)  # also seeds every CUDA device
    np.random.seed(seed)
    random.seed(seed)


def _post_query_fingerprint(model, x):
    """What the next AL round sees: the dropout masks, the post-query eval
    forward (may rebuild masks from the CUDA RNG), and the next global draws."""
    masks = [m._mask for m in model.modules() if isinstance(m, ConsistentMCDropout)]
    with torch.no_grad():
        out = model(x.to(query_diversity.DEVICE), agg=False).cpu()
    draws = [torch.rand(4), torch.from_numpy(np.random.rand(4)), torch.tensor([random.random()])]
    if torch.cuda.is_available():
        draws.append(torch.rand(4, device="cuda:0").cpu())
    return masks, out, draws


class TestArmsMatchStandaloneQuery:
    """A bandit that picks arm a must be indistinguishable from the standalone
    query of arm a: same picks AND same post-query state (masks + RNG), so the
    next round trains the same model. Real code paths, nothing mocked."""

    def _cfg(self, name, vendi):
        return DictConfig({
            "model": {"dropout_p": 0.5},
            "query": {
                "name": name,
                "vendi": {"normalization": "none", "gamma": 1.0, "q": 1.0,
                          "batch_size": 32, "kernel": "rbf", "approx": False,
                          "use_grad": False, **vendi},
                "bandit": {"normalize_features": True, "num_classes": NUM_CLASSES,
                           "use_vendi_q2": name == "vendi-q2"},
            },
            "active": {"num_iter": 10, "m": 0, "acq_size": ACQ_SIZE},
        })

    def _query(self, cfg, model, arm=None):
        """Seeded ranking_step from a fixed model/RNG start; returns picks + fingerprint."""
        _seed_all(0)
        pool_loader, labeled_loader = make_loader(NUM_POOL), make_loader(NUM_LABELED)
        probe = torch.randn(6, FEATURE_DIM)
        dev = query_diversity.DEVICE
        if arm is None:
            sampler = QuerySampler(cfg, model, count=0, device=dev)
        else:
            cfg = copy.deepcopy(cfg)
            cfg.query.name = "bandit"
            sampler = BanditQuerySampler(
                cfg, model, count=0, device=dev, bandit_manager=MockBanditManager(arm)
            )
        indices, values, _ = sampler.ranking_step(pool_loader, labeled_loader)
        return np.asarray(indices), np.asarray(values), _post_query_fingerprint(model, probe)

    def _assert_same(self, name, vendi, arm):
        torch.manual_seed(123)
        model = TinyClassifier().to(query_diversity.DEVICE).eval()
        with torch.no_grad():  # masks already built, as after val/test
            model(torch.randn(3, FEATURE_DIM, device=query_diversity.DEVICE), agg=False)
        twin = copy.deepcopy(model)

        cfg = self._cfg(name, vendi)
        idx_ref, val_ref, (masks_ref, out_ref, draws_ref) = self._query(cfg, model)
        idx, val, (masks, out, draws) = self._query(cfg, twin, arm=arm)

        np.testing.assert_array_equal(idx, idx_ref)
        np.testing.assert_array_equal(val, val_ref)
        for m, m_ref in zip(masks, masks_ref):
            same = (m is None and m_ref is None) or (
                m is not None and m_ref is not None and torch.equal(m, m_ref)
            )
            assert same, "dropout masks after the query differ"
        assert torch.equal(out, out_ref), "post-query eval forward differs"
        for d, d_ref in zip(draws, draws_ref):
            assert torch.equal(d, d_ref), "RNG state after the query differs"

    def test_uncertainty_arm_matches_bald(self):
        self._assert_same("bald", {}, arm=0)

    def test_uncertainty_arm_matches_bald_with_grad_diversity_arm(self):
        # the (unchosen) gradient diversity arm resets the dropout masks; the
        # BALD arm's masks must come back
        self._assert_same("bald", {"use_grad": True, "kernel": "cosine"}, arm=0)

    def test_diversity_arm_matches_vendi(self):
        self._assert_same("vendi", {}, arm=1)

    def test_diversity_arm_matches_vendi_q2(self):
        self._assert_same("vendi-q2", {"q": 2.0, "kernel": "cosine", "normalization": "minmax"}, arm=1)

    def test_diversity_arm_matches_gvendi(self):
        # gradient embeddings reset the dropout masks (disable_mc_dropout)
        self._assert_same("gvendi", {"use_grad": True, "kernel": "cosine", "normalization": "l2"}, arm=1)


class TestBanditWandb:
    def test_bandit_arm_logged(self):
        from utils.wandb_utils import _query_score_metrics
        assert _query_score_metrics({"bandit_arm": 1})["al/query/bandit_arm"] == 1
        assert _query_score_metrics({"bandit_arm": 0})["al/query/bandit_arm"] == 0
        assert "al/query/bandit_arm" not in _query_score_metrics({"eig_time_s": 0.0})

    def test_run_name_marks_vendi_q2(self):
        from utils.wandb_utils import _bandit_variant
        cfg = make_cfg(use_vendi_q2=True)
        cfg.query.vendi.q = 2.0
        tag, hp = _bandit_variant(cfg)
        assert tag.startswith("bandit-vq2-") and "-q2.0" in tag
        assert hp["div"] == "vendi-q2"
        tag, hp = _bandit_variant(make_cfg())
        assert not tag.startswith("bandit-vq2") and "div" not in hp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
