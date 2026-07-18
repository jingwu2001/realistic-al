import sys
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("src"))

from query.query_diversity import (
    _get_vendi,
    get_grad_embedding,
    vendi_from_features,
    kernel_self_similarity,
    resolve_gamma,
    DEVICE,
)

INPUT_DIM = 8
FEATURE_DIM = 16
NUM_CLASSES = 4


class MockModel(torch.nn.Module):
    """Minimal model satisfying the get_grad_embedding interface:
    get_features, .model.classifier, and hparams.model.small_head."""

    def __init__(self):
        super().__init__()
        inner = torch.nn.Module()
        inner.classifier = torch.nn.Linear(FEATURE_DIM, NUM_CLASSES)
        self.model = inner
        self.encoder = torch.nn.Linear(INPUT_DIM, FEATURE_DIM)
        self.hparams = OmegaConf.create({"model": {"small_head": True}})

    def get_features(self, x):
        return self.encoder(x)


def make_loader(n, seed):
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(n, INPUT_DIM, generator=g)
    targets = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return DataLoader(TensorDataset(data, targets), batch_size=16)


def make_cfg(kernel="rbf", normalization="minmax", **overrides):
    vendi = {
        "batch_size": 32,
        "normalization": normalization,
        "q": 1.0,
        "gamma": 1.0,
        "kernel": kernel,
        "approx": False,
        "use_grad": True,
        "grad_embedding_type": "linear",
        "labeled_true_labels": True,
    }
    vendi.update(overrides)
    return OmegaConf.create({"query": {"name": "gvendi", "vendi": vendi}})


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MockModel().to(DEVICE)


@pytest.fixture
def loaders():
    return make_loader(40, seed=1), make_loader(120, seed=2)


@pytest.mark.parametrize("kernel", ["rbf", "cosine", "linear"])
@pytest.mark.parametrize("normalization", ["l2", "minmax", "zscore", "none"])
def test_gvendi_kernels_and_normalizations(model, loaders, kernel, normalization):
    """Every kernel x normalization combination runs and returns sane output."""
    labeled_loader, pool_loader = loaders
    acq_size = 10
    cfg = make_cfg(kernel=kernel, normalization=normalization)

    indices, scores, extra_info = _get_vendi(
        cfg, model, labeled_loader, pool_loader, acq_size=acq_size
    )

    assert len(indices) == acq_size
    assert len(scores) == acq_size
    assert len(np.unique(indices)) == acq_size  # no duplicate acquisitions
    assert np.all(np.isfinite(scores))
    assert np.all(np.diff(scores) <= 0)  # descending ranking

    # raw gradient norms are always reported, regardless of normalization
    assert extra_info["grad_norms_acquired"].shape == (acq_size,)
    assert np.all(extra_info["grad_norms_acquired"] > 0)
    assert extra_info["grad_norm_acq"].shape == (3,)  # [max, min, median]
    assert extra_info["grad_norm_else"].shape == (3,)


def test_grad_norms_independent_of_normalization(model, loaders):
    """Norms must come from the raw gradients: identical across normalizations
    (under l2 normalization, norms taken afterwards would all be 1)."""
    labeled_loader, pool_loader = loaders
    norms = {}
    for normalization in ["l2", "none"]:
        cfg = make_cfg(kernel="rbf", normalization=normalization)
        _, _, extra_info = _get_vendi(
            cfg, model, labeled_loader, pool_loader, acq_size=10
        )
        # sort: acquisition order may differ between normalizations
        norms[normalization] = np.sort(extra_info["grad_norms_acquired"])
    assert not np.allclose(norms["l2"], 1.0)


def test_linear_kernel_on_l2_matches_cosine(model, loaders):
    """L2 normalization + linear kernel is exactly the cosine kernel, so the
    Vendi scores (and hence the acquired indices) must coincide."""
    labeled_loader, pool_loader = loaders
    results = {}
    for kernel in ["linear", "cosine"]:
        cfg = make_cfg(kernel=kernel, normalization="l2")
        indices, scores, _ = _get_vendi(
            cfg, model, labeled_loader, pool_loader, acq_size=10
        )
        results[kernel] = (indices, scores)
    np.testing.assert_allclose(
        results["linear"][1], results["cosine"][1], rtol=1e-6
    )
    np.testing.assert_array_equal(results["linear"][0], results["cosine"][0])


def test_feature_vendi_has_no_grad_norms(model, loaders):
    """use_grad=False (classic feature vendi) runs through the same code path
    and must not emit gradient-norm keys."""
    labeled_loader, pool_loader = loaders
    cfg = make_cfg(kernel="rbf", normalization="minmax", use_grad=False)
    indices, scores, extra_info = _get_vendi(
        cfg, model, labeled_loader, pool_loader, acq_size=10
    )
    assert len(indices) == 10
    assert np.all(np.isfinite(scores))
    assert "grad_norms_acquired" not in extra_info
    assert "grad_norm_acq" not in extra_info


def test_resolve_gamma():
    """rbf bandwidth heuristics: gamma = 1/(2 sigma^2) conventions."""
    torch.manual_seed(5)
    x = torch.randn(50, 8)

    assert resolve_gamma(0.5, x) == 0.5
    assert resolve_gamma(2, x) == 2.0  # int accepted

    # dim heuristic: 1 / (2 D)
    assert resolve_gamma("dim", x) == pytest.approx(1.0 / 16)

    # median heuristic: 1 / (2 median^2) — NOT the median itself (P3.1 fix)
    dists = torch.cdist(x, x)
    med = dists[torch.triu(torch.ones(50, 50, dtype=torch.bool), diagonal=1)].median().item()
    assert resolve_gamma("median", x) == pytest.approx(1.0 / (2 * med * med))

    with pytest.raises(ValueError):
        resolve_gamma("nonsense", x)


@pytest.mark.parametrize("gamma", ["median", "dim"])
def test_vendi_runs_with_gamma_heuristics(model, loaders, gamma):
    """String gamma settings work through the full acquisition path."""
    labeled_loader, pool_loader = loaders
    cfg = make_cfg(kernel="rbf", normalization="zscore", gamma=gamma)
    indices, scores, _ = _get_vendi(
        cfg, model, labeled_loader, pool_loader, acq_size=10
    )
    assert len(indices) == 10
    assert np.all(np.isfinite(scores))


def test_kernel_self_similarity():
    x = torch.randn(7, 5)
    assert torch.all(kernel_self_similarity("rbf", x) == 1.0)
    assert torch.all(kernel_self_similarity("cosine", x) == 1.0)
    torch.testing.assert_close(
        kernel_self_similarity("linear", x), (x**2).sum(dim=1)
    )


def test_grad_embedding_true_vs_pseudo_labels(model):
    """Ground-truth labels must change the gradient wherever the model's argmax
    prediction disagrees with the label."""
    torch.manual_seed(3)
    data = torch.randn(30, INPUT_DIM)
    # deliberately random labels so some disagree with the prediction
    targets = torch.randint(0, NUM_CLASSES, (30,))
    loader = DataLoader(TensorDataset(data, targets), batch_size=16)

    grad_pseudo = get_grad_embedding(model, loader, device=DEVICE)
    grad_true = get_grad_embedding(model, loader, device=DEVICE, use_true_labels=True)

    assert grad_pseudo.shape == (30, NUM_CLASSES * FEATURE_DIM)
    assert grad_true.shape == (30, NUM_CLASSES * FEATURE_DIM)
    # neither should require grad (results are detached)
    assert not grad_pseudo.requires_grad and not grad_true.requires_grad

    with torch.no_grad():
        preds = model.model.classifier(model.get_features(data.to(DEVICE))).argmax(1).cpu()
    disagree = preds != targets
    assert disagree.any(), "test setup degenerate: all pseudo-labels correct"
    assert not torch.allclose(grad_pseudo[disagree], grad_true[disagree])
    torch.testing.assert_close(grad_pseudo[~disagree], grad_true[~disagree])


def test_grad_embedding_default_matches_badge_formula(model):
    """BADGE regression guard: the default call (pseudo-labels, as _get_badge
    uses it) must equal the closed form (softmax(out) - onehot(argmax)) ⊗ h.
    Protects BADGE from behavior drift in get_grad_embedding."""
    torch.manual_seed(4)
    data = torch.randn(20, INPUT_DIM)
    targets = torch.randint(0, NUM_CLASSES, (20,))
    loader = DataLoader(TensorDataset(data, targets), batch_size=8)

    emb = get_grad_embedding(model, loader, device=DEVICE)

    with torch.no_grad():
        h = model.get_features(data.to(DEVICE))
        out = model.model.classifier(h)
        p = torch.softmax(out, dim=1)
        err = p - F.one_hot(out.argmax(dim=1), NUM_CLASSES).to(p.dtype)
        # get_grad_embedding lays the C*Z gradient out as (C, Z) row-major
        ref = torch.einsum("bc,bz->bcz", err, h).reshape(20, -1).cpu()

    torch.testing.assert_close(emb, ref, rtol=1e-5, atol=1e-6)


def test_vendi_from_features_takes_subconfig():
    """vendi_from_features accepts the method sub-config directly (shared by
    vendi and gvendi)."""
    cfg = make_cfg(kernel="linear", normalization="none")
    feat_l = torch.randn(20, 6, device=DEVICE)
    feat_u = torch.randn(50, 6, device=DEVICE)
    scores, eig_time = vendi_from_features(cfg.query.vendi, feat_l, feat_u)
    assert scores.shape == (50,)
    assert np.all(np.isfinite(scores))
    assert eig_time >= 0
