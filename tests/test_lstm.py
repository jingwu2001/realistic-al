"""Tests for the Harutyunyan et al. 2017 benchmark LSTM (model=lstm).

Covers the BayesianModule contract the AL loop relies on (deterministic
backbone at eval, consistent MC-dropout head), the Keras initialisation, the
BADGE/gvendi gradient embedding through the LSTM, and the wandb data name that
keeps LSTM runs from pooling with the SAND runs.
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.networks import build_model
from query.query_diversity import DEVICE, get_grad_embedding
from utils import wandb_utils

T, D_INP, D_STATIC = 24, 387, 2


def _cfg(**model_overrides):
    model = OmegaConf.load("src/config/model/lstm.yaml")
    model.update(model_overrides)
    return OmegaConf.create({"model": model})


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, D_INP, generator=g), torch.randn(n, D_STATIC, generator=g)


@pytest.mark.parametrize(
    "depth,use_static,num_classes",
    [(1, True, 1), (1, False, 25), (2, True, 1), (3, False, 2)],
)
def test_lstm_bayesian_contract(depth, use_static, num_classes):
    torch.manual_seed(0)
    cfg = _cfg(depth=depth, use_static=use_static)
    model = build_model(cfg, num_classes=num_classes)
    x = _batch()
    dim = cfg.model.dim

    model.eval()
    feats = model.get_features(x)
    assert feats.shape == (6, dim * (2 if use_static else 1))
    assert torch.equal(feats, model.get_features(x)), "backbone must be deterministic at eval"

    out = model(x, 10)
    assert out.shape == (6, 10, num_classes)
    assert not torch.allclose(out[:, 0], out[:, 1]), "MC samples should differ"
    assert torch.equal(out, model(x, 10)), "ConsistentMCDropout masks must be reused"

    model.train()
    assert not torch.equal(model.get_features(x), model.get_features(x)), (
        "LSTM input dropout should be active in training"
    )
    model(x, 1).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("use_static", [True, False])
def test_sand_use_static(use_static):
    """SAND with use_static=False is time-series only: features drop the hid_dim
    demo block and are invariant to the demo input."""
    torch.manual_seed(0)
    model_cfg = OmegaConf.load("src/config/model/sand.yaml")
    model_cfg.update(use_static=use_static)
    model = build_model(OmegaConf.create({"model": model_cfg}), num_classes=1).eval()
    ts, demo = _batch()
    m = model_cfg
    feats = model.get_features((ts, demo))
    assert feats.shape == (6, m.hid_dim * m.M + (m.hid_dim if use_static else 0))
    assert model.classifier.in_features == feats.shape[1]
    assert hasattr(model, "demo_emb") == use_static
    same = torch.equal(feats, model.get_features((ts, demo + 1.0)))
    assert same != use_static, "demo must matter iff use_static"
    assert model((ts, demo), 10).shape == (6, 10, 1)


def test_sand_use_static_default_is_backward_compatible():
    """Omitting use_static (old configs) builds the original demo-concat SAND,
    with an identical parameter layout and init stream."""
    model_cfg = OmegaConf.load("src/config/model/sand.yaml")
    del model_cfg["use_static"]
    torch.manual_seed(0)
    old = build_model(OmegaConf.create({"model": model_cfg}), num_classes=1)
    model_cfg.use_static = True
    torch.manual_seed(0)
    new = build_model(OmegaConf.create({"model": model_cfg}), num_classes=1)
    sd_old, sd_new = old.state_dict(), new.state_dict()
    assert list(sd_old) == list(sd_new)
    assert all(torch.equal(sd_old[k], sd_new[k]) for k in sd_old)


def test_lstm_keras_init():
    model = build_model(_cfg(depth=2), num_classes=1)
    for lstm in [*model.bilstms, model.lstm]:
        h = lstm.hidden_size
        for name, p in lstm.named_parameters():
            if name.startswith("bias_ih"):
                # unit_forget_bias: gate order (i, f, g, o)
                assert torch.all(p[h : 2 * h] == 1) and torch.all(p[:h] == 0)
                assert torch.all(p[2 * h :] == 0)
            elif name.startswith("bias_hh"):
                assert torch.all(p == 0)
            elif name.startswith("weight_hh"):
                assert torch.allclose(p.T @ p, torch.eye(h), atol=1e-5)  # orthogonal


class _TupleDataset(Dataset):
    """Yields ((ts, demo), y) like MIMIC3PhenotypeDataset in single-task mode."""

    def __init__(self, ts, demo, y):
        self.ts, self.demo, self.y = ts, demo, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.ts[i], self.demo[i]), self.y[i]


class _Wrapper(torch.nn.Module):
    """The slice of models.bayesian.BayesianModule that get_grad_embedding uses."""

    def __init__(self, net):
        super().__init__()
        self.model = net
        self.hparams = OmegaConf.create({"model": {"small_head": False}})
        self.multilabel = True

    def get_features(self, x):
        return self.model.get_features(x)


@pytest.mark.parametrize("use_true_labels", [True, False])
def test_lstm_grad_embedding_matches_analytic(use_true_labels):
    """Single-task BADGE embedding through the LSTM = (sigmoid(z) - y) * h_x,
    with h_x the deterministic LSTM(+demo) features (bare Linear classifier,
    so no classifier[:-1] slicing even though small_head is False)."""
    torch.manual_seed(0)
    wrapper = _Wrapper(build_model(_cfg(), num_classes=1)).to(DEVICE).eval()
    ts, demo = _batch(n=10)
    y = torch.randint(0, 2, (10, 1)).float()
    loader = DataLoader(_TupleDataset(ts, demo, y), batch_size=4)

    emb = get_grad_embedding(
        wrapper, loader, grad_embedding_type="linear", device=DEVICE,
        use_true_labels=use_true_labels,
    )
    with torch.no_grad():
        h = wrapper.get_features((ts.to(DEVICE), demo.to(DEVICE)))
        p = torch.sigmoid(wrapper.model.classifier(h))
        target = y.to(DEVICE) if use_true_labels else (p > 0.5).float()
        expected = ((p - target) * h).cpu()
    assert emb.shape == (10, h.shape[1])
    assert torch.allclose(emb, expected, atol=1e-5)


@pytest.mark.parametrize(
    "data_name,model_name,task_id,use_static,expected",
    [
        ("mimic3_pheno", "sand", 8, None, "mimic3_pheno_t08"),       # unchanged
        ("mimic3_pheno", "sand", 8, True, "mimic3_pheno_t08"),       # explicit default
        ("mimic3_pheno", "lstm", 8, None, "mimic3_pheno_t08_lstm"),
        ("mimic3_pheno", "lstm", None, None, "mimic3_pheno_lstm"),   # multi-task
        ("mimic3_los", "lstm", None, None, "mimic3_los_lstm"),
        ("cifar10", "resnet", None, None, "cifar10"),                # non-MIMIC untouched
        # time-series-only (no demographics) variants
        ("mimic3_pheno", "sand", 8, False, "mimic3_pheno_t08_nodemo"),
        ("mimic3_pheno", "lstm", 11, False, "mimic3_pheno_t11_lstm_nodemo"),
        ("p12_transformer", "transformer2", None, False, "p12_transformer"),  # non-MIMIC untouched
    ],
)
def test_wandb_data_name_tags_non_sand_mimic_models(data_name, model_name, task_id, use_static, expected):
    data = {"name": data_name}
    if task_id is not None:
        data.update(multitask=False, task_id=task_id)
    model = {"name": model_name}
    if use_static is not None:
        model["use_static"] = use_static
    cfg = OmegaConf.create({
        "data": data,
        "model": model,
        "query": {"name": "random"},
        "active": {"num_labelled": 100, "acq_size": 100, "num_iter": 10},
        "trainer": {
            "use_wandb": True, "dry_run": False, "seed": 12345,
            "experiment_id": "2026-10-04_12-00-00-000000", "wandb_notes": "",
            "wandb_project": "test", "is_test": False, "wandb_test": False,
        },
    })
    hydra = SimpleNamespace(runtime=SimpleNamespace(choices={"active": "mimic3_med", "data": data_name}))
    with mock.patch.object(wandb_utils.HydraConfig, "get", return_value=hydra), \
         mock.patch.object(wandb_utils.wandb, "init") as init:
        wandb_utils.init_wandb(cfg)
    kwargs = init.call_args.kwargs
    assert kwargs["name"] == f"{expected}/random/seed-12345"
    assert kwargs["group"] == f"{expected}/mimic3_med/random"
    assert expected in kwargs["tags"]
    assert kwargs["config"]["data_name"] == expected
