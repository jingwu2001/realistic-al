"""Tests for SAND's banded causal self-attention.

model=sand is real attention (nn.TransformerEncoder). model=sand_legacy keeps
the STraTS combine step, which never applies the attention weights (o_t = v_t,
then a head/time shuffle), so all SAND runs before 2026-10-24 reproduce.
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.networks import build_model
from utils import wandb_utils

T, D_INP, D_STATIC = 24, 387, 2


def _sand(seed=0, num_classes=1, **model_overrides):
    # sand_legacy.yaml is sand.yaml with another name (via the Hydra defaults list)
    model_cfg = OmegaConf.load("src/config/model/sand.yaml")
    model_cfg.update(model_overrides)
    torch.manual_seed(seed)
    return build_model(OmegaConf.create({"model": model_cfg}), num_classes=num_classes)


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, D_INP, generator=g), torch.randn(n, D_STATIC, generator=g)


def _hidden(n=3, d=64, seed=1):
    return torch.randn(n, T, d, generator=torch.Generator().manual_seed(seed))


def _changed_positions(layer, x, hour):
    """Output positions of one layer that move when input `hour` does."""
    x2 = x.clone()
    x2[:, hour] += 10.0
    delta = (layer(x2) - layer(x)).abs().amax(dim=(0, 2))
    return (delta > 1e-5).nonzero().flatten().tolist()


@torch.no_grad()
@pytest.mark.parametrize("r,hour,expected", [
    (24, 10, list(range(10, 24))),  # r = T: plain causal, every later hour sees it
    (3, 10, [10, 11, 12, 13]),       # banded: only t with t-3 <= 10 <= t
    (3, 23, [23]),
])
def test_attention_receptive_field(r, hour, expected):
    model = _sand(r=r).eval()
    layer = lambda h: model.transformer.layers[0](h, src_mask=model.mask)
    assert _changed_positions(layer, _hidden(), hour) == expected


@torch.no_grad()
def test_attention_uses_query_key_weights():
    model = _sand().eval()
    x = _batch()
    feats = model.get_features(x)
    model.transformer.layers[0].self_attn.in_proj_weight[:64].normal_()  # Wq
    assert not torch.allclose(model.get_features(x), feats, atol=1e-4)


@torch.no_grad()
def test_legacy_attention_ignores_weights():
    """Pin the STraTS behaviour (model=sand_legacy) so earlier runs reproduce: at
    eval the output is Wo of the values reshaped (B, h, T, dk) -> (B, T, d),
    independent of Wq/Wk, and each input hour lands at h shuffled positions."""
    model = _sand(name="sand_legacy").eval()
    mha, x = model.transformer[0].mha, _hidden()
    B, h, dk = x.shape[0], mha.num_heads, mha.dk
    v = (x @ mha.Wv).view(B, T, h, dk).permute(0, 2, 1, 3)
    out = mha(x, model.mask)
    assert torch.allclose(out, mha.Wo(v.reshape(B, T, h * dk)), atol=1e-5)

    mha.Wq.normal_()
    mha.Wk.normal_()
    assert torch.allclose(mha(x, model.mask), out, atol=1e-5), "Wq/Wk must not matter"
    layer = lambda h: mha(h, model.mask)
    assert _changed_positions(layer, x, 0) == [0, 6, 12, 18]
    assert _changed_positions(layer, x, 23) == [5, 11, 17, 23]


@pytest.mark.parametrize("name", ["sand", "sand_legacy"])
def test_bayesian_contract(name):
    model = _sand(name=name, num_classes=25)
    m = OmegaConf.load("src/config/model/sand.yaml")
    x = _batch()

    model.eval()
    feats = model.get_features(x)
    assert feats.shape == (6, m.hid_dim * m.M + m.hid_dim)
    assert torch.equal(feats, model.get_features(x)), "backbone must be deterministic at eval"
    out = model(x, 10)
    assert out.shape == (6, 10, 25)
    assert not torch.allclose(out[:, 0], out[:, 1]), "MC samples should differ"
    assert torch.equal(out, model(x, 10)), "ConsistentMCDropout masks must be reused"

    model.train()
    model(x, 1).sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "data_name,task_id,model_overrides,expected",
    [
        ("mimic3_pheno", 8, {"name": "sand_legacy"}, "mimic3_pheno_t08"),  # earlier runs' name
        ("mimic3_pheno", 8, {}, "mimic3_pheno_t08_attnfix"),
        ("mimic3_pheno", 8, {"d_static_out": 2}, "mimic3_pheno_t08_demo2_attnfix"),
        ("mimic3_pheno", 11, {"use_static": False}, "mimic3_pheno_t11_nodemo_attnfix"),
        ("mimic3_los", None, {}, "mimic3_los_attnfix"),
    ],
)
def test_wandb_data_name_tags_sand(data_name, task_id, model_overrides, expected):
    data = {"name": data_name}
    if task_id is not None:
        data.update(multitask=False, task_id=task_id)
    cfg = OmegaConf.create({
        "data": data,
        "model": {"name": "sand", **model_overrides},
        "query": {"name": "random"},
        "active": {"num_labelled": 100, "acq_size": 100, "num_iter": 10},
        "trainer": {
            "use_wandb": True, "dry_run": False, "seed": 12345,
            "experiment_id": "2026-10-11_12-00-00-000000", "wandb_notes": "",
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


@torch.no_grad()
def test_trailing_padding_excluded_from_features():
    """Right-padded stays (benchmark arrays): the causal mask hides padded hours
    from real ones, and the interpolation drops them, so anything that only acts
    at padded positions (here their positional encodings) leaves the features
    unchanged."""
    model = _sand(use_static=False).eval()
    ts, demo = _batch()
    ts[:, 15:] = 0                                    # stays of 15 h, padded to T = 24
    feats = model.get_features((ts, demo))
    model.positional_encoding[:, 15:] += 5.0
    assert torch.allclose(model.get_features((ts, demo)), feats, atol=1e-6)
    model.positional_encoding[:, 14] += 5.0           # a real hour does change them
    assert not torch.allclose(model.get_features((ts, demo)), feats, atol=1e-4)
