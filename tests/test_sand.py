"""Tests for SAND's masked self-attention (model=sand, model.attn_fix).

attn_fix=False keeps the STraTS combine step, which never applies the attention
weights (o_t = v_t, then a head/time shuffle), so all earlier SAND runs
reproduce; attn_fix=True is real banded causal multi-head attention. Both share
one parameter layout and init stream.
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
    model_cfg = OmegaConf.load("src/config/model/sand.yaml")
    model_cfg.update(model_overrides)
    torch.manual_seed(seed)
    return build_model(OmegaConf.create({"model": model_cfg}), num_classes=num_classes)


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, D_INP, generator=g), torch.randn(n, D_STATIC, generator=g)


def _hidden(n=3, d=64, seed=1):
    return torch.randn(n, T, d, generator=torch.Generator().manual_seed(seed))


def _changed_positions(mha, mask, x, hour):
    """Output positions of one attention layer that move when input `hour` does."""
    x2 = x.clone()
    x2[:, hour] += 10.0
    delta = (mha(x2, mask) - mha(x, mask)).abs().amax(dim=(0, 2))
    return (delta > 1e-5).nonzero().flatten().tolist()


def _textbook_attention(mha, x, r):
    """Masked multi-head attention one head at a time: head i owns columns
    i*dk:(i+1)*dk of Wq/Wk/Wv, attends from t to keys t-r..t, heads concatenated."""
    idx = torch.arange(x.shape[1])
    allowed = (idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - r)
    heads = []
    for i in range(mha.num_heads):
        cols = slice(i * mha.dk, (i + 1) * mha.dk)
        q, k, v = x @ mha.Wq[:, cols], x @ mha.Wk[:, cols], x @ mha.Wv[:, cols]
        scores = (q @ k.transpose(1, 2) / mha.dk**0.5).masked_fill(~allowed, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ v)
    return mha.Wo(torch.cat(heads, dim=-1))


@torch.no_grad()
@pytest.mark.parametrize("r", [24, 3])
def test_fixed_attention_matches_textbook(r):
    model = _sand(attn_fix=True, r=r).eval()
    mha, x = model.transformer[0].mha, _hidden()
    assert torch.allclose(mha(x, model.mask), _textbook_attention(mha, x, r), atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("r,hour,expected", [
    (24, 10, list(range(10, 24))),  # r = T: plain causal, every later hour sees it
    (3, 10, [10, 11, 12, 13]),       # banded: only t with t-3 <= 10 <= t
    (3, 23, [23]),
])
def test_fixed_attention_receptive_field(r, hour, expected):
    model = _sand(attn_fix=True, r=r).eval()
    assert _changed_positions(model.transformer[0].mha, model.mask, _hidden(), hour) == expected


@torch.no_grad()
def test_legacy_attention_ignores_weights():
    """Pin the STraTS behaviour (attn_fix=False) so earlier runs reproduce: at
    eval the output is Wo of the values reshaped (B, h, T, dk) -> (B, T, d),
    independent of Wq/Wk, and each input hour lands at h shuffled positions."""
    model = _sand().eval()
    mha, x = model.transformer[0].mha, _hidden()
    B, h, dk = x.shape[0], mha.num_heads, mha.dk
    v = (x @ mha.Wv).view(B, T, h, dk).permute(0, 2, 1, 3)
    out = mha(x, model.mask)
    assert torch.allclose(out, mha.Wo(v.reshape(B, T, h * dk)), atol=1e-5)

    mha.Wq.normal_()
    mha.Wk.normal_()
    assert torch.allclose(mha(x, model.mask), out, atol=1e-5), "Wq/Wk must not matter"
    assert _changed_positions(mha, model.mask, x, 0) == [0, 6, 12, 18]
    assert _changed_positions(mha, model.mask, x, 23) == [5, 11, 17, 23]


@torch.no_grad()
def test_attn_fix_default_is_backward_compatible():
    """The yaml default and a config without the key build the legacy SAND;
    attn_fix=True has the identical parameter layout and init stream, so the
    flag changes only the forward pass."""
    x = _batch()
    default = _sand().eval()
    model_cfg = OmegaConf.load("src/config/model/sand.yaml")
    assert model_cfg.attn_fix is False
    del model_cfg["attn_fix"]
    torch.manual_seed(0)
    keyless = build_model(OmegaConf.create({"model": model_cfg}), num_classes=1).eval()
    fixed = _sand(attn_fix=True).eval()

    sd = default.state_dict()
    for other in (keyless, fixed):
        assert list(other.state_dict()) == list(sd)
        assert all(torch.equal(sd[k], other.state_dict()[k]) for k in sd)
    assert torch.equal(default.get_features(x), keyless.get_features(x))
    assert not torch.allclose(default.get_features(x), fixed.get_features(x))


def test_attn_fix_bayesian_contract():
    model = _sand(attn_fix=True, num_classes=25)
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


def test_attn_fix_trains_query_key_weights():
    """Without dropout noise the legacy attention gives Wq/Wk (numerically) no
    gradient: they only scale v_t by rowsum(softmax) = 1. With attn_fix they
    get a real one."""
    x = _batch()

    def wq_grad_norm(attn_fix):
        model = _sand(attn_fix=attn_fix).eval()
        model.get_features(x).pow(2).sum().backward()
        return sum(b.mha.Wq.grad.norm() for b in model.transformer)

    assert wq_grad_norm(False) < 1e-4 * wq_grad_norm(True)


@pytest.mark.parametrize(
    "data_name,task_id,model_overrides,expected",
    [
        ("mimic3_pheno", 8, {}, "mimic3_pheno_t08"),                     # unchanged
        ("mimic3_pheno", 8, {"attn_fix": False}, "mimic3_pheno_t08"),    # explicit default
        ("mimic3_pheno", 8, {"attn_fix": True}, "mimic3_pheno_t08_attnfix"),
        ("mimic3_pheno", 8, {"attn_fix": True, "d_static_out": 2}, "mimic3_pheno_t08_demo2_attnfix"),
        ("mimic3_pheno", 11, {"attn_fix": True, "use_static": False}, "mimic3_pheno_t11_nodemo_attnfix"),
        ("mimic3_los", None, {"attn_fix": True}, "mimic3_los_attnfix"),
    ],
)
def test_wandb_data_name_tags_attn_fix(data_name, task_id, model_overrides, expected):
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
