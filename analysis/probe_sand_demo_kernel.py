"""
Does the demographics block dominate the vendi kernel on SAND features?

SAND features = concat(ts_emb (hid*M), demo_emb (d_demo)). For each AL loop
checkpoint, rebuild the labelled set L and pool U of that loop, embed both
(penultimate features, and the BADGE-style grad embedding = (p - y_hat) h), and
for every kernel config report:

  demo_share    E||d_u - d_l||^2 / E||x_u - x_l||^2 over all (u, l) pairs, after
                normalization. For rbf this is exactly the demo part of the
                kernel exponent (K = K_ts * K_demo at a shared gamma); for cosine
                it is the demo share of ||x||^2 instead (cosine is not additive).
  top_ts        overlap of the vendi-q2 top-B batch with full features vs with
  top_demo      the TS block alone / demo block alone (each block normalized and
                bandwidth-resolved on its own, as vendi would do with it).
  rho_ts        Spearman of the vendi-q2 scores, full vs block.
  rho_demo
  rbf_arg_med   median rbf exponent on K_LL (healthy ~0.1-5).

vendi-q2 (non-sequential) picks the B candidates with the smallest
sum_l K(u, l)^2: with unit-diagonal kernels ||K_aug||_F^2 = ||K_LL||_F^2 +
2 sum_l K(u, l)^2 + 1, and VS_2 = (L+1)^2 / ||K_aug||_F^2.

Random-overlap reference for top-B: B / |U|.

Usage:
  python analysis/probe_sand_demo_kernel.py RUN_DIR [RUN_DIR ...] \
      --loops 0 4 9 --out analysis/cache/sand_demo_kernel.csv
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.bayesian import BayesianModule  # noqa: E402
from query.query_diversity import (  # noqa: E402
    get_embeddings,
    get_grad_embedding,
    normalize_features,
    resolve_gamma,
)
from run_training import get_torchvision_dm  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# MIMIC-10 (docs/exp.md): (embedding, kernel, gamma, normalization)
CONFIGS = [
    ("feat", "rbf", "median", "l2"),
    ("feat", "rbf", "median", "minmax"),
    ("feat", "rbf", "median", "zscore"),
    ("feat", "rbf", "median", "none"),
    ("feat", "rbf", "dim", "zscore"),
    ("feat", "cosine", None, "zscore"),
    ("feat", "cosine", None, "none"),
    ("grad", "rbf", "median", "l2"),
    ("grad", "cosine", None, "zscore"),
    ("grad", "cosine", None, "none"),
]


def load_loop(loop_dir: Path):
    cfg = OmegaConf.load(loop_dir / "hparams.yaml")
    ckpt = sorted((loop_dir / "checkpoints").glob("*.ckpt"))[-1]
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    if any(".mha.Wq" in k for k in sd):
        # pre-2026-10-24 SAND checkpoints were saved as model=sand
        cfg.model.name = "sand_legacy"
    model = BayesianModule(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert all(k.startswith("loss_fct.") for k in missing + unexpected), (missing, unexpected)
    return cfg, model.to(DEVICE).eval()


def build_sets(cfg, loop_dir: Path):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.trainer.num_workers = 4
    cfg.trainer.persistent_workers = False
    dm = get_torchvision_dm(cfg)
    with open(loop_dir / "data_ckpt.pkl", "rb") as f:
        dm.train_set.load_state_dict(pickle.load(f))
    dm.train_set.init_state()  # load_state_dict does not rebuild labelled_set
    return dm.labeled_dataloader(batch_size=256), dm.pool_dataloader(batch_size=256)


def kernel(kind, gamma, x, y):
    if kind == "cosine":
        return F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
    return torch.exp(-gamma * torch.cdist(x, y).pow(2))


def vq2_scores(kind, gamma_cfg, norm, L, U):
    """vendi-q2 ranking score (lower = picked first) and the resolved gamma."""
    L, U = normalize_features(L, U, norm)
    gamma = resolve_gamma(gamma_cfg, L) if kind == "rbf" else None
    s = torch.zeros(U.shape[0], dtype=torch.float64, device=U.device)
    for i in range(0, U.shape[0], 2048):
        s[i:i + 2048] = kernel(kind, gamma, U[i:i + 2048], L).double().pow(2).sum(1)
    return s, gamma, L, U


def demo_share(kind, L, U, d_ts):
    if kind == "cosine":
        X = torch.cat([L, U])
        return (X[:, d_ts:].pow(2).sum(1) / X.pow(2).sum(1)).mean().item()
    ts = torch.cdist(U[:, :d_ts], L[:, :d_ts]).pow(2).mean()
    dm = torch.cdist(U[:, d_ts:], L[:, d_ts:]).pow(2).mean()
    return (dm / (ts + dm)).item()


def top_overlap(a, b, B):
    ia, ib = torch.argsort(a)[:B], torch.argsort(b)[:B]
    return np.intersect1d(ia.cpu().numpy(), ib.cpu().numpy()).size / B


def probe(run_dir: Path, loop: int):
    loop_dir = run_dir / f"loop-{loop}"
    cfg, model = load_loop(loop_dir)
    lab_loader, pool_loader = build_sets(cfg, loop_dir)
    d_ts = cfg.model.hid_dim * cfg.model.M
    B = cfg.active.acq_size

    emb = {
        "feat": (get_embeddings(model, lab_loader), get_embeddings(model, pool_loader)),
        "grad": tuple(
            get_grad_embedding(model, ld, "linear", device=DEVICE).to(DEVICE)
            for ld in (lab_loader, pool_loader)
        ),
    }
    rows = []
    for e, kind, g, norm in CONFIGS:
        L, U = (t.float() for t in emb[e])
        s_full, gamma, Ln, Un = vq2_scores(kind, g, norm, L, U)
        s_ts, *_ = vq2_scores(kind, g, norm, L[:, :d_ts], U[:, :d_ts])
        s_demo, *_ = vq2_scores(kind, g, norm, L[:, d_ts:], U[:, d_ts:])
        row = dict(
            run=run_dir.name, model=cfg.model.name, loop=loop, n_lab=L.shape[0],
            n_pool=U.shape[0], emb=e, kernel=kind, gamma=g, norm=norm,
            demo_share=demo_share(kind, Ln, Un, d_ts),
            top_ts=top_overlap(s_full, s_ts, B), top_demo=top_overlap(s_full, s_demo, B),
            rho_ts=spearmanr(s_full.cpu(), s_ts.cpu())[0],
            rho_demo=spearmanr(s_full.cpu(), s_demo.cpu())[0],
            top_random=B / U.shape[0],
        )
        if kind == "rbf":
            sub = Ln[:2000]
            arg = gamma * torch.cdist(sub, sub).pow(2)
            row["rbf_arg_med"] = arg[torch.triu(torch.ones_like(arg, dtype=bool), 1)].median().item()
        rows.append(row)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--loops", nargs="+", type=int, default=[0, 4, 9])
    p.add_argument("--out", type=Path, default=Path("analysis/cache/sand_demo_kernel.csv"))
    args = p.parse_args()

    rows = []
    for r in args.run_dirs:
        for k in args.loops:
            print(f"[probe] {r.name} loop-{k}", flush=True)
            rows += probe(r, k)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.drop(columns=["run", "n_pool"]).round(3).to_string(index=False))
