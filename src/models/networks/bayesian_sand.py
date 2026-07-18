"""
Bayesian SAnD model (Simply Attend and Diagnose) for dense hourly-binned
clinical time series, wrapped in the BayesianModule interface for MC-dropout
uncertainty estimation.

Adapted from the STraTS repository (src/modeling_sand.py), which follows
Song et al., "Attend and Diagnose: Clinical Time Series Analysis Using
Attention Models" (AAAI 2018, arXiv:1711.03905).

Architecture (det part):
  Input embedding: Conv1d(3V -> hid_dim, kernel 1)   [per-timestep linear]
  + learned positional encoding (1, T, hid_dim)
  num_layers x TransformerBlock with a causal band mask
    (position t attends to positions t-r .. t)
  DenseInterpolation over time -> (B, hid_dim * M)
  Static/demographic embedding: Linear(D -> 2*hid) + Tanh + Linear(2*hid -> hid)
  Concatenate -> features of dim hid_dim * M + hid_dim

Stochastic head (same pattern as ResNet / GRU-D / transformers):
  ConsistentMCDropout(dropout_p)
  Linear(hid_dim * M + hid_dim -> n_classes)

Input format expected from the dataloader (MIMIC3SandDataset):
  x = (ts, demo)
    ts   : (B, T, 3V) float32 — concat([values, obs_mask, delta], -1)
    demo : (B, D)     float32 — normalised static features

Fixes applied vs. the original modeling_sand.py:
  • MultiHeadAttention applied attention dropout via F.dropout(A, p) without
    the training flag, so it stayed active during eval. We pass self.training
    so the backbone is deterministic at eval time (only the MC head samples).
  • binary BCE head replaced by an n_classes softmax head to match the
    framework's loss/metric plumbing (weighted_loss supported via config).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_module import BayesianModule, ConsistentMCDropout


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim: int, num_heads: int, dropout_p: float) -> None:
        super().__init__()
        assert hid_dim % num_heads == 0
        self.dk = hid_dim // num_heads
        self.Wq = nn.Parameter(torch.empty((hid_dim, hid_dim)), requires_grad=True)
        self.Wk = nn.Parameter(torch.empty((hid_dim, hid_dim)), requires_grad=True)
        self.Wv = nn.Parameter(torch.empty((hid_dim, hid_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self.Wo = nn.Linear(hid_dim, hid_dim, bias=False)
        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d); mask: (T, T) additive (-inf outside the band)
        bsz, T, d = x.size()
        queries = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk) / np.sqrt(self.dk)
        keys    = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        values  = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        A = torch.einsum("bthd,blhd->bhtl", queries, keys) + mask  # (B, h, T, T)
        A = F.softmax(A, dim=-1)
        A = F.dropout(A, self.dropout_p, self.training)
        x = torch.einsum("bhtl,bthd->bhtd", A, values)
        return self.Wo(x.reshape((bsz, T, d)))


class FeedForward(nn.Module):
    """Position-wise FFN implemented with 1x1 convolutions (as in SAnD)."""

    def __init__(self, hid_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hid_dim * 2, hid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, hid_dim: int, num_heads: int, dropout_p: float) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(hid_dim, num_heads, dropout_p)
        self.ffn = FeedForward(hid_dim)
        self.norm_mha = nn.LayerNorm(hid_dim)
        self.norm_ffn = nn.LayerNorm(hid_dim)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x2 = F.dropout(self.mha(x, mask), self.dropout_p, self.training)
        x = self.norm_mha(x + x2)
        x2 = F.dropout(self.ffn(x), self.dropout_p, self.training)
        return self.norm_ffn(x + x2)


class DenseInterpolation(nn.Module):
    """Fixed (T, M) interpolation weights collapsing time to M steps."""

    def __init__(self, T: int, M: int) -> None:
        super().__init__()
        cols = torch.arange(M).reshape((1, M)) / M
        rows = torch.arange(T).reshape((T, 1)) / T
        W = (1 - torch.abs(rows - cols)) ** 2
        self.W = nn.Parameter(W, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size()[0]
        x = torch.matmul(x.transpose(1, 2), self.W)  # (B, hid, M)
        return x.reshape((bsz, -1))                   # (B, hid*M)


class BayesianSANDModel(BayesianModule):
    """SAnD backbone + Bayesian classification head.

    Args:
        d_inp     : time-series input dim per timestep (3V; 387 for MIMIC-III).
        T         : number of time steps (24 hourly bins for MIMIC-III).
        d_static  : static feature dim (2 for MIMIC-III: Age, Gender).
        hid_dim   : transformer hidden dimension.
        num_heads : attention heads (must divide hid_dim).
        num_layers: number of transformer blocks.
        r         : attention band width — t attends to t-r .. t.
        M         : dense-interpolation output steps.
        dropout_p : dropout used in the backbone and the MC head.
        n_classes : number of output classes.
    """

    def __init__(
        self,
        d_inp:      int,
        T:          int,
        d_static:   int,
        hid_dim:    int,
        num_heads:  int,
        num_layers: int,
        r:          int,
        M:          int,
        dropout_p:  float,
        n_classes:  int,
    ) -> None:
        super().__init__()

        self.input_embedding = nn.Conv1d(d_inp, hid_dim, 1)
        self.positional_encoding = nn.Parameter(
            torch.empty((1, T, hid_dim)), requires_grad=True
        )
        nn.init.normal_(self.positional_encoding)

        indices = torch.arange(T)
        # t attends to t-r,...,t
        mask = torch.logical_and(
            indices[None, :] <= indices[:, None],
            indices[None, :] >= indices[:, None] - r,
        ).float()
        mask = (1 - mask) * torch.finfo(mask.dtype).min
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.dropout_p = dropout_p
        self.transformer = nn.ModuleList(
            [TransformerBlock(hid_dim, num_heads, dropout_p) for _ in range(num_layers)]
        )
        self.dense_interpolation = DenseInterpolation(T, M)

        self.demo_emb = nn.Sequential(
            nn.Linear(d_static, hid_dim * 2),
            nn.Tanh(),
            nn.Linear(hid_dim * 2, hid_dim),
        )

        d_fi = hid_dim * M + hid_dim
        self.d_fi = d_fi
        self.mc_dropout = ConsistentMCDropout(p=dropout_p)
        self.classifier = nn.Linear(d_fi, n_classes)

    # ------------------------------------------------------------------
    # BayesianModule interface
    # ------------------------------------------------------------------

    def forward(self, x: tuple, k: int) -> torch.Tensor:
        """Override BayesianModule.forward to handle tuple input.

        Args:
            x: 2-tuple (ts, demo) — see module docstring.
            k: number of MC samples.

        Returns:
            logits of shape (B, k, n_classes).
        """
        BayesianModule.k = k
        features = self._encode(x)                        # (B, d_fi)
        mc_feat  = BayesianModule.mc_tensor(features, k)  # (B*k, d_fi)
        mc_out   = self.mc_forward_impl(mc_feat)          # (B*k, n_classes)
        return BayesianModule.unflatten_tensor(mc_out, k)  # (B, k, n_classes)

    def det_forward_impl(self, x: tuple) -> torch.Tensor:
        return self._encode(x)

    def mc_forward_impl(self, h_BK: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.mc_dropout(h_BK))

    def get_features(self, x: tuple) -> torch.Tensor:
        return self._encode(x)

    # ------------------------------------------------------------------
    # Internal encoder
    # ------------------------------------------------------------------

    def _encode(self, x: tuple) -> torch.Tensor:
        ts, demo = x
        # ts: (B, T, 3V)  demo: (B, D)
        h = self.input_embedding(ts.permute((0, 2, 1))).permute((0, 2, 1))  # (B, T, hid)
        h = h + self.positional_encoding
        if self.dropout_p > 0:
            h = F.dropout(h, self.dropout_p, self.training)
        for layer in self.transformer:
            h = layer(h, self.mask)
        ts_emb = self.dense_interpolation(h)   # (B, hid*M)
        demo_emb = self.demo_emb(demo)          # (B, hid)
        return torch.cat((ts_emb, demo_emb), dim=-1)  # (B, hid*M + hid)
