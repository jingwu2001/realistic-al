"""
Bayesian SAnD model (Simply Attend and Diagnose) for dense hourly-binned
clinical time series, wrapped in the BayesianModule interface for MC-dropout
uncertainty estimation.

Song et al., "Attend and Diagnose: Clinical Time Series Analysis Using
Attention Models" (AAAI 2018, arXiv:1711.03905), as configured in the STraTS
repository (src/modeling_sand.py). The STraTS attention never applied its
attention weights; runs before 2026-10-24 used that code, which is kept as
bayesian_sand_legacy.py (model=sand_legacy). This file uses torch's
nn.TransformerEncoder, so the attention is the standard one.

Architecture (det part):
  Input embedding: Linear(3V -> hid_dim) per time step
  + learned positional encoding (1, T, hid_dim)
  num_layers x nn.TransformerEncoderLayer (post-norm, FFN width 2*hid_dim)
    with a causal band mask: position t attends to positions t-r .. t
  DenseInterpolation over time -> (B, hid_dim * M)
    (trailing all-zero input rows = padding, zeroed before the interpolation)
  [use_static] Static/demographic embedding: Linear(D -> 2*hid) + Tanh + Linear(2*hid -> hid)
    [d_static_out set] ... + Linear(2*hid -> d_static_out) + Tanh instead (as model=lstm)
  Concatenate -> features of dim hid_dim * M + d_demo (hid_dim * M without use_static)

Stochastic head (same pattern as ResNet / GRU-D / transformers):
  ConsistentMCDropout(dropout_p)
  Linear(d_features -> n_classes)

Input format expected from the dataloader (MIMIC3ICUDataset / MIMIC3LOSDataset):
  x = (ts, demo)
    ts   : (B, T, 3V) float32 — concat([values, obs_mask, delta], -1)
    demo : (B, D)     float32 — normalised static features
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_module import BayesianModule, ConsistentMCDropout


def dense_interpolation_weights(T: int, M: int) -> torch.Tensor:
    """Fixed (T, M) SAnD weights collapsing T time steps to M."""
    cols = torch.arange(M).reshape((1, M)) / M
    rows = torch.arange(T).reshape((T, 1)) / T
    return (1 - torch.abs(rows - cols)) ** 2


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
        use_static: concatenate the demographics embedding to the time-series
                    features; False = time series only (demo input ignored).
        d_static_out: width of a tanh-bounded demographics embedding; None =
                    the original hid_dim-wide linear-output embedding.
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
        use_static: bool = True,
        d_static_out: int | None = None,
    ) -> None:
        super().__init__()

        self.use_static = use_static
        self.dropout_p = dropout_p
        self.input_embedding = nn.Linear(d_inp, hid_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, T, hid_dim))

        # True = not allowed: t attends to t-r,...,t
        t = torch.arange(T)
        self.register_buffer("mask", (t[None, :] > t[:, None]) | (t[None, :] < t[:, None] - r))

        layer = nn.TransformerEncoderLayer(
            hid_dim, num_heads, dim_feedforward=2 * hid_dim, dropout=dropout_p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.register_buffer("interp", dense_interpolation_weights(T, M))

        d_fi = hid_dim * M
        if use_static:
            d_demo = hid_dim if d_static_out is None else d_static_out
            layers = [
                nn.Linear(d_static, hid_dim * 2),
                nn.Tanh(),
                nn.Linear(hid_dim * 2, d_demo),
            ]
            if d_static_out is not None:
                layers.append(nn.Tanh())  # bounded to (-1, 1), as model=lstm
            self.demo_emb = nn.Sequential(*layers)
            d_fi += d_demo
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
        h = self.input_embedding(ts) + self.positional_encoding  # (B, T, hid)
        h = F.dropout(h, self.dropout_p, self.training)
        h = self.transformer(h, mask=self.mask)
        # trailing all-zero input rows are padding (right-padded benchmark arrays):
        # the causal mask already hides them from real hours; drop them from the
        # interpolation too. The STraTS tensor has none (multiply by 1).
        real = (ts.abs().amax(dim=-1) > 0).flip(1).cummax(dim=1).values.flip(1)  # (B, T)
        h = h * real.unsqueeze(-1)
        ts_emb = (h.transpose(1, 2) @ self.interp).flatten(1)  # (B, hid*M)
        if not self.use_static:
            return ts_emb
        demo_emb = self.demo_emb(demo)          # (B, hid) or (B, d_static_out)
        return torch.cat((ts_emb, demo_emb), dim=-1)  # (B, hid*M + d_demo)
