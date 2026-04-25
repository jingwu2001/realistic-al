"""
Bayesian Transformer models for temporal / sequential data, wrapped in the
BayesianModule interface for MC-dropout uncertainty estimation.

Two model variants are provided, both adapted from the Raindrop codebase
(models.py at the repository root) for use in the active-learning framework:

  BayesianTransformerModel   — uses a single d_model-dimensional embedding with
                               positional encoding added before the encoder, and
                               prepends a static-feature context token.

  BayesianTransformerModel2  — split-dimension design: positional encoding
                               (d_pe=16) and feature encoding (d_enc=d_inp) are
                               concatenated before the transformer encoder.
                               Static features are concatenated to the pooled
                               output rather than prepended as a token.

Both models follow the same Bayesian head pattern used throughout the codebase
(ResNet, GRU-D): the transformer backbone + intermediate MLP hidden layer are
deterministic (det_forward_impl), and only the final linear projection has
ConsistentMCDropout applied (mc_forward_impl).

Input format expected from the dataloader (P12TransformerDataset):
  x = (ts, static, times, lengths)
    ts      : (B, T, F)   float32 — normalised time-series (0 where missing)
    static  : (B, D)      float32 — normalised static features
    times   : (B, T)      float32 — timestamps in minutes (0-padded)
    lengths : (B,)        int64   — actual sequence length per sample

Fixes applied vs. the original models.py:
  • PositionalEncodingTF.forward: replaced hard-coded .cuda() with .to(device)
  • TransformerModel2: removed import pdb / pdb.set_trace() debug call
  • TransformerModel2: dropped the src[:, :, :half] halving step (our dataset
    already provides clean feature tensors without a concatenated mask channel)
  • forward signatures accept the 4-tuple (ts, static, times, lengths) and
    override BayesianModule.forward so that mc_tensor is called only on the
    already-extracted feature vector, not on the raw tuple input
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.bayesian_module import BayesianModule, ConsistentMCDropout


# ---------------------------------------------------------------------------
# Positional encoding (time-aware, from Raindrop)
# ---------------------------------------------------------------------------

class PositionalEncodingTF(nn.Module):
    """Time-aware sinusoidal positional encoding.

    Args:
        d_model: embedding dimension (must be even).
        max_len: scaling base for the timescales.
        MAX:     period scaling constant.
    """

    def __init__(self, d_model: int, max_len: int = 500, MAX: int = 10000) -> None:
        super().__init__()
        self.max_len         = max_len
        self.d_model         = d_model
        self.MAX             = MAX
        self._num_timescales = d_model // 2

    def forward(self, P_time: torch.Tensor) -> torch.Tensor:
        """Compute positional encoding for a batch of time sequences.

        Args:
            P_time: (T, B) float tensor of timestamps.

        Returns:
            pe: (T, B, d_model) positional encoding tensor on same device as P_time.
        """
        import numpy as np
        timescales  = self.max_len ** np.linspace(0, 1, self._num_timescales)  # (d//2,)
        times       = P_time.float().unsqueeze(2)                              # (T, B, 1)
        scale       = torch.tensor(
            timescales[None, None, :], dtype=torch.float32, device=P_time.device
        )                                                                        # (1, 1, d//2)
        scaled_time = times / scale                                             # (T, B, d//2)
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)  # (T, B, d_model)
        return pe


# ---------------------------------------------------------------------------
# BayesianTransformerModel  (variant 1)
# ---------------------------------------------------------------------------

class BayesianTransformerModel(BayesianModule):
    """Transformer with static context token, wrapped as a BayesianModule.

    Architecture (det part):
      Input projection: Linear(d_inp → d_model)
      + time-aware PE
      Prepend static embedding as the first token  (→ sequence length T+1)
      TransformerEncoder (nlayers × nhead, FFN dim nhid)
      Masked mean pooling over valid positions
      MLP hidden layer: Linear(d_model → d_model) + ReLU

    Stochastic head:
      ConsistentMCDropout(dropout_p)
      Linear(d_model → n_classes)

    Args:
        d_inp       : number of input time-series features (36 for P12).
        d_model     : transformer hidden / embedding dimension.
        nhead       : number of attention heads.
        nhid        : feedforward network hidden dimension.
        nlayers     : number of encoder layers.
        dropout_p   : dropout probability used in both transformer FFN layers
                      and the Bayesian classification head.
        max_len     : positional-encoding scaling base (passed to PE module).
        d_static    : dimension of static feature vector (9 for P12).
        MAX         : positional-encoding period constant.
        n_classes   : number of output classes.
    """

    def __init__(
        self,
        d_inp:     int,
        d_model:   int,
        nhead:     int,
        nhid:      int,
        nlayers:   int,
        dropout_p: float,
        max_len:   int,
        d_static:  int,
        MAX:       int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.pos_encoder       = PositionalEncodingTF(d_model, max_len, MAX)
        encoder_layer          = TransformerEncoderLayer(d_model, nhead, nhid, dropout_p,
                                                         batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        self.encoder    = nn.Linear(d_inp, d_model)
        self.d_model    = d_model
        self.emb        = nn.Linear(d_static, d_model)

        self.mlp_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.mc_dropout  = ConsistentMCDropout(p=dropout_p)
        self.classifier  = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        self.encoder.weight.data.uniform_(-1e-10, 1e-10)

    # ------------------------------------------------------------------
    # BayesianModule interface
    # ------------------------------------------------------------------

    def forward(self, x: tuple, k: int) -> torch.Tensor:
        """Override BayesianModule.forward to handle tuple input.

        Args:
            x: 4-tuple (ts, static, times, lengths) — see module docstring.
            k: number of MC samples.

        Returns:
            logits of shape (B, k, n_classes).
        """
        BayesianModule.k = k
        features  = self._encode(x)                          # (B, d_model)
        mc_feat   = BayesianModule.mc_tensor(features, k)   # (B*k, d_model)
        mc_out    = self.mc_forward_impl(mc_feat)            # (B*k, n_classes)
        return BayesianModule.unflatten_tensor(mc_out, k)    # (B, k, n_classes)

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
        ts, static, times, lengths = x
        # ts: (B, T, F)  static: (B, D)  times: (B, T)  lengths: (B,)
        B, T, _ = ts.shape

        src      = ts.permute(1, 0, 2)      # (T, B, F)
        times_TB = times.permute(1, 0)      # (T, B)

        src = self.encoder(src) * math.sqrt(self.d_model)  # (T, B, d_model)
        pe  = self.pos_encoder(times_TB)                    # (T, B, d_model)
        src = src + pe

        emb   = self.emb(static)                            # (B, d_model)
        x_cat = torch.cat([emb.unsqueeze(0), src], dim=0)  # (T+1, B, d_model)

        # key_padding_mask: True at positions to be ignored
        # valid positions: 0 .. lengths (inclusive; position 0 is static token)
        positions = torch.arange(T + 1, device=ts.device).unsqueeze(0)  # (1, T+1)
        pad_mask  = positions >= (lengths.unsqueeze(1) + 1)              # (B, T+1)

        out = self.transformer_encoder(x_cat, src_key_padding_mask=pad_mask)  # (T+1, B, d)
        out = out.permute(1, 0, 2)                                             # (B, T+1, d)

        # Masked mean over valid positions (static token + actual timesteps)
        valid      = (~pad_mask).unsqueeze(2).float()                # (B, T+1, 1)
        n_valid    = valid.sum(dim=1).clamp(min=1)                   # (B, 1)
        pooled     = (out * valid).sum(dim=1) / n_valid              # (B, d_model)

        return self.mlp_hidden(pooled)                               # (B, d_model)


# ---------------------------------------------------------------------------
# BayesianTransformerModel2  (variant 2 — split-dimension PE)
# ---------------------------------------------------------------------------

class BayesianTransformerModel2(BayesianModule):
    """Transformer with split-dimension PE + static concatenation at output.

    Architecture (det part):
      Feature projection: Linear(d_inp → d_enc)  where d_enc = d_inp
      Positional encoding with d_pe = 16 (narrower than d_model)
      Concatenate [PE(16), features(d_enc)] → total dim d_pe + d_enc
      Input dropout
      TransformerEncoder (nlayers × nhead, FFN dim nhid)
      Mean or max pooling over valid positions
      Concatenate pooled output with static embedding → (d_enc + d_pe + d_inp)
      MLP hidden layer: Linear(d_fi → d_fi) + ReLU

    Stochastic head:
      ConsistentMCDropout(dropout_p)
      Linear(d_fi → n_classes)

    Args:
        d_inp       : number of input time-series features (36 for P12).
        d_model     : unused (kept for API parity); actual encoder dim = d_pe + d_inp.
        nhead       : number of attention heads.
        nhid        : feedforward network hidden dimension.
        nlayers     : number of encoder layers.
        dropout_p   : MC-dropout probability.
        max_len     : positional-encoding scaling base.
        d_static    : static feature dimension (9 for P12).
        MAX         : positional-encoding period constant.
        perc        : unused (kept for API parity with original).
        aggreg      : ``"mean"`` or ``"max"`` pooling over time.
        n_classes   : number of output classes.
        use_static  : whether to concatenate static features to pooled output.
    """

    _D_PE = 16  # positional encoding dimension

    def __init__(
        self,
        d_inp:       int,
        d_model:     int,
        nhead:       int,
        nhid:        int,
        nlayers:     int,
        dropout_p:   float,
        max_len:     int,
        d_static:    int,
        MAX:         int,
        perc:        float,
        aggreg:      str,
        n_classes:   int,
        use_static:  bool = True,
    ) -> None:
        super().__init__()

        d_enc        = d_inp                             # feature projection dim
        d_enc_total  = self._D_PE + d_enc               # transformer input dim

        self.pos_encoder        = PositionalEncodingTF(self._D_PE, max_len, MAX)
        encoder_layer           = TransformerEncoderLayer(d_enc_total, nhead, nhid, dropout_p,
                                                          batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        self.encoder      = nn.Linear(d_inp, d_enc)
        self.use_static   = use_static
        self.aggreg       = aggreg
        self.input_dropout = nn.Dropout(dropout_p)

        if use_static:
            self.emb = nn.Linear(d_static, d_inp)
            d_fi = d_enc + self._D_PE + d_inp
        else:
            d_fi = d_enc + self._D_PE

        self.d_fi = d_fi

        self.mlp_hidden  = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
        )

        self.mc_dropout  = ConsistentMCDropout(p=dropout_p)
        self.classifier  = nn.Linear(d_fi, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        self.encoder.weight.data.uniform_(-1e-10, 1e-10)
        if self.use_static:
            self.emb.weight.data.uniform_(-1e-10, 1e-10)

    # ------------------------------------------------------------------
    # BayesianModule interface
    # ------------------------------------------------------------------

    def forward(self, x: tuple, k: int) -> torch.Tensor:
        BayesianModule.k = k
        features = self._encode(x)
        mc_feat  = BayesianModule.mc_tensor(features, k)
        mc_out   = self.mc_forward_impl(mc_feat)
        return BayesianModule.unflatten_tensor(mc_out, k)

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
        ts, static, times, lengths = x
        B, T, _ = ts.shape

        src      = ts.permute(1, 0, 2)   # (T, B, F)
        times_TB = times.permute(1, 0)   # (T, B)

        src = self.encoder(src)           # (T, B, d_enc)
        pe  = self.pos_encoder(times_TB) # (T, B, d_pe)
        src = torch.cat([pe, src], dim=2) # (T, B, d_pe + d_enc)
        src = self.input_dropout(src)

        # key_padding_mask: True at padding positions
        positions = torch.arange(T, device=ts.device).unsqueeze(0)  # (1, T)
        pad_mask  = positions >= lengths.unsqueeze(1)                # (B, T)

        out = self.transformer_encoder(src, src_key_padding_mask=pad_mask)  # (T, B, d_enc_total)
        out = out.permute(1, 0, 2)                                           # (B, T, d_enc_total)

        valid = (~pad_mask).unsqueeze(2).float()  # (B, T, 1)
        if self.aggreg == "mean":
            n_valid = valid.sum(dim=1).clamp(min=1)     # (B, 1)
            pooled  = (out * valid).sum(dim=1) / n_valid  # (B, d_enc_total)
        else:  # max
            out_masked = out * valid + (valid - 1) * 1e9  # large negative for pads
            pooled, _  = out_masked.max(dim=1)             # (B, d_enc_total)

        if self.use_static:
            static_emb = self.emb(static)                    # (B, d_inp)
            pooled     = torch.cat([pooled, static_emb], dim=1)  # (B, d_fi)

        return self.mlp_hidden(pooled)   # (B, d_fi)


