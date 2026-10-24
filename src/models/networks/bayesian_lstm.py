"""
Bayesian LSTM baseline of the MIMIC-III benchmark, Harutyunyan et al.,
"Multitask learning and benchmarking with clinical time series data"
(arXiv:1703.07771, 2017; Scientific Data 2019), wrapped in the BayesianModule
interface for MC-dropout uncertainty estimation.

Ported from the benchmark's Keras model
(third_party/mimic3-benchmarks/mimic3models/keras_models/lstm.py) with
deep_supervision=False, target_repl=False (the per-stay tasks: ihm, phenotyping).

Architecture (det part):
  (depth - 1) x Bidirectional LSTM(dim // 2 per direction), returning sequences
  LSTM(dim) -> last hidden state                              (B, dim)
  [use_static] + demo embedding Linear(D -> 2*dim) + Tanh + Linear(2*dim -> d_static_out) + Tanh
  -> features of dim `dim` (+ `d_static_out` with use_static), all in (-1, 1)

Stochastic head (same pattern as SAND / GRU-D / transformers):
  ConsistentMCDropout(head_dropout_p)   <- the Keras Dropout(dropout) after the LSTM
  Linear(features -> n_classes)

Input format expected from the dataloader (MIMIC3*Dataset):
  x = (ts, demo)
    ts   : (B, T, 3V) float32 — concat([values, obs_mask, delta], -1)
    demo : (B, D)     float32 — normalised static features

Deviations from the benchmark:
  • Input is the shared SAND tensor (24 hourly bins x 387 = values/mask/delta of
    129 STraTS variables, mean-filled + z-normalised) rather than the
    benchmark's 76-dim carry-forward features, so SAND and LSTM see identical
    inputs and only the architecture differs.
  • use_static concatenates the STraTS demo embedding to the last hidden state,
    as STraTS's GRU/GRU-D baselines and our SAND do. The benchmark LSTM has no
    static input: use_static=false gives the exact benchmark architecture.
  • The demo embedding is narrowed to d_static_out (default 2) instead of
    STraTS's `dim`, and ends in a Tanh: at 256 dims with a linear output (the
    runs before 2026-10-09) it was ~96% of the pairwise get_features distance
    (|h| ~0.1/dim, tanh-bounded, vs |demo_emb| ~0.5/dim, max 10.7), so
    kcenter/vendi/BADGE diversified over Age/Gender. The bound puts the demo
    dims on the scale of h; how much Age/Gender matter is left to the
    classifier weights, which no feature distance sees.
  • Keras `dropout` (input dropout of every LSTM layer, one mask per sequence
    shared across time steps) is a variational input mask here too, but shared
    across the four gates (Keras draws one per gate). `rec_dropout` is not
    ported: the benchmark default is 0.0 and no published command sets it.
  • No Masking layer: the 24 hourly bins are never padded.
  • Keras initialisation is kept (glorot-uniform input kernel, orthogonal
    recurrent kernel, zero bias with forget-gate bias 1); PyTorch's default
    LSTM init is uniform(+-1/sqrt(h)) with no forget bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_module import BayesianModule, ConsistentMCDropout


def _keras_lstm_init(lstm: nn.LSTM) -> None:
    """Keras LSTM defaults: glorot_uniform kernel, orthogonal recurrent kernel,
    zero bias with unit_forget_bias. PyTorch gate order is (i, f, g, o), and the
    two bias vectors are summed, so the forget slice of bias_ih carries the 1."""
    h = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(param)
        elif name.startswith("weight_hh"):
            nn.init.orthogonal_(param)
        elif name.startswith("bias"):
            nn.init.zeros_(param)
            if name.startswith("bias_ih"):
                param.data[h : 2 * h] = 1.0


class BayesianLSTMModel(BayesianModule):
    """Benchmark LSTM backbone + Bayesian classification head.

    Args:
        d_inp     : time-series input dim per timestep (3V; 387 for MIMIC-III).
        d_static_in : static feature dim (2 for MIMIC-III: Age, Gender).
        d_static_out: width of the demographics embedding concatenated to h.
        dim         : LSTM units of the final layer (BiLSTMs use dim // 2 per direction).
        depth       : total LSTM layers, (depth - 1) bidirectional + 1 unidirectional.
        dropout_p   : LSTM input dropout and MC head dropout (benchmark --dropout).
        n_classes   : number of outputs.
        use_static  : concatenate the demographics embedding to the last hidden state.
        input_dropout_p : input dropout only; None = dropout_p.
        head_dropout_p  : MC head dropout only; None = dropout_p.
    """

    def __init__(
        self,
        d_inp:        int,
        d_static_in:  int,
        d_static_out: int,
        dim:          int,
        depth:        int,
        dropout_p:    float,
        n_classes:    int,
        use_static:   bool = True,
        input_dropout_p: float | None = None,
        head_dropout_p:  float | None = None,
    ) -> None:
        super().__init__()
        assert depth >= 1

        self.dropout_p = dropout_p
        self.input_dropout_p = dropout_p if input_dropout_p is None else input_dropout_p
        self.head_dropout_p = dropout_p if head_dropout_p is None else head_dropout_p
        self.use_static = use_static

        self.bilstms = nn.ModuleList()
        d_in = d_inp
        for _ in range(depth - 1):
            self.bilstms.append(
                nn.LSTM(d_in, dim // 2, batch_first=True, bidirectional=True)
            )
            d_in = 2 * (dim // 2)
        self.lstm = nn.LSTM(d_in, dim, batch_first=True)
        for lstm in [*self.bilstms, self.lstm]:
            _keras_lstm_init(lstm)

        d_fi = dim
        if use_static:
            self.demo_emb = nn.Sequential(
                nn.Linear(d_static_in, dim * 2),
                nn.Tanh(),
                nn.Linear(dim * 2, d_static_out),
                nn.Tanh(),  # bounded like h = o * tanh(c), so it can't dominate distances
            )
            d_fi += d_static_out
        self.d_fi = d_fi
        self.mc_dropout = ConsistentMCDropout(p=self.head_dropout_p)
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

    def _input_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Keras LSTM(dropout=p): one input mask per sequence, reused at every
        time step. Training only, so the backbone is deterministic at eval."""
        if not self.training or self.input_dropout_p == 0:
            return x
        mask = F.dropout(x.new_ones(x.shape[0], 1, x.shape[2]), self.input_dropout_p, True)
        return x * mask

    def _encode(self, x: tuple) -> torch.Tensor:
        ts, demo = x
        # ts: (B, T, 3V)  demo: (B, D)
        h = ts
        for bilstm in self.bilstms:
            h = bilstm(self._input_dropout(h))[0]       # (B, T, 2 * (dim // 2))
        _, (h_n, _) = self.lstm(self._input_dropout(h))
        feat = h_n[-1]                                   # (B, dim) last time step
        if self.use_static:
            feat = torch.cat((feat, self.demo_emb(demo)), dim=-1)  # (B, dim + d_static_out)
        return feat
