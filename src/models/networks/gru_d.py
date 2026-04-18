"""
Batched GRU-D model for the P12 dataset, wrapped in the BayesianModule interface.

Original GRU-D paper: Che et al., "Recurrent Neural Networks for Multivariate
Time Series with Missing Values", Scientific Reports 2018.
Original single-sample implementation: Raindrop/code/baselines/models.py.

This file re-implements the forward pass in batched form so that standard
PyTorch DataLoaders (batch_size > 1) work correctly, and wraps the model in
the BayesianModule / AbstractClassifier interface expected by the AL framework.

Data layout (per sample):  Tensor[3, input_size, T]
  dim 0 == 0 : X     — normalised feature values (0 where unobserved)
  dim 0 == 1 : Mask  — 1 where observed, 0 where missing
  dim 0 == 2 : Delta — hours since last observation per feature

Constraint: input_size == hidden_size (required by element-wise GRU-D update).
For P12 both are 33.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayesian_module import BayesianModule, ConsistentMCDropout
from .registry import register_model


class GRUDModel(BayesianModule):
    """Batched GRU-D wrapped in the BayesianModule interface.

    The deterministic backbone (det_forward_impl) runs the full GRU-D
    recurrence and returns the final hidden state [B, hidden_size].
    The stochastic head (mc_forward_impl) applies ConsistentMCDropout and a
    linear classifier to produce logits [B*k, num_classes].

    Dropout is applied during both training and inference via ConsistentMCDropout,
    mirroring the ResNet/CIFAR10 pattern: set dropout_p=0 for deterministic query
    strategies (random, entropy, badge, kcentergreedy, vendi) and dropout_p=0.5
    for strategies that require MC uncertainty estimates (bald, bandit).

    Args:
        input_size  (int): Number of input features (= hidden_size for GRU-D).
        hidden_size (int): Hidden state dimension (must equal input_size).
        num_classes (int): Number of output classes.
        x_mean      (Tensor[input_size]): Per-feature means used for imputation.
        dropout_p   (float): MC-dropout probability (training and inference).
                             Use 0.0 for deterministic strategies; 0.5 for BALD/bandit.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        x_mean: torch.Tensor,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        assert input_size == hidden_size, (
            f"GRU-D requires input_size == hidden_size, got {input_size} vs {hidden_size}"
        )
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Register x_mean as a non-trainable buffer so it moves to the right device.
        self.register_buffer("x_mean", x_mean.float())

        # ---- GRU-D recurrence parameters ----
        # Decay rates
        self.weight_dg_x = nn.Parameter(torch.Tensor(input_size))
        self.weight_dg_h = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_dg_x   = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_dg_h   = nn.Parameter(torch.Tensor(hidden_size))

        # Update gate z
        self.weight_xz = nn.Parameter(torch.Tensor(input_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_mz = nn.Parameter(torch.Tensor(input_size))
        self.bias_z    = nn.Parameter(torch.Tensor(hidden_size))

        # Reset gate r
        self.weight_xr = nn.Parameter(torch.Tensor(input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_mr = nn.Parameter(torch.Tensor(input_size))
        self.bias_r    = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate hidden state h̃
        self.weight_xh = nn.Parameter(torch.Tensor(input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_mh = nn.Parameter(torch.Tensor(input_size))
        self.bias_h    = nn.Parameter(torch.Tensor(hidden_size))

        # ---- Classification head ----
        # Mirrors ResNet: ConsistentMCDropout is active during both training
        # (standard dropout) and eval (consistent MC mask for uncertainty estimation).
        # Set dropout_p=0 for deterministic query strategies; 0.5 for BALD/bandit.
        self.mc_dropout  = ConsistentMCDropout(p=dropout_p)
        self.classifier  = nn.Linear(hidden_size, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if "classifier" not in name:
                nn.init.uniform_(param, -stdv, stdv)
        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    # BayesianModule interface
    # ------------------------------------------------------------------

    def det_forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GRU-D recurrence on a batch of samples.

        Args:
            x: Tensor of shape (B, 3, input_size, T).

        Returns:
            h: Final hidden state, shape (B, hidden_size).
        """
        B, _, _, T = x.shape
        X = x[:, 0]   # (B, input_size, T)
        M = x[:, 1]   # (B, input_size, T)
        D = x[:, 2]   # (B, input_size, T)

        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        # x_last tracks the last imputed value per feature per sample.
        # The dataset stores normalized(0) for missing entries — NOT the last observed
        # value — so we must maintain this state ourselves.  Initialise to x_mean so
        # that features missing from the very first timestep are imputed to the mean.
        x_last = self.x_mean.unsqueeze(0).expand(B, -1)  # (B, input_size)

        for t in range(T):
            x_t = X[:, :, t]   # (B, input_size)
            m_t = M[:, :, t]   # (B, input_size)
            d_t = D[:, :, t]   # (B, input_size)

            # Exponential decay rates (eq. 4) — clamp replaces max(0, ·)
            gamma_x = torch.exp(-F.relu(self.weight_dg_x * d_t + self.bias_dg_x))
            gamma_h = torch.exp(-F.relu(self.weight_dg_h * d_t + self.bias_dg_h))

            # Imputation (eq. 5): observed values kept; missing replaced by the
            # decayed last imputed value blended toward the feature mean.
            # x_last is used (not the raw x_t which stores normalized(0) for
            # missing timesteps), so the decay is from the correct prior value.
            x_t = m_t * x_t + (1.0 - m_t) * (
                gamma_x * x_last + (1.0 - gamma_x) * self.x_mean
            )
            x_last = x_t   # store imputed value for next timestep's decay

            # GRU-D hidden state update (eq. 6)
            h = gamma_h * h

            z = torch.sigmoid(
                self.weight_xz * x_t + self.weight_hz * h + self.weight_mz * m_t + self.bias_z
            )
            r = torch.sigmoid(
                self.weight_xr * x_t + self.weight_hr * h + self.weight_mr * m_t + self.bias_r
            )
            h_tilde = torch.tanh(
                self.weight_xh * x_t
                + self.weight_hh * (r * h)
                + self.weight_mh * m_t
                + self.bias_h
            )
            h = (1.0 - z) * h + z * h_tilde

        return h   # (B, hidden_size)

    def mc_forward_impl(self, h_BK: torch.Tensor) -> torch.Tensor:
        """Apply MC-dropout + linear classifier to the hidden state.

        Mirrors ResNet.mc_forward_impl: ConsistentMCDropout handles both modes
        — standard dropout during training, consistent mask during eval.

        Args:
            h_BK: Tensor of shape (B*k, hidden_size).

        Returns:
            logits: Tensor of shape (B*k, num_classes).
        """
        return self.classifier(self.mc_dropout(h_BK))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final hidden state (used by diversity-based query strategies)."""
        return self.det_forward_impl(x)


# ---------------------------------------------------------------------------
# Model registry entry
# ---------------------------------------------------------------------------

@register_model
def gru_d(config, num_classes: int = 2, data_shape=None, **kwargs) -> GRUDModel:
    """Factory function registered under the name 'gru_d'.

    Expected config fields:
      config.model.hidden_size   – hidden / input size (33 for P12)
      config.model.dropout_p     – MC-dropout probability
      config.data.x_mean_path    – path to x_mean_aft_nor.npy
    """
    if data_shape is None:
        data_shape = [3, 33, 49]
    input_size = data_shape[1]   # 33

    x_mean_path = config.data.x_mean_path
    x_mean = torch.tensor(
        np.load(x_mean_path), dtype=torch.float32
    )

    return GRUDModel(
        input_size  = input_size,
        hidden_size = config.model.hidden_size,
        num_classes = num_classes,
        x_mean      = x_mean,
        dropout_p   = config.model.dropout_p,
    )
