"""Registry entry for BayesianSANDModel.

The model name is derived from this filename: 'sand'.
All architecture code lives in bayesian_sand.py.
"""

from .bayesian_sand import BayesianSANDModel
from .registry import register_model


@register_model
def sand(
    config,
    num_classes: int = 2,
    data_shape=None,
    **kwargs,
) -> BayesianSANDModel:
    """Factory registered under the name ``'sand'``.

    Expected config fields (all under ``config.model``):
      d_inp, T, d_static, hid_dim, num_heads, num_layers, r, M, dropout_p
    """
    m = config.model
    return BayesianSANDModel(
        d_inp      = m.d_inp,
        T          = m.T,
        d_static   = m.d_static,
        hid_dim    = m.hid_dim,
        num_heads  = m.num_heads,
        num_layers = m.num_layers,
        r          = m.r,
        M          = m.M,
        dropout_p  = m.dropout_p,
        n_classes  = num_classes,
    )
