"""Registry entry for BayesianLSTMModel.

The model name is derived from this filename: 'lstm'.
All architecture code lives in bayesian_lstm.py.
"""

from .bayesian_lstm import BayesianLSTMModel
from .registry import register_model


@register_model
def lstm(
    config,
    num_classes: int = 2,
    data_shape=None,
    **kwargs,
) -> BayesianLSTMModel:
    """Factory registered under the name ``'lstm'``.

    Expected config fields (all under ``config.model``):
      d_inp, d_static, dim, depth, dropout_p, use_static
    """
    m = config.model
    return BayesianLSTMModel(
        d_inp      = m.d_inp,
        d_static   = m.d_static,
        dim        = m.dim,
        depth      = m.depth,
        dropout_p  = m.dropout_p,
        n_classes  = num_classes,
        use_static = m.use_static,
    )
