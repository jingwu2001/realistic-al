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
      d_inp, d_static_in, d_static_out, dim, depth, dropout_p, use_static
    Optional: input_dropout_p, head_dropout_p (null / missing = dropout_p)
    """
    m = config.model
    return BayesianLSTMModel(
        d_inp           = m.d_inp,
        d_static_in     = m.d_static_in,
        d_static_out    = m.d_static_out,
        dim             = m.dim,
        depth           = m.depth,
        dropout_p       = m.dropout_p,
        n_classes       = num_classes,
        use_static      = m.use_static,
        input_dropout_p = m.get("input_dropout_p", None),
        head_dropout_p  = m.get("head_dropout_p", None),
    )
