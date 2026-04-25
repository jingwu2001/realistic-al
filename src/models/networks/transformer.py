"""Registry entry for BayesianTransformerModel (variant 1).

The model name is derived from this filename: 'transformer'.
All architecture code lives in bayesian_transformer.py.
"""

from .bayesian_transformer import BayesianTransformerModel
from .registry import register_model


@register_model
def transformer(
    config,
    num_classes: int = 2,
    data_shape=None,
    **kwargs,
) -> BayesianTransformerModel:
    """Factory registered under the name ``'transformer'``.

    Expected config fields (all under ``config.model``):
      d_inp, d_model, nhead, nhid, nlayers, dropout_p, max_len, d_static, MAX
    """
    m = config.model
    return BayesianTransformerModel(
        d_inp     = m.d_inp,
        d_model   = m.d_model,
        nhead     = m.nhead,
        nhid      = m.nhid,
        nlayers   = m.nlayers,
        dropout_p = m.dropout_p,
        max_len   = m.max_len,
        d_static  = m.d_static,
        MAX       = m.MAX,
        n_classes = num_classes,
    )
