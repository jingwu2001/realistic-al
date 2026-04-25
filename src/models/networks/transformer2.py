"""Registry entry for BayesianTransformerModel2 (variant 2).

The model name is derived from this filename: 'transformer2'.
All architecture code lives in bayesian_transformer.py.
"""

from .bayesian_transformer import BayesianTransformerModel2
from .registry import register_model


@register_model
def transformer2(
    config,
    num_classes: int = 2,
    data_shape=None,
    **kwargs,
) -> BayesianTransformerModel2:
    """Factory registered under the name ``'transformer2'``.

    Expected config fields (all under ``config.model``):
      d_inp, d_model, nhead, nhid, nlayers, dropout_p, max_len,
      d_static, MAX, perc, aggreg, use_static
    """
    m = config.model
    return BayesianTransformerModel2(
        d_inp      = m.d_inp,
        d_model    = m.d_model,
        nhead      = m.nhead,
        nhid       = m.nhid,
        nlayers    = m.nlayers,
        dropout_p  = m.dropout_p,
        max_len    = m.max_len,
        d_static   = m.d_static,
        MAX        = m.MAX,
        perc       = getattr(m, "perc", 0.25),
        aggreg     = getattr(m, "aggreg", "mean"),
        n_classes  = num_classes,
        use_static = getattr(m, "use_static", True),
    )
