"""Registry entry for BayesianSANDLegacyModel (the broken STraTS attention).

The model name is derived from this filename: 'sand_legacy'.
All architecture code lives in bayesian_sand_legacy.py; only for reproducing
SAND runs from before 2026-10-24.
"""

from .bayesian_sand_legacy import BayesianSANDLegacyModel
from .registry import register_model


@register_model
def sand_legacy(
    config,
    num_classes: int = 2,
    data_shape=None,
    **kwargs,
) -> BayesianSANDLegacyModel:
    """Factory registered under the name ``'sand_legacy'``.

    Expected config fields (all under ``config.model``):
      d_inp, T, d_static, hid_dim, num_heads, num_layers, r, M, dropout_p,
      use_static (optional, default True), d_static_out (optional, default None)
    """
    m = config.model
    return BayesianSANDLegacyModel(
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
        use_static = getattr(m, "use_static", True),
        d_static_out = getattr(m, "d_static_out", None),
    )
