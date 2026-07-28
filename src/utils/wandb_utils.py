"""Centralized wandb tracking for the active-learning loop.

All wandb wiring lives here so it stays out of ``main.py``/``active_loop``.

Metric-namespace conventions
----------------------------
- ``al/*`` metrics are per-AL-iteration and use ``step_metric="al_iter"``
  (registered once in :func:`init_wandb`). Anything logged once per active
  learning round belongs under ``al/``.
- ``al/query/*`` carries query-time score diagnostics (acquired / unacquired
  score distribution) so drift across rounds is visible directly in the wandb
  UI instead of only in the ``extra_info_{i}.npz`` files on disk.
- ``init_label_dist/*`` summary keys capture the per-split label distribution
  once at the start of the run.

The full resolved Hydra config is logged to ``wandb.config`` so sweeps over
kernel / normalization / quality / q / alpha are filterable in the UI without
touching this code every time a new option is added.
"""

import math
import os
import subprocess

import numpy as np
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf


# --------------------------------------------------------------------------- #
# Run naming: method-variant tags from query hyperparameters
# --------------------------------------------------------------------------- #

# Short display abbreviations so wandb doesn't truncate names in the sidebar.
_KERNEL_ABBR = {"cosine": "cos", "cos": "cos", "rbf": "rbf", "linear": "lin", "lin": "lin"}
_NORM_ABBR = {"l2": "l2", "minmax": "minmax", "zscore": "zscore", "none": "none"}


def _kernel_hp(vendi_cfg: DictConfig):
    """Extract the shared vendi kernel hyperparameters ``(emb, kernel, norm, q)``.

    Both vendi and bandit use the same vendi kernel machinery for diversity
    (bandit inherits the ``vendi`` config block), so this is factored out and
    reused by both method-variant builders.
    """
    # embedding tag from use_grad ("grad" = BADGE-style last-layer gradients,
    # plus the embedding type when it deviates from the default full linear-
    # layer gradient); an explicit `embedding`/`emb` key still wins.
    if vendi_cfg.get("use_grad", False):
        default_emb = "grad"
        if str(vendi_cfg.get("grad_embedding_type", "linear")) != "linear":
            default_emb += str(vendi_cfg.grad_embedding_type)
    else:
        default_emb = "feat"
    emb = str(vendi_cfg.get("embedding", vendi_cfg.get("emb", default_emb)))
    kernel = _KERNEL_ABBR.get(str(vendi_cfg.kernel).lower(), str(vendi_cfg.kernel).lower())
    if kernel == "rbf":
        # gamma is an rbf sweep axis (float / "dim" / "median"); encode it in
        # the kernel token so gamma variants don't collide in wandb run names
        # (e.g. rbf1.0 / rbfdim / rbfmed)
        gamma = str(vendi_cfg.get("gamma", "")).replace("median", "med")
        if gamma:
            kernel = f"rbf{gamma}"
    norm = _NORM_ABBR.get(str(vendi_cfg.normalization).lower(), str(vendi_cfg.normalization).lower())
    q = vendi_cfg.q
    return emb, kernel, norm, q


def _vendi_variant(cfg: DictConfig):
    """Encode the vendi settings that distinguish one run from another.

    Returns ``(method_tag, hyperparam_tags)`` where ``method_tag`` looks like
    ``vendi-grad-cos-l2-q1.0-a0.5-sgradnorm`` and ``hyperparam_tags`` is a flat
    dict used for cross-cutting wandb tags (``kernel:cos``, ``norm:l2``, ...).
    Fields that don't exist yet (``alpha``/``quality`` for the gradient work)
    are simply omitted, so this degrades gracefully.
    """
    v = cfg.query.vendi
    emb, kernel, norm, q = _kernel_hp(v)

    # gvendi shares this builder (same vendi block, use_grad=true); the query
    # name keeps the variants distinct: vendi-feat-... vs gvendi-grad-...
    tag = f"{cfg.query.name}-{emb}-{kernel}-{norm}-q{q}"
    hp = {"emb": emb, "kernel": kernel, "norm": norm, "q": str(q)}

    alpha = v.get("alpha", None)
    if alpha is not None:
        tag += f"-a{alpha}"
        hp["alpha"] = str(alpha)

    quality = v.get("quality", None)
    if quality is not None:
        tag += f"-s{quality}"
        hp["quality"] = str(quality)

    return tag, hp


def _bandit_variant(cfg: DictConfig):
    """Encode the bandit settings that distinguish one run from another.

    Bandit selects among diversity contexts built on the vendi kernel, so it
    carries the same ``emb/kernel/norm/q`` tags as vendi (when the inherited
    ``vendi`` block is present), plus the bandit-specific ``alpha`` (LinUCB
    exploration) and ``context_dim``. e.g. ``bandit-feat-rbf-minmax-q1.0-a1.0-c6``.
    """
    b = cfg.query.bandit

    tag = "bandit"
    hp = {}

    v = cfg.query.get("vendi", None)
    if v is not None:
        emb, kernel, norm, q = _kernel_hp(v)
        tag += f"-{emb}-{kernel}-{norm}-q{q}"
        hp.update({"emb": emb, "kernel": kernel, "norm": norm, "q": str(q)})

    alpha = b.get("alpha", None)
    if alpha is not None:
        tag += f"-a{alpha}"
        hp["alpha"] = str(alpha)

    context_dim = b.get("context_dim", None)
    if context_dim is not None:
        tag += f"-c{context_dim}"
        hp["context_dim"] = str(context_dim)

    return tag, hp


# Per-method registry. Methods without extra hyperparameters (badge/bald/...)
# fall through to the default and keep their short query name.
_METHOD_VARIANT = {
    "vendi": _vendi_variant,
    "gvendi": _vendi_variant,  # config alias of vendi (use_grad=true)
    "bandit": _bandit_variant,
}


def method_variant(cfg: DictConfig):
    """Return ``(method_tag, hyperparam_tags)`` for the configured query method."""
    fn = _METHOD_VARIANT.get(cfg.query.name)
    if fn is None:
        return cfg.query.name, {}
    return fn(cfg)


def make_run_name(cfg: DictConfig, data_name: str = None) -> str:
    """e.g. ``cifar10_imb/vendi-feat-rbf-minmax-q1.0/seed-12345``.

    ``data_name`` should be the hydra data-config choice (e.g.
    ``p12_transformer_baleval``): several data configs share one
    ``cfg.data.name`` (baleval vs natural split, cifar10 vs cifar10_imb),
    which would otherwise collide in wandb.

    Well under wandb's 128-char run-name limit even for the longest variants.
    """
    tag, _ = method_variant(cfg)
    return f"{data_name or cfg.data.name}/{tag}/seed-{cfg.trainer.seed}"


def make_group(cfg: DictConfig, active_name: str, data_name: str = None) -> str:
    """e.g. ``cifar10_imb/cifar10_low/vendi-feat-rbf-minmax-q1.0``.

    Grouping by method-variant keeps a sweep from lumping all vendi variants
    together.
    """
    tag, _ = method_variant(cfg)
    return f"{data_name or cfg.data.name}/{active_name}/{tag}"


def make_tags(cfg: DictConfig, active_name: str, session_tag: str, data_name: str = None) -> list:
    """Build wandb tags: the coarse identity tags plus per-hyperparameter tags
    (``kernel:cos``, ``norm:l2``, ...) for cross-cutting filtering."""
    _, hp = method_variant(cfg)
    tags = [data_name or cfg.data.name, active_name, cfg.query.name, session_tag]
    tags += [f"{k}:{val}" for k, val in hp.items()]
    if cfg.trainer.is_test:
        tags.append("test")
    if cfg.trainer.get("wandb_test", False):
        # wandb wiring checks: tagged so they can be bulk-selected and deleted
        tags.append("wandb_test")
    return tags


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #

def init_wandb(cfg: DictConfig):
    """Initialize the wandb run, or return ``None`` if wandb is disabled.

    Logs the full resolved Hydra config plus a handful of flat convenience
    aliases, and registers the ``al/*`` step metric.
    """
    if not cfg.trainer.use_wandb or cfg.trainer.dry_run:
        return None

    hydra_choices = HydraConfig.get().runtime.choices
    active_name = hydra_choices.get("active", "")
    # the data-config choice (e.g. "p12_transformer_baleval", "cifar10_imb")
    # distinguishes eval-split/imbalance variants that share cfg.data.name
    data_name = hydra_choices.get("data", None) or cfg.data.name

    # git commit for the session tag; fall back gracefully outside a checkout
    # (e.g. a bare copy on a cluster).
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode()
        )
    except Exception:
        logger.warning("git rev-parse failed; using 'nogit' for the session tag")
        commit = "nogit"

    _eid = cfg.trainer.experiment_id  # "2026-05-23_18-53-00-411537"
    date_str = _eid[:10].replace("-", "")  # "20260523"
    time_str = _eid[11:19].replace("-", "")  # "185300"
    session_tag = f"{date_str}-{time_str}_{commit}"  # "20260523-185300_f8fe30ab"

    # Guard notes: str(None) -> "None" (truthy), so an unset value must become
    # a real None, not the string "None".
    notes = cfg.trainer.get("wandb_notes", None)
    notes = str(notes) if notes not in (None, "") else None

    # Log the entire resolved config so any option is filterable in sweeps,
    # then add flat aliases that used to be hand-picked (kept for dashboards).
    config = OmegaConf.to_container(cfg, resolve=True)
    config.update(
        {
            "query_name": cfg.query.name,
            "model_name": cfg.model.name,
            "data_name": data_name,
            "active_name": active_name,
            "seed": cfg.trainer.seed,
            "num_labelled": cfg.active.num_labelled,
            "acq_size": cfg.active.acq_size,
            "num_iter": cfg.active.num_iter,
        }
    )

    wandb_run = wandb.init(
        project=cfg.trainer.wandb_project,
        name=make_run_name(cfg, data_name),
        group=make_group(cfg, active_name, data_name),
        tags=make_tags(cfg, active_name, session_tag, data_name),
        notes=notes,
        config=config,
    )
    # n_labelled is the natural budget axis for AL curves — make it the
    # default x-axis of every auto-generated al/* panel (al_iter is still
    # logged alongside for reference / exact-round alignment).
    wandb_run.define_metric("al/n_labelled")
    wandb_run.define_metric("al/*", step_metric="al/n_labelled")
    return wandb_run


# --------------------------------------------------------------------------- #
# Per-iteration and summary logging
# --------------------------------------------------------------------------- #

def _read_loop_metrics(log_dir) -> dict:
    """Return al/-prefixed metrics from a loop's metrics.csv, aggregated per AL iteration.

    val/* -> max across epochs (best checkpoint value)
    train/* and test/* -> last non-NaN value
    """
    try:
        df = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
        result = {}
        for col in df.columns:
            if col in ("epoch", "step"):
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            if col.startswith("val/"):
                val = series.max()
            else:
                val = series.iloc[-1]
            result[f"al/{col.replace('/', '_')}"] = val
        return result
    except Exception as e:
        logger.warning(f"Failed to read loop metrics from {log_dir}: {e}")
    return {}


def _query_score_metrics(extra_info) -> dict:
    """Route query-time score diagnostics into ``al/query/*`` metrics.

    ``extra_info`` stores acquired/unacquired score ``[max, min, median]``
    arrays (see ``query_diversity.calculate_extra_info``). Surfacing them as
    metrics makes score-distribution drift across rounds visible in wandb.
    """
    out = {}
    if not extra_info:
        return out
    labels = ["max", "min", "median"]
    if "acq" in extra_info:
        for name, v in zip(labels, np.asarray(extra_info["acq"]).ravel()):
            out[f"al/query/acquired_score_{name}"] = float(v)
    if "else" in extra_info:
        for name, v in zip(labels, np.asarray(extra_info["else"]).ravel()):
            out[f"al/query/unacquired_score_{name}"] = float(v)
    return out


def log_al_iteration(
    wandb_run,
    i: int,
    cfg: DictConfig,
    train_timer,
    query_timer=None,
    eig_time=float("nan"),
    log_dir=None,
    extra_info=None,
):
    """Log one active-learning iteration.

    Unifies the normal path and the pool-exhausted early-exit path so the two
    can't drift. On the early-exit path there is no query, so ``query_timer``
    is ``None`` and ``eig_time`` is NaN.
    """
    if wandb_run is None:
        return

    query_time_s = float("nan") if query_timer is None else round(query_timer.elapsed, 4)
    log_dict = {
        "al_iter": i,
        "al/n_labelled": cfg.active.num_labelled,
        "al/train_time_s": round(train_timer.elapsed, 4),
        "al/query_time_s": query_time_s,
        "al/eig_time_s": eig_time,
    }
    if log_dir is not None:
        log_dict.update(_read_loop_metrics(log_dir))
    log_dict.update(_query_score_metrics(extra_info))
    wandb_run.log(log_dict)


def compute_aubc(x, y) -> float:
    """Area under the budget curve, normalized by the budget span.

    Trapezoidal integral of metric ``y`` against labeled-set size ``x``,
    divided by ``x[-1] - x[0]`` — i.e. the budget-weighted mean of the
    metric, living on the same scale as the metric itself. The standard AL
    summary statistic: it rewards methods that reach good performance
    *early* in the budget, not just at the final iteration.

    Non-finite pairs are dropped; returns NaN with fewer than 2 valid points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x[-1] == x[0]:
        return float("nan")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def aubc_summary(metrics_df, num_samples) -> dict:
    """AUBC for every per-iteration test metric column.

    ``metrics_df``: one row per AL iteration (columns like ``test/acc``,
    ``test/auroc``, ``test/auprc``); ``num_samples``: labeled-set size at each
    iteration. Returns ``{"al_summary/aubc_test_acc": ..., ...}`` suitable for
    ``wandb_run.summary.update``.
    """
    out = {}
    n = min(len(metrics_df), len(num_samples))
    if n < 2:
        return out
    x = np.asarray(num_samples)[:n]
    for col in metrics_df.columns:
        y = pd.to_numeric(metrics_df[col], errors="coerce").values[:n]
        v = compute_aubc(x, y)
        if np.isfinite(v):
            out[f"al_summary/aubc_{col.replace('/', '_')}"] = round(v, 6)
    return out


def log_label_distributions(wandb_run, datamodule):
    """Log per-split label distributions as ``init_label_dist/*`` summary keys."""
    if wandb_run is None:
        return

    def _dist_counts(targets) -> dict:
        tgts = np.asarray(targets)
        classes, counts = np.unique(tgts, return_counts=True)
        total = len(tgts)
        d = {"n": int(total)}
        for c, n in zip(classes, counts):
            d[f"class_{int(c)}_count"] = int(n)
            d[f"class_{int(c)}_pct"] = round(100 * n / total, 2)
        return d

    def _subset_targets(subset):
        # val_set is a torch Subset (carved from train); test_set is often the
        # raw dataset (e.g. CIFAR10/MNIST) with no .indices/.dataset.
        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            return np.asarray(subset.dataset.targets)[subset.indices]
        return np.asarray(subset.targets)

    ts = datamodule.train_set
    for split, tgts in [
        ("L", ts.labelled_set.targets),
        ("U", ts.pool.targets),
        ("L+U", ts._dataset.targets),
        ("val", _subset_targets(datamodule.val_set)),
        ("test", _subset_targets(datamodule.test_set)),
    ]:
        for k, v in _dist_counts(tgts).items():
            wandb_run.summary[f"init_label_dist/{split}/{k}"] = v
