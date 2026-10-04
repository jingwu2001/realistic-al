"""Tests for the MIMIC-III multi-label phenotyping task (mimic3_pheno).

Covers the three pieces that are not just wiring:
  1. the phenotype label builder (ICD9 -> 25 CCS conditions multi-hot),
  2. multi-label BALD == sum of per-label binary mutual information,
  3. the multi-label BADGE last-layer gradient embedding == analytic BCE gradient.
Plus the single-task mode (data.multitask=false: one condition, one output).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(os.path.abspath("src"))

from data.mimic_dataset import (
    PHENO_CONDITION_NAMES,
    PHENO_NUM_CLASSES,
    MIMIC3PhenotypeDataset,
    pheno_single_task_id,
)
from data.preprocess_mimic3_sand import DEFAULT_PHENO_DEFS, build_phenotype_matrix
from models.callbacks.metrics_callback import MetricCallback, MultiLabelMetricCallback
from query.query_diversity import DEVICE, get_grad_embedding
from query.query_uncertainty import mutual_bald_multilabel
from utils.wandb_utils import _read_loop_metrics

INPUT_DIM = 8
FEATURE_DIM = 16
NUM_CLASSES = 4


# --------------------------------------------------------------------------- #
# 1. Phenotype label builder                                                  #
# --------------------------------------------------------------------------- #
def test_phenotype_label_builder(tmp_path):
    """ICD9 codes -> 25-condition multi-hot, joined ts_id->HADM_ID->code->CCS.

    Uses the REAL hcup_ccs_2015_definitions.yaml (25 benchmark groups) with a
    tiny synthetic oc + DIAGNOSES_ICD to verify the mapping, column order, and
    all-negative handling.
    """
    # stay 10 -> admission 100 has CHF (4280) + septicemia (0389);
    # stay 11 -> admission 101 has only a non-benchmark code;
    # stay 12 -> admission 102 has no diagnosis rows at all.
    oc = pd.DataFrame({"ts_id": [10, 11, 12], "HADM_ID": [100, 101, 102]})
    diag = pd.DataFrame(
        {
            "HADM_ID": [100, 100, 101],
            "ICD9_CODE": ["4280", "0389", "99999"],  # last is not a CCS code
        }
    )
    diag_csv = tmp_path / "DIAGNOSES_ICD.csv"
    diag.to_csv(diag_csv, index=False)

    all_ids = np.array([10, 11, 12])
    pheno, names = build_phenotype_matrix(oc, str(diag_csv), DEFAULT_PHENO_DEFS, all_ids)

    assert pheno.shape == (3, 25)
    assert len(names) == 25 and names == sorted(names)  # canonical alpha order

    chf = names.index("Congestive heart failure; nonhypertensive")
    sep = names.index("Septicemia (except in labor)")
    # stay 10: exactly CHF + septicemia set
    assert pheno[0, chf] == 1 and pheno[0, sep] == 1 and pheno[0].sum() == 2
    # stay 11: only a non-benchmark code -> all-negative row
    assert pheno[1].sum() == 0
    # stay 12: no diagnoses -> all-negative row
    assert pheno[2].sum() == 0
    # row order follows all_ids
    assert pheno.dtype == np.int8
    # the hard-coded PHENO_CONDITION_NAMES (used only to name per-label metrics)
    # must match the builder's canonical column order exactly.
    assert names == PHENO_CONDITION_NAMES


# --------------------------------------------------------------------------- #
# 1b. Per-label phenotype metrics -> al/phenotype/* subtab                     #
# --------------------------------------------------------------------------- #
def _run_val_epoch(cb, n_batches=4, batch=16, seed=0, num_labels=PHENO_NUM_CLASSES):
    """Drive a callback through one validation epoch, capturing logged scalars."""

    class FakePL:
        def __init__(self):
            self.logged = {}

        def _put(self, k, v):
            self.logged[k] = v.item() if hasattr(v, "item") else v

        def log(self, key, val, **kw):
            self._put(key, val)

        def log_dict(self, d, **kw):
            for k, v in d.items():
                self._put(k, v)

    pl = FakePL()
    torch.manual_seed(seed)
    cb.on_validation_epoch_start(None, pl)
    for _ in range(n_batches):
        probs = torch.rand(batch, num_labels)
        y = torch.randint(0, 2, (batch, num_labels))
        cb.on_validation_batch_end(None, pl, (probs, y), None, 0)
    MetricCallback.on_validation_epoch_end(cb, None, pl)
    return pl.logged


def test_multilabel_callback_logs_per_label_metrics():
    """MultiLabelMetricCallback emits a per-condition acc/auroc/auprc/f1 for all
    25 labels (val), named from PHENO_CONDITION_NAMES, alongside the macro
    summaries — and never for train."""
    cb = MultiLabelMetricCallback(
        num_classes=PHENO_NUM_CLASSES, label_names=PHENO_CONDITION_NAMES
    )
    assert len(cb.label_tokens) == PHENO_NUM_CLASSES == len(set(cb.label_tokens))
    assert cb.label_tokens[8].startswith("08_congestive_heart_failure")

    logged = _run_val_epoch(cb)
    pheno = {k for k in logged if "/phenotype/" in k}
    # 25 labels x {acc, auroc, auprc, f1}
    assert len(pheno) == PHENO_NUM_CLASSES * 4
    assert {k.rsplit("/", 1)[1] for k in pheno} == {"acc", "auroc", "auprc", "f1"}
    # macro summaries still present, and no per-label train metrics exist.
    assert "val/macro_auroc" in logged
    assert not any(k.startswith("train/") for k in logged)
    # all logged per-label values are finite probabilities in [0, 1].
    assert all(0.0 <= logged[k] <= 1.0 for k in pheno)


def test_per_label_metrics_fall_back_to_indices_without_names():
    """No label_names -> cls_NN tokens (still 25x4 keys)."""
    cb = MultiLabelMetricCallback(num_classes=PHENO_NUM_CLASSES, label_names=None)
    assert cb.label_tokens[0] == "cls_00" and cb.label_tokens[24] == "cls_24"
    logged = _run_val_epoch(cb)
    assert sum("/phenotype/" in k for k in logged) == PHENO_NUM_CLASSES * 4


def test_single_task_callback_logs_binary_metrics():
    """num_classes=1 (single-task): binary val/auroc|auprc|f1 headline, no macro
    keys, and the per-label keys carry the condition's original index token so
    they match the multi-task runs' per-condition keys."""
    cb = MultiLabelMetricCallback(
        num_classes=1, label_names=[PHENO_CONDITION_NAMES[8]], label_ids=[8]
    )
    multi = MultiLabelMetricCallback(
        num_classes=PHENO_NUM_CLASSES, label_names=PHENO_CONDITION_NAMES
    )
    assert cb.label_tokens == [multi.label_tokens[8]]

    logged = _run_val_epoch(cb, num_labels=1)
    assert {"val/auroc", "val/auprc", "val/f1"} <= set(logged)
    assert not any("macro" in k or "micro" in k for k in logged)
    token = cb.label_tokens[0]
    pheno = {k for k in logged if "/phenotype/" in k}
    assert pheno == {f"val/phenotype/{token}/{m}" for m in ("acc", "auroc", "auprc", "f1")}
    assert logged[f"val/phenotype/{token}/auroc"] == pytest.approx(logged["val/auroc"])
    assert all(0.0 <= logged[k] <= 1.0 for k in pheno)


def test_pheno_single_task_id():
    """multitask (default) -> None; multitask=false -> validated task_id."""
    assert pheno_single_task_id(OmegaConf.create({})) is None
    assert pheno_single_task_id(OmegaConf.create({"multitask": True, "task_id": 8})) is None
    assert pheno_single_task_id(OmegaConf.create({"multitask": False, "task_id": 11})) == 11
    for bad in (None, -1, PHENO_NUM_CLASSES):
        with pytest.raises(ValueError):
            pheno_single_task_id(OmegaConf.create({"multitask": False, "task_id": bad}))


_PHENO_ROOT = os.path.join(os.environ.get("DATA_ROOT", ""), "mimic3_sand")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_PHENO_ROOT, "all", "test", "phenotype.npy")),
    reason="needs $DATA_ROOT/mimic3_sand phenotype objects",
)
@pytest.mark.parametrize("task_id", [8, 11])
def test_single_task_dataset_selects_one_column(task_id):
    """task_id keeps only that phenotype.npy column: targets (n, 1), label (1,)."""
    full = MIMIC3PhenotypeDataset(_PHENO_ROOT, train=False)
    one = MIMIC3PhenotypeDataset(_PHENO_ROOT, train=False, task_id=task_id)
    assert full.num_classes == PHENO_NUM_CLASSES and one.num_classes == 1
    assert one.targets.shape == (len(full), 1)
    assert np.array_equal(one.targets[:, 0], full.targets[:, task_id])
    (_, _), label = one[0]
    assert label.shape == (1,) and label.dtype == torch.float32


def test_read_loop_metrics_routes_phenotype_to_subtab(tmp_path):
    """metrics.csv per-label columns route to al/phenotype/{label}/{mode}_{metric};
    val aggregates as max, test as last non-NaN, others keep flat al/ naming."""
    nan = float("nan")
    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "val/macro_auroc": [0.7, 0.8, 0.75],
            "test/macro_auroc": [nan, nan, 0.72],
            "val/phenotype/08_chf/auroc": [0.6, 0.9, 0.85],
            "test/phenotype/08_chf/auroc": [nan, nan, 0.66],
            "test/phenotype/08_chf/acc": [nan, nan, 0.90],
        }
    )
    df.to_csv(tmp_path / "metrics.csv", index=False)
    out = _read_loop_metrics(str(tmp_path))

    assert out["al/val_macro_auroc"] == 0.8
    assert out["al/test_macro_auroc"] == 0.72
    assert out["al/phenotype/08_chf/val_auroc"] == 0.9  # val -> max over epochs
    assert out["al/phenotype/08_chf/test_auroc"] == 0.66  # test -> last non-NaN
    assert out["al/phenotype/08_chf/test_acc"] == 0.90


# --------------------------------------------------------------------------- #
# 2. Multi-label BALD                                                         #
# --------------------------------------------------------------------------- #
def test_multilabel_bald_equals_sum_of_binary_mi():
    """mutual_bald_multilabel == sum over labels of per-label binary MI,
    computed independently in numpy; and MI is non-negative."""
    torch.manual_seed(0)
    B, K, C = 6, 8, 5
    logits = torch.randn(B, K, C)
    got = mutual_bald_multilabel(logits).numpy()

    p = torch.sigmoid(logits).numpy()  # (B, K, C)

    def bent(q):
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -(q * np.log(q) + (1 - q) * np.log(1 - q))

    p_bar = p.mean(axis=1)  # (B, C)
    expected = bent(p_bar).sum(axis=1) - bent(p).mean(axis=1).sum(axis=1)

    assert np.allclose(got, expected, atol=1e-5)
    assert (got >= -1e-5).all()  # mutual information is non-negative


def test_multilabel_bald_zero_when_no_disagreement():
    """If all MC samples give identical logits, epistemic MI is ~0."""
    B, K, C = 4, 10, 5
    base = torch.randn(B, 1, C)
    logits = base.expand(B, K, C).contiguous()  # identical across k
    got = mutual_bald_multilabel(logits)
    assert torch.allclose(got, torch.zeros(B), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Multi-label BADGE gradient embedding                                     #
# --------------------------------------------------------------------------- #
class MultiLabelMockModel(torch.nn.Module):
    """Minimal multi-label model for get_grad_embedding: get_features,
    .model.classifier, hparams.model.small_head, and multilabel=True."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        inner = torch.nn.Module()
        inner.classifier = torch.nn.Linear(FEATURE_DIM, num_classes)
        self.model = inner
        self.encoder = torch.nn.Linear(INPUT_DIM, FEATURE_DIM)
        self.hparams = OmegaConf.create({"model": {"small_head": True}})
        self.multilabel = True

    def get_features(self, x):
        return self.encoder(x)


def _analytic_linear_embedding(model, data, targets):
    """Analytic (sigmoid(logits) - target) outer-product feature embedding."""
    with torch.no_grad():
        feats = model.get_features(data.to(DEVICE))
        logits = model.model.classifier(feats)
        l0 = torch.sigmoid(logits) - targets.to(DEVICE)  # (n, C)
        emb = torch.einsum("bc,bz->bcz", l0, feats).reshape(len(data), -1)
    return emb.cpu()


@pytest.mark.parametrize("num_classes", [NUM_CLASSES, 1])  # 1 = single-task
@pytest.mark.parametrize("use_true_labels", [True, False])
def test_multilabel_badge_grad_embedding_matches_analytic(use_true_labels, num_classes):
    """The 'linear' multi-label BADGE embedding equals the analytic BCE
    last-layer gradient: g_x = concat_c[(sigmoid(logit_c) - y_c) * h_x]  (C*Z).
    True-labels path uses the multi-hot target; pool path uses the (p>0.5)
    pseudo-label."""
    torch.manual_seed(0)
    model = MultiLabelMockModel(num_classes).to(DEVICE)
    n = 12
    data = torch.randn(n, INPUT_DIM)
    targets = torch.randint(0, 2, (n, num_classes)).float()
    loader = DataLoader(TensorDataset(data, targets), batch_size=5)

    emb = get_grad_embedding(
        model,
        loader,
        grad_embedding_type="linear",
        device=DEVICE,
        use_true_labels=use_true_labels,
    )
    assert emb.shape == (n, num_classes * FEATURE_DIM)

    if use_true_labels:
        y = targets
    else:
        with torch.no_grad():
            logits = model.model.classifier(model.get_features(data.to(DEVICE)))
            y = (torch.sigmoid(logits) > 0.5).float().cpu()
    expected = _analytic_linear_embedding(model, data, y)
    assert torch.allclose(emb, expected, atol=1e-5)
