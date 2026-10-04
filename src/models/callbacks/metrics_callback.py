import re
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torchmetrics import AUROC, AveragePrecision, ConfusionMatrix
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)


def _label_token(idx: int, name: Optional[str]) -> str:
    """Compact, wandb-safe per-label key: ``NN`` prefix + slugged condition name.

    The zero-padded index keeps tokens ordered and unique even if two slugs
    collide or truncate to the same text; ``name=None`` falls back to
    ``cls_NN``. e.g. ``08_congestive_heart_failure_nonhyperte``.
    """
    if name is None:
        return f"cls_{idx:02d}"
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()[:32].strip("_")
    return f"{idx:02d}_{slug}"


def weighted_cohen_kappa(conf_mat: torch.Tensor, weights: str = "linear") -> torch.Tensor:
    """Weighted Cohen's kappa computed directly from a KxK confusion matrix.

    ``weights='linear'`` uses |i-j|/(K-1) penalties (the convention used by the
    MIMIC-III benchmark LOS task); ``'quadratic'`` uses ((i-j)/(K-1))**2.
    Robust to empty rows/columns: a class with no support contributes no mass to
    the observed/expected agreement, so it cannot produce NaN as long as at least
    two classes carry mass. Returns NaN only for a degenerate (single-class or
    empty) matrix.
    """
    C = conf_mat.float()
    N = C.sum()
    K = C.shape[0]
    if N == 0 or K < 2:
        return torch.tensor(float("nan"))
    O = C / N                              # observed agreement
    row = O.sum(dim=1, keepdim=True)       # (K, 1)
    col = O.sum(dim=0, keepdim=True)       # (1, K)
    E = row * col                          # expected agreement under independence
    idx = torch.arange(K, dtype=torch.float, device=C.device)
    diff = torch.abs(idx[:, None] - idx[None, :]) / (K - 1)
    W = diff ** 2 if weights == "quadratic" else diff
    denom = (W * E).sum()
    if denom == 0:
        return torch.tensor(float("nan"))
    return 1.0 - (W * O).sum() / denom


class MetricCallback(Callback):
    """Class for using using multiple different metrics including automated resets.
    IMPLEMENTATION: add additional metrics here."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        mode = "train"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        mode = "val"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        mode = "test"
        self.reset_metric_dict(self.auc_dict, mode)
        self.reset_metric_dict(self.pred_dict, mode)
        self.reset_metric_dict(self.pred_conf, mode)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        mode = "train"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        mode = "val"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        mode = "test"
        self.log_metric_dict(self.auc_dict, pl_module, mode=mode)
        self.log_metric_dict(self.pred_dict, pl_module, mode=mode)
        self.log_metrics(pl_module, mode)

    def log_metrics(self, pl_module: pl.LightningModule, mode: str) -> None:
        pass

    @staticmethod
    def log_metric_dict(
        metric_dict: Dict[str, Any], pl_module: pl.LightningModule, mode: str
    ) -> None:
        """Log values in metric dict with correct mode to pytorch lightning module logger.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            pl_module (pl.LightningModule): module used
            mode (str): prefix name of logdict e.g. train, val, test
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                pl_module.log(
                    key, metric_dict[key].compute(), on_step=False, on_epoch=True
                )

    @staticmethod
    def reset_metric_dict(metric_dict: Dict[str, Any], mode: str) -> None:
        """Reset metrics in metric dict for a specific mode.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            mode (str): prefix name of logdict e.g. train, val, test
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].reset()

    @staticmethod
    def update_metric_dict(
        metric_dict: Dict[str, Any], mode: str, preds: torch.Tensor, y: torch.Tensor
    ):
        """Update metrics in metric dict for a specific mode.

        Args:
            metric_dict (Dict[str, Any]): dictionary with keys: {mode/metric_name: metric}
            mode (str): prefix name of logdict e.g. train, val, test
            preds (torch.Tensor): predictions
            y (torch.Tensor): labels
        """
        for key in metric_dict:
            if key.split("/")[0] == mode:
                metric_dict[key].update(preds, y)


### IMPLEMENTATION
# New callbacks for metrics can be implemented here following the examples of ImbClassMetricCallback


class ImbClassMetricCallback(MetricCallback):
    def __init__(self, num_classes: int):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision and values from Confusion Matrix (balanced accuracy, precision, F1 etc.).

        Args:
            num_classes (int): #of classes in dataset
        """
        super().__init__()
        modes = ["train", "val", "test"]
        self.binary = num_classes == 2
        self.pred_dict = {}
        self.pred_conf = {
            mode: ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize=None)
            for mode in modes
        }

        self.auc_dict = {}
        if self.binary:
            self.auc_dict.update({f"{mode}/auroc": AUROC(task="binary") for mode in modes})
            self.auc_dict.update({f"{mode}/auprc": AveragePrecision(task="binary") for mode in modes})
        else:
            self.auc_dict.update(
                {
                    f"{mode}/auroc": AUROC(task="multiclass", num_classes=num_classes, average="macro")
                    for mode in modes
                }
            )
            self.auc_dict.update(
                {
                    f"{mode}/auprc": AveragePrecision(
                        task="multiclass", num_classes=num_classes, average="macro"
                    )
                    for mode in modes
                }
            )

    def compute_pred_metrics(
        self, mode: str, class_wise: bool = False
    ) -> Dict[str, float]:
        """Compute prediction values based on confusion matrix.
        When class_wise is true, return rec, prec and f1 for every class.

        Args:
            mode (str): prefix name of logdict e.g. train, val, test
            class_wise (bool, optional): add values per class. Defaults to False.

        Returns:
            dict: values for logging in subsequent steps
        """
        conf_mat = self.pred_conf[mode].compute()
        acc = conf_mat.diag().sum() / conf_mat.sum()
        # Note: balanced multiclass accuracy = mean of diagonal of conf matrix, divided by positive incidences
        w_acc = (conf_mat.diag() / conf_mat.sum(dim=1)).mean()
        av_prec = (conf_mat.diag() / conf_mat.sum(dim=0)).mean()
        av_f1 = (
            2 * conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0))
        ).mean()

        out_dict = {
            f"{mode}/acc": acc,
            f"{mode}/w_acc": w_acc,
            f"{mode}/av_prec": av_prec,
            f"{mode}/av_f1": av_f1,
            # Linear-weighted Cohen's kappa (ordinal agreement), the headline
            # metric for the length-of-stay task; matches Harutyunyan et al.'s
            # cohen_kappa_score(..., weights='linear'). Purely a function of the
            # confusion matrix, so it costs nothing extra to log for every task.
            f"{mode}/kappa": weighted_cohen_kappa(conf_mat, weights="linear"),
        }
        if class_wise:
            class_wise_dict = {
                "rec": conf_mat.diag() / conf_mat.sum(dim=1),
                "prec": conf_mat.diag() / conf_mat.sum(dim=0),
                "f1": 2 * conf_mat.diag() / (conf_mat.sum(dim=1) + conf_mat.sum(dim=0)),
            }
            for metric in class_wise_dict:
                for cls, val in enumerate(class_wise_dict[metric]):
                    out_dict[f"{mode}/{metric}/cls_{cls}"] = class_wise_dict[metric][
                        cls
                    ]

        return out_dict

    def log_metrics(self, pl_module: pl.LightningModule, mode: str):
        """Compute log_metrics and write them to loggers.

        Args:
            pl_module (pl.LightningModule): Module for training
            mode (str): prefix name of logdict e.g. train, val, test
        """
        if mode in self.pred_conf:
            log_dict = self.compute_pred_metrics(mode)
            pl_module.log_dict(log_dict, on_epoch=True, on_step=False)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, torch.Tensor],
        batch,
        batch_idx,
        unused: int = 0,
    ) -> None:
        """Update confusion matrix and auc dicts with data from train batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        pred = torch.argmax(logprob, dim=-1)
        probs = torch.exp(logprob)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, probs[:, 1] if self.binary else probs, y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        """Update confusion matrix and auc dicts with data from validation batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "val"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        probs = torch.exp(logprob)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, probs[:, 1] if self.binary else probs, y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        """Update confusion matrix and auc dicts with data from test batch.

        Args:
            outputs (Dict[str, torch.Tensor]): contains keys [logprob, label]
            ...
        """
        mode = "test"
        logprob, y = outputs
        pred = torch.argmax(logprob, dim=-1)
        probs = torch.exp(logprob)
        self.update_metric_dict(self.pred_conf, mode, pred.to("cpu"), y.to("cpu"))
        self.update_metric_dict(self.auc_dict, mode, probs[:, 1] if self.binary else probs, y)


class MultiLabelMetricCallback(MetricCallback):
    # per-label metrics are only tracked for these modes (train is too noisy on
    # the small labelled set to be worth 25x the columns).
    PERLABEL_MODES = ("val", "test")

    def __init__(
        self,
        num_classes: int,
        label_names: Optional[List[str]] = None,
        label_ids: Optional[List[int]] = None,
    ):
        """Multi-label metrics for the MIMIC-III phenotyping task (25 independent
        Bernoulli conditions, Harutyunyan et al. 2019).

        Consumes **sigmoid probabilities** ``(B, C)`` (the multi-label model's
        ``step`` returns MC-marginal probs in the ``logprob`` slot) and an int
        multi-hot target. Headline = **macro-AUROC** (``{mode}/macro_auroc``),
        the checkpoint monitor; also logs micro-AUROC, macro-AUPRC, macro-F1.
        No confusion matrix / kappa (undefined for multi-label).

        On top of the macro/micro summaries it also logs **per-label** acc,
        AUROC, AUPRC and F1 for every one of the ``num_classes`` binary tasks
        (val + test only), under keys ``{mode}/phenotype/{label}/{metric}``.
        ``wandb_utils`` routes those into the ``al/phenotype/*`` subtab.

        ``num_classes == 1`` is single-task phenotyping (``data.multitask=false``):
        one binary condition, so the summaries are the binary ``{mode}/auroc``,
        ``{mode}/auprc``, ``{mode}/f1`` (torchmetrics' multilabel metrics need
        >= 2 labels; ``{mode}/auroc`` is the checkpoint monitor). The per-label
        keys are still logged under the condition's token, so they line up with
        the same condition's curves in the multi-task runs.

        Args:
            num_classes (int): number of labels (25 for phenotyping, 1 single-task).
            label_names (list[str], optional): human-readable condition names,
                one per label; used only to name the per-label metric keys.
                ``None`` -> ``cls_00``..``cls_NN``.
            label_ids (list[int], optional): phenotype.npy column of each label,
                used as the per-label key's index prefix. Defaults to
                ``range(num_classes)``; single-task passes ``[task_id]``.
        """
        super().__init__()
        modes = ["train", "val", "test"]
        self.num_labels = num_classes
        self.binary = num_classes == 1
        if label_ids is None:
            label_ids = list(range(num_classes))
        if label_names is not None and len(label_names) != num_classes:
            label_names = None  # length mismatch -> fall back to indices
        self.label_tokens = [
            _label_token(label_ids[i], None if label_names is None else label_names[i])
            for i in range(num_classes)
        ]
        # pred_conf / pred_dict are unused for multi-label; keep them empty so the
        # base epoch hooks (reset/log) iterate over nothing.
        self.pred_conf = {}
        self.pred_dict = {}
        self.auc_dict = {}
        for mode in modes:
            if self.binary:
                self.auc_dict[f"{mode}/auroc"] = BinaryAUROC()
                self.auc_dict[f"{mode}/auprc"] = BinaryAveragePrecision()
                self.auc_dict[f"{mode}/f1"] = BinaryF1Score(threshold=0.5)
                continue
            self.auc_dict[f"{mode}/macro_auroc"] = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.auc_dict[f"{mode}/micro_auroc"] = MultilabelAUROC(
                num_labels=num_classes, average="micro"
            )
            self.auc_dict[f"{mode}/macro_auprc"] = MultilabelAveragePrecision(
                num_labels=num_classes, average="macro"
            )
            self.auc_dict[f"{mode}/macro_f1"] = MultilabelF1Score(
                num_labels=num_classes, average="macro", threshold=0.5
            )

        # Per-label metrics (average=None -> a (num_labels,) vector per compute).
        # Kept in their own dict so the base scalar-logging path never touches
        # them (it would choke on a length-25 tensor); logged element-wise in
        # log_metrics below.
        self.perlabel_dict = {}
        for mode in self.PERLABEL_MODES:
            if self.binary:
                self.perlabel_dict[f"{mode}/auroc"] = BinaryAUROC()
                self.perlabel_dict[f"{mode}/auprc"] = BinaryAveragePrecision()
                self.perlabel_dict[f"{mode}/f1"] = BinaryF1Score(threshold=0.5)
                self.perlabel_dict[f"{mode}/acc"] = BinaryAccuracy(threshold=0.5)
                continue
            self.perlabel_dict[f"{mode}/auroc"] = MultilabelAUROC(
                num_labels=num_classes, average=None
            )
            self.perlabel_dict[f"{mode}/auprc"] = MultilabelAveragePrecision(
                num_labels=num_classes, average=None
            )
            self.perlabel_dict[f"{mode}/f1"] = MultilabelF1Score(
                num_labels=num_classes, average=None, threshold=0.5
            )
            self.perlabel_dict[f"{mode}/acc"] = MultilabelAccuracy(
                num_labels=num_classes, average=None, threshold=0.5
            )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.reset_metric_dict(self.perlabel_dict, "val")

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.reset_metric_dict(self.perlabel_dict, "test")

    def log_metrics(self, pl_module: pl.LightningModule, mode: str) -> None:
        """Log per-label acc/auroc/auprc/f1 for every binary task (val+test).

        Called by the base epoch-end hooks after the scalar dicts are logged.
        Each ``average=None`` metric computes to a (num_labels,) vector, which
        we scatter into ``{mode}/phenotype/{label}/{metric}`` keys.
        """
        if mode not in self.PERLABEL_MODES:
            return
        log_dict = {}
        for key, metric in self.perlabel_dict.items():
            key_mode, metric_name = key.split("/")
            if key_mode != mode:
                continue
            values = metric.compute().reshape(-1)  # (num_labels,); binary: (1,)
            for i, token in enumerate(self.label_tokens):
                log_dict[f"{mode}/phenotype/{token}/{metric_name}"] = values[i]
        if log_dict:
            pl_module.log_dict(log_dict, on_epoch=True, on_step=False)

    def _update(self, mode: str, probs: torch.Tensor, y: torch.Tensor) -> None:
        # torchmetrics multilabel metrics want probs (B,C) float + target int (B,C).
        probs = probs.detach().to("cpu").float()
        y = y.detach().to("cpu").int()
        if self.binary:
            # (B, 1) -> (B,) for the torchmetrics binary metrics
            probs, y = probs.reshape(-1), y.reshape(-1)
        self.update_metric_dict(self.auc_dict, mode, probs, y)
        # perlabel_dict only holds val/test keys, so a "train" update is a no-op.
        self.update_metric_dict(self.perlabel_dict, mode, probs, y)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        # outputs["logprob"] carries sigmoid probs for the multi-label model.
        self._update("train", outputs["logprob"], outputs["label"])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        probs, y = outputs
        self._update("val", probs, y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        probs, y = outputs
        self._update("test", probs, y)


# This code is not used in benchmark, but maybe it is useful for some people.
class ISIC2016MetricCallback(MetricCallback):
    def __init__(self):
        """Callback which creates and tracks the torchmetrics AUROC and Average Prescision."""
        super().__init__()
        modes = ["train", "val", "test"]
        self.auc_dict = {f"{mode}/auroc": AUROC(task="binary") for mode in modes}
        self.auc_dict.update(
            {f"{mode}/auprc": AveragePrecision(task="binary") for mode in modes}
        )
        self.pred_dict = {}
        self.pred_conf = {}

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "train"
        logprob = outputs["logprob"]
        y = outputs["label"]
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "val"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused: int = 0
    ) -> None:
        mode = "test"
        logprob, y = outputs
        self.update_metric_dict(self.auc_dict, mode, torch.exp(logprob), y)
