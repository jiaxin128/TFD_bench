# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class FPRx(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    conf: list[Tensor]
    targets: list[Tensor]

    def __init__(self, recall_level: float, pos_label: int, **kwargs) -> None:
        r"""Compute the False Positive Rate at x% Recall.

        The False Positive Rate at x% Recall (FPR@x) is a metric used in tasks like anomaly detection, out-of-distribution (OOD) detection, and binary classification. It measures the proportion of false positives (normal samples misclassified as anomalies) when the model achieves a specified recall level for the positive class (e.g., anomalies or OOD samples).

        Args:
            recall_level (float): The recall level at which to compute the FPR.
            pos_label (int): The positive label.
            kwargs: Additional arguments to pass to the metric class.

        Reference:
            Improved from https://github.com/hendrycks/anomaly-seg and translated to torch.

        Example:
            .. code-block:: python

                from src.metrics.classification import FPRx

                # Initialize the metric with 95% recall and positive label as 1 (e.g., OOD)
                metric = FPRx(recall_level=0.95, pos_label=1)

                # Simulated model predictions (confidence scores) and ground-truth labels
                conf = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1])
                targets = torch.tensor([1, 0, 1, 0, 0, 1, 0])  # 1: OOD, 0: In-Distribution

                # Update the metric with predictions and labels
                metric.update(conf, targets)

                # Compute FPR at 95% recall
                result = metric.compute()
                print(f"FPR at 95% Recall: {result.item()}")
                # output : FPR at 95% Recall: 0.75
        """
        super().__init__(**kwargs)

        if recall_level < 0 or recall_level > 1:
            raise ValueError(f"Recall level must be between 0 and 1. Got {recall_level}.")
        self.recall_level = recall_level
        self.pos_label = pos_label
        self.add_state("conf", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            f"Metric `FPR{int(recall_level * 100)}` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, conf: Tensor, target: Tensor) -> None:
        """Update the metric state.

        Args:
            conf (Tensor): The confidence scores.
            target (Tensor): The target labels, 0 if ID, 1 if OOD.
        """
        self.conf.append(conf)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the False Positive Rate at x% Recall.

        Returns:
            Tensor: The value of the FPRx.
        """
        conf = dim_zero_cat(self.conf).flatten()
        labels = dim_zero_cat(self.targets).flatten() == self.pos_label
        if conf.shape != labels.shape:
            raise ValueError("Expected `conf` and `target` to have the same shape.")

        num_positive = labels.sum()
        num_negative = (~labels).sum()
        if num_positive == 0 or num_negative == 0:
            dtype = conf.dtype if conf.is_floating_point() else torch.float32
            return torch.tensor(torch.nan, device=conf.device, dtype=dtype)
        if self.recall_level == 0:
            return conf.new_tensor(0.0, dtype=torch.float32)

        order = conf.argsort(descending=True)
        sorted_scores = conf[order]
        sorted_labels = labels[order]

        # Evaluate thresholds only after complete groups of tied scores.
        threshold_idxs = torch.cat(
            (
                torch.where(sorted_scores[1:] != sorted_scores[:-1])[0],
                torch.tensor([labels.numel() - 1], device=labels.device),
            )
        )
        true_positive = sorted_labels.cumsum(dim=0)[threshold_idxs]
        false_positive = threshold_idxs + 1 - true_positive
        recall = true_positive / num_positive

        cutoff = torch.where(recall >= self.recall_level)[0][0]
        return false_positive[cutoff] / num_negative


class FPR95(FPRx):
    def __init__(self, pos_label: int, **kwargs) -> None:
        r"""Compute the False Positive Rate at 95% Recall.

        This is a specific case of the more general FPRx metric, where the recall level is fixed at 95%.

        Args:
            pos_label (int): The positive label (e.g., 1 for OOD samples).
            kwargs: Additional arguments to pass to the FPRx metric class.

        .. seealso::
            - :class:`FPRx` - The base metric that allows customization of the recall level.
        """
        super().__init__(recall_level=0.95, pos_label=pos_label, **kwargs)
