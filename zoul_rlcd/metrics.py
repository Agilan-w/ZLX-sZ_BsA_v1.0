# Copyright 2026 ZLX-CDE V2.5
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibration-aware metric tracking for RLCD evaluation.

Provides :class:`CalibrationTracker`, a stateful metric aggregator that
computes Top-1 Accuracy, Macro F1, Brier Score, and Expected Calibration
Error (ECE) across an entire evaluation epoch.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassF1Score,
)

logger = logging.getLogger(__name__)


class CalibrationTracker:
    """Epoch-level calibration and performance metric tracker.

    Wraps TorchMetrics classifiers alongside a manual Brier Score
    accumulator to compute the full RLCD evaluation suite in a single
    pass over validation data.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    n_bins : int
        Number of equal-width confidence bins for ECE computation.
        Recommended range is 10–15; defaults to ``15`` for finer
        granularity.
    device : str | torch.device
        Device on which metric tensors are allocated.

    Examples
    --------
    >>> tracker = CalibrationTracker(num_classes=5, device="cuda")
    >>> for batch in val_loader:
    ...     logits = model(**batch).logits
    ...     tracker.update(logits, batch["labels"])
    >>> metrics = tracker.compute()
    >>> print(metrics)
    {'accuracy': 0.82, 'macro_f1': 0.79, 'brier_score': 0.31, 'ece': 0.04}
    """

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 15,
        device: Optional[str | torch.device] = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

        effective_device = device or "cpu"

        self.num_classes = num_classes

        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        ).to(effective_device)

        self.f1 = MulticlassF1Score(
            num_classes=num_classes, average="macro"
        ).to(effective_device)

        self.ece = MulticlassCalibrationError(
            num_classes=num_classes, n_bins=n_bins, norm="l1"
        ).to(effective_device)

        # Manual Brier score accumulation (running sums)
        self._brier_sum = torch.tensor(0.0, device=effective_device)
        self._sample_count = torch.tensor(0, device=effective_device)

        self._device = effective_device

    def update(self, logits: Tensor, targets: Tensor) -> None:
        """Accumulate a batch of predictions.

        Parameters
        ----------
        logits : Tensor
            Raw model outputs of shape ``(N, C)``.  Softmax is applied
            internally — do **not** pre-apply it.
        targets : Tensor
            Integer class labels of shape ``(N,)``.
        """
        probs: Tensor = F.softmax(logits.float(), dim=-1)
        targets = targets.to(self._device)

        self.accuracy.update(probs, targets)
        self.f1.update(probs, targets)
        self.ece.update(probs, targets)

        # Brier score: per-sample sum of squared differences
        one_hot = F.one_hot(
            targets, num_classes=self.num_classes
        ).float()
        batch_brier: Tensor = ((probs - one_hot) ** 2).sum(dim=-1).sum()
        self._brier_sum += batch_brier
        self._sample_count += targets.size(0)

    def compute(self) -> dict[str, float]:
        """Compute all metrics over accumulated data.

        Returns
        -------
        dict[str, float]
            Dictionary with keys ``"accuracy"``, ``"macro_f1"``,
            ``"brier_score"``, and ``"ece"``.
        """
        if self._sample_count.item() == 0:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "brier_score": 0.0,
                "ece": 0.0,
            }

        brier: float = (self._brier_sum / self._sample_count).item()

        return {
            "accuracy": self.accuracy.compute().item(),
            "macro_f1": self.f1.compute().item(),
            "brier_score": brier,
            "ece": self.ece.compute().item(),
        }

    def reset(self) -> None:
        """Reset all internal metric states for a new epoch."""
        self.accuracy.reset()
        self.f1.reset()
        self.ece.reset()
        self._brier_sum.zero_()
        self._sample_count.zero_()

    def log_summary(self, epoch: int) -> dict[str, float]:
        """Compute, log, and return all metrics.

        Convenience method that calls :meth:`compute`, logs the results,
        and returns the metrics dict.

        Parameters
        ----------
        epoch : int
            Current epoch number (used in the log message).

        Returns
        -------
        dict[str, float]
            Same as :meth:`compute`.
        """
        metrics = self.compute()
        logger.info(
            "[Epoch %d] Validation -- Acc: %.4f | F1: %.4f | "
            "Brier: %.4f | ECE: %.4f",
            epoch,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["brier_score"],
            metrics["ece"],
        )
        return metrics
