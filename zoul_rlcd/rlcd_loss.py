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

"""RLCD loss functions for calibrated decision training.

Provides a Brier Score loss and a compound loss that combines standard
cross-entropy with the Brier calibration penalty:

    L_total = L_CE + lambda_cal * L_Brier
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BrierScoreLoss(nn.Module):
    """Multi-class Brier Score loss.

    Computes the mean squared error between predicted class probabilities
    and one-hot encoded ground truth labels:

        BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2

    This is a strictly proper scoring rule that encourages calibrated
    probability estimates — the model minimises this loss only when its
    predicted probabilities match the true data-generating distribution.

    Parameters
    ----------
    reduction : str
        Reduction mode: ``"mean"`` (default) averages over the batch,
        ``"sum"`` returns the un-normalised sum, ``"none"`` returns
        per-sample scores.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                "Expected one of: 'mean', 'sum', 'none'."
            )
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Brier Score loss.

        Parameters
        ----------
        logits : Tensor
            Raw (un-normalised) model outputs of shape ``(N, C)`` where
            *N* is the batch size and *C* is the number of classes.
        targets : Tensor
            Integer class labels of shape ``(N,)`` with values in
            ``[0, C)``.

        Returns
        -------
        Tensor
            Scalar loss (if ``reduction`` is ``"mean"`` or ``"sum"``)
            or per-sample scores of shape ``(N,)``.
        """
        num_classes: int = logits.size(-1)
        probs: Tensor = F.softmax(logits, dim=-1)
        one_hot: Tensor = F.one_hot(targets, num_classes=num_classes).float()

        # Per-sample Brier score: sum over classes of squared differences
        per_sample: Tensor = ((probs - one_hot) ** 2).sum(dim=-1)

        if self.reduction == "mean":
            return per_sample.mean()
        elif self.reduction == "sum":
            return per_sample.sum()
        return per_sample


class RLCDCompoundLoss(nn.Module):
    """Compound RLCD training loss.

    Combines standard cross-entropy with a Brier calibration penalty:

        L_total = L_CE + lambda_cal * L_Brier

    The cross-entropy drives discriminative performance while the Brier
    term acts as a calibration regulariser, encouraging the model to
    output probabilities that faithfully reflect empirical frequencies.

    Parameters
    ----------
    lambda_cal : float
        Weight for the Brier calibration term.  Higher values place
        more emphasis on calibration at the potential expense of raw
        discriminative accuracy.
    label_smoothing : float
        Label smoothing factor for the cross-entropy term.
    """

    def __init__(
        self,
        lambda_cal: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if lambda_cal < 0.0:
            raise ValueError(
                f"lambda_cal must be non-negative, got {lambda_cal}."
            )
        self.lambda_cal = lambda_cal
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.brier_loss = BrierScoreLoss(reduction="mean")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the compound loss.

        Parameters
        ----------
        logits : Tensor
            Raw model outputs of shape ``(N, C)``.
        targets : Tensor
            Integer class labels of shape ``(N,)``.

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            A 2-tuple of ``(total_loss, components)`` where *components*
            is a dict with keys ``"ce"``, ``"brier"``, and ``"total"``
            holding the individual (detached) scalars for logging.
        """
        ce: Tensor = self.ce_loss(logits, targets)
        brier: Tensor = self.brier_loss(logits, targets)
        total: Tensor = ce + self.lambda_cal * brier

        components: dict[str, Tensor] = {
            "ce": ce.detach(),
            "brier": brier.detach(),
            "total": total.detach(),
        }
        return total, components
