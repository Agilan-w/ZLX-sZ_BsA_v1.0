# Copyright 2026 Studio Zoul
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

"""Post-hoc temperature scaling module for calibrated decision heads.

Implements Temperature Scaling (Guo et al., 2017), a simple and effective
post-processing calibration method. It learns a single scalar parameter T > 0
on validation logits using L-BFGS to minimize Negative Log-Likelihood (NLL):

    p_i = softmax(z_i / T)

Temperature scaling does not alter the maximum of the softmax function,
meaning that Top-1 Accuracy and class predictions remain identical while
probabilities become significantly better calibrated.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelWithTemperature(nn.Module):
    """Wraps a classification model and applies learned temperature scaling.

    Parameters
    ----------
    model : nn.Module
        A trained PyTorch model producing raw classification logits.
    initial_temperature : float
        Initial value for temperature parameter (default: 1.5).
    """

    def __init__(
        self,
        model: nn.Module,
        initial_temperature: float = 1.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(
            torch.ones(1) * max(initial_temperature, 0.01)
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through base model followed by temperature scaling.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to base model.
        **kwargs : Any
            Keyword arguments passed to base model.

        Returns
        -------
        torch.Tensor
            Temperature-scaled logits.
        """
        outputs = self.model(*args, **kwargs)
        # Handle Hugging Face ModelOutput objects
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by current temperature parameter.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape (N, C).

        Returns
        -------
        torch.Tensor
            Scaled logits of shape (N, C).
        """
        # Ensure temperature is positive and broadcastable
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp

    @torch.no_grad()
    def collect_logits_and_labels(
        self,
        data_loader: DataLoader,
        device: Union[torch.device, str] = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract unscaled logits and ground truth labels from a DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            Validation or calibration data loader.
        device : torch.device | str
            Execution device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (all_logits, all_labels)
        """
        self.model.eval()
        device_obj = torch.device(device)

        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for batch in data_loader:
            input_dict = {
                k: v.to(device_obj)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            outputs = self.model(**input_dict)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            logits_list.append(logits.detach().cpu())
            labels_list.append(batch["labels"].detach().cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        return all_logits, all_labels

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Optimize temperature scalar on pre-computed logits and labels.

        Parameters
        ----------
        logits : torch.Tensor
            Validation logits of shape (N, C).
        labels : torch.Tensor
            Validation integer labels of shape (N,).
        max_iter : int
            Maximum L-BFGS iterations (default: 50).
        lr : float
            Learning rate for L-BFGS (default: 0.01).

        Returns
        -------
        float
            Learned optimal temperature scalar T.
        """
        nll_criterion = nn.CrossEntropyLoss()

        logits_device = logits.to(self.temperature.device)
        labels_device = labels.to(self.temperature.device)

        # Calculate initial NLL
        with torch.no_grad():
            initial_nll = nll_criterion(logits_device, labels_device).item()

        # Initialize L-BFGS optimizer specifically for single scalar T
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _eval_step() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(logits_device)
            loss = nll_criterion(scaled_logits, labels_device)
            loss.backward()
            return loss

        optimizer.step(_eval_step)

        # Ensure temperature remains strictly positive
        with torch.no_grad():
            self.temperature.data.clamp_(min=1e-3)
            final_nll = nll_criterion(
                self.temperature_scale(logits_device), labels_device
            ).item()

        optimal_t = float(self.temperature.item())
        logger.info(
            "Temperature scaling complete -- Optimal T: %.4f | "
            "NLL: %.4f -> %.4f (improvement: %+.4f)",
            optimal_t,
            initial_nll,
            final_nll,
            final_nll - initial_nll,
        )
        return optimal_t

    def set_temperature(
        self,
        valid_loader: DataLoader,
        device: Union[torch.device, str] = "cpu",
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Convenience method to extract validation logits and fit temperature.

        Parameters
        ----------
        valid_loader : DataLoader
            Validation DataLoader.
        device : torch.device | str
            Execution device.
        max_iter : int
            Maximum optimization iterations.
        lr : float
            Optimizer learning rate.

        Returns
        -------
        float
            Optimal temperature T.
        """
        logger.info("Extracting validation logits for temperature scaling...")
        logits, labels = self.collect_logits_and_labels(
            valid_loader, device=device
        )
        return self.fit(logits, labels, max_iter=max_iter, lr=lr)
