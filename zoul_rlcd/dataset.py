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

"""Synthetic smoke-test dataset for the RLCD pipeline.

Generates mock tokenised inputs (input IDs, attention masks, and class
labels) so the full training loop can be validated end-to-end without
downloading real data.  The data is intentionally random and carries no
semantic signal — it exists solely for integration testing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticDecisionDataset(Dataset[dict[str, Tensor]]):
    """Random synthetic dataset mimicking tokenised NLP inputs.

    Each sample is a dictionary with:

    - ``input_ids``:      ``(max_length,)``  — random token IDs
    - ``attention_mask``:  ``(max_length,)``  — all ones (no padding)
    - ``labels``:          scalar             — random class label

    Parameters
    ----------
    num_samples : int
        Total number of samples to generate.
    num_classes : int
        Number of distinct target classes.
    max_length : int
        Sequence length for input IDs and attention masks.
    vocab_size : int
        Upper bound for random token IDs.  Defaults to ``50_265``
        (ModernBERT / RoBERTa default vocabulary).
    seed : int | None
        Random seed for reproducibility.  Pass ``None`` for
        non-deterministic generation.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 5,
        max_length: int = 128,
        vocab_size: int = 50_265,
        seed: Optional[int] = 42,
    ) -> None:
        if num_samples < 1:
            raise ValueError(
                f"num_samples must be >= 1, got {num_samples}."
            )
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )
        if max_length < 1:
            raise ValueError(
                f"max_length must be >= 1, got {max_length}."
            )

        super().__init__()

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        # Pre-generate all data in one shot for speed and reproducibility
        self.input_ids: Tensor = torch.randint(
            low=1,
            high=vocab_size,
            size=(num_samples, max_length),
            generator=generator,
        )
        self.attention_mask: Tensor = torch.ones(
            num_samples, max_length, dtype=torch.long
        )
        self.labels: Tensor = torch.randint(
            low=0,
            high=num_classes,
            size=(num_samples,),
            generator=generator,
        )

        logger.info(
            "SyntheticDecisionDataset created -- %d samples, %d classes, "
            "seq_len=%d, vocab=%d.",
            num_samples,
            num_classes,
            max_length,
            vocab_size,
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.labels.size(0)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a single tokenised sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``"input_ids"``, ``"attention_mask"``, ``"labels"``.
        """
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }
