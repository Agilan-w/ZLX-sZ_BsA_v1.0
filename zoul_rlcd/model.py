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
#
# MODIFICATION NOTICE: This file has been modified from the original Laya
# implementation by Studio Zoul on 2026-10-12 to introduce RLCD
# fine-tuning capabilities for calibrated decision heads.

"""Model builder for the Zoul AI calibrated decision pipeline.

Loads a pre-trained ModernBERT-based encoder (default: Laya) via
Hugging Face ``AutoModelForSequenceClassification`` and configures it
for RLCD fine-tuning with optional backbone freezing and gradient
checkpointing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


def build_calibrated_decision_model(
    model_name_or_path: str = "convaiinnovations/laya",
    num_classes: int = 5,
    freeze_backbone: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> AutoModelForSequenceClassification:
    """Build a calibrated decision model from a pre-trained encoder.

    Loads the encoder via Hugging Face, optionally freezes the backbone
    parameters (training only the classification head), and enables
    gradient checkpointing when the full model is being fine-tuned to
    reduce peak VRAM consumption on memory-constrained GPUs.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face model identifier or local path.  Defaults to the
        Laya English checkpoint (``convaiinnovations/laya``).
    num_classes : int
        Number of output classes for the decision head.
    freeze_backbone : bool
        If ``True``, freeze all encoder parameters and train only the
        classification head.  When ``False``, gradient checkpointing is
        enabled to keep memory usage within 8 GB VRAM.
    torch_dtype : torch.dtype | None
        Data type for the model weights.  If ``None``, defaults to
        ``torch.float32`` (master weights are kept in full precision;
        AMP handles the forward-pass casting).

    Returns
    -------
    AutoModelForSequenceClassification
        The configured model ready for RLCD fine-tuning.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded from ``model_name_or_path``.
    """
    effective_dtype = torch_dtype or torch.float32

    logger.info(
        "Loading model '%s' with %d classes (dtype=%s).",
        model_name_or_path,
        num_classes,
        effective_dtype,
    )

    try:
        pretrained_kwargs: dict = {
            "num_labels": num_classes,
            "torch_dtype": effective_dtype,
            "ignore_mismatched_sizes": True,
        }
        if id2label is not None:
            pretrained_kwargs["id2label"] = id2label
        if label2id is not None:
            pretrained_kwargs["label2id"] = label2id

        model: AutoModelForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                **pretrained_kwargs,
            )
        )

        # Ensure label mapping is explicitly stored on the config so it is
        # serialized into config.json upon save_pretrained()
        if id2label is not None:
            model.config.id2label = id2label
        if label2id is not None:
            model.config.label2id = label2id
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{model_name_or_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------ #
    #  Backbone freezing / gradient checkpointing                        #
    # ------------------------------------------------------------------ #
    if freeze_backbone:
        frozen_count: int = 0
        for name, param in model.named_parameters():
            if "classifier" not in name and "cls" not in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(
            "Backbone frozen: %d parameter groups set to requires_grad=False.",
            frozen_count,
        )
    else:
        # Enable gradient checkpointing to trade compute for memory —
        # critical for fitting a ~421M param model within 8 GB VRAM
        # during full fine-tuning.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (full fine-tuning).")
        else:
            logger.warning(
                "Model does not support gradient_checkpointing_enable(). "
                "Training may require more VRAM than expected."
            )

    # ------------------------------------------------------------------ #
    #  Parameter count summary                                           #
    # ------------------------------------------------------------------ #
    trainable: int = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total: int = sum(p.numel() for p in model.parameters())
    frozen: int = total - trainable

    logger.info(
        "Parameter summary -- Total: %s | Trainable: %s (%.1f%%) | "
        "Frozen: %s (%.1f%%)",
        f"{total:,}",
        f"{trainable:,}",
        100.0 * trainable / total if total > 0 else 0.0,
        f"{frozen:,}",
        100.0 * frozen / total if total > 0 else 0.0,
    )

    return model
