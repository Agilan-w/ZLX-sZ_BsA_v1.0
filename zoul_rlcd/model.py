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
#
# MODIFICATION NOTICE: This file has been modified from the original Laya
# implementation on 2026-10-12 to introduce RLCD
# fine-tuning capabilities for calibrated decision heads and native
# Flash Attention 2 support for full-parameter 32GB VRAM training.

"""Model builder for the ZLX-CDE V2.5 calibrated decision pipeline.

Loads a pre-trained ModernBERT-based encoder (default: Laya) via
Hugging Face ``AutoModelForSequenceClassification`` with native
Flash Attention 2 support (``attn_implementation="flash_attention_2"``)
and configures it for high-throughput RLCD fine-tuning.
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
    gradient_checkpointing: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    torch_dtype: Optional[torch.dtype] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> AutoModelForSequenceClassification:
    """Build a calibrated decision model from a pre-trained encoder.

    Loads the encoder via Hugging Face with native Flash Attention 2
    acceleration, configures classification head parameters, and allows
    toggling backbone freezing and gradient checkpointing.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face model identifier or local path. Defaults to the
        Laya English checkpoint (``convaiinnovations/laya``).
    num_classes : int
        Number of output classes for the decision head.
    freeze_backbone : bool
        If ``True``, freeze all encoder parameters and train only the
        classification head. Default is ``False`` (full-parameter training).
    gradient_checkpointing : bool
        If ``True``, enable gradient checkpointing to conserve VRAM at
        the cost of compute throughput. Default is ``False`` (disabled for
        accelerated training on 32GB VRAM hardware such as RTX 5090).
    attn_implementation : str | None
        Attention mechanism backend. Defaults to ``"flash_attention_2"``.
        Automatically falls back to default attention if unsupported by
        the model architecture or execution hardware.
    torch_dtype : torch.dtype | None
        Data type for the model weights. Defaults to ``torch.float32``
        (master weights in full precision; AMP handles forward casting).
    id2label : dict[int, str] | None
        Optional class index to label string mapping.
    label2id : dict[str, int] | None
        Optional label string to class index mapping.

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
        "Loading model '%s' with %d classes (dtype=%s, attn_implementation=%s).",
        model_name_or_path,
        num_classes,
        effective_dtype,
        attn_implementation,
    )

    try:
        pretrained_kwargs: dict = {
            "num_labels": num_classes,
            "torch_dtype": effective_dtype,
            "ignore_mismatched_sizes": True,
        }
        if attn_implementation is not None:
            pretrained_kwargs["attn_implementation"] = attn_implementation
        if id2label is not None:
            pretrained_kwargs["id2label"] = id2label
        if label2id is not None:
            pretrained_kwargs["label2id"] = label2id

        try:
            model: AutoModelForSequenceClassification = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    **pretrained_kwargs,
                )
            )
        except (ValueError, KeyError, ImportError) as err:
            # Fall back gracefully if flash_attention_2 is not supported on this model or device
            if attn_implementation == "flash_attention_2":
                logger.warning(
                    "Flash Attention 2 requested but not supported by '%s' or environment (%s). "
                    "Falling back to default attention implementation.",
                    model_name_or_path,
                    err,
                )
                pretrained_kwargs.pop("attn_implementation", None)
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name_or_path,
                        **pretrained_kwargs,
                    )
                except Exception as inner_err:
                    if "laya" in str(inner_err).lower() or "laya" in model_name_or_path.lower():
                        logger.warning(
                            "Model '%s' uses proprietary Laya architecture. "
                            "Falling back to ModernBERT base encoder ('answerdotai/ModernBERT-base').",
                            model_name_or_path,
                        )
                        model = AutoModelForSequenceClassification.from_pretrained(
                            "answerdotai/ModernBERT-base",
                            **pretrained_kwargs,
                        )
                    else:
                        raise
            elif "laya" in str(err).lower() or "laya" in model_name_or_path.lower():
                logger.warning(
                    "Model '%s' uses proprietary Laya architecture. "
                    "Falling back to ModernBERT base encoder ('answerdotai/ModernBERT-base').",
                    model_name_or_path,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    "answerdotai/ModernBERT-base",
                    **pretrained_kwargs,
                )
            else:
                raise

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
    #  Backbone freezing / full-parameter configuration                  #
    # ------------------------------------------------------------------ #
    if freeze_backbone:
        frozen_count: int = 0
        for name, param in model.named_parameters():
            if not any(k in name for k in ("classifier", "cls", "score", "head")):
                param.requires_grad = False
                frozen_count += 1
        logger.info(
            "Backbone frozen: %d parameter groups set to requires_grad=False.",
            frozen_count,
        )
    else:
        # Full fine-tuning: ensure all model parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Full-parameter training mode active (backbone unfrozen).")

    # ------------------------------------------------------------------ #
    #  Gradient checkpointing toggle                                     #
    # ------------------------------------------------------------------ #
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        else:
            logger.warning(
                "Model does not support gradient_checkpointing_enable()."
            )
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        logger.info(
            "Gradient checkpointing disabled for maximum training throughput (32GB VRAM)."
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
