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

"""RLCD training orchestrator for the Zoul AI Decision Engine.

Manages the full training and validation lifecycle with:

- PyTorch AMP (BF16 where supported, FP16 fallback) + GradScaler
- Gradient accumulation for effective large-batch training on 8 GB VRAM
- Gradient checkpointing during full fine-tuning
- Best-checkpoint saving guarded by ECE improvement **and** accuracy
  stability (prevents saving a well-calibrated but degenerate model)

Usage::

    python -m zoul_rlcd.train --epochs 10 --batch_size 4 --lambda_cal 0.5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from zoul_rlcd.data_loader import load_decision_dataset
from zoul_rlcd.dataset import SyntheticDecisionDataset
from zoul_rlcd.metrics import CalibrationTracker
from zoul_rlcd.model import build_calibrated_decision_model
from zoul_rlcd.rlcd_loss import RLCDCompoundLoss

# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zoul_rlcd.train")


# --------------------------------------------------------------------------- #
#  AMP dtype selection                                                         #
# --------------------------------------------------------------------------- #
def _select_amp_dtype() -> torch.dtype:
    """Return BF16 if supported by the current GPU, else FP16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("AMP dtype: torch.bfloat16 (hardware supported).")
        return torch.bfloat16
    logger.info("AMP dtype: torch.float16 (BF16 not available).")
    return torch.float16


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="zoul_rlcd.train",
        description=(
            "Zoul AI Decision Engine -- RLCD fine-tuning pipeline for "
            "calibrated decision heads on top of the Laya model."
        ),
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="convaiinnovations/laya",
        help="HuggingFace model ID or local path (default: convaiinnovations/laya).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of output classes for the decision head.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze encoder weights; train only the classification head.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Micro-batch size per forward pass (default: 4).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * this).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--lambda_cal",
        type=float,
        default=0.5,
        help="Weight for the Brier calibration term in the compound loss.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to local dataset file (.csv, .jsonl, .json, .parquet), "
            "directory, or Hugging Face Hub dataset ID. If not provided, "
            "synthetic data is generated."
        ),
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column containing input text (default: 'text').",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Column containing target labels (default: 'label').",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional sub-configuration name for Hugging Face Hub datasets.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate when --data_path is None.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length threshold (default: 128).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="best_calibrated_model",
        help="Directory to save the best checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoint guard
    parser.add_argument(
        "--acc_degradation_tol",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed drop in validation accuracy (relative to "
            "the best seen) when saving a checkpoint based on ECE "
            "improvement.  Set to 0.0 for strict no-degradation."
        ),
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Data helpers                                                                #
# --------------------------------------------------------------------------- #
def _build_dataloaders(
    args: argparse.Namespace,
    vocab_size: int = 50265,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from synthetic data."""
    full_dataset = SyntheticDecisionDataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        max_length=args.max_length,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    val_size: int = max(1, int(len(full_dataset) * args.val_split))
    train_size: int = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    logger.info(
        "Dataset split -- Train: %d | Val: %d", train_size, val_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,  # No gradients → can double
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader


# --------------------------------------------------------------------------- #
#  Training loop                                                               #
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> None:
    """Execute the full RLCD training and validation lifecycle."""

    # ------------------------------------------------------------------ #
    #  Device & reproducibility                                          #
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training will run on CPU (very slow)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    #  Tokenizer (instantiated early so save_pretrained works later)     #
    # ------------------------------------------------------------------ #
    logger.info("Loading tokenizer from '%s'.", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ------------------------------------------------------------------ #
    #  Data                                                              #
    # ------------------------------------------------------------------ #
    if args.data_path:
        logger.info("Loading real dataset from '%s'...", args.data_path)
        (
            train_loader,
            val_loader,
            id2label,
            label2id,
            num_classes,
        ) = load_decision_dataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            text_column=args.text_column,
            label_column=args.label_column,
            batch_size=args.batch_size,
            max_length=args.max_length,
            val_split=args.val_split,
            seed=args.seed,
            dataset_config=args.dataset_config,
        )
        args.num_classes = num_classes
    else:
        logger.info(
            "Generating %d synthetic samples (%d classes, seq_len=%d).",
            args.num_samples,
            args.num_classes,
            args.max_length,
        )
        tok_vocab = getattr(tokenizer, "vocab_size", 50265)
        train_loader, val_loader = _build_dataloaders(args, vocab_size=tok_vocab)
        id2label = {i: f"class_{i}" for i in range(args.num_classes)}
        label2id = {f"class_{i}": i for i in range(args.num_classes)}

    # ------------------------------------------------------------------ #
    #  Model                                                             #
    # ------------------------------------------------------------------ #
    model = build_calibrated_decision_model(
        model_name_or_path=args.model_name_or_path,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # ------------------------------------------------------------------ #
    #  Loss, optimiser, scaler                                           #
    # ------------------------------------------------------------------ #
    criterion = RLCDCompoundLoss(lambda_cal=args.lambda_cal)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    amp_dtype: torch.dtype = _select_amp_dtype()
    # GradScaler is only needed for float16; bfloat16 doesn't require it
    # but it is harmless and simplifies the code path.
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ------------------------------------------------------------------ #
    #  Checkpoint guard state                                            #
    # ------------------------------------------------------------------ #
    best_ece: float = float("inf")
    best_accuracy: float = 0.0  # Track best accuracy seen so far
    effective_batch: int = args.batch_size * args.grad_accum_steps

    logger.info(
        "Training config -- Epochs: %d | Micro-batch: %d | "
        "Grad accum: %d | Effective batch: %d | LR: %s | "
        "Lambda_cal: %s | AMP dtype: %s",
        args.epochs,
        args.batch_size,
        args.grad_accum_steps,
        effective_batch,
        args.lr,
        args.lambda_cal,
        amp_dtype,
    )

    # ------------------------------------------------------------------ #
    #  Training loop                                                     #
    # ------------------------------------------------------------------ #
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        epoch_steps: int = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            # Move batch to device
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=torch.cuda.is_available(),
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss, components = criterion(outputs.logits, batch["labels"])
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                epoch_steps += 1

            epoch_loss += components["total"].item()

        # Handle remaining gradients if the last accumulation window
        # is incomplete
        if len(train_loader) % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_steps += 1

        avg_train_loss: float = (
            epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )
        logger.info(
            "[Epoch %d/%d] Train loss: %.4f (over %d optimiser steps)",
            epoch,
            args.epochs,
            avg_train_loss,
            epoch_steps,
        )

        # -------------------------------------------------------------- #
        #  Validation                                                    #
        # -------------------------------------------------------------- #
        model.eval()
        tracker = CalibrationTracker(
            num_classes=args.num_classes, device=str(device)
        )

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=torch.cuda.is_available(),
                ):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                # Cast logits to float32 for stable metric computation
                tracker.update(
                    outputs.logits.float(), batch["labels"]
                )

        metrics: dict[str, float] = tracker.log_summary(epoch)

        # -------------------------------------------------------------- #
        #  Guarded checkpoint saving                                     #
        #                                                                #
        #  Save only when ECE improves AND validation accuracy has not   #
        #  degraded beyond the tolerance threshold.  This prevents       #
        #  saving a model that is "well-calibrated" only because it      #
        #  has collapsed to near-uniform predictions.                    #
        # -------------------------------------------------------------- #
        current_ece: float = metrics["ece"]
        current_acc: float = metrics["accuracy"]

        # Update the best accuracy watermark
        best_accuracy = max(best_accuracy, current_acc)

        ece_improved: bool = current_ece < best_ece
        acc_acceptable: bool = (
            current_acc >= best_accuracy - args.acc_degradation_tol
        )

        if ece_improved and acc_acceptable:
            best_ece = current_ece
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Ensure label mappings are stored in the configuration before saving
            model.config.id2label = id2label
            model.config.label2id = label2id

            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            logger.info(
                "[Epoch %d] [OK] Checkpoint saved to '%s' -- "
                "ECE: %.4f (best) | Acc: %.4f",
                epoch,
                output_path,
                best_ece,
                current_acc,
            )
        elif ece_improved and not acc_acceptable:
            logger.warning(
                "[Epoch %d] ECE improved (%.4f -> %.4f) but accuracy "
                "degraded beyond tolerance (%.4f < %.4f - %.4f). "
                "Checkpoint NOT saved.",
                epoch,
                best_ece,
                current_ece,
                current_acc,
                best_accuracy,
                args.acc_degradation_tol,
            )
        else:
            logger.info(
                "[Epoch %d] No ECE improvement (current: %.4f, "
                "best: %.4f). Checkpoint not saved.",
                epoch,
                current_ece,
                best_ece,
            )

        tracker.reset()

    logger.info("Training complete. Best ECE: %.4f", best_ece)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
