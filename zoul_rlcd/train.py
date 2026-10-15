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
# fine-tuning capabilities for calibrated decision heads, differential
# learning rates, native Weights & Biases (wandb) logging, and high-throughput
# 32GB VRAM optimization (RTX 5090).

"""RLCD training orchestrator for the ZLX-CDE V2.5 Decision Engine.

Manages the full training and validation lifecycle optimized for full-parameter
training on 32GB VRAM hardware (such as RTX 5090):

- Native Flash Attention 2 acceleration
- Differential learning rates (AdamW: 1e-5 backbone, 1e-4 classification head)
- Gradient checkpointing disabled by default for maximum throughput
- Batch size 32 with max sequence length 1024
- Weights & Biases (wandb) experiment tracking (Train Loss, Val Loss, Acc, ECE)
- PyTorch AMP (BF16 auto-detected on modern hardware)
- Guarded checkpoint saving on ECE improvement and accuracy stability

Usage::

    python -m zoul_rlcd.train \
        --data_path data/decisions.csv \
        --batch_size 32 \
        --max_length 1024 \
        --use_wandb
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
#  CLI Argument Parser                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="zoul_rlcd.train",
        description=(
            "ZLX-CDE V2.5 Decision Engine -- RLCD fine-tuning pipeline for "
            "calibrated decision heads on top of the Laya model (RTX 5090 optimized)."
        ),
    )

    # Model configuration
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
        help="Freeze encoder weights; train head only (default: False for full-parameter training).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (default: False; disabled for maximum 32GB throughput).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (default: flash_attention_2).",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Micro-batch size per forward pass (default: 32).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1; effective batch = batch_size * grad_accum_steps).",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-5,
        help="Peak learning rate for the ModernBERT encoder backbone (default: 1e-5).",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-4,
        help="Peak learning rate for the classification head (default: 1e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01).",
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

    # Data parameters
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
        default=1024,
        help="Maximum sequence length threshold (default: 1024).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )

    # Weights & Biases (wandb) experiment tracking
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="zlx-cde-v2.5",
        help="Weights & Biases project name (default: zlx-cde-v2.5).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional run name for Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional entity/team for Weights & Biases.",
    )

    # Output & Checkpoints
    parser.add_argument(
        "--output_dir",
        type=str,
        default="best_calibrated_model",
        help="Directory to save the best checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--acc_degradation_tol",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed drop in validation accuracy relative to "
            "the best seen when saving a checkpoint based on ECE improvement."
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

    logger.info("Dataset split -- Train: %d | Val: %d", train_size, val_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
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
    #  Weights & Biases (wandb) initialization                           #
    # ------------------------------------------------------------------ #
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
            logger.info("Weights & Biases initialized successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to initialize Weights & Biases: %s. Continuing without wandb.",
                exc,
            )
            wandb_run = None

    # ------------------------------------------------------------------ #
    #  Tokenizer (instantiated early so save_pretrained works later)     #
    # ------------------------------------------------------------------ #
    logger.info("Loading tokenizer from '%s'.", args.model_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception as exc:
        if "laya" in args.model_name_or_path.lower():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name_or_path, subfolder="tokenizer"
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        else:
            raise

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
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # ------------------------------------------------------------------ #
    #  Loss & Optimizer (Differential Learning Rates)                    #
    # ------------------------------------------------------------------ #
    criterion = RLCDCompoundLoss(lambda_cal=args.lambda_cal)

    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(head_key in name for head_key in ("classifier", "cls", "score", "head")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_grouped_parameters = []
    if backbone_params:
        optimizer_grouped_parameters.append(
            {
                "params": backbone_params,
                "lr": args.lr_backbone,
                "weight_decay": args.weight_decay,
            }
        )
    if head_params:
        optimizer_grouped_parameters.append(
            {
                "params": head_params,
                "lr": args.lr_head,
                "weight_decay": args.weight_decay,
            }
        )

    optimizer = AdamW(optimizer_grouped_parameters)

    logger.info(
        "Differential LR configured -- Backbone: %d params (lr=%.2e) | "
        "Head: %d params (lr=%.2e)",
        sum(p.numel() for p in backbone_params),
        args.lr_backbone,
        sum(p.numel() for p in head_params),
        args.lr_head,
    )

    amp_dtype: torch.dtype = _select_amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ------------------------------------------------------------------ #
    #  Checkpoint guard state                                            #
    # ------------------------------------------------------------------ #
    best_ece: float = float("inf")
    best_accuracy: float = 0.0
    effective_batch: int = args.batch_size * args.grad_accum_steps

    logger.info(
        "Training config -- Epochs: %d | Micro-batch: %d | "
        "Grad accum: %d | Effective batch: %d | "
        "LR Backbone: %.2e | LR Head: %.2e | Lambda_cal: %s | "
        "Gradient Checkpointing: %s | AMP: %s | Wandb: %s",
        args.epochs,
        args.batch_size,
        args.grad_accum_steps,
        effective_batch,
        args.lr_backbone,
        args.lr_head,
        args.lambda_cal,
        args.gradient_checkpointing,
        amp_dtype,
        args.use_wandb,
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

        # Handle remaining gradient accumulation step
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
            "[Epoch %d/%d] Train loss: %.4f (over %d optimizer steps)",
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
        val_loss_total: float = 0.0

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
                    v_loss, v_comp = criterion(outputs.logits, batch["labels"])
                    val_loss_total += v_comp["total"].item()

                # Cast logits to float32 for stable metric computation
                tracker.update(
                    outputs.logits.float(), batch["labels"]
                )

        avg_val_loss: float = (
            val_loss_total / len(val_loader) if len(val_loader) > 0 else 0.0
        )
        metrics: dict[str, float] = tracker.log_summary(epoch)
        logger.info(
            "[Epoch %d/%d] Validation -- Loss: %.4f | Acc: %.4f | F1: %.4f | "
            "Brier: %.4f | ECE: %.4f",
            epoch,
            args.epochs,
            avg_val_loss,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["brier_score"],
            metrics["ece"],
        )

        vram_alloc_mb = (
            torch.cuda.memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        vram_res_mb = (
            torch.cuda.memory_reserved() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        vram_max_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        logger.info(
            "[Epoch %d/%d] VRAM Usage -- Allocated: %.1f MB | Reserved: %.1f MB | Peak: %.1f MB",
            epoch,
            args.epochs,
            vram_alloc_mb,
            vram_res_mb,
            vram_max_mb,
        )

        # -------------------------------------------------------------- #
        #  Weights & Biases (wandb) logging                              #
        # -------------------------------------------------------------- #
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "brier_score": metrics["brier_score"],
                    "ece": metrics["ece"],
                    "best_ece": min(best_ece, metrics["ece"]),
                    "vram_allocated_mb": vram_alloc_mb,
                    "vram_reserved_mb": vram_res_mb,
                    "vram_peak_mb": vram_max_mb,
                },
                step=epoch,
            )

        # -------------------------------------------------------------- #
        #  Guarded checkpoint saving                                     #
        # -------------------------------------------------------------- #
        current_ece: float = metrics["ece"]
        current_acc: float = metrics["accuracy"]

        best_accuracy = max(best_accuracy, current_acc)
        ece_improved: bool = current_ece < best_ece
        acc_acceptable: bool = (
            current_acc >= best_accuracy - args.acc_degradation_tol
        )

        if ece_improved and acc_acceptable:
            best_ece = current_ece
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

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

    if wandb_run is not None:
        wandb_run.finish()

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
