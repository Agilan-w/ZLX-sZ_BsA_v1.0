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

"""Evaluation and reliability diagram generator for calibrated decisions.

Evaluates a trained decision checkpoint against a validation dataset:
- Computes Top-1 Accuracy, Macro F1, Brier Score, and Expected Calibration Error (ECE)
- Learns post-hoc optimal temperature scaling via L-BFGS
- Plots a publication-quality Reliability Diagram (Confidence vs. Empirical Accuracy)
- Outputs a comprehensive calibration diagnostic report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from zoul_rlcd.data_loader import load_decision_dataset
from zoul_rlcd.dataset import SyntheticDecisionDataset
from zoul_rlcd.temperature_scaling import ModelWithTemperature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zoul_rlcd.evaluate")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation and visualization."""
    parser = argparse.ArgumentParser(
        prog="zoul_rlcd.evaluate",
        description=(
            "Zoul AI Decision Engine -- Checkpoint evaluation, post-hoc "
            "temperature scaling, and reliability diagram generator."
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_calibrated_model",
        help="Path to trained checkpoint directory (default: best_calibrated_model).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to evaluation dataset (.csv, .jsonl, .json, .parquet) or "
            "HF Hub ID. If None, synthetic data is evaluated."
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
        help="Optional sub-configuration name for HF Hub datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length threshold (default: 128).",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=15,
        help="Number of confidence bins for ECE and reliability diagrams (default: 15).",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="reliability_diagram.png",
        help="Output path for reliability diagram plot (default: reliability_diagram.png).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split fraction if unpartitioned (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)


def compute_calibration_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Calculate bin statistics and ECE for reliability diagram plotting.

    Parameters
    ----------
    probs : torch.Tensor
        Predicted softmax probabilities of shape (N, C).
    labels : torch.Tensor
        Integer true labels of shape (N,).
    n_bins : int
        Number of equal-width confidence intervals.

    Returns
    -------
    dict[str, Any]
        Dictionary with bin edges, accuracies, confidences, sample counts, and ECE.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_sizes: list[int] = []

    ece = 0.0

    for lower, upper in zip(bin_lowers, bin_uppers):
        # Samples falling into [lower, upper)
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop_in_bin = in_bin.float().mean().item()
        bin_count = in_bin.sum().item()
        bin_sizes.append(bin_count)

        if bin_count > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower.item() + upper.item()) / 2.0)

    return {
        "bin_edges": bin_boundaries.cpu().numpy(),
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_sizes": np.array(bin_sizes),
        "ece": float(ece),
    }


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics from logits and labels."""
    probs = F.softmax(logits, dim=-1)
    num_classes = logits.size(-1)

    # 1. Top-1 Accuracy
    preds = torch.argmax(probs, dim=-1)
    acc = (preds == labels).float().mean().item()

    # 2. Macro F1
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")
    macro_f1 = f1_metric(preds, labels).item()

    # 3. Brier Score
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

    # 4. NLL
    nll = nn.CrossEntropyLoss()(logits, labels).item()

    # 5. Calibration curve & ECE
    curve_data = compute_calibration_curve(probs, labels, n_bins=n_bins)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "brier_score": brier,
        "nll": nll,
        "ece": curve_data["ece"],
        "curve_data": curve_data,
    }


def plot_reliability_diagrams(
    pre_curve: dict[str, Any],
    post_curve: dict[str, Any],
    pre_metrics: dict[str, Any],
    post_metrics: dict[str, Any],
    optimal_t: float,
    output_path: str | Path,
    n_bins: int = 15,
) -> None:
    """Generate and save publication-quality reliability diagrams."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    bin_width = 1.0 / n_bins
    bin_centers = np.linspace(bin_width / 2, 1.0 - bin_width / 2, n_bins)

    plots_config = [
        {
            "ax": axes[0],
            "title": "Uncalibrated Baseline (T = 1.000)",
            "curve": pre_curve,
            "metrics": pre_metrics,
            "bar_color": "#2563EB",
            "gap_color": "#EF4444",
        },
        {
            "ax": axes[1],
            "title": f"Post-Hoc Temperature Scaled (T = {optimal_t:.3f})",
            "curve": post_curve,
            "metrics": post_metrics,
            "bar_color": "#059669",
            "gap_color": "#F59E0B",
        },
    ]

    for p in plots_config:
        ax = p["ax"]
        curve = p["curve"]
        metrics = p["metrics"]

        accs = curve["bin_accuracies"]
        confs = curve["bin_confidences"]
        sizes = curve["bin_sizes"]

        # 1. Diagonal perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=2, label="Perfect Calibration (y = x)")

        # 2. Empirical accuracy bars
        ax.bar(
            bin_centers,
            accs,
            width=bin_width * 0.85,
            color=p["bar_color"],
            alpha=0.75,
            edgecolor=p["bar_color"],
            linewidth=1.2,
            label="Empirical Accuracy",
        )

        # 3. Highlight calibration gap
        gaps = np.abs(accs - confs)
        # Only draw gap where there are samples
        gap_heights = np.where(sizes > 0, gaps, 0.0)
        gap_bottoms = np.minimum(accs, confs)
        ax.bar(
            bin_centers,
            gap_heights,
            bottom=gap_bottoms,
            width=bin_width * 0.85,
            color=p["gap_color"],
            alpha=0.35,
            hatch="//",
            edgecolor=p["gap_color"],
            linewidth=0.8,
            label="Calibration Gap (ECE)",
        )

        # Labels and limits
        ax.set_title(p["title"], fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Confidence Bin", fontsize=11, fontweight="medium")
        ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="medium")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))

        # Metrics statistics box
        stats_text = (
            f"ECE: {metrics['ece']:.4f}\n"
            f"Brier: {metrics['brier_score']:.4f}\n"
            f"NLL: {metrics['nll']:.4f}\n"
            f"Acc: {metrics['accuracy']:.4f}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
        )
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="#E2E8F0")

    plt.suptitle("Zoul AI Decision Engine -- Calibration Reliability Analysis", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved reliability diagram to '%s'.", out_file)


def print_comparison_table(
    pre_metrics: dict[str, Any],
    post_metrics: dict[str, Any],
    optimal_t: float,
) -> None:
    """Print an ASCII comparison table of pre- vs post-temperature scaling metrics."""
    ece_diff = post_metrics["ece"] - pre_metrics["ece"]
    brier_diff = post_metrics["brier_score"] - pre_metrics["brier_score"]
    nll_diff = post_metrics["nll"] - pre_metrics["nll"]

    print("\n" + "=" * 76)
    print("           ZOUL AI DECISION ENGINE -- CALIBRATION EVALUATION REPORT")
    print("=" * 76)
    print(f" {'Metric':<28} | {'Before Scaling':<18} | {'After Scaling (T=' + f'{optimal_t:.3f})':<22}")
    print("-" * 76)
    print(f" {'Top-1 Accuracy':<28} | {pre_metrics['accuracy']:<18.4f} | {post_metrics['accuracy']:<18.4f} (Exact)")
    print(f" {'Macro F1 Score':<28} | {pre_metrics['macro_f1']:<18.4f} | {post_metrics['macro_f1']:<18.4f} (Exact)")
    print(f" {'Expected Calib Error (ECE)':<28} | {pre_metrics['ece']:<18.4f} | {post_metrics['ece']:<18.4f} ({ece_diff:+.4f})")
    print(f" {'Brier Score':<28} | {pre_metrics['brier_score']:<18.4f} | {post_metrics['brier_score']:<18.4f} ({brier_diff:+.4f})")
    print(f" {'Negative Log-Likelihood':<28} | {pre_metrics['nll']:<18.4f} | {post_metrics['nll']:<18.4f} ({nll_diff:+.4f})")
    print("=" * 76 + "\n")


def evaluate(args: argparse.Namespace) -> None:
    """Run full evaluation lifecycle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    # 1. Load Tokenizer & Model
    logger.info("Loading checkpoint from '%s'...", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # 2. Load Evaluation Data
    if args.data_path:
        logger.info("Loading evaluation dataset from '%s'...", args.data_path)
        _, val_loader, id2label, _, _ = load_decision_dataset(
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
    else:
        logger.warning("No --data_path provided. Evaluating on synthetic decision dataset.")
        num_classes = getattr(model.config, "num_labels", 5)
        vocab_size = getattr(model.config, "vocab_size", getattr(tokenizer, "vocab_size", 30522))
        syn_ds = SyntheticDecisionDataset(
            num_samples=200,
            num_classes=num_classes,
            max_length=args.max_length,
            vocab_size=vocab_size,
            seed=args.seed,
        )
        val_loader = torch.utils.data.DataLoader(
            syn_ds, batch_size=args.batch_size, shuffle=False
        )

    # 3. Collect Raw Validation Logits & Labels
    logger.info("Collecting model logits across evaluation dataset...")
    temp_wrapper = ModelWithTemperature(model)
    raw_logits, true_labels = temp_wrapper.collect_logits_and_labels(
        val_loader, device=device
    )

    # 4. Compute Baseline (Pre-Scaling) Metrics
    logger.info("Computing pre-scaling calibration metrics (n_bins=%d)...", args.n_bins)
    pre_metrics = compute_metrics(raw_logits, true_labels, n_bins=args.n_bins)

    # 5. Fit Optimal Temperature Scaling via L-BFGS
    logger.info("Optimizing post-hoc temperature scalar T...")
    optimal_t = temp_wrapper.fit(raw_logits, true_labels)

    # 6. Compute Post-Scaling Metrics
    scaled_logits = temp_wrapper.temperature_scale(raw_logits)
    post_metrics = compute_metrics(scaled_logits, true_labels, n_bins=args.n_bins)

    # 7. Print Terminal Diagnostic Report
    print_comparison_table(pre_metrics, post_metrics, optimal_t)

    # 8. Generate & Save Publication Reliability Diagram
    logger.info("Rendering publication-quality reliability diagrams...")
    plot_reliability_diagrams(
        pre_curve=pre_metrics["curve_data"],
        post_curve=post_metrics["curve_data"],
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        optimal_t=optimal_t,
        output_path=args.output_plot,
        n_bins=args.n_bins,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
