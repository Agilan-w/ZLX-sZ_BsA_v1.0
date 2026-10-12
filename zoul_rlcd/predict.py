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
# MODIFICATION NOTICE: This file has been created by Studio Zoul on 2026-10-12
# to serve calibrated decision predictions with selective classification.

"""Production inference engine and calibrated decision scorer.

Provides both a reusable Python API (``DecisionEngine``) and a standalone CLI
for scoring input texts against trained decision checkpoints. Outputs
calibrated probabilities, predictive entropy, and selective classification
(abstention) flags formatted strictly as JSON for webhook and workflow
automation pipelines (e.g., n8n, Zapier).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logger to output only to stderr so CLI stdout remains pure JSON
logger = logging.getLogger("zoul_rlcd.predict")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class DecisionEngine:
    """Production serving engine for calibrated decision scoring.

    Loads a trained classification checkpoint, restores label mappings
    from ``config.json``, and provides calibrated probability estimates
    with predictive entropy and selective classification (abstention) logic.

    Parameters
    ----------
    model_path : str | Path
        Path to checkpoint directory or Hugging Face Hub model identifier.
    device : str | torch.device | None
        Target compute device. If None, auto-selects CUDA if available.
    max_length : int
        Maximum sequence length threshold (default: 512).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = str(model_path)
        self.max_length = max_length

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info("Initializing DecisionEngine on device: %s", self.device)
        logger.info("Loading tokenizer from '%s'...", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        logger.info("Loading model weights from '%s'...", self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        # Extract label mapping from model config
        self.id2label: dict[int, str] = self._extract_id2label()
        logger.info(
            "DecisionEngine ready with %d classes: %s",
            len(self.id2label),
            list(self.id2label.values()),
        )

    def _extract_id2label(self) -> dict[int, str]:
        """Extract and normalize id2label mappings from the model configuration."""
        raw_id2label = getattr(self.model.config, "id2label", None)
        num_labels = getattr(self.model.config, "num_labels", 2)

        if raw_id2label and isinstance(raw_id2label, dict):
            # Keys in serialized JSON config are strings, cast back to ints
            return {int(k): str(v) for k, v in raw_id2label.items()}

        # Fallback to class indices if unconfigured
        logger.warning(
            "id2label not found in config.json. Using fallback 'class_i' mapping."
        )
        return {i: f"class_{i}" for i in range(num_labels)}

    @torch.no_grad()
    def predict(
        self,
        text: str,
        threshold: float = 0.85,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Score input text and produce a calibrated decision recommendation.

        Parameters
        ----------
        text : str
            Input text to evaluate.
        threshold : float
            Confidence threshold for selective classification. If max probability
            is below this value, status is set to 'ABSTAIN / REVIEW_REQUIRED'.
        temperature : float
            Post-hoc temperature scalar T > 0. Values > 1 soften overconfident
            logits, while values < 1 sharpen probabilities.

        Returns
        -------
        dict[str, Any]
            Decision dictionary matching the specification schema.
        """
        # 1. Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)  # Shape: (C,)

        # 3. Temperature scaling
        t = max(float(temperature), 1e-3)
        scaled_logits = logits / t

        # 4. Probabilities via softmax
        probs = F.softmax(scaled_logits, dim=-1)

        # 5. Determine winning decision
        best_idx = int(torch.argmax(probs).item())
        best_confidence = float(probs[best_idx].item())
        decision = self.id2label.get(best_idx, f"class_{best_idx}")

        # 6. Compute predictive entropy: H(p) = -sum_c p_c * ln(p_c + eps)
        eps = 1e-12
        entropy = float(-torch.sum(probs * torch.log(probs + eps)).item())

        # 7. Selective classification (abstention)
        if best_confidence >= threshold:
            status = "CONFIDENT"
        else:
            status = "ABSTAIN / REVIEW_REQUIRED"

        # 8. Full probabilities dictionary mapped to class names
        probabilities: dict[str, float] = {
            self.id2label.get(i, f"class_{i}"): round(float(probs[i].item()), 4)
            for i in range(len(probs))
        }

        return {
            "text": text,
            "decision": decision,
            "confidence": round(best_confidence, 4),
            "entropy": round(entropy, 4),
            "status": status,
            "probabilities": probabilities,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: list[str],
        threshold: float = 0.85,
        temperature: float = 1.0,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Score a list of text inputs in batches.

        Parameters
        ----------
        texts : list[str]
            List of input texts.
        threshold : float
            Confidence threshold for abstention.
        temperature : float
            Temperature scaling scalar.
        batch_size : int
            Batch size for evaluation.

        Returns
        -------
        list[dict[str, Any]]
            List of prediction dictionaries.
        """
        results: list[dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for text in chunk:
                results.append(
                    self.predict(
                        text, threshold=threshold, temperature=temperature
                    )
                )
        return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(
        prog="zoul_rlcd.predict",
        description=(
            "Zoul AI Decision Engine -- Calibrated inference scorer and "
            "selective classification CLI tool. Outputs strict JSON."
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="best_calibrated_model",
        help="Path to trained model checkpoint directory (default: best_calibrated_model).",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text string to score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for selective classification (default: 0.85).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling scalar T (default: 1.0).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2; set 0 for compact JSON).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all informational logs on stderr.",
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point. Emits strictly valid JSON to stdout."""
    args = parse_args()

    if args.quiet:
        logger.setLevel(logging.ERROR)

    try:
        engine = DecisionEngine(model_path=args.model_path)
        result = engine.predict(
            text=args.text,
            threshold=args.threshold,
            temperature=args.temperature,
        )
        # Output ONLY formatted JSON to stdout for webhook consumption
        indent = args.indent if args.indent > 0 else None
        sys.stdout.write(json.dumps(result, indent=indent) + "\n")
        sys.stdout.flush()
    except Exception as exc:
        logger.error("Inference failure: %s", exc)
        error_payload = {
            "error": str(exc),
            "status": "ERROR",
        }
        sys.stdout.write(json.dumps(error_payload) + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
