# ZLX-CDE V2.5 Decision Engine

**RLCD Fine-Tuning Pipeline for Calibrated Decision Heads**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org)

Pipeline for training **calibrated decision heads** on top of the [Laya](https://huggingface.co/convaiinnovations/laya) model (~421M parameter ModernBERT-large encoder) using **Reinforcement Learning for Calibrated Decisions (RLCD)**.

---

## Overview

Classifiers trained only on cross-entropy often produce overconfident probability estimates. RLCD adds a **Brier Score** penalty to train for calibration alongside classification accuracy:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{cal}} \cdot \mathcal{L}_{\text{Brier}}$$

### Features

- Mixed-precision training (BF16 / FP16) and gradient accumulation
- Tracks Accuracy, Macro F1, Brier Score, and Expected Calibration Error (ECE)
- Saves checkpoints on ECE improvement without accuracy regression
- Post-hoc temperature scaling via L-BFGS

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/zlx-cde-v2.5.git
cd zlx-cde-v2.5

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with development & testing dependencies
pip install -e ".[dev]"
```

### Smoke Test (Synthetic Data)

```bash
python -m zoul_rlcd.train \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --num_classes 5 \
    --lambda_cal 0.5 \
    --num_samples 500 \
    --max_length 128
```

This runs a full training loop on random synthetic data to validate the pipeline end-to-end without downloading real data.

---

## Architecture

```
zoul_rlcd/
├── __init__.py            # Package exports & version
├── rlcd_loss.py           # BrierScoreLoss + RLCDCompoundLoss
├── model.py               # build_calibrated_decision_model()
├── metrics.py             # CalibrationTracker (Acc, F1, Brier, ECE)
├── dataset.py             # SyntheticDecisionDataset
├── data_loader.py         # Real-world data loading & dynamic batch collation
├── temperature_scaling.py # Post-hoc temperature calibration (L-BFGS)
├── evaluate.py            # Diagnostic reporting & reliability diagram plotting
├── predict.py             # Production inference & calibrated decision scorer
└── train.py               # Training orchestrator with CLI
```

### Data Flow

```
Input Tokens → [ModernBERT Encoder] → [Classification Head] → Logits
                                                                 │
                                              ┌──────────────────┤
                                              ▼                  ▼
                                        CrossEntropy        BrierScore
                                              │                  │
                                              └──── L_total ─────┘
                                                (λ-weighted sum)
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model_name_or_path` | `convaiinnovations/laya` | HuggingFace model ID or local path |
| `--num_classes` | `5` | Number of output classes (auto-detected when using `--data_path`) |
| `--freeze_backbone` | `False` | Freeze encoder; train head only (default: False for full-parameter training) |
| `--gradient_checkpointing` | `False` | Enable gradient checkpointing (disabled by default for 32GB VRAM speed) |
| `--attn_implementation` | `flash_attention_2` | Attention mechanism backend |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Micro-batch size per forward pass (RTX 5090 default: 32) |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (effective batch = batch_size * this) |
| `--lr_backbone` | `1e-5` | Peak learning rate for ModernBERT backbone |
| `--lr_head` | `1e-4` | Peak learning rate for classification head |
| `--weight_decay` | `0.01` | Weight decay for AdamW |
| `--lambda_cal` | `0.5` | Brier calibration weight |
| `--max_grad_norm` | `1.0` | Gradient clipping threshold |
| `--data_path` | `None` | Path to dataset (.csv, .jsonl, .parquet) or Hugging Face Hub dataset ID |
| `--text_column` | `text` | Column name containing input text |
| `--label_column` | `label` | Column name containing target labels |
| `--dataset_config` | `None` | Optional sub-configuration name for HF Hub datasets |
| `--num_samples` | `1000` | Synthetic dataset size (used only when `--data_path` is omitted) |
| `--max_length` | `1024` | Sequence length threshold (RTX 5090 default: 1024) |
| `--val_split` | `0.2` | Validation split fraction |
| `--use_wandb` | `False` | Enable Weights & Biases (wandb) experiment tracking |
| `--wandb_project` | `zlx-cde-v2.5` | Weights & Biases project name |
| `--wandb_run_name` | `None` | Optional run name for Weights & Biases |
| `--output_dir` | `best_calibrated_model` | Checkpoint directory |
| `--seed` | `42` | Random seed |
| `--acc_degradation_tol` | `0.05` | Max accuracy drop allowed when saving on ECE improvement |

---

## Training on Custom Data

The engine natively ingests `.csv`, `.jsonl`, `.parquet` files and Hugging Face Hub datasets with automated label mapping (`id2label` / `label2id`) and dynamic batch padding via `DataCollatorWithPadding`:

### Example: Training on a Local CSV / JSONL File

```bash
python -m zoul_rlcd.train \
    --data_path data/decisions.csv \
    --text_column query_text \
    --label_column decision_label \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum_steps 8
```

### Example: Training on a Hugging Face Hub Dataset

```bash
python -m zoul_rlcd.train \
    --data_path ag_news \
    --text_column text \
    --label_column label \
    --epochs 3
```

When training completes, `id2label` and `label2id` mappings are automatically serialized into the model's `config.json` inside `--output_dir`.

---

## Evaluation & Reliability Diagrams

Use `zoul_rlcd.evaluate` to inspect calibration quality, learn optimal post-hoc temperature scaling ($T > 0$ via L-BFGS), and render publication-ready Reliability Diagrams:

```bash
python -m zoul_rlcd.evaluate \
    --model_path best_calibrated_model \
    --data_path data/decisions.csv \
    --text_column query_text \
    --label_column decision_label \
    --n_bins 15 \
    --output_plot reliability_diagram.png
```

### Features
- **Strict Metric Invariants**: Top-1 Accuracy and Macro F1 are strictly preserved under temperature scaling while ECE, Brier score, and NLL are optimized.
- **Reliability Diagram Generation**: Produces a side-by-side plot of empirical accuracy vs. confidence bins ($y = x$ diagonal) with calibration gaps visually highlighted.
- **Diagnostic Table**: Prints a before-and-after comparison in the terminal.

---

## Inference

Run calibrated inference either via Python or through the CLI.

### Python API (`DecisionEngine`)

```python
from zoul_rlcd import DecisionEngine

# Load checkpoint and restore id2label mappings from config.json
engine = DecisionEngine("best_calibrated_model")

# Score input text with post-hoc temperature scaling and selective abstention
result = engine.predict(
    text="Suspicious wire transfer initiated from unrecognized location",
    threshold=0.85,
    temperature=1.0,
)

print(result)
```

**Output schema**:
```json
{
  "text": "Suspicious wire transfer initiated from unrecognized location",
  "decision": "FLAGGED",
  "confidence": 0.924,
  "entropy": 0.31,
  "status": "CONFIDENT",
  "probabilities": {
    "APPROVED": 0.026,
    "FLAGGED": 0.924,
    "REJECTED": 0.05
  }
}
```

### Selective Classification (Abstention Logic)

If the maximum predicted class probability ($\max_c p_c$) is below `--threshold`, the decision status is set to:
`"ABSTAIN / REVIEW_REQUIRED"`
Otherwise, when $\max_c p_c \ge \text{threshold}$, the status is marked as:
`"CONFIDENT"`

### CLI (JSON Output)

Logging is sent to `stderr` and JSON output to `stdout`:

```bash
python -m zoul_rlcd.predict \
    --model_path best_calibrated_model \
    --text "Customer requested urgent refund for order #12345" \
    --threshold 0.85 \
    --temperature 1.0 \
    --quiet
```

### Batch Inference

```python
# Score multiple texts efficiently
batch_results = engine.predict_batch(
    texts=["Transaction A", "Transaction B"],
    threshold=0.85,
    batch_size=16,
)
```

---

### Optional: Flash Attention 2

For Ampere+ GPUs (A100, RTX 3090/4090), install Flash Attention for significantly faster training:

```bash
pip install flash-attn --no-build-isolation
```

The model will automatically use it if available.

---

## Memory Budget (8 GB VRAM)

| Technique | Memory Saving |
|---|---|
| AMP (BF16/FP16) | ~50% reduction in activation memory |
| Gradient accumulation (4 × 8) | Effective batch 32 without storing 32 activations |
| Gradient checkpointing | ~60% reduction in activation memory (trades compute) |
| `set_to_none=True` | Avoids zeroing gradients; frees memory immediately |

With all optimisations enabled, the ~421M parameter Laya model fits comfortably within 8 GB VRAM at `max_length=128` and `batch_size=4`.

---

## Testing

Run tests and linter:

```bash
# Run linter
ruff check .

# Execute full test suite
pytest tests/ -v
```

CI runs automatically on push and pull requests to `main` via GitHub Actions (`.github/workflows/ci.yml`) across Python 3.10, 3.11, and 3.12.

---

## Legal & Licensing

This project is licensed under the **Apache License, Version 2.0**. See the [`LICENSE`](LICENSE) file for the full terms.

This software is provided **"AS IS"**, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

### Attribution

This project is built upon the **Laya** model by [Convai Innovations](https://convaiinnovations.com) (Copyright 2026 Nandakishor Mukkunnoth / Convai Innovations) and the **ModernBERT** architecture by [Answer.AI](https://huggingface.co/answerdotai/ModernBERT-large) and collaborators. See the [`NOTICE`](NOTICE) file for complete attribution details.

This is an **independent derivative work**. "ZLX-CDE V2.5 Decision Engine" does not claim endorsement from, or affiliation with, the original Laya or ModernBERT creators. "Laya" is a trademark of Convai Innovations and is used here solely to identify the upstream model for transparency and compliance with Apache 2.0 Section 6.

### Modification Notices

Per Apache 2.0 Section 4(b), files that directly wrap, modify, or derive from original Laya code carry prominent modification notices in their headers. See `model.py` and `train.py` for examples.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{zlx_cde_v2_5_2026,
    title  = {ZLX-CDE V2.5 Decision Engine: RLCD Fine-Tuning Pipeline},
    author = {ZLX-CDE V2.5},
    year   = {2026},
    url    = {https://github.com/your-org/zlx-cde-v2.5},
}
```
