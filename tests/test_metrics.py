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

"""Unit tests for CalibrationTracker and Expected Calibration Error (ECE) metrics."""

from __future__ import annotations

import math

import pytest
import torch

from zoul_rlcd.metrics import CalibrationTracker


class TestCalibrationTracker:
    """Tests for calibration metrics, ECE binning accuracy, and zero-division safeguards."""

    def test_invalid_parameters(self):
        """Invalid num_classes or n_bins must raise ValueError."""
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            CalibrationTracker(num_classes=1)

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            CalibrationTracker(num_classes=3, n_bins=0)

    def test_zero_samples_safeguard(self):
        """Calling compute() before any update() must not raise ZeroDivisionError."""
        tracker = CalibrationTracker(num_classes=3)
        metrics = tracker.compute()

        assert metrics["brier_score"] == 0.0
        assert not math.isnan(metrics["brier_score"])
        assert not math.isinf(metrics["brier_score"])

    def test_empty_bins_safeguard(self):
        """ECE binning must handle empty confidence bins without generating NaNs."""
        # When all predictions have identical confidence, most bins are empty
        tracker = CalibrationTracker(num_classes=3, n_bins=15)
        batch_size = 20

        # Uniform logits -> all probabilities = 1/3 ~ 0.333 -> all fall into bin 5
        logits = torch.zeros(batch_size, 3)
        targets = torch.zeros(batch_size, dtype=torch.long)

        tracker.update(logits, targets)
        metrics = tracker.compute()

        assert "ece" in metrics
        assert not math.isnan(metrics["ece"])
        assert not math.isinf(metrics["ece"])
        assert 0.0 <= metrics["ece"] <= 1.0

    def test_metric_value_ranges(self):
        """All computed metrics must fall strictly within valid theoretical ranges."""
        tracker = CalibrationTracker(num_classes=4, n_bins=10)

        logits = torch.randn(50, 4)
        targets = torch.randint(0, 4, (50,))

        tracker.update(logits, targets)
        metrics = tracker.compute()

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 2.0
        assert 0.0 <= metrics["ece"] <= 1.0

    def test_perfect_calibration_case(self):
        """Well-calibrated probabilities on true labels must yield very low ECE."""
        tracker = CalibrationTracker(num_classes=2, n_bins=10)

        # 100 samples with 99% confident correct predictions
        logits = torch.zeros(100, 2)
        targets = torch.zeros(100, dtype=torch.long)
        logits[:, 0] = 10.0  # ~99.99% confidence on correct label

        tracker.update(logits, targets)
        metrics = tracker.compute()

        assert metrics["accuracy"] == 1.0
        assert metrics["ece"] < 0.02
        assert metrics["brier_score"] < 0.01

    def test_overconfident_miscalibration_case(self):
        """Completely confident predictions on WRONG labels must produce high ECE."""
        tracker = CalibrationTracker(num_classes=2, n_bins=10)

        # 100 samples with 99% confidence on INCORRECT class
        logits = torch.zeros(100, 2)
        targets = torch.zeros(100, dtype=torch.long)
        logits[:, 1] = 10.0  # Confident on class 1, but true is class 0

        tracker.update(logits, targets)
        metrics = tracker.compute()

        assert metrics["accuracy"] == 0.0
        # Confidence is ~1.0 but accuracy is 0.0 -> ECE is near 1.0
        assert metrics["ece"] > 0.90
        assert metrics["brier_score"] > 1.90

    def test_multi_batch_accumulation_and_reset(self):
        """Tracker must correctly accumulate across batches and reset state cleanly."""
        tracker = CalibrationTracker(num_classes=3)

        # Batch 1
        logits_1 = torch.randn(16, 3)
        targets_1 = torch.randint(0, 3, (16,))
        tracker.update(logits_1, targets_1)

        # Batch 2
        logits_2 = torch.randn(16, 3)
        targets_2 = torch.randint(0, 3, (16,))
        tracker.update(logits_2, targets_2)

        metrics_accumulated = tracker.compute()
        assert tracker._sample_count.item() == 32
        assert "accuracy" in metrics_accumulated

        # Reset
        tracker.reset()
        assert tracker._sample_count.item() == 0
        assert tracker._brier_sum.item() == 0.0

        # Post-reset compute
        post_reset = tracker.compute()
        assert post_reset["brier_score"] == 0.0

    def test_log_summary_returns_metrics(self):
        """log_summary() should log and return the metrics dictionary."""
        tracker = CalibrationTracker(num_classes=2)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        tracker.update(logits, targets)

        summary = tracker.log_summary(epoch=1)
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {"accuracy", "macro_f1", "brier_score", "ece"}
