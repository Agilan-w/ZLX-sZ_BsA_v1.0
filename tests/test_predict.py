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

"""Unit and integration tests for DecisionEngine and selective classification."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from zoul_rlcd.predict import DecisionEngine


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    """Instantiate DecisionEngine once for test module."""
    return DecisionEngine("bert-base-uncased")


class TestDecisionEngine:
    """Tests for production inference engine and selective classification thresholds."""

    def test_predict_schema(self, engine: DecisionEngine):
        """Verify returned dictionary matches production JSON schema."""
        result = engine.predict("Payment reconciliation initiated", threshold=0.5)

        assert isinstance(result, dict)
        assert result["text"] == "Payment reconciliation initiated"
        assert isinstance(result["decision"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["entropy"], float)
        assert result["status"] in ("CONFIDENT", "ABSTAIN / REVIEW_REQUIRED")
        assert isinstance(result["probabilities"], dict)

        # Probabilities sum to 1.0 within floating point rounding
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

        # Winning class probability matches confidence
        winning_class = result["decision"]
        assert result["probabilities"][winning_class] == result["confidence"]

    def test_selective_classification_abstention(self, engine: DecisionEngine):
        """Thresholding must trigger abstention when max probability is below threshold."""
        # Extremely high threshold guarantees abstention
        res_abstain = engine.predict("Borderline ambiguous case", threshold=0.9999)
        assert res_abstain["status"] == "ABSTAIN / REVIEW_REQUIRED"

        # Extremely low threshold guarantees confident status
        res_confident = engine.predict("Borderline ambiguous case", threshold=0.0001)
        assert res_confident["status"] == "CONFIDENT"

    def test_temperature_scaling_impact_on_entropy(self, engine: DecisionEngine):
        """Higher temperatures should increase predictive entropy (more dispersed)."""
        text = "Evaluation text for temperature effects"
        res_sharp = engine.predict(text, temperature=0.2)
        res_smooth = engine.predict(text, temperature=5.0)

        assert res_smooth["entropy"] >= res_sharp["entropy"] - 1e-4

    def test_batch_prediction(self, engine: DecisionEngine):
        """predict_batch must process lists of texts and return list of result dicts."""
        texts = ["Text sample A", "Text sample B", "Text sample C"]
        results = engine.predict_batch(texts, threshold=0.85, batch_size=2)

        assert len(results) == 3
        for i, res in enumerate(results):
            assert res["text"] == texts[i]
            assert "decision" in res
            assert "confidence" in res


class TestPredictCLI:
    """Tests for standalone CLI execution and strict JSON output."""

    def test_cli_strict_json_stdout(self):
        """CLI stdout must be strictly parseable as valid JSON."""
        cmd = [
            sys.executable,
            "-m",
            "zoul_rlcd.predict",
            "--model_path",
            "bert-base-uncased",
            "--text",
            "Urgent webhook transaction query",
            "--threshold",
            "0.80",
            "--temperature",
            "1.0",
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"

        # stdout must parse as JSON without extra lines or characters
        payload = json.loads(proc.stdout.strip())
        assert payload["text"] == "Urgent webhook transaction query"
        assert "decision" in payload
        assert "confidence" in payload
        assert payload["status"] in ("CONFIDENT", "ABSTAIN / REVIEW_REQUIRED")
