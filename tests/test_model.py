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

"""Unit tests for calibrated decision model construction, freezing, and checkpointing."""

from __future__ import annotations

import pytest
import torch

from zoul_rlcd.model import build_calibrated_decision_model


@pytest.fixture(scope="module")
def model_name() -> str:
    """Use cached bert-base-uncased for fast, reproducible unit testing."""
    return "bert-base-uncased"


class TestModelArchitectureAndFreezing:
    """Tests for model building, parameter freezing, and configuration injection."""

    def test_backbone_freezing(self, model_name: str):
        """When freeze_backbone=True, encoder is frozen and only classification head trains."""
        model = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=3,
            freeze_backbone=True,
        )

        trainable_names = []
        frozen_names = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        # Classifier parameters must be trainable
        assert len(trainable_names) > 0
        for name in trainable_names:
            assert "classifier" in name or "cls" in name or "score" in name

        # Backbone parameters must be frozen
        assert len(frozen_names) > 0
        for name in frozen_names:
            assert "classifier" not in name and "cls" not in name

    def test_full_finetuning_unfrozen(self, model_name: str):
        """When freeze_backbone=False, all parameters remain trainable."""
        model = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=3,
            freeze_backbone=False,
        )

        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())

        assert trainable_count == total_count
        assert all(p.requires_grad for p in model.parameters())

    def test_gradient_checkpointing_toggle(self, model_name: str):
        """Gradient checkpointing can be toggled on or off."""
        model_enabled = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=3,
            freeze_backbone=False,
            gradient_checkpointing=True,
        )
        is_gc_enabled = getattr(model_enabled, "is_gradient_checkpointing", False) or getattr(
            model_enabled.base_model, "gradient_checkpointing", False
        )
        assert is_gc_enabled is True

        model_disabled = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=3,
            freeze_backbone=False,
            gradient_checkpointing=False,
        )
        is_gc_disabled = getattr(model_disabled, "is_gradient_checkpointing", False) or getattr(
            model_disabled.base_model, "gradient_checkpointing", False
        )
        assert is_gc_disabled is False

    def test_id2label_and_label2id_injection(self, model_name: str):
        """Custom label mappings must be injected into model.config for serialization."""
        id2label = {0: "REJECT", 1: "REVIEW", 2: "APPROVE"}
        label2id = {"REJECT": 0, "REVIEW": 1, "APPROVE": 2}

        model = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=3,
            id2label=id2label,
            label2id=label2id,
        )

        assert model.config.num_labels == 3
        # In transformers config, keys might be normalized to ints or strings
        config_id2label = {int(k): v for k, v in model.config.id2label.items()}
        assert config_id2label == id2label
        assert model.config.label2id == label2id

    def test_forward_pass_output_shape(self, model_name: str):
        """Model forward pass must return logits of shape (batch_size, num_classes)."""
        num_classes = 4
        model = build_calibrated_decision_model(
            model_name_or_path=model_name,
            num_classes=num_classes,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, num_classes)
        assert not torch.isnan(outputs.logits).any()
