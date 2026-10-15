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

"""Unit tests for RLCD loss functions and mathematical invariants."""

from __future__ import annotations

import pytest
import torch

from zoul_rlcd.rlcd_loss import BrierScoreLoss, RLCDCompoundLoss


class TestBrierScoreLoss:
    """Mathematical invariants and gradient properties of BrierScoreLoss."""

    def test_reduction_modes(self):
        """Verify mean, sum, and none reduction modes."""
        loss_mean = BrierScoreLoss(reduction="mean")
        loss_sum = BrierScoreLoss(reduction="sum")
        loss_none = BrierScoreLoss(reduction="none")

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        out_mean = loss_mean(logits, targets)
        out_sum = loss_sum(logits, targets)
        out_none = loss_none(logits, targets)

        assert out_mean.dim() == 0
        assert out_sum.dim() == 0
        assert out_none.shape == (8,)
        assert torch.allclose(out_mean, out_none.mean())
        assert torch.allclose(out_sum, out_none.sum())

    def test_invalid_reduction(self):
        """Verify invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            BrierScoreLoss(reduction="invalid_reduction")

    def test_non_negativity_and_bounds(self):
        """Brier score must be non-negative and bounded in [0, 2] for one-hot targets."""
        loss_fn = BrierScoreLoss(reduction="none")
        for _ in range(5):
            logits = torch.randn(16, 5) * 5.0
            targets = torch.randint(0, 5, (16,))
            scores = loss_fn(logits, targets)

            assert torch.all(scores >= 0.0)
            assert torch.all(scores <= 2.0 + 1e-6)

    def test_perfect_prediction_limit(self):
        """When predicted probability for true class is ~1.0, Brier score approaches 0."""
        loss_fn = BrierScoreLoss(reduction="mean")
        num_classes = 4
        targets = torch.tensor([0, 1, 2, 3])

        # Extremely confident logits on true classes
        logits = torch.zeros(4, num_classes)
        for i, target in enumerate(targets):
            logits[i, target] = 50.0

        loss = loss_fn(logits, targets)
        assert loss.item() < 1e-6

    def test_worst_prediction_limit(self):
        """When predicted probability for a wrong class is ~1.0, Brier score approaches 2.0."""
        loss_fn = BrierScoreLoss(reduction="mean")
        targets = torch.tensor([0, 1])

        # Extremely confident logits on WRONG classes
        logits = torch.zeros(2, 2)
        logits[0, 1] = 50.0  # Target is 0
        logits[1, 0] = 50.0  # Target is 1

        loss = loss_fn(logits, targets)
        assert abs(loss.item() - 2.0) < 1e-4

    def test_uniform_prediction_invariant(self):
        """For uniform probabilities p_c = 1/C, Brier score is exactly 1 - 1/C."""
        loss_fn = BrierScoreLoss(reduction="mean")
        for num_classes in [2, 3, 5, 10]:
            logits = torch.zeros(10, num_classes)
            targets = torch.randint(0, num_classes, (10,))
            loss = loss_fn(logits, targets)

            expected = 1.0 - (1.0 / num_classes)
            assert abs(loss.item() - expected) < 1e-5

    def test_autograd_gradient_flow(self):
        """Verify backpropagation computes valid, finite gradients."""
        loss_fn = BrierScoreLoss(reduction="mean")
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (8, 4)
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()
        assert torch.any(logits.grad != 0.0)


class TestRLCDCompoundLoss:
    """Behavior and invariants of the compound RLCD loss."""

    def test_invalid_lambda(self):
        """Negative lambda_cal must raise ValueError."""
        with pytest.raises(ValueError, match="lambda_cal must be non-negative"):
            RLCDCompoundLoss(lambda_cal=-0.1)

    def test_pure_ce_limit_when_lambda_zero(self):
        """When lambda_cal is 0.0, compound loss equals pure CrossEntropyLoss."""
        compound_loss = RLCDCompoundLoss(lambda_cal=0.0)
        ce_loss_fn = torch.nn.CrossEntropyLoss()

        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        total, components = compound_loss(logits, targets)
        expected_ce = ce_loss_fn(logits, targets)

        assert torch.allclose(total, expected_ce)
        assert torch.allclose(components["ce"], expected_ce)
        assert components["brier"] > 0.0  # Brier is still calculated for logging

    def test_calibration_penalty_weighting(self):
        """Total loss must be exactly L_CE + lambda_cal * L_Brier."""
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        for lambda_val in [0.1, 0.5, 1.0, 2.0]:
            compound = RLCDCompoundLoss(lambda_cal=lambda_val)
            total, components = compound(logits, targets)

            expected = components["ce"] + lambda_val * components["brier"]
            assert torch.allclose(total, expected, atol=1e-5)

    def test_gradient_propagation_both_branches(self):
        """Verify gradients propagate from both CE and Brier terms into logits."""
        compound = RLCDCompoundLoss(lambda_cal=0.5)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])

        total, _ = compound(logits, targets)
        total.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert logits.grad.shape == (4, 3)
