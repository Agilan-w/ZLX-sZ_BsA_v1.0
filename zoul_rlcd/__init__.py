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

"""Zoul AI Decision Engine -- RLCD Fine-Tuning Pipeline.

A production-grade pipeline for training calibrated decision heads on top
of the Laya model (ModernBERT-large encoder) using Reinforcement Learning
for Calibrated Decisions (RLCD).
"""

__version__ = "0.1.0"

from zoul_rlcd.data_loader import load_decision_dataset
from zoul_rlcd.dataset import SyntheticDecisionDataset
from zoul_rlcd.metrics import CalibrationTracker
from zoul_rlcd.model import build_calibrated_decision_model
from zoul_rlcd.rlcd_loss import BrierScoreLoss, RLCDCompoundLoss
from zoul_rlcd.temperature_scaling import ModelWithTemperature

__all__ = [
    "BrierScoreLoss",
    "RLCDCompoundLoss",
    "build_calibrated_decision_model",
    "CalibrationTracker",
    "SyntheticDecisionDataset",
    "load_decision_dataset",
    "ModelWithTemperature",
    "DecisionEngine",
]


def __getattr__(name: str):
    """Lazy-load modules to prevent runpy execution warnings when running CLI commands."""
    if name == "DecisionEngine":
        from zoul_rlcd.predict import DecisionEngine

        return DecisionEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
