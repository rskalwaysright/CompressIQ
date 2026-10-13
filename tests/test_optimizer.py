"""Unit tests for Budget-aware Greedy Optimizer in CompressIQ."""

import pytest
import torch
import torch.nn as nn

from compressiq.optimization.sensitivity import SensitivityProfile, LayerSensitivityRecord
from compressiq.optimization.budget_optimizer import BudgetAwareOptimizer, apply_optimization_result


def create_dummy_profile():
    """Helper creating synthetic sensitivity profile for testing greedy decisions."""
    records = [
        # Layer 1: Highly tolerant (low loss increase per MB saved)
        LayerSensitivityRecord(
            layer_name="layer1",
            compression_type="quantization",
            setting_value=8,
            val_loss=1.05,
            loss_increase=0.05,
            val_acc=78.5,
            acc_drop=0.5,
            layer_size_mb=1.0,
            mb_saved=3.0,
            sensitivity_score=0.0167
        ),
        LayerSensitivityRecord(
            layer_name="layer1",
            compression_type="quantization",
            setting_value=4,
            val_loss=1.12,
            loss_increase=0.12,
            val_acc=77.0,
            acc_drop=2.0,
            layer_size_mb=0.5,
            mb_saved=3.5,
            sensitivity_score=0.0342
        ),
        # Layer 2: Fragile (high loss increase per MB saved)
        LayerSensitivityRecord(
            layer_name="layer2",
            compression_type="quantization",
            setting_value=8,
            val_loss=1.80,
            loss_increase=0.80,
            val_acc=60.0,
            acc_drop=19.0,
            layer_size_mb=1.0,
            mb_saved=3.0,
            sensitivity_score=0.2667
        ),
    ]
    return SensitivityProfile(
        baseline_loss=1.00,
        baseline_acc=79.0,
        baseline_size_mb=8.0,
        records=records
    )


def test_optimizer_respects_feasible_budget():
    """Verify greedy optimizer satisfies budget and picks lowest sensitivity actions."""
    profile = create_dummy_profile()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Linear(16, 10)
    )

    optimizer = BudgetAwareOptimizer(model, profile, target_budget_mb=5.0)
    result = optimizer.optimize_greedy_sensitivity()

    assert result.is_feasible is True
    assert result.final_size_mb <= 5.0
    assert len(result.decision_trace) > 0
    # First action should be layer1 because of lower sensitivity_score
    assert result.decision_trace[0].layer_name == "layer1"


def test_optimizer_handles_infeasible_budget():
    """Verify explicit diagnosis when target budget is unattainable."""
    profile = create_dummy_profile()
    model = nn.Sequential(nn.Conv2d(3, 16, 3))

    # Impossibly small budget (e.g. 0.001 MB)
    optimizer = BudgetAwareOptimizer(model, profile, target_budget_mb=0.001)
    result = optimizer.optimize_greedy_sensitivity()

    assert result.is_feasible is False
    assert result.infeasibility_reason is not None
    assert "infeasible" in result.infeasibility_reason.lower()
