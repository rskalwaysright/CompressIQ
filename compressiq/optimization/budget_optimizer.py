"""Budget-aware greedy optimizer for CompressIQ.

Implements deterministic greedy search guided by per-layer sensitivity:
- Selects layer-wise precision and structured pruning configurations under a model-size budget (MB).
- Prioritizes layers with lowest sensitivity (least accuracy/loss penalty per MB saved).
- Handles infeasible budgets explicitly with informative error diagnostics.
- Records full decision trace for transparency and research auditability.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn

from compressiq.optimization.sensitivity import SensitivityProfile, LayerSensitivityRecord
from compressiq.compression.quantization import apply_simulated_quantization, apply_uniform_quantization
from compressiq.compression.pruning import apply_structured_pruning, apply_uniform_pruning
from compressiq.evaluation.metrics import compute_model_size_mb


@dataclass
class DecisionStep:
    step: int
    layer_name: str
    compression_type: str
    selected_setting: Any
    marginal_mb_saved: float
    marginal_loss_penalty: float
    sensitivity_score: float
    cumulative_size_mb: float


@dataclass
class OptimizationResult:
    strategy: str
    target_budget_mb: float
    initial_size_mb: float
    final_size_mb: float
    is_feasible: bool
    quantization_plan: Dict[str, int]
    pruning_plan: Dict[str, float]
    decision_trace: List[DecisionStep]
    infeasibility_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "target_budget_mb": self.target_budget_mb,
            "initial_size_mb": self.initial_size_mb,
            "final_size_mb": self.final_size_mb,
            "is_feasible": self.is_feasible,
            "quantization_plan": self.quantization_plan,
            "pruning_plan": self.pruning_plan,
            "decision_trace": [asdict(s) for s in self.decision_trace],
            "infeasibility_reason": self.infeasibility_reason
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class BudgetAwareOptimizer:
    """Deterministic greedy search optimizer guided by layer sensitivity."""

    def __init__(
        self,
        base_model: nn.Module,
        sensitivity_profile: SensitivityProfile,
        target_budget_mb: float
    ):
        self.base_model = base_model
        self.profile = sensitivity_profile
        self.target_budget_mb = target_budget_mb
        self.initial_size_mb = sensitivity_profile.baseline_size_mb

    def optimize_greedy_sensitivity(self) -> OptimizationResult:
        """Execute deterministic greedy search to find layer-wise configuration satisfying budget."""
        curr_size = self.initial_size_mb
        if curr_size <= self.target_budget_mb:
            # Already satisfies budget without compression
            return OptimizationResult(
                strategy="greedy_sensitivity",
                target_budget_mb=self.target_budget_mb,
                initial_size_mb=self.initial_size_mb,
                final_size_mb=curr_size,
                is_feasible=True,
                quantization_plan={},
                pruning_plan={},
                decision_trace=[]
            )

        # Sort all candidate actions by sensitivity_score ascending (least penalty per MB saved first)
        # Tie-breaker: larger mb_saved first, then layer_name for determinism
        sorted_candidates = sorted(
            self.profile.records,
            key=lambda r: (r.sensitivity_score, -r.mb_saved, r.layer_name)
        )

        current_quant_plan: Dict[str, int] = {}
        current_prune_plan: Dict[str, float] = {}
        decision_trace: List[DecisionStep] = []
        step_idx = 1

        applied_actions = set()

        for cand in sorted_candidates:
            if curr_size <= self.target_budget_mb:
                break

            action_key = (cand.layer_name, cand.compression_type)
            # Only apply if setting is stricter than what was already assigned
            if cand.compression_type == "quantization":
                existing_bits = current_quant_plan.get(cand.layer_name, 32)
                if cand.setting_value < existing_bits:
                    # Calculate incremental size savings
                    prev_layer_size = (existing_bits / 32.0) * cand.layer_size_mb if existing_bits < 32 else cand.layer_size_mb + cand.mb_saved
                    new_layer_size = (cand.setting_value / 32.0) * (cand.layer_size_mb + cand.mb_saved)
                    inc_saved = max(0.001, prev_layer_size - new_layer_size)

                    current_quant_plan[cand.layer_name] = cand.setting_value
                    curr_size = max(0.1, curr_size - inc_saved)

                    decision_trace.append(DecisionStep(
                        step=step_idx,
                        layer_name=cand.layer_name,
                        compression_type="quantization",
                        selected_setting=f"{cand.setting_value}-bit",
                        marginal_mb_saved=round(inc_saved, 4),
                        marginal_loss_penalty=cand.loss_increase,
                        sensitivity_score=cand.sensitivity_score,
                        cumulative_size_mb=round(curr_size, 3)
                    ))
                    step_idx += 1

            elif cand.compression_type == "pruning":
                existing_ratio = current_prune_plan.get(cand.layer_name, 0.0)
                if cand.setting_value > existing_ratio:
                    inc_saved = max(0.001, cand.mb_saved * (cand.setting_value - existing_ratio) / max(cand.setting_value, 0.01))
                    current_prune_plan[cand.layer_name] = cand.setting_value
                    curr_size = max(0.1, curr_size - inc_saved)

                    decision_trace.append(DecisionStep(
                        step=step_idx,
                        layer_name=cand.layer_name,
                        compression_type="pruning",
                        selected_setting=f"{int(cand.setting_value * 100)}% prune",
                        marginal_mb_saved=round(inc_saved, 4),
                        marginal_loss_penalty=cand.loss_increase,
                        sensitivity_score=cand.sensitivity_score,
                        cumulative_size_mb=round(curr_size, 3)
                    ))
                    step_idx += 1

        is_feasible = curr_size <= self.target_budget_mb
        infeasibility_reason = None
        if not is_feasible:
            infeasibility_reason = (
                f"Requested budget {self.target_budget_mb:.2f} MB is infeasible under available compression candidates. "
                f"Minimum attainable model size is {curr_size:.2f} MB."
            )

        return OptimizationResult(
            strategy="greedy_sensitivity",
            target_budget_mb=self.target_budget_mb,
            initial_size_mb=self.initial_size_mb,
            final_size_mb=round(curr_size, 3),
            is_feasible=is_feasible,
            quantization_plan=current_quant_plan,
            pruning_plan=current_prune_plan,
            decision_trace=decision_trace,
            infeasibility_reason=infeasibility_reason
        )

    def optimize_uniform_quantization(self) -> OptimizationResult:
        """Find uniform quantization setting closest to budget."""
        # Candidates: 8-bit (~4x), 4-bit (~8x), 2-bit (~16x)
        for bits in [8, 4, 2]:
            est_size = self.initial_size_mb * (bits / 32.0)
            if est_size <= self.target_budget_mb:
                # Found satisfying uniform precision
                plan = {name: bits for name, m in self.base_model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
                return OptimizationResult(
                    strategy="uniform_quantization",
                    target_budget_mb=self.target_budget_mb,
                    initial_size_mb=self.initial_size_mb,
                    final_size_mb=round(est_size, 3),
                    is_feasible=True,
                    quantization_plan=plan,
                    pruning_plan={},
                    decision_trace=[]
                )

        # Infeasible uniform quant
        min_size = self.initial_size_mb * (2.0 / 32.0)
        return OptimizationResult(
            strategy="uniform_quantization",
            target_budget_mb=self.target_budget_mb,
            initial_size_mb=self.initial_size_mb,
            final_size_mb=round(min_size, 3),
            is_feasible=False,
            quantization_plan={name: 2 for name, m in self.base_model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))},
            pruning_plan={},
            decision_trace=[],
            infeasibility_reason=f"Uniform quantization cannot meet {self.target_budget_mb:.2f} MB (min size: {min_size:.2f} MB)."
        )

    def optimize_uniform_pruning(self) -> OptimizationResult:
        """Find uniform pruning setting closest to budget."""
        for ratio in [0.15, 0.30, 0.45]:
            # Pruning reduces expansion filters ~ (1-ratio)
            est_size = self.initial_size_mb * (1.0 - (ratio * 0.7))
            if est_size <= self.target_budget_mb:
                plan = {name: ratio for name, m in self.base_model.named_modules() if hasattr(m, "conv") and len(getattr(m, "conv")) >= 4}
                return OptimizationResult(
                    strategy="uniform_pruning",
                    target_budget_mb=self.target_budget_mb,
                    initial_size_mb=self.initial_size_mb,
                    final_size_mb=round(est_size, 3),
                    is_feasible=True,
                    quantization_plan={},
                    pruning_plan=plan,
                    decision_trace=[]
                )

        min_size = self.initial_size_mb * (1.0 - (0.45 * 0.7))
        return OptimizationResult(
            strategy="uniform_pruning",
            target_budget_mb=self.target_budget_mb,
            initial_size_mb=self.initial_size_mb,
            final_size_mb=round(min_size, 3),
            is_feasible=False,
            quantization_plan={},
            pruning_plan={name: 0.45 for name, m in self.base_model.named_modules() if hasattr(m, "conv") and len(getattr(m, "conv")) >= 4},
            decision_trace=[],
            infeasibility_reason=f"Uniform pruning alone cannot reach {self.target_budget_mb:.2f} MB without excessive structural damage."
        )


def apply_optimization_result(
    model: nn.Module,
    result: OptimizationResult
) -> nn.Module:
    """Instantiate the optimized model by applying the resulting pruning and quantization plans."""
    opt_model = copy.deepcopy(model)

    # 1. Apply structured pruning first (physically alters layer dimensions)
    if result.pruning_plan:
        opt_model = apply_structured_pruning(opt_model, result.pruning_plan)

    # 2. Apply simulated layer-wise quantization
    if result.quantization_plan:
        opt_model = apply_simulated_quantization(opt_model, result.quantization_plan)

    return opt_model
