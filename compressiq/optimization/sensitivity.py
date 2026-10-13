"""Sensitivity analysis module for CompressIQ.

Measures layer-by-layer tolerance to quantization and structured pruning:
- Evaluates individual layer degradation on validation loss and Top-1 accuracy.
- Computes sensitivity scores (Degradation / Memory Saved).
- Produces layer sensitivity rankings and structured profiles for budget-aware optimization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compressiq.compression.quantization import SimulatedQuantizedConv2d, SimulatedQuantizedLinear
from compressiq.compression.pruning import prune_inverted_residual_block, compute_filter_l1_norms, select_retained_channels, prune_conv_out_channels
from compressiq.evaluation.metrics import evaluate_accuracy, compute_model_size_mb, count_parameters


@dataclass
class LayerSensitivityRecord:
    layer_name: str
    compression_type: str  # 'quantization' or 'pruning'
    setting_value: Any  # bit width (int) or prune ratio (float)
    val_loss: float
    loss_increase: float
    val_acc: float
    acc_drop: float
    layer_size_mb: float
    mb_saved: float
    sensitivity_score: float  # loss_increase / max(mb_saved, 1e-5)


@dataclass
class SensitivityProfile:
    baseline_loss: float
    baseline_acc: float
    baseline_size_mb: float
    records: List[LayerSensitivityRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_loss": self.baseline_loss,
            "baseline_acc": self.baseline_acc,
            "baseline_size_mb": self.baseline_size_mb,
            "records": [asdict(r) for r in self.records]
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SensitivityProfile":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [LayerSensitivityRecord(**r) for r in data["records"]]
        return cls(
            baseline_loss=data["baseline_loss"],
            baseline_acc=data["baseline_acc"],
            baseline_size_mb=data["baseline_size_mb"],
            records=records
        )


def analyze_layer_sensitivity(
    model: nn.Module,
    val_loader: DataLoader,
    candidate_bits: List[int] = [8, 4, 2],
    candidate_prune_ratios: List[float] = [0.15, 0.30, 0.45],
    device: str = "cpu",
    max_eval_batches: int = 20
) -> SensitivityProfile:
    """Perform comprehensive per-layer sensitivity profiling under candidate compression settings.
    
    Args:
        model: Base uncompressed model.
        val_loader: Validation DataLoader for loss/accuracy evaluation.
        candidate_bits: Candidate bit-widths to test per layer.
        candidate_prune_ratios: Candidate pruning ratios to test per block.
        device: Device to run evaluation on.
        max_eval_batches: Number of batches to evaluate for speed during calibration.
        
    Returns:
        SensitivityProfile containing baseline metrics and per-layer records.
    """
    model.eval()
    model.to(device)

    # 1. Baseline evaluation
    base_metrics = evaluate_accuracy(model, val_loader, device=device, max_batches=max_eval_batches)
    base_loss = base_metrics["loss"]
    base_acc = base_metrics["top1_acc"]
    base_size_mb = compute_model_size_mb(model)

    records: List[LayerSensitivityRecord] = []

    # 2. Sensitivity to Quantization
    quant_targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for idx, (name, module) in enumerate(quant_targets, start=1):
        parent, attr = _get_parent_and_attr(model, name)
        orig_layer = getattr(parent, attr)

        for bits in candidate_bits:
            if isinstance(module, nn.Conv2d):
                quant_layer = SimulatedQuantizedConv2d(module, num_bits=bits)
            else:
                quant_layer = SimulatedQuantizedLinear(module, num_bits=bits)
            
            # Swap in
            setattr(parent, attr, quant_layer)
            
            # Evaluate
            metrics = evaluate_accuracy(model, val_loader, device=device, max_batches=max_eval_batches)
            test_size_mb = compute_model_size_mb(model)

            # Restore original layer immediately
            setattr(parent, attr, orig_layer)

            loss_inc = max(0.0, metrics["loss"] - base_loss)
            acc_drop = max(0.0, base_acc - metrics["top1_acc"])
            mb_saved = max(0.0, base_size_mb - test_size_mb)
            score = loss_inc / max(mb_saved, 1e-5)

            records.append(LayerSensitivityRecord(
                layer_name=name,
                compression_type="quantization",
                setting_value=bits,
                val_loss=round(metrics["loss"], 4),
                loss_increase=round(loss_inc, 4),
                val_acc=round(metrics["top1_acc"], 2),
                acc_drop=round(acc_drop, 2),
                layer_size_mb=round(test_size_mb, 4),
                mb_saved=round(mb_saved, 4),
                sensitivity_score=round(score, 4)
            ))

    # 3. Sensitivity to Structured Pruning
    prune_targets = [(n, m) for n, m in model.named_modules() if hasattr(m, "conv") and len(getattr(m, "conv")) >= 4]
    for idx, (name, module) in enumerate(prune_targets, start=1):
        parent, attr = _get_parent_and_attr(model, name)
        orig_block = getattr(parent, attr)

        for ratio in candidate_prune_ratios:
            # Deepcopy only this single block
            test_block = copy.deepcopy(orig_block)
            prune_inverted_residual_block(test_block, prune_ratio=ratio)
            
            # Swap in
            setattr(parent, attr, test_block)

            metrics = evaluate_accuracy(model, val_loader, device=device, max_batches=max_eval_batches)
            test_size_mb = compute_model_size_mb(model)

            # Restore original block
            setattr(parent, attr, orig_block)

            loss_inc = max(0.0, metrics["loss"] - base_loss)
            acc_drop = max(0.0, base_acc - metrics["top1_acc"])
            mb_saved = max(0.0, base_size_mb - test_size_mb)
            score = loss_inc / max(mb_saved, 1e-5)

            records.append(LayerSensitivityRecord(
                layer_name=name,
                compression_type="pruning",
                setting_value=ratio,
                val_loss=round(metrics["loss"], 4),
                loss_increase=round(loss_inc, 4),
                val_acc=round(metrics["top1_acc"], 2),
                acc_drop=round(acc_drop, 2),
                layer_size_mb=round(test_size_mb, 4),
                mb_saved=round(mb_saved, 4),
                sensitivity_score=round(score, 4)
            ))

    return SensitivityProfile(
        baseline_loss=round(base_loss, 4),
        baseline_acc=round(base_acc, 2),
        baseline_size_mb=round(base_size_mb, 4),
        records=records
    )


def _get_parent_and_attr(root: nn.Module, target_name: str) -> Tuple[nn.Module, str]:
    parts = target_name.split(".")
    curr = root
    for p in parts[:-1]:
        curr = getattr(curr, p)
    return curr, parts[-1]
