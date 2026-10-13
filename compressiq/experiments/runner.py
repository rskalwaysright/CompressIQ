"""Experiment runner and pipeline orchestration for CompressIQ.

Executes the end-to-end benchmark suite:
1. FP32 Baseline evaluation & checkpointing.
2. Per-layer sensitivity profiling & metric recording.
3. Budget-aware sensitivity-guided greedy optimization.
4. Uniform baseline comparisons (Uniform Quant, Uniform Pruning, Uniform Hybrid, Native INT8).
5. Inference validation on exported/reconstructed models.
6. Generation of research tables and high-resolution plots.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compressiq.configs.schema import CompressIQConfig
from compressiq.configs.default_config import get_default_config
from compressiq.data.dataset import get_cifar100_dataloaders
from compressiq.models.factory import build_model, save_checkpoint
from compressiq.compression.quantization import (
    apply_uniform_quantization,
    apply_native_int8_quantization,
    apply_simulated_quantization
)
from compressiq.compression.pruning import apply_uniform_pruning, apply_structured_pruning
from compressiq.optimization.sensitivity import analyze_layer_sensitivity, SensitivityProfile
from compressiq.optimization.budget_optimizer import (
    BudgetAwareOptimizer,
    OptimizationResult,
    apply_optimization_result
)
from compressiq.evaluation.metrics import evaluate_model_full
from compressiq.evaluation.plots import (
    plot_accuracy_vs_size,
    plot_accuracy_vs_latency,
    plot_compression_vs_acc_drop,
    plot_layer_sensitivity_bars,
    plot_ablation_summary
)


def run_benchmark_pipeline(
    config: Optional[CompressIQConfig] = None,
    target_budget_mb: Optional[float] = None,
    max_eval_batches: int = 25,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Execute end-to-end compression and optimization experiment pipeline."""
    cfg = config or get_default_config()
    budget = target_budget_mb if target_budget_mb is not None else cfg.optimization.target_budget_mb

    device = "cuda" if (cfg.evaluation.device == "cuda" or (cfg.evaluation.device == "auto" and torch.cuda.is_available())) else "cpu"
    
    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    res_dir = out_dir / "results"
    for d in [ckpt_dir, plot_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def log(msg: str, step: float = 0.0):
        if progress_callback:
            progress_callback(msg, step)
        print(f"[CompressIQ] {msg}", flush=True)

    # Step 1: Dataloaders
    log("Loading CIFAR-100 dataset splits...", 0.05)
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(cfg.data)

    # Step 2: Build FP32 Baseline
    log(f"Building FP32 baseline ({cfg.model.architecture})...", 0.15)
    base_model = build_model(cfg.model, device=device)
    save_checkpoint(base_model, str(ckpt_dir / "baseline_fp32.pt"), {"architecture": cfg.model.architecture, "type": "fp32"})

    log("Evaluating FP32 baseline metrics...", 0.25)
    base_metrics = evaluate_model_full(
        base_model,
        test_loader,
        device=device,
        target_budget_mb=budget,
        max_eval_batches=max_eval_batches
    )
    base_metrics["Method"] = "FP32 Baseline"
    base_metrics["Type"] = "FP32 Uncompressed"

    # Step 3: Sensitivity Profiling
    log("Executing per-layer sensitivity profiling...", 0.40)
    profile = analyze_layer_sensitivity(
        base_model,
        val_loader,
        candidate_bits=cfg.candidates.candidate_precisions,
        candidate_prune_ratios=cfg.candidates.candidate_prune_ratios,
        device=device,
        max_eval_batches=max_eval_batches
    )
    profile.save(str(res_dir / "sensitivity_profile.json"))

    # Step 4: Budget-Aware Greedy Search
    log(f"Running deterministic budget-aware optimizer for budget <= {budget:.2f} MB...", 0.60)
    optimizer = BudgetAwareOptimizer(base_model, profile, target_budget_mb=budget)
    opt_result = optimizer.optimize_greedy_sensitivity()
    opt_result.save(str(res_dir / "optimization_result.json"))

    # Instantiate optimized model
    opt_model = apply_optimization_result(base_model, opt_result)
    save_checkpoint(opt_model, str(ckpt_dir / "sensitivity_guided_opt.pt"), opt_result.to_dict())

    # Step 5: Evaluate Proposed Method
    log("Evaluating sensitivity-guided compressed model...", 0.70)
    opt_metrics = evaluate_model_full(
        opt_model,
        test_loader,
        device=device,
        baseline_metrics=base_metrics,
        target_budget_mb=budget,
        bit_config=opt_result.quantization_plan,
        max_eval_batches=max_eval_batches
    )
    opt_metrics["Method"] = "Sensitivity-Guided (Proposed)"
    opt_metrics["Type"] = "Layer-wise Mixed Precision + Pruning"

    # Step 6: Uniform Baselines for honest comparison
    log("Evaluating comparison baselines (Uniform INT8, Uniform Pruning, Uniform Hybrid, Native INT8)...", 0.80)

    # 6a: Uniform INT8
    u_quant_model = apply_uniform_quantization(base_model, num_bits=8)
    u_quant_metrics = evaluate_model_full(
        u_quant_model,
        test_loader,
        device=device,
        baseline_metrics=base_metrics,
        target_budget_mb=budget,
        bit_config={n: 8 for n, _ in base_model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))},
        max_eval_batches=max_eval_batches
    )
    u_quant_metrics["Method"] = "Uniform INT8"
    u_quant_metrics["Type"] = "Simulated Uniform 8-bit"

    # 6b: Uniform Pruning (30%)
    u_prune_model = apply_uniform_pruning(base_model, prune_ratio=0.30)
    u_prune_metrics = evaluate_model_full(
        u_prune_model,
        test_loader,
        device=device,
        baseline_metrics=base_metrics,
        target_budget_mb=budget,
        max_eval_batches=max_eval_batches
    )
    u_prune_metrics["Method"] = "Uniform Pruning"
    u_prune_metrics["Type"] = "Structured Filter Pruning (30%)"

    # 6c: Uniform Hybrid (Uniform 8-bit + 15% Pruning)
    u_hybrid_model = apply_uniform_quantization(
        apply_uniform_pruning(base_model, prune_ratio=0.15),
        num_bits=8
    )
    u_hybrid_metrics = evaluate_model_full(
        u_hybrid_model,
        test_loader,
        device=device,
        baseline_metrics=base_metrics,
        target_budget_mb=budget,
        bit_config={n: 8 for n, _ in base_model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))},
        max_eval_batches=max_eval_batches
    )
    u_hybrid_metrics["Method"] = "Uniform Hybrid"
    u_hybrid_metrics["Type"] = "Uniform 8-bit + 15% Prune"

    # 6d: Native INT8 PTQ
    native_int8_model = apply_native_int8_quantization(base_model)
    native_int8_metrics = evaluate_model_full(
        native_int8_model,
        test_loader,
        device="cpu",  # Native PyTorch INT8 dynamic quant runs on CPU
        baseline_metrics=base_metrics,
        target_budget_mb=budget,
        max_eval_batches=max_eval_batches
    )
    native_int8_metrics["Method"] = "Native INT8 PTQ"
    native_int8_metrics["Type"] = "PyTorch Dynamic Quant (CPU)"

    # Step 7: Inference Validation
    log("Validating inference execution across all exported models...", 0.90)
    test_dummy = torch.randn((2, 3, 32, 32), device=device)
    validation_status = {}
    for name, m in [
        ("FP32", base_model),
        ("Sensitivity-Guided", opt_model),
        ("Uniform-INT8", u_quant_model),
        ("Uniform-Pruning", u_prune_model),
        ("Uniform-Hybrid", u_hybrid_model),
    ]:
        try:
            m.eval()
            out = m(test_dummy)
            assert out.shape == (2, cfg.model.num_classes), f"Output shape mismatch: {out.shape}"
            assert not torch.isnan(out).any(), "NaN values detected in output"
            validation_status[name] = "PASSED"
        except Exception as e:
            validation_status[name] = f"FAILED: {str(e)}"

    # Native INT8 test on CPU
    try:
        native_int8_model.eval()
        out_n = native_int8_model(torch.randn((2, 3, 32, 32), device="cpu"))
        assert out_n.shape == (2, cfg.model.num_classes)
        validation_status["Native-INT8"] = "PASSED"
    except Exception as e:
        validation_status["Native-INT8"] = f"FAILED: {str(e)}"

    # Step 8: Compile Results Table
    all_runs = [
        base_metrics,
        u_quant_metrics,
        u_prune_metrics,
        u_hybrid_metrics,
        native_int8_metrics,
        opt_metrics
    ]
    results_df = pd.DataFrame(all_runs)
    
    # Save results
    results_df.to_csv(str(res_dir / "benchmark_summary.csv"), index=False)
    with open(res_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    # Step 9: Render Plots
    log("Rendering publication figures...", 0.95)
    plot_accuracy_vs_size(results_df, target_budget_mb=budget, save_path=str(plot_dir / "acc_vs_size.png"))
    plot_accuracy_vs_latency(results_df, save_path=str(plot_dir / "acc_vs_latency.png"))
    plot_compression_vs_acc_drop(results_df, save_path=str(plot_dir / "compression_vs_drop.png"))
    plot_layer_sensitivity_bars(profile.to_dict(), save_path=str(plot_dir / "layer_sensitivity.png"))
    plot_ablation_summary(results_df, save_path=str(plot_dir / "ablation_summary.png"))

    log("Benchmark pipeline completed successfully!", 1.0)

    return {
        "results_df": results_df,
        "summary": all_runs,
        "profile": profile.to_dict(),
        "optimization_result": opt_result.to_dict(),
        "validation_status": validation_status,
        "target_budget_mb": budget,
        "device": device
    }
