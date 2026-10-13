"""Configuration schemas and data classes for CompressIQ."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    dataset_name: str = "cifar100"
    data_dir: str = "data"
    batch_size: int = 64
    eval_batch_size: int = 128
    val_split: float = 0.1  # 10% of 50,000 = 5,000 validation samples
    num_workers: int = 2
    seed: int = 42
    image_size: int = 32  # Standard CIFAR-100 resolution, resized or interpolated if required


@dataclass
class ModelConfig:
    architecture: str = "mobilenet_v2"  # 'mobilenet_v2' or 'resnet18'
    num_classes: int = 100
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class CompressionCandidateConfig:
    candidate_precisions: List[int] = field(default_factory=lambda: [8, 4, 2])
    candidate_prune_ratios: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.30, 0.45])


@dataclass
class OptimizationConfig:
    target_budget_mb: float = 4.0  # User-defined model size budget in MB
    search_strategy: str = "greedy_sensitivity"  # 'greedy_sensitivity', 'uniform_pruning', 'uniform_quant'
    metric: str = "val_loss"  # 'val_loss' or 'val_acc'
    calibration_batches: int = 20  # Batches to use for calibration/sensitivity evaluation


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "top1_acc", "model_size_mb", "compression_ratio",
        "param_count", "macs_flops", "median_latency_ms", "p95_latency_ms"
    ])
    num_latency_warmup: int = 10
    num_latency_runs: int = 50
    device: str = "auto"  # 'auto', 'cuda', 'cpu'


@dataclass
class CompressIQConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    candidates: CompressionCandidateConfig = field(default_factory=CompressionCandidateConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    output_dir: str = "outputs"
