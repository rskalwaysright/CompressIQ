"""Default configurations for CompressIQ experiments."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

from compressiq.configs.schema import (
    CompressIQConfig,
    DataConfig,
    ModelConfig,
    CompressionCandidateConfig,
    OptimizationConfig,
    EvaluationConfig
)


def get_default_config() -> CompressIQConfig:
    """Return default experiment configuration."""
    return CompressIQConfig(
        data=DataConfig(
            dataset_name="cifar100",
            data_dir="data",
            batch_size=64,
            eval_batch_size=128,
            val_split=0.1,
            seed=42,
        ),
        model=ModelConfig(
            architecture="mobilenet_v2",
            num_classes=100,
            pretrained=True,
        ),
        candidates=CompressionCandidateConfig(
            candidate_precisions=[8, 4, 2],
            candidate_prune_ratios=[0.0, 0.15, 0.30, 0.45],
        ),
        optimization=OptimizationConfig(
            target_budget_mb=4.0,  # ~3.5x to 4x compression from ~14 MB baseline
            search_strategy="greedy_sensitivity",
            metric="val_loss",
            calibration_batches=20,
        ),
        evaluation=EvaluationConfig(
            num_latency_warmup=10,
            num_latency_runs=50,
            device="auto",
        ),
        seed=42,
        output_dir="outputs",
    )


def save_config(config: CompressIQConfig, filepath: str) -> None:
    """Serialize config to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def load_config(filepath: str) -> CompressIQConfig:
    """Load config from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CompressIQConfig(
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        candidates=CompressionCandidateConfig(**data.get("candidates", {})),
        optimization=OptimizationConfig(**data.get("optimization", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        seed=data.get("seed", 42),
        output_dir=data.get("output_dir", "outputs"),
    )
