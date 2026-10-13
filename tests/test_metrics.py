"""Unit tests for Metrics and evaluation routines in CompressIQ."""

import pytest
import torch
import torch.nn as nn

from compressiq.evaluation.metrics import (
    count_parameters,
    compute_model_size_mb,
    estimate_macs,
    measure_inference_latency
)


def test_metrics_on_linear_model():
    """Verify parameter count, model size, and MAC calculations."""
    # Simple linear model: 100 in -> 50 out -> 10 out
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.Linear(50, 10)
    )
    # Params: (100*50 + 50) + (50*10 + 10) = 5050 + 510 = 5560
    params = count_parameters(model)
    assert params == 5560

    # FP32 size: 5560 * 4 / (1024*1024) = 0.0212 MB
    size_mb = compute_model_size_mb(model)
    assert 0.020 < size_mb < 0.025

    # 8-bit size: 5560 bytes / 1024^2 ~= 0.0053 MB
    size_8bit = compute_model_size_mb(model, bit_config={"0": 8, "1": 8})
    assert size_8bit < size_mb
    assert 0.004 < size_8bit < 0.008

    # MACs: (100*50) + (50*10) = 5500
    macs = estimate_macs(model, input_size=(1, 100))
    assert macs == 5500


def test_latency_measurement():
    """Verify latency benchmark returns expected keys and valid values."""
    model = nn.Sequential(nn.Linear(32, 10))
    lat = measure_inference_latency(model, input_size=(1, 32), device="cpu", num_warmup=2, num_runs=5)

    assert "median_latency_ms" in lat
    assert "p95_latency_ms" in lat
    assert lat["median_latency_ms"] >= 0.0
    assert lat["p95_latency_ms"] >= lat["median_latency_ms"]
