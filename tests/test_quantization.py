"""Unit tests for Quantization modules in CompressIQ."""

import pytest
import torch
import torch.nn as nn

from compressiq.compression.quantization import (
    quantize_tensor_uniform,
    SimulatedQuantizedConv2d,
    SimulatedQuantizedLinear,
    apply_simulated_quantization,
    apply_uniform_quantization
)


def test_quantize_tensor_uniform_properties():
    """Verify uniform quantization preserves shape and bounds."""
    tensor = torch.randn(10, 10) * 5.0
    for bits in [8, 4, 2]:
        q_dq, scale, zp = quantize_tensor_uniform(tensor, num_bits=bits, symmetric=True)
        assert q_dq.shape == tensor.shape
        assert scale > 0.0
        # Quantized values should have at most 2^bits unique levels
        unique_levels = torch.unique(torch.round(tensor / scale)).numel()
        assert unique_levels <= (2 ** bits) + 1


def test_simulated_quantized_conv2d_forward():
    """Verify SimulatedQuantizedConv2d executes forward pass accurately."""
    conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    quant_conv = SimulatedQuantizedConv2d(conv, num_bits=4)

    dummy_input = torch.randn(2, 16, 8, 8)
    out = quant_conv(dummy_input)

    assert out.shape == (2, 32, 8, 8)
    assert not torch.isnan(out).any()
    assert quant_conv.num_bits == 4


def test_simulated_quantized_linear_forward():
    """Verify SimulatedQuantizedLinear executes forward pass accurately."""
    linear = nn.Linear(in_features=64, out_features=10)
    quant_linear = SimulatedQuantizedLinear(linear, num_bits=8)

    dummy_input = torch.randn(4, 64)
    out = quant_linear(dummy_input)

    assert out.shape == (4, 10)
    assert not torch.isnan(out).any()
    assert quant_linear.num_bits == 8


def test_apply_simulated_quantization_model():
    """Verify recursive substitution on a neural network."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 5)
    )
    bit_plan = {"0": 4, "5": 8}
    q_model = apply_simulated_quantization(model, bit_plan)

    assert isinstance(q_model[0], SimulatedQuantizedConv2d)
    assert q_model[0].num_bits == 4
    assert isinstance(q_model[5], SimulatedQuantizedLinear)
    assert q_model[5].num_bits == 8

    # Forward pass verification
    out = q_model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 5)
