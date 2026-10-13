"""Unit tests for Structured Pruning modules in CompressIQ."""

import pytest
import torch
import torch.nn as nn

from compressiq.compression.pruning import (
    compute_filter_l1_norms,
    select_retained_channels,
    prune_conv_out_channels,
    prune_bn_channels,
    prune_conv_in_channels,
    prune_inverted_residual_block
)
from torchvision.models.mobilenetv2 import InvertedResidual


def test_filter_l1_norms_and_selection():
    """Verify filter L1 norm computation and selection ordering."""
    conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    # Assign known weight magnitudes to filter 0 and filter 1
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0] = 5.0
        conv.weight[1] = 10.0

    norms = compute_filter_l1_norms(conv)
    assert norms.shape == (16,)
    assert norms[1] > norms[0]

    # Select top 50%
    retained = select_retained_channels(norms, prune_ratio=0.5, min_channels=4)
    assert len(retained) == 8
    assert 0 in retained
    assert 1 in retained


def test_physical_conv_pruning():
    """Verify that pruning reconstructs a genuinely smaller Conv2d layer."""
    conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    keep_indices = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])

    pruned_conv = prune_conv_out_channels(conv, keep_indices)
    assert pruned_conv.out_channels == 8
    assert pruned_conv.in_channels == 16
    assert pruned_conv.weight.shape == (8, 16, 3, 3)

    # Test forward pass
    x = torch.randn(2, 16, 10, 10)
    out = pruned_conv(x)
    assert out.shape == (2, 8, 10, 10)


def test_inverted_residual_physical_pruning():
    """Verify InvertedResidual block intermediate channel pruning and execution."""
    # InvertedResidual: inp=16, oup=24, stride=1, expand_ratio=6 (expansion out = 96)
    block = InvertedResidual(inp=16, oup=24, stride=1, expand_ratio=6)
    initial_params = sum(p.numel() for p in block.parameters())

    # Prune expansion channels by 40%
    pruned_block = prune_inverted_residual_block(block, prune_ratio=0.4, min_channels=8)
    pruned_params = sum(p.numel() for p in pruned_block.parameters())

    # Parameters must be strictly smaller
    assert pruned_params < initial_params

    # Check forward pass with residual connection intact
    x = torch.randn(2, 16, 16, 16)
    out = pruned_block(x)
    assert out.shape == (2, 24, 16, 16)
    assert not torch.isnan(out).any()
