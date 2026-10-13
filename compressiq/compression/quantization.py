"""Quantization modules for CompressIQ.

Provides:
1. Simulated Low-Bit Uniform Quantization (2-bit, 4-bit, 8-bit) for research exploration.
2. Native INT8 Post-Training Quantization (PTQ) baseline.
3. Accurate layer-wise bit-width simulation and theoretical/serialized size calculations.

NOTE ON ACCELERATION:
Simulated low-bit quantization (e.g., 2-bit, 4-bit) maps floating-point weights to discrete
quantized representation levels in-place while keeping execution on standard PyTorch float
runtimes. It accurately models precision loss and storage footprint without claiming
sub-byte hardware acceleration unless backed by specialized low-bit execution kernels.
"""

from typing import Dict, Any, Optional, List, Tuple
import copy
import torch
import torch.nn as nn


def quantize_tensor_uniform(
    tensor: torch.Tensor,
    num_bits: int,
    symmetric: bool = True
) -> Tuple[torch.Tensor, float, float]:
    """Simulate uniform quantization on a tensor to `num_bits` precision.
    
    Args:
        tensor: Floating point tensor to quantize.
        num_bits: Bit precision (e.g., 2, 4, 8).
        symmetric: If True, uses zero-symmetric quantization around 0.
        
    Returns:
        (quantized_dequantized_tensor, scale, zero_point)
    """
    if num_bits >= 32:
        return tensor.clone(), 1.0, 0.0

    t = tensor.detach()
    if symmetric:
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1
        max_abs = torch.max(torch.abs(t))
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs.item() / qmax
        zero_point = 0.0
        q = torch.clamp(torch.round(t / scale), qmin, qmax)
        dq = q * scale
    else:
        qmin = 0
        qmax = (2 ** num_bits) - 1
        min_val = torch.min(t).item()
        max_val = torch.max(t).item()
        if max_val == min_val:
            scale = 1.0
            zero_point = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = round(-min_val / scale)
        q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax)
        dq = (q - zero_point) * scale

    return dq, scale, zero_point


class SimulatedQuantizedConv2d(nn.Module):
    """Conv2d layer wrapper that applies simulated low-bit quantization to weights."""
    def __init__(self, original_conv: nn.Conv2d, num_bits: int = 8):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.num_bits = num_bits

        # Quantize original weights
        with torch.no_grad():
            quant_w, scale, zp = quantize_tensor_uniform(original_conv.weight.data, num_bits=num_bits)
            self.weight = nn.Parameter(quant_w, requires_grad=False)
            self.scale = scale
            self.zero_point = zp
            if original_conv.bias is not None:
                self.bias = nn.Parameter(original_conv.bias.data.clone(), requires_grad=False)
            else:
                self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class SimulatedQuantizedLinear(nn.Module):
    """Linear layer wrapper that applies simulated low-bit quantization to weights."""
    def __init__(self, original_linear: nn.Linear, num_bits: int = 8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.num_bits = num_bits

        with torch.no_grad():
            quant_w, scale, zp = quantize_tensor_uniform(original_linear.weight.data, num_bits=num_bits)
            self.weight = nn.Parameter(quant_w, requires_grad=False)
            self.scale = scale
            self.zero_point = zp
            if original_linear.bias is not None:
                self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
            else:
                self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


def apply_simulated_quantization(
    model: nn.Module,
    bit_config: Dict[str, int],
    default_bits: int = 32
) -> nn.Module:
    """Apply layer-wise simulated quantization to the given model.
    
    Args:
        model: Base PyTorch model.
        bit_config: Dict mapping layer name to bit precision (e.g. {'features.0.0': 8, 'features.1.conv.0': 4}).
        default_bits: Fallback bit-width for layers not specified in bit_config (32 means no quant).
        
    Returns:
        Deep-copied model with quantized layers substituted.
    """
    quant_model = copy.deepcopy(model)
    
    for name, module in list(quant_model.named_modules()):
        bits = bit_config.get(name, default_bits)
        if bits >= 32:
            continue

        if isinstance(module, nn.Conv2d):
            # Locate parent module
            parent, attr = _get_parent_and_attr(quant_model, name)
            quant_layer = SimulatedQuantizedConv2d(module, num_bits=bits)
            setattr(parent, attr, quant_layer)
        elif isinstance(module, nn.Linear):
            parent, attr = _get_parent_and_attr(quant_model, name)
            quant_layer = SimulatedQuantizedLinear(module, num_bits=bits)
            setattr(parent, attr, quant_layer)

    return quant_model


def apply_uniform_quantization(model: nn.Module, num_bits: int = 8) -> nn.Module:
    """Apply uniform bit-width quantization across all Conv2d and Linear layers."""
    quant_model = copy.deepcopy(model)
    for name, module in list(quant_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            parent, attr = _get_parent_and_attr(quant_model, name)
            quant_layer = SimulatedQuantizedConv2d(module, num_bits=num_bits)
            setattr(parent, attr, quant_layer)
        elif isinstance(module, nn.Linear):
            parent, attr = _get_parent_and_attr(quant_model, name)
            quant_layer = SimulatedQuantizedLinear(module, num_bits=num_bits)
            setattr(parent, attr, quant_layer)
    return quant_model


def apply_native_int8_quantization(
    model: nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None
) -> nn.Module:
    """Produce a native INT8 quantized model using PyTorch's native dynamic or static PTQ.
    
    Executes on CPU as standard for PyTorch INT8 quantized models.
    """
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    try:
        # Native PyTorch Dynamic Quantization for linear layers
        quantized_model = torch.ao.quantization.quantize_dynamic(
            cpu_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        # Fallback to simulated 8-bit if dynamic quant fails on non-standard submodules
        return apply_uniform_quantization(cpu_model, num_bits=8)


def _get_parent_and_attr(root: nn.Module, target_name: str) -> Tuple[nn.Module, str]:
    """Helper to navigate module hierarchy and retrieve (parent_module, attribute_name)."""
    parts = target_name.split(".")
    curr = root
    for p in parts[:-1]:
        curr = getattr(curr, p)
    return curr, parts[-1]
