"""Evaluation metrics: accuracy, parameter count, MACs/FLOPs, model size, and latency."""

import time
import io
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from compressiq.compression.quantization import SimulatedQuantizedConv2d, SimulatedQuantizedLinear


def count_parameters(model: nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def estimate_macs(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cpu"
) -> int:
    """Compute true theoretical multiply-accumulate operations (MACs).
    
    Traverses forward execution hooks for Conv2d, Linear, and BatchNorm layers.
    """
    total_macs = 0
    hooks = []

    def conv_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        # out shape: (N, C_out, H_out, W_out)
        out_shape = out.shape
        if len(out_shape) == 4:
            batch_size, out_c, out_h, out_w = out_shape
            # in_c_per_group
            in_c = module.in_channels // module.groups
            k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            macs = batch_size * out_c * out_h * out_w * (in_c * k_h * k_w)
            total_macs += macs

    def linear_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total_macs
        batch_size = inp[0].shape[0] if inp[0].dim() > 1 else 1
        macs = batch_size * module.in_features * module.out_features
        total_macs += macs

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, SimulatedQuantizedConv2d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, (nn.Linear, SimulatedQuantizedLinear)):
            hooks.append(mod.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    dummy_input = torch.zeros(input_size, device=device)
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                model.train()

    return total_macs


def compute_model_size_mb(
    model: nn.Module,
    bit_config: Optional[Dict[str, int]] = None
) -> float:
    """Compute theoretical and serialized model size in MB based on per-layer bit precision.
    
    For each parameter:
    - If layer is in bit_config or is a SimulatedQuantized module, weight bytes = numel * bits / 8.
    - Bias and batch norm parameters remain FP32 (32 bits = 4 bytes).
    - Unspecified layers default to FP32 (4 bytes).
    """
    total_bytes = 0.0
    bit_dict = bit_config or {}

    for name, module in model.named_modules():
        # Check if layer is simulated quantized
        bits = getattr(module, "num_bits", None) or bit_dict.get(name, 32)
        
        # Immediate parameters of this module (without recursing children)
        for param_name, param in module.named_parameters(recurse=False):
            if param_name == "weight":
                total_bytes += param.numel() * (bits / 8.0)
            else:
                total_bytes += param.numel() * 4.0  # FP32 bias or norm

    if total_bytes == 0.0:
        # Fallback to total param elements * 4
        total_bytes = count_parameters(model) * 4.0

    return total_bytes / (1024.0 * 1024.0)


def measure_disk_serialized_size_mb(model: nn.Module) -> float:
    """Measure serialized size in MB by saving state_dict to a BytesIO buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = buf.getbuffer().nbytes / (1024.0 * 1024.0)
    buf.close()
    return size_mb


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, float]:
    """Measure empirical inference latency in milliseconds."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    times_arr = np.array(times)
    return {
        "median_latency_ms": float(np.median(times_arr)),
        "p95_latency_ms": float(np.percentile(times_arr, 95)),
        "mean_latency_ms": float(np.mean(times_arr)),
        "std_latency_ms": float(np.std(times_arr)),
    }


def evaluate_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate Top-1 and Top-5 accuracy and average loss on a dataloader."""
    model.eval()
    model.to(device)
    crit = criterion or nn.CrossEntropyLoss()

    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for b_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = crit(outputs, targets)

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            # Top-1
            _, pred_top1 = outputs.max(1)
            total_correct_top1 += pred_top1.eq(targets).sum().item()

            # Top-5
            _, pred_top5 = outputs.topk(min(5, outputs.size(1)), 1, largest=True, sorted=True)
            total_correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()

    acc_top1 = (total_correct_top1 / total_samples) * 100.0 if total_samples > 0 else 0.0
    acc_top5 = (total_correct_top5 / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0

    return {
        "top1_acc": float(acc_top1),
        "top5_acc": float(acc_top5),
        "loss": float(avg_loss),
        "total_samples": int(total_samples)
    }


def compute_peak_memory_mb(device: str = "cpu") -> float:
    """Measure peak memory in MB if CUDA is available, else return 0.0."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return 0.0


def evaluate_model_full(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    baseline_metrics: Optional[Dict[str, Any]] = None,
    target_budget_mb: Optional[float] = None,
    bit_config: Optional[Dict[str, int]] = None,
    max_eval_batches: Optional[int] = None
) -> Dict[str, Any]:
    """Run full evaluation suite on a model."""
    acc_metrics = evaluate_accuracy(model, dataloader, device=device, max_batches=max_eval_batches)
    param_cnt = count_parameters(model)
    macs = estimate_macs(model, input_size=(1, 3, 32, 32), device=device)
    model_size_mb = compute_model_size_mb(model, bit_config=bit_config)
    latency_metrics = measure_inference_latency(model, input_size=(1, 3, 32, 32), device=device)
    peak_mem = compute_peak_memory_mb(device=device)

    res = {
        "top1_acc": acc_metrics["top1_acc"],
        "top5_acc": acc_metrics["top5_acc"],
        "loss": acc_metrics["loss"],
        "param_count": param_cnt,
        "macs": macs,
        "model_size_mb": round(model_size_mb, 3),
        "median_latency_ms": round(latency_metrics["median_latency_ms"], 2),
        "p95_latency_ms": round(latency_metrics["p95_latency_ms"], 2),
        "peak_memory_mb": round(peak_mem, 2),
    }

    if baseline_metrics:
        base_size = baseline_metrics.get("model_size_mb", model_size_mb)
        base_acc = baseline_metrics.get("top1_acc", res["top1_acc"])
        res["compression_ratio"] = round(base_size / max(model_size_mb, 1e-5), 2)
        res["accuracy_drop"] = round(base_acc - res["top1_acc"], 2)
    else:
        res["compression_ratio"] = 1.0
        res["accuracy_drop"] = 0.0

    if target_budget_mb is not None:
        res["target_budget_mb"] = target_budget_mb
        res["budget_satisfied"] = bool(model_size_mb <= target_budget_mb)

    return res
