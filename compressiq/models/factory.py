"""Model construction, checkpoint loading and layer inspection utilities."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn

from compressiq.configs.schema import ModelConfig
from compressiq.models.mobilenet import get_mobilenet_v2
from compressiq.models.resnet import get_resnet_18


def build_model(
    config: Optional[ModelConfig] = None,
    architecture: str = "mobilenet_v2",
    num_classes: int = 100,
    pretrained: bool = True,
    device: str = "cpu"
) -> nn.Module:
    """Build classifier model based on architecture name or ModelConfig."""
    arch = config.architecture if config else architecture
    classes = config.num_classes if config else num_classes
    pretrain = config.pretrained if config else pretrained

    if arch.lower() in ["mobilenet_v2", "mobilenetv2", "mobilenet"]:
        model = get_mobilenet_v2(num_classes=classes, pretrained=pretrain)
    elif arch.lower() in ["resnet18", "resnet_18"]:
        model = get_resnet_18(num_classes=classes, pretrained=pretrain)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}. Choose 'mobilenet_v2' or 'resnet18'.")

    if config and config.checkpoint_path and Path(config.checkpoint_path).exists():
        state_dict = torch.load(config.checkpoint_path, map_location=device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    model = model.to(device)
    return model


def save_checkpoint(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save model weights and metadata to file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    torch.save(payload, str(path))


def load_checkpoint(
    filepath: str,
    model: Optional[nn.Module] = None,
    device: str = "cpu"
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """Load model weights and metadata from checkpoint."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    payload = torch.load(str(path), map_location=device, weights_only=False)
    state_dict = payload.get("state_dict", payload)
    metadata = payload.get("metadata", {})

    if model is not None:
        model.load_state_dict(state_dict)
        model.to(device)
        return model, metadata
    return None, metadata


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """Return all standard 2D convolutional layers eligible for structured channel pruning.
    
    Excludes depthwise convolutions (groups > 1 and groups == in_channels) from direct
    independent output channel pruning because depthwise filters are tightly bound to input channels.
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Normal convolution or pointwise conv (not depthwise)
            if module.groups == 1 and module.out_channels > 4:
                prunable.append((name, module))
    return prunable
