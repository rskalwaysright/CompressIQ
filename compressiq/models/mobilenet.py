"""MobileNetV2 architecture configuration and CIFAR-100 adaptation."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


class CIFAR100MobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR-100 classification.
    
    Optionally adapts the initial convolution stride for low-resolution 32x32 inputs
    so that spatial feature maps do not collapse prematurely.
    """
    def __init__(self, num_classes: int = 100, pretrained: bool = True, adapt_cifar: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v2(weights=weights)

        self.features = base_model.features
        if adapt_cifar:
            # Modify first conv: stride 1 instead of 2 to preserve spatial dimensions for 32x32 images
            first_conv = self.features[0][0]
            if first_conv.stride == (2, 2):
                new_conv = nn.Conv2d(
                    in_channels=first_conv.in_channels,
                    out_channels=first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=1,
                    padding=first_conv.padding,
                    bias=first_conv.bias is not None
                )
                if pretrained:
                    new_conv.weight.data.copy_(first_conv.weight.data)
                self.features[0][0] = new_conv

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_mobilenet_v2(num_classes: int = 100, pretrained: bool = True, adapt_cifar: bool = True) -> nn.Module:
    """Factory for CIFAR-100 MobileNetV2."""
    return CIFAR100MobileNetV2(num_classes=num_classes, pretrained=pretrained, adapt_cifar=adapt_cifar)
