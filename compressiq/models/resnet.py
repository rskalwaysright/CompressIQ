"""ResNet-18 architecture configuration and CIFAR-100 adaptation."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class CIFAR100ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-100 validation."""
    def __init__(self, num_classes: int = 100, pretrained: bool = True, adapt_cifar: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        if adapt_cifar:
            # Replace 7x7 conv with 3x3 conv stride 1 and remove maxpool for CIFAR 32x32 resolution
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if pretrained:
                # Initialize using center crop or mean of weights
                with torch.no_grad():
                    self.conv1.weight.copy_(base_model.conv1.weight[:, :, 2:5, 2:5])
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = nn.Identity()
        else:
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_resnet_18(num_classes: int = 100, pretrained: bool = True, adapt_cifar: bool = True) -> nn.Module:
    """Factory for CIFAR-100 ResNet-18."""
    return CIFAR100ResNet18(num_classes=num_classes, pretrained=pretrained, adapt_cifar=adapt_cifar)
