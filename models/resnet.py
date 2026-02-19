"""
ResNet-18 implementation from scratch for CIFAR-10.

No torchvision.models imports. Built from He et al., "Deep Residual Learning
for Image Recognition" (2015). Architecture: 4 residual groups [2,2,2,2],
batch normalization, skip connections with downsampling via 1x1 convolutions.

Target: >=93% clean accuracy on CIFAR-10 test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Residual block: two 3x3 conv layers with batch norm and skip connection.

    If dimensions change (stride > 1 or channel mismatch), the skip connection
    uses a 1x1 convolution to match dimensions.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: identity if dimensions match, 1x1 conv otherwise
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 for CIFAR-10 (32x32 input, 10 classes).

    Modified from ImageNet version:
    - First conv is 3x3 (not 7x7) since CIFAR images are small
    - No max pool after first conv
    - 4 residual groups with [2, 2, 2, 2] blocks
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.in_channels = 64

        # Initial convolution (3x3 for CIFAR, not 7x7 for ImageNet)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 residual groups
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)   # 32x32
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  # 16x16
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  # 8x8
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Kaiming initialization
        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-d feature vector (before FC layer). Used by CLIP-lite."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)
