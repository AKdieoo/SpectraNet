"""
Model zoo for RF spectrogram classification.

Includes:
  - CustomCNN        : lightweight from-scratch baseline
  - ResNet18          \
  - VGG16              |  torchvision backbones, first conv layer adapted
  - EfficientNetB0      >  to accept 1 or 2 channel spectrogram input
  - MobileNetV3Small   |  instead of 3-channel RGB, final FC re-headed
                       /   for num_classes

`build_model(name, ...)` is the single entry point used by the training
script and the benchmarking script so architectures stay swappable via
config alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# --------------------------------------------------------------------------- #
# Custom CNN baseline
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CustomCNN(nn.Module):
    """
    Small, fast, from-scratch CNN baseline purpose-built for RF spectrograms.
    Designed to be cheap enough for edge deployment while still serving as a
    meaningful accuracy baseline against the transfer-learned backbones.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.stem = ConvBlock(in_channels, width)
        self.layer1 = nn.Sequential(ConvBlock(width, width), ConvBlock(width, width * 2, stride=2))
        self.layer2 = nn.Sequential(ConvBlock(width * 2, width * 2), ConvBlock(width * 2, width * 4, stride=2))
        self.layer3 = nn.Sequential(ConvBlock(width * 4, width * 4), ConvBlock(width * 4, width * 8, stride=2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# Backbone adaptation helpers
# --------------------------------------------------------------------------- #

def _replace_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a conv layer with a different input channel count, keeping
    other hyperparameters (kernel, stride, padding, bias) identical, and
    average-initialize weights from the pretrained RGB filters so transfer
    learning still has a sane starting point."""
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        avg_weight = conv.weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
        new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def resnet18(in_channels: int = 1, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.conv1 = _replace_first_conv(model.conv1, in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def vgg16(in_channels: int = 1, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = tvm.VGG16_Weights.DEFAULT if pretrained else None
    model = tvm.vgg16(weights=weights)
    first_conv = model.features[0]
    model.features[0] = _replace_first_conv(first_conv, in_channels)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def efficientnet_b0(in_channels: int = 1, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    first_conv = model.features[0][0]
    model.features[0][0] = _replace_first_conv(first_conv, in_channels)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def mobilenet_v3_small(in_channels: int = 1, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)
    first_conv = model.features[0][0]
    model.features[0][0] = _replace_first_conv(first_conv, in_channels)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY = {
    "custom_cnn": CustomCNN,
    "resnet18": resnet18,
    "vgg16": vgg16,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v3_small": mobilenet_v3_small,
}


def build_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Single entry point for constructing any registered architecture.

    Example
    -------
    >>> model = build_model("resnet18", in_channels=1, num_classes=11)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    if name == "custom_cnn":
        return CustomCNN(in_channels=in_channels, num_classes=num_classes, **kwargs)

    builder = MODEL_REGISTRY[name]
    return builder(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
