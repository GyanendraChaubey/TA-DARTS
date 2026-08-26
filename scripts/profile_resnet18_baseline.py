"""
Independently profile the ResNet-18(28) baseline referenced in
src/reporting.py (RESNET18_PARAMS, RESNET18_FLOPS_M), using the same thop
profiler used for our own models, instead of trusting the hardcoded literals.

Architecture: the small-image ResNet-18 adaptation used by the MedMNIST v2
benchmark paper (3x3 stride-1 stem, no initial maxpool — a standard 7x7
stride-2 stem + maxpool would collapse a 28x28 input to ~1x1 before any
residual stage runs). BasicBlock x2 per stage, channels [64,128,256,512],
strides [1,2,2,2].

Profiles at both 28x28 (the paper's resolution, to check the hardcoded
constants) and 64x64 (our pipeline's resolution, for a same-resolution
efficiency comparison against our searched architectures).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from thop import profile as thop_profile
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet18Small(nn.Module):
    """ResNet-18 adapted for small (28px-class) inputs: 3x3 stride-1 stem,
    no maxpool — matches the MedMNIST v2 benchmark's ResNet-18(28) baseline."""

    def __init__(self, in_channels: int = 3, num_classes: int = 9) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet50Small(nn.Module):
    """ResNet-50 adapted for small (28px-class) inputs — same 3x3 stride-1
    stem / no-maxpool adaptation as ResNet18Small, with Bottleneck blocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 9) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = planes * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = out_planes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


HARDCODED_PARAMS   = 11_200_000
HARDCODED_FLOPS_M  = 1_820.0

if __name__ == "__main__":
    for arch_name, arch_cls in [("ResNet18Small", ResNet18Small), ("ResNet50Small", ResNet50Small)]:
        for num_classes, task in [(9, "PathMNIST"), (14, "ChestMNIST"), (7, "DermaMNIST")]:
            model = arch_cls(num_classes=num_classes).eval()
            n_params = sum(p.numel() for p in model.parameters())
            for size in (28, 64):
                dummy = torch.zeros(1, 3, size, size)
                flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
                print(f"{arch_name:14s} {task:12s} nc={num_classes:2d}  {size}px  "
                      f"params={n_params:,}  flops_m={flops/1e6:.2f}")
    print()
    print(f"Hardcoded RESNET18_PARAMS  = {HARDCODED_PARAMS:,}")
    print(f"Hardcoded RESNET18_FLOPS_M = {HARDCODED_FLOPS_M}")
