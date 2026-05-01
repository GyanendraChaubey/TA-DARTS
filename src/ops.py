"""
Primitive operations for the 7-op search space, plus MixedOp.

Search space:  MBConv3x3 | MBConv5x5 | MBConvSE | DilatedConv3x3 | SepConv5x5
             | SkipConnect | Zero
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .utils import _make_divisible


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        ks:       int = 3,
        stride:   int = 1,
        groups:   int = 1,
        dilation: int = 1,
    ) -> None:
        pad = (ks - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=pad,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class MBConv(nn.Module):
    """MobileNetV3-style Inverted Residual block — requires C_in == C_out."""

    def __init__(
        self,
        channels:   int,
        kernel_size: int = 3,
        expansion:  int = 4,
        stride:     int = 1,
    ) -> None:
        super().__init__()
        hidden       = _make_divisible(channels * expansion)
        self.use_res = (stride == 1)
        layers: List[nn.Module] = []
        if expansion != 1:
            layers.append(_ConvBNReLU(channels, hidden, ks=1))
        layers.append(
            _ConvBNReLU(hidden, hidden, ks=kernel_size, stride=stride,
                        groups=hidden)
        )
        layers += [
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out + x if self.use_res else out


class MBConv3x3(MBConv):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=3, expansion=4)


class MBConv5x5(MBConv):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=5, expansion=6)


class DilatedConv3x3(nn.Module):
    """Depthwise-separable dilated conv, dilation=2  →  effective RF ≈ 5×5."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x) + x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(_make_divisible(channels // reduction), 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class MBConvSE(nn.Module):
    """MBConv3x3 with Squeeze-and-Excitation attention — best for texture tasks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = _make_divisible(channels * 4)
        self.conv = nn.Sequential(
            _ConvBNReLU(channels, hidden, ks=1),
            _ConvBNReLU(hidden, hidden, ks=3, groups=hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.se(self.conv(x)) + x


class SepConv5x5(nn.Module):
    """Depthwise-separable 5×5 conv — wider receptive field for skin lesions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x) + x


class SkipConnect(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Zero(nn.Module):
    """
    Drops this edge entirely — returns zeros_like so the alpha gradient
    for this op is always zero; the optimizer can freely push its weight
    to –∞ without producing NaN.
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


# ──────────────────────────────────────────────────────────────────────────────
# Search space registry
# ──────────────────────────────────────────────────────────────────────────────

OP_NAMES: List[str] = [
    "MBConv3x3",
    "MBConv5x5",
    "MBConvSE",
    "DilatedConv3x3",
    "SepConv5x5",
    "SkipConnect",
    "Zero",
]
NUM_OPS = len(OP_NAMES)  # 7

_OP_REGISTRY = {
    "MBConv3x3":      MBConv3x3,
    "MBConv5x5":      MBConv5x5,
    "MBConvSE":       MBConvSE,
    "DilatedConv3x3": DilatedConv3x3,
    "SepConv5x5":     SepConv5x5,
    "SkipConnect":    lambda c: SkipConnect(),
    "Zero":           lambda c: Zero(),
}


# ──────────────────────────────────────────────────────────────────────────────
# MixedOp
# ──────────────────────────────────────────────────────────────────────────────

class MixedOp(nn.Module):
    """
    Differentiable mixture:  output = Σ_i sparsemax(α)_i · op_i(x).

    All ops are evaluated eagerly so every op receives gradient signal
    regardless of its current weight magnitude.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ops = nn.ModuleList(
            [_OP_REGISTRY[name](channels) for name in OP_NAMES]
        )

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        return sum(w * op(x) for w, op in zip(weights, self.ops))
