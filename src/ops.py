"""
Primitive operations for the 10-op search space, plus MixedOp.

Search space:  MBConv3x3 | MBConv5x5 | MBConvSE | DilatedConv3x3 | DilatedConv5x5
             | SepConv3x3 | SepConv5x5 | ResidualBN | AvgPool3x3 | MaxPool3x3

Note on ResidualBN (replaces SkipConnect):
  Pure identity (SkipConnect) caused NAS collapse — the search consistently
  assigned near-all weight to skip ops because they never increase training loss.
  ResidualBN = identity + BatchNorm preserves the near-skip shortcut path while
  forcing the NAS to learn a meaningful normalisation at every layer, breaking
  the zero-cost incentive that drives skip dominance.

Note: Zero was removed from the search space. In a sequential chain architecture
a Zero op kills all downstream gradient flow; AvgPool3x3 is the replacement.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
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
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


class DilatedConv5x5(nn.Module):
    """Depthwise-separable dilated 5×5 conv, dilation=2  →  effective RF ≈ 9×9."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=4, dilation=2,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


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
        out = self.se(self.conv(x))
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


class SepConv3x3(nn.Module):
    """Depthwise-separable 3×3 conv — efficient local feature extraction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


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
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


class ResidualBN(nn.Module):
    """
    Identity shortcut + BatchNorm (replaces SkipConnect).

    Pure SkipConnect has zero training cost, so sparsemax always collapses
    toward it regardless of task.  Adding BatchNorm introduces learnable
    parameters (γ, β) and running statistics that must be optimised, giving
    the search a real signal to weigh this op against others.

    Forward: out = BN(x) + x  (residual form keeps the shortcut path open)
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x) + x


class AvgPool3x3(nn.Module):
    """3×3 average pooling — smooth spatial feature aggregation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


class MaxPool3x3(nn.Module):
    """3×3 max pooling — preserves dominant activations and sharp features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.op(x)
        if hasattr(self, 'drop_prob'):
            out = drop_path(out, self.drop_prob, self.training)
        return out + x


# ──────────────────────────────────────────────────────────────────────────────
# Search space registry
# ──────────────────────────────────────────────────────────────────────────────

OP_NAMES: List[str] = [
    "MBConv3x3",
    "MBConv5x5",
    "MBConvSE",
    "DilatedConv3x3",
    "DilatedConv5x5",
    "SepConv3x3",
    "SepConv5x5",
    "ResidualBN",      # replaces SkipConnect — see module docstring
    "AvgPool3x3",
    "MaxPool3x3",
]
NUM_OPS = len(OP_NAMES)  # 10

_OP_REGISTRY = {
    "MBConv3x3":      MBConv3x3,
    "MBConv5x5":      MBConv5x5,
    "MBConvSE":       MBConvSE,
    "DilatedConv3x3": DilatedConv3x3,
    "DilatedConv5x5": DilatedConv5x5,
    "SepConv3x3":     SepConv3x3,
    "SepConv5x5":     SepConv5x5,
    "ResidualBN":     ResidualBN,
    "AvgPool3x3":     AvgPool3x3,
    "MaxPool3x3":     MaxPool3x3,
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
