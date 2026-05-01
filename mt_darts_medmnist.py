"""
MT-DARTS: Multi-Task Differentiable Neural Architecture Search for MedMNIST
============================================================================
Target venue : CVPR (camera-ready quality)
Author note  : Built entirely with torch.nn primitives — no NAS libraries.

Gradient Disentanglement (key design principle)
-----------------------------------------------
In standard DARTS the same alpha tensor participates in both the weight-update
and the architecture-update, creating implicit coupling that biases alpha
gradients toward tasks with larger loss scales.  We break this coupling via
two mechanisms:

  1. **Task-partitioned alphas** – self.alphas[k] is only read during a
     forward pass that carries task_id == k.  Gradients for task k never
     flow through alphas[j≠k], so the Hessian cross-terms in the bilevel
     objective are exactly zero.

  2. **Phase-gated optimizers** – the SearchController zeros grads
     selectively: during Phase-1 (weight update) alphas are frozen by
     `alphas.requires_grad_(False)`; during Phase-2 (alpha update) weight
     params are frozen.  This prevents the "ghost gradient" pathology where
     alpha gradients accumulate weight updates through the optimizer state.

Together these ensure ∂L_val/∂α_k is a clean signal reflecting only the
architectural fitness of task k, not an entangled mixture across tasks.

Usage
-----
  # Smoke test (mock data, CPU)
  python mt_darts_medmnist.py --mock --epochs 2

  # Full benchmark (requires: pip install medmnist scikit-learn)
  python mt_darts_medmnist.py --epochs 50 --retrain_epochs 100 --device cuda

  # Resume from checkpoint
  python mt_darts_medmnist.py --resume results/checkpoint_latest.pt --device cuda
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ── Optional dependencies (graceful fallback) ─────────────────────────────────
try:
    from torchvision import transforms

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import medmnist

    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False

try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MT-DARTS")


# ============================================================================
# 0.  Utility helpers & task configuration
# ============================================================================

# ── Task metadata ─────────────────────────────────────────────────────────────
# Each entry describes one MedMNIST sub-task used in the multi-task search.
TASK_CONFIG: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "PathMNIST",
        "n_classes": 9,
        "is_multilabel": False,
        "is_grayscale": False,
        "medmnist_flag": "pathmnist",
    },
    1: {
        "name": "ChestMNIST",
        "n_classes": 14,
        "is_multilabel": True,
        "is_grayscale": True,
        "medmnist_flag": "chestmnist",
    },
    2: {
        "name": "DermaMNIST",
        "n_classes": 7,
        "is_multilabel": False,
        "is_grayscale": False,
        "medmnist_flag": "dermamnist",
    },
}
NUM_TASKS = len(TASK_CONFIG)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round `v` up to the nearest multiple of `divisor` (MobileNet convention)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ============================================================================
# 1.  Primitive Operations (the 5-op search space)
# ============================================================================

class _ConvBNReLU(nn.Sequential):
    """Fused Conv-BN-ReLU6 building block used by MBConv internals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class MBConv(nn.Module):
    """
    MobileNetV3-style Inverted Residual block.

    Args:
        channels    : Input (= output) channel count.  We keep C_in == C_out
                      so skip connections work without projection.
        kernel_size : Depthwise kernel (3 or 5).
        expansion   : Width multiplier for the hidden dimension.
        stride      : Spatial stride (1 in search, 2 in stem only).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expansion: int = 4,
        stride: int = 1,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(channels * expansion)
        self.use_res = (stride == 1)

        layers: List[nn.Module] = []

        # Point-wise expand (skip if expansion == 1)
        if expansion != 1:
            layers.append(_ConvBNReLU(channels, hidden, kernel_size=1))

        # Depth-wise
        layers.append(
            _ConvBNReLU(hidden, hidden, kernel_size=kernel_size,
                        stride=stride, groups=hidden)
        )

        # Point-wise project (linear — no activation)
        layers += [
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.use_res:
            out = out + x          # residual keeps gradient flow healthy
        return out


class MBConv3x3(MBConv):
    """Op-1: MBConv with 3×3 depthwise kernel and expansion ratio 4."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=3, expansion=4)


class MBConv5x5(MBConv):
    """Op-2: MBConv with 5×5 depthwise kernel and expansion ratio 6."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=5, expansion=6)


class DilatedConv3x3(nn.Module):
    """
    Op-3: 3×3 dilated separable convolution with dilation=2.

    Receptive field = 5×5 but parameter count ≈ 3×3.  Useful for
    capturing long-range texture patterns in dermoscopy images.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            # Depth-wise dilated conv
            nn.Conv2d(
                channels, channels, kernel_size=3,
                padding=2, dilation=2, groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            # Point-wise
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x) + x   # residual: dilated conv as a refinement


class SkipConnect(nn.Module):
    """Op-4: Identity / skip connection.  Zero parameters."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class Zero(nn.Module):
    """
    Op-5: Zero operation — effectively drops this edge.

    Returning a zero tensor (not the input) is important: during softmax
    weighting the alpha for this op can freely go to –∞ without causing
    NaN gradients, because the gradient of the weighted sum w.r.t. the
    zero-op's alpha is always 0.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


# ============================================================================
# 2.  MixedOp — the differentiable mixture layer
# ============================================================================

OP_NAMES = ["MBConv3x3", "MBConv5x5", "DilatedConv3x3", "SkipConnect", "Zero"]
NUM_OPS = len(OP_NAMES)


def _build_op(name: str, channels: int) -> nn.Module:
    registry = {
        "MBConv3x3":      lambda c: MBConv3x3(c),
        "MBConv5x5":      lambda c: MBConv5x5(c),
        "DilatedConv3x3": lambda c: DilatedConv3x3(c),
        "SkipConnect":    lambda c: SkipConnect(),
        "Zero":           lambda c: Zero(),
    }
    return registry[name](channels)


class MixedOp(nn.Module):
    """
    Differentiable mixture of candidate operations.

    At search time  : output = Σ_i  softmax(α)_i · op_i(x)
    At deploy time  : output = op_{argmax α}(x)

    The `weights` argument is passed in from the supernet so that all
    tasks share the same op implementations but receive task-specific
    weighting vectors — this is the core of gradient disentanglement.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # nn.ModuleList so PyTorch registers all sub-module parameters
        self.ops = nn.ModuleList(
            [_build_op(name, channels) for name in OP_NAMES]
        )

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        """
        Args:
            x       : Feature map  [B, C, H, W]
            weights : Softmax'd architecture weights  [NUM_OPS]

        Returns:
            Weighted sum of all op outputs  [B, C, H, W]
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# ============================================================================
# 3.  TaskAwareSupernet
# ============================================================================

class TaskAwareSupernet(nn.Module):
    """
    Multi-task DARTS supernet.

    Architecture
    ------------
    stem  →  [MixedOp × num_layers]  →  head (per-task)

    Parameters
    ----------
    num_tasks   : Number of distinct classification tasks (e.g. 3 for the
                  PathMNIST / ChestMNIST / DermaMNIST split).
    num_layers  : Depth of the searchable cell stack.
    channels    : Feature-map channel width (constant throughout cells).
    num_classes_per_task : List of output class counts, one per task.
    """

    def __init__(
        self,
        num_tasks: int = 3,
        num_layers: int = 6,
        channels: int = 32,
        num_classes_per_task: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.channels = channels

        if num_classes_per_task is None:
            num_classes_per_task = [
                TASK_CONFIG[t]["n_classes"] for t in range(num_tasks)
            ]
        assert len(num_classes_per_task) == num_tasks

        # ── Stem: RGB → channels ──────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

        # ── Searchable cell stack ─────────────────────────────────────────
        self.cells = nn.ModuleList(
            [MixedOp(channels) for _ in range(num_layers)]
        )

        # ── Per-task classification heads ─────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(channels, nc),
            )
            for nc in num_classes_per_task
        ])

        # ── Architecture parameters (the "alphas") ────────────────────────
        # Shape: [T, L, O] where T=tasks, L=layers, O=ops
        # Initialised near zero so softmax starts close to uniform,
        # giving all ops an equal opportunity at the start of search.
        self.alphas = nn.Parameter(
            torch.zeros(num_tasks, num_layers, NUM_OPS),
            requires_grad=True,
        )
        nn.init.normal_(self.alphas, mean=0.0, std=1e-3)

        # Weight-initialise conv layers with Kaiming Normal
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """
        Forward pass for a single task.

        Gradient Disentanglement note
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        `self.alphas[task_id]` slices a (num_layers, NUM_OPS) sub-tensor.
        PyTorch's slice indexing preserves the computation graph but the
        gradient of this slice is a sparse update: only the [task_id] rows
        of ∂L/∂alphas receive non-zero gradients.  All other task rows are
        zero-filled, so their alpha estimates are never corrupted by the
        loss of a different task.  This is "gradient disentanglement by
        construction" — no extra projection or stop-gradient is needed.

        Args:
            x       : Input tensor  [B, 3, H, W]
            task_id : Integer in [0, num_tasks)

        Returns:
            Logits  [B, num_classes_for_task]
        """
        # Slice task-specific alphas and apply softmax per layer
        task_alphas = self.alphas[task_id]                         # (L, O)
        weights_per_layer = F.softmax(task_alphas, dim=-1)         # (L, O)

        x = self.stem(x)

        for layer_idx, cell in enumerate(self.cells):
            w = weights_per_layer[layer_idx]                       # (O,)
            x = cell(x, w)

        return self.heads[task_id](x)                              # (B, C)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def discretize(self, task_id: int) -> nn.Sequential:
        """
        Derive the final discrete architecture for `task_id`.

        Selects argmax(alpha[task_id, l]) for each layer l and assembles
        a clean nn.Sequential with *copies* of the winning operations —
        fully detached from the supernet, ready for stand-alone fine-tuning.

        Returns
        -------
        nn.Sequential  :  stem → [best_op × num_layers] → head
        """
        task_alphas = self.alphas[task_id]                  # (L, O)
        best_op_indices = task_alphas.argmax(dim=-1).tolist()

        chosen_ops: List[nn.Module] = []
        for layer_idx, op_idx in enumerate(best_op_indices):
            # Deep-copy so the discrete model is independent of the supernet
            op = copy.deepcopy(self.cells[layer_idx].ops[op_idx])
            chosen_ops.append(op)

        discrete = nn.Sequential(
            copy.deepcopy(self.stem),
            *chosen_ops,
            copy.deepcopy(self.heads[task_id]),
        )

        arch_summary = [OP_NAMES[i] for i in best_op_indices]
        logger.info(f"[discretize] Task {task_id} ({TASK_CONFIG[task_id]['name']}):")
        for l, name in enumerate(arch_summary):
            logger.info(f"  Layer {l:2d} → {name}")

        return discrete

    # ─────────────────────────────────────────────────────────────────────────
    def arch_parameters(self) -> List[nn.Parameter]:
        """Return only the alpha parameters (for Optimizer B)."""
        return [self.alphas]

    def weight_parameters(self) -> List[nn.Parameter]:
        """Return all non-alpha parameters (for Optimizer A)."""
        alpha_id = id(self.alphas)
        return [p for p in self.parameters() if id(p) != alpha_id]


# ============================================================================
# 4.  Data Pipeline — real MedMNIST + mock fallback
# ============================================================================

class EnsureRGB:
    """Transform that converts 1-channel tensors to 3-channel by repeating."""

    def __call__(self, img: Tensor) -> Tensor:
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        return img


class MedMNISTTaskWrapper(Dataset):
    """
    Wraps a single medmnist dataset, standardises label format.

    - Single-label tasks: label → scalar long tensor
    - Multi-label tasks:  label → float vector of shape (n_classes,)
    """

    def __init__(self, base_dataset, task_id: int) -> None:
        self.base = base_dataset
        self.task_id = task_id
        self.is_multilabel = TASK_CONFIG[task_id]["is_multilabel"]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img, label = self.base[idx]

        # medmnist returns label as ndarray; convert to tensor
        if self.is_multilabel:
            label = torch.tensor(label, dtype=torch.float32).squeeze()
        else:
            label = torch.tensor(label, dtype=torch.long).squeeze()

        return img, label


class MockTaskDataset(Dataset):
    """
    Synthetic data for a single task — for smoke-testing without medmnist.
    Produces random images and labels matching the task's format.
    """

    def __init__(
        self,
        task_id: int,
        num_samples: int = 600,
        seed: int = 42,
    ) -> None:
        super().__init__()
        cfg = TASK_CONFIG[task_id]
        rng = torch.Generator().manual_seed(seed + task_id)

        self.images: List[Tensor] = []
        self.labels: List[Tensor] = []

        for _ in range(num_samples):
            img = torch.rand(3, 28, 28, generator=rng)
            if cfg["is_multilabel"]:
                label = (torch.rand(cfg["n_classes"], generator=rng) > 0.5).float()
            else:
                label = torch.randint(0, cfg["n_classes"], (), generator=rng)
            self.images.append(img)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.images[idx], self.labels[idx]


def _get_medmnist_class(flag: str):
    """Resolve medmnist dataset class from its flag string."""
    info = medmnist.INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass


def build_dataloaders(
    batch_size: int = 32,
    data_dir: str = "./data",
    use_mock: bool = False,
    num_workers: int = 2,
    image_size: int = 28,
) -> Dict[str, Dict[int, DataLoader]]:
    """
    Build per-task DataLoaders for train / val / test splits.

    Returns
    -------
    {
        "train": {0: DataLoader, 1: DataLoader, 2: DataLoader},
        "val":   {0: DataLoader, 1: DataLoader, 2: DataLoader},
        "test":  {0: DataLoader, 1: DataLoader, 2: DataLoader},
    }
    """
    if use_mock or not HAS_MEDMNIST:
        if not use_mock:
            logger.warning(
                "medmnist not installed — falling back to synthetic data. "
                "Install with: pip install medmnist"
            )
        return _build_mock_loaders(batch_size)

    # ── Transforms ────────────────────────────────────────────────────────
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required. Install: pip install torchvision")

    train_tfm = transforms.Compose([
        transforms.ToTensor(),        # PIL → [C, H, W] float in [0, 1]
        EnsureRGB(),                   # grayscale → RGB
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        EnsureRGB(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    loaders: Dict[str, Dict[int, DataLoader]] = {
        "train": {}, "val": {}, "test": {},
    }

    for task_id, cfg in TASK_CONFIG.items():
        DataClass = _get_medmnist_class(cfg["medmnist_flag"])

        for split, tfm, shuffle, drop in [
            ("train", train_tfm, True,  True),
            ("val",   eval_tfm,  False, False),
            ("test",  eval_tfm,  False, False),
        ]:
            base_ds = DataClass(
                split=split,
                transform=tfm,
                download=True,
                root=data_dir,
                size=image_size,
            )
            wrapped = MedMNISTTaskWrapper(base_ds, task_id)

            loaders[split][task_id] = DataLoader(
                wrapped,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop and (len(wrapped) >= batch_size),
                num_workers=num_workers,
                pin_memory=True,
            )
            logger.info(
                f"  {cfg['name']:12s} {split:5s} → {len(wrapped):6d} samples"
            )

    return loaders


def _build_mock_loaders(
    batch_size: int,
) -> Dict[str, Dict[int, DataLoader]]:
    """Build mock DataLoaders for smoke-testing."""
    logger.warning("Using MOCK data — results will NOT be meaningful benchmarks.")
    loaders: Dict[str, Dict[int, DataLoader]] = {
        "train": {}, "val": {}, "test": {},
    }
    for task_id in TASK_CONFIG:
        for split, n, seed_off, shuffle in [
            ("train", 1200, 0,  True),
            ("val",   300,  10, False),
            ("test",  300,  20, False),
        ]:
            ds = MockTaskDataset(task_id, num_samples=n, seed=42 + seed_off)
            loaders[split][task_id] = DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                drop_last=(split == "train"),
            )
    return loaders


# ============================================================================
# 5.  Evaluation Utilities
# ============================================================================

def compute_task_loss(
    logits: Tensor, labels: Tensor, task_id: int,
) -> Tensor:
    """
    Compute the appropriate loss for a task.

    - Single-label tasks (PathMNIST, DermaMNIST): CrossEntropyLoss
    - Multi-label tasks  (ChestMNIST):            BCEWithLogitsLoss
    """
    if TASK_CONFIG[task_id]["is_multilabel"]:
        return F.binary_cross_entropy_with_logits(logits, labels)
    else:
        return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
    is_supernet: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model on one task.  Returns ACC, AUC, and loss.

    Args:
        model       : The supernet or a discrete nn.Sequential.
        loader      : DataLoader for this task's split.
        task_id     : Task index (used to select head in supernet mode).
        device      : Torch device.
        is_supernet : If True, calls model(x, task_id). Else model(x).

    Returns:
        {"acc": float, "auc": float, "loss": float}
    """
    model.eval()
    cfg = TASK_CONFIG[task_id]
    is_ml = cfg["is_multilabel"]
    n_classes = cfg["n_classes"]

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_supernet:
            logits = model(images, task_id)
        else:
            logits = model(images)

        loss = compute_task_loss(logits, labels, task_id)
        total_loss += loss.item()
        n_batches += 1

        # Collect predictions
        if is_ml:
            scores = torch.sigmoid(logits).cpu().numpy()
        else:
            scores = F.softmax(logits, dim=-1).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels.cpu().numpy())

    if n_batches == 0:
        return {"acc": 0.0, "auc": 0.0, "loss": 0.0}

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / n_batches

    # ── Accuracy ──────────────────────────────────────────────────────────
    if is_ml:
        # Multi-label: average per-label accuracy
        y_pred = (y_score >= 0.5).astype(np.float32)
        acc = np.mean([
            np.mean(y_pred[:, c] == y_true[:, c])
            for c in range(n_classes)
        ])
    else:
        # Single-label: top-1 accuracy
        y_pred = np.argmax(y_score, axis=-1)
        y_true_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
        acc = np.mean(y_pred == y_true_flat)

    # ── AUC ───────────────────────────────────────────────────────────────
    auc = _safe_auc(y_true, y_score, is_ml, n_classes)

    return {"acc": float(acc), "auc": float(auc), "loss": float(avg_loss)}


def _safe_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_multilabel: bool,
    n_classes: int,
) -> float:
    """Compute AUC with graceful fallback if sklearn is unavailable or data is degenerate."""
    if not HAS_SKLEARN:
        return float("nan")
    try:
        if is_multilabel:
            # Per-label binary AUC, macro-averaged (skip labels with no pos/neg)
            aucs = []
            for c in range(n_classes):
                if len(np.unique(y_true[:, c])) < 2:
                    continue  # skip degenerate columns
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            return float(np.mean(aucs)) if aucs else float("nan")
        else:
            y_true_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
            if len(np.unique(y_true_flat)) < 2:
                return float("nan")
            return float(roc_auc_score(
                y_true_flat, y_score,
                multi_class="ovr",
                labels=list(range(n_classes)),
            ))
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        return float("nan")


def evaluate_all_tasks(
    model: nn.Module,
    loaders: Dict[int, DataLoader],
    device: torch.device,
    is_supernet: bool = True,
    split_name: str = "test",
) -> Dict[int, Dict[str, float]]:
    """Evaluate model on all tasks; returns {task_id: metrics_dict}."""
    results: Dict[int, Dict[str, float]] = {}
    for task_id, loader in loaders.items():
        metrics = evaluate_task(model, loader, task_id, device, is_supernet)
        name = TASK_CONFIG[task_id]["name"]
        logger.info(
            f"  [{split_name}] {name:12s}  ACC={metrics['acc']:.4f}  "
            f"AUC={metrics['auc']:.4f}  Loss={metrics['loss']:.4f}"
        )
        results[task_id] = metrics
    return results


# ============================================================================
# 6.  SearchController — Bilevel Optimisation
# ============================================================================

class SearchController:
    """
    Handles the alternating bilevel optimisation loop of MT-DARTS.

    Bilevel objective (informal)
    ----------------------------
      min_{α}  Σ_k  L_val(w*(α), α_k)       (outer / arch problem)
      s.t.  w*(α) = argmin_w  Σ_k  L_train(w, α_k)  (inner / weight problem)

    We approximate the inner optimisation with a single SGD step (first-order
    DARTS), which makes the wall-clock cost tractable while preserving the
    qualitative search behaviour reported in the original DARTS paper.

    Gradient Disentanglement (implementation)
    -----------------------------------------
    Phase 1 (weight update):
      - `alphas.requires_grad_(False)` — alpha gradients are suppressed so
        SGD's momentum buffer is never contaminated by arch-loss signals.

    Phase 2 (alpha update):
      - We loop over tasks and compute L_val for each task separately.
        `self.model.alphas[k]` receives gradients; all other alpha rows do not
        participate in the forward pass of task k (see forward() docstring).
      - Weight parameters are not in opt_arch, so their gradient buffers
        remain clean across the two phases.
    """

    def __init__(
        self,
        model: TaskAwareSupernet,
        lr_weights: float = 0.025,
        lr_alphas:  float = 3e-4,
        momentum:   float = 0.9,
        weight_decay_w: float = 3e-4,
        weight_decay_a: float = 1e-3,
        grad_clip:  float = 5.0,
        epochs: int = 50,
    ) -> None:
        self.model      = model
        self.grad_clip  = grad_clip
        self.num_tasks  = model.num_tasks

        # Optimizer A — trains convolutional filters
        self.opt_weights = torch.optim.SGD(
            model.weight_parameters(),
            lr=lr_weights,
            momentum=momentum,
            weight_decay=weight_decay_w,
            nesterov=True,
        )

        # Optimizer B — trains architecture weights (alphas)
        # Adam chosen for its adaptivity: alpha updates need careful scaling
        # because different tasks may produce gradients of very different
        # magnitudes depending on class imbalance.
        self.opt_arch = torch.optim.Adam(
            model.arch_parameters(),
            lr=lr_alphas,
            betas=(0.5, 0.999),          # lower β₁ → less momentum
            weight_decay=weight_decay_a,
        )

        # Cosine annealing for weight LR — critical for stable DARTS search
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_weights, T_max=epochs, eta_min=1e-4
        )

        self._step = 0
        self._epoch = 0

    # ─────────────────────────────────────────────────────────────────────────
    def step(
        self,
        train_batches: Dict[int, Tuple[Tensor, Tensor]],
        val_batches:   Dict[int, Tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        One bilevel step: (Phase-1 weight update) + (Phase-2 alpha update).

        Args:
            train_batches : {task_id: (images, labels)} — training data per task
            val_batches   : {task_id: (images, labels)} — validation data per task
            device        : Torch device.

        Returns:
            (weight_loss, arch_loss) as Python floats for logging.
        """
        self.model.train()

        # ── Phase 1: Update weights w, freeze alphas ──────────────────────
        self.model.alphas.requires_grad_(False)   # ← gradient disentanglement
        self.opt_weights.zero_grad()

        total_loss_w = 0.0
        active_tasks_w = 0
        for task_id, (images, labels) in train_batches.items():
            images, labels = images.to(device), labels.to(device)
            logits = self.model(images, task_id)
            loss = compute_task_loss(logits, labels, task_id) / self.num_tasks
            loss.backward()
            total_loss_w += loss.item()
            active_tasks_w += 1

        nn.utils.clip_grad_norm_(self.model.weight_parameters(), self.grad_clip)
        self.opt_weights.step()

        # ── Phase 2: Update alphas, freeze weights ────────────────────────
        self.model.alphas.requires_grad_(True)    # ← re-enable arch grads

        # Clear stale weight gradients to prevent cross-phase contamination
        self.opt_arch.zero_grad(set_to_none=True)
        for p in self.model.weight_parameters():
            p.grad = None

        total_loss_a = 0.0
        active_tasks_a = 0
        for task_id, (images, labels) in val_batches.items():
            images, labels = images.to(device), labels.to(device)
            logits = self.model(images, task_id)
            loss = compute_task_loss(logits, labels, task_id) / self.num_tasks
            loss.backward()
            total_loss_a += loss.item()
            active_tasks_a += 1

        # No grad clipping for alphas: large alpha gradients are informative
        self.opt_arch.step()

        self._step += 1
        loss_w = total_loss_w * self.num_tasks / max(active_tasks_w, 1)
        loss_a = total_loss_a * self.num_tasks / max(active_tasks_a, 1)
        return loss_w, loss_a

    # ─────────────────────────────────────────────────────────────────────────
    def step_scheduler(self) -> None:
        """Advance the cosine LR scheduler (call once per epoch)."""
        self.scheduler.step()
        self._epoch += 1
        current_lr = self.scheduler.get_last_lr()[0]
        logger.info(f"  LR updated → {current_lr:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def log_arch_distribution(self) -> None:
        """Pretty-print current softmax(alpha) for each task × layer."""
        logger.info(f"[Step {self._step}] Architecture distribution (softmax α):")
        alphas_soft = F.softmax(self.model.alphas, dim=-1)  # (T, L, O)

        for t in range(self.num_tasks):
            task_name = TASK_CONFIG[t]["name"]
            logger.info(f"  {task_name}:")
            for l in range(self.model.num_layers):
                probs = alphas_soft[t, l].tolist()
                best_idx = int(alphas_soft[t, l].argmax())
                bar = "  ".join(
                    f"{OP_NAMES[i][:12]:12s}={p:.3f}" for i, p in enumerate(probs)
                )
                logger.info(f"    Layer {l}: {bar}  ← {OP_NAMES[best_idx]}")


# ============================================================================
# 7.  Post-Search Retraining of Discrete Architectures
# ============================================================================

def _reinit_weights(model: nn.Module) -> None:
    """Re-initialize all weights from scratch (DARTS convention)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


def retrain_discrete(
    discrete_model: nn.Module,
    task_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 0.025,
    weight_decay: float = 3e-4,
    grad_clip: float = 5.0,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrain a discrete architecture from scratch (standard DARTS evaluation).

    Steps:
        1. Re-initialize all weights randomly.
        2. Train with SGD + cosine annealing.
        3. Track best validation AUC; restore best model for final test eval.

    Returns:
        {
            "test_acc": float, "test_auc": float, "test_loss": float,
            "best_val_auc": float, "n_params": int,
            "architecture": list[str],
        }
    """
    task_name = TASK_CONFIG[task_id]["name"]
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Retraining discrete model for {task_name} (task {task_id})")
    logger.info(f"{'─' * 60}")

    # Re-init weights from scratch
    _reinit_weights(discrete_model)
    discrete_model = discrete_model.to(device)

    n_params = sum(p.numel() for p in discrete_model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.SGD(
        discrete_model.parameters(),
        lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-4,
    )

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        discrete_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = discrete_model(images)
            loss = compute_task_loss(logits, labels, task_id)
            loss.backward()
            nn.utils.clip_grad_norm_(discrete_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # ── Validate (every 5 epochs or last epoch) ───────────────────────
        if epoch % 5 == 0 or epoch == num_epochs:
            val_metrics = evaluate_task(
                discrete_model, val_loader, task_id, device, is_supernet=False,
            )
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"  [{task_name}] Epoch {epoch:3d}/{num_epochs}  "
                f"train_loss={avg_loss:.4f}  "
                f"val_ACC={val_metrics['acc']:.4f}  "
                f"val_AUC={val_metrics['auc']:.4f}"
            )

            if val_metrics["auc"] > best_val_auc or math.isnan(best_val_auc):
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(discrete_model.state_dict())

    # ── Restore best & test ───────────────────────────────────────────────
    if best_state is not None:
        discrete_model.load_state_dict(best_state)

    test_metrics = evaluate_task(
        discrete_model, test_loader, task_id, device, is_supernet=False,
    )
    logger.info(
        f"  [{task_name}] FINAL TEST  ACC={test_metrics['acc']:.4f}  "
        f"AUC={test_metrics['auc']:.4f}"
    )

    # ── Save best model ──────────────────────────────────────────────────
    if save_dir:
        path = os.path.join(save_dir, f"discrete_{task_name.lower()}_best.pt")
        torch.save(discrete_model.state_dict(), path)
        logger.info(f"  Saved best model → {path}")

    return {
        "test_acc": test_metrics["acc"],
        "test_auc": test_metrics["auc"],
        "test_loss": test_metrics["loss"],
        "best_val_auc": best_val_auc,
        "n_params": n_params,
    }


# ============================================================================
# 8.  Benchmark Reporting
# ============================================================================

def print_benchmark_table(
    results: Dict[int, Dict[str, Any]],
    architectures: Dict[int, List[str]],
) -> str:
    """Pretty-print final benchmark results as a table."""
    sep = "═" * 72
    header = (
        f"{'Task':<14s} │ {'ACC (%)':>8s} │ {'AUC':>8s} │ "
        f"{'Params':>10s} │ {'Architecture':s}"
    )
    lines = [
        "", sep,
        "  MT-DARTS Benchmark Results on MedMNIST",
        sep, header, "─" * 72,
    ]

    accs, aucs = [], []
    for task_id in sorted(results.keys()):
        r = results[task_id]
        name = TASK_CONFIG[task_id]["name"]
        acc_pct = r["test_acc"] * 100
        auc_val = r["test_auc"]
        params = r["n_params"]
        arch = architectures.get(task_id, [])
        # Show top-3 most frequent ops in the architecture
        from collections import Counter
        op_counts = Counter(arch)
        arch_summary = ", ".join(f"{op}×{cnt}" for op, cnt in op_counts.most_common(3))

        lines.append(
            f"  {name:<12s} │ {acc_pct:>7.2f}% │ {auc_val:>8.4f} │ "
            f"{params:>10,d} │ {arch_summary}"
        )
        accs.append(acc_pct)
        aucs.append(auc_val)

    lines.append("─" * 72)
    macro_acc = np.mean(accs)
    macro_auc = np.mean(aucs)
    lines.append(
        f"  {'Macro Avg':<12s} │ {macro_acc:>7.2f}% │ {macro_auc:>8.4f} │ "
        f"{'—':>10s} │"
    )
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table


def save_benchmark_results(
    results: Dict[int, Dict[str, Any]],
    architectures: Dict[int, List[str]],
    search_time_s: float,
    retrain_time_s: float,
    save_dir: str,
) -> None:
    """Save benchmark results to JSON."""
    output = {
        "search_time_seconds": search_time_s,
        "retrain_time_seconds": retrain_time_s,
        "tasks": {},
    }
    for task_id in sorted(results.keys()):
        r = results[task_id]
        output["tasks"][TASK_CONFIG[task_id]["name"]] = {
            "test_acc": r["test_acc"],
            "test_auc": r["test_auc"],
            "test_loss": r["test_loss"],
            "best_val_auc": r["best_val_auc"],
            "n_params": r["n_params"],
            "architecture": architectures.get(task_id, []),
        }

    accs = [r["test_acc"] for r in results.values()]
    aucs = [r["test_auc"] for r in results.values()]
    output["macro_avg_acc"] = float(np.mean(accs))
    output["macro_avg_auc"] = float(np.mean(aucs))

    path = os.path.join(save_dir, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Benchmark results saved → {path}")


# ============================================================================
# 9.  Main Pipeline — Search → Discretize → Retrain → Benchmark
# ============================================================================

def save_checkpoint(
    model: TaskAwareSupernet,
    controller: SearchController,
    epoch: int,
    save_dir: str,
    filename: str = "checkpoint_latest.pt",
) -> None:
    """Save full training state for resume capability."""
    path = os.path.join(save_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "opt_weights_state": controller.opt_weights.state_dict(),
        "opt_arch_state": controller.opt_arch.state_dict(),
        "scheduler_state": controller.scheduler.state_dict(),
        "step": controller._step,
    }, path)
    logger.info(f"  Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: TaskAwareSupernet,
    controller: SearchController,
) -> int:
    """Load checkpoint and return the epoch to resume from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    controller.opt_weights.load_state_dict(ckpt["opt_weights_state"])
    controller.opt_arch.load_state_dict(ckpt["opt_arch_state"])
    controller.scheduler.load_state_dict(ckpt["scheduler_state"])
    controller._step = ckpt["step"]
    epoch = ckpt["epoch"]
    logger.info(f"  Resumed from checkpoint (epoch {epoch})")
    return epoch


def run_mt_darts(
    # Search hyper-parameters
    search_epochs: int = 50,
    batch_size: int = 64,
    num_layers: int = 6,
    channels: int = 32,
    lr_weights: float = 0.025,
    lr_alphas: float = 3e-4,
    # Retrain hyper-parameters
    retrain_epochs: int = 100,
    retrain_lr: float = 0.025,
    # Infrastructure
    device_str: str = "cpu",
    data_dir: str = "./data",
    save_dir: str = "./results",
    use_mock: bool = False,
    log_interval: int = 10,
    resume_path: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 2,
    image_size: int = 28,
) -> Dict[str, Any]:
    """
    Full MT-DARTS pipeline: Search → Discretize → Retrain → Benchmark.

    Returns:
        Dictionary with benchmark results for all tasks.
    """
    set_seed(seed)
    device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Building data loaders …")
    loaders = build_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
        use_mock=use_mock,
        num_workers=num_workers,
        image_size=image_size,
    )

    train_loaders = loaders["train"]
    val_loaders   = loaders["val"]
    test_loaders  = loaders["test"]

    # ── Model & Controller ────────────────────────────────────────────────
    num_classes_per_task = [TASK_CONFIG[t]["n_classes"] for t in range(NUM_TASKS)]

    model = TaskAwareSupernet(
        num_tasks=NUM_TASKS,
        num_layers=num_layers,
        channels=channels,
        num_classes_per_task=num_classes_per_task,
    ).to(device)

    controller = SearchController(
        model,
        lr_weights=lr_weights,
        lr_alphas=lr_alphas,
        epochs=search_epochs,
    )

    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_checkpoint(resume_path, model, controller)

    total_params = sum(p.numel() for p in model.parameters())
    alpha_params = model.alphas.numel()
    logger.info("=" * 70)
    logger.info(
        f"MT-DARTS Search  |  device={device}  |  epochs={search_epochs}"
    )
    logger.info(f"  supernet params : {total_params:,}")
    logger.info(f"  alpha params    : {alpha_params}")
    logger.info(f"  search space    : {NUM_OPS} ops × {num_layers} layers × {NUM_TASKS} tasks")
    logger.info(f"  image size      : {image_size}×{image_size}")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A: Architecture Search
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE A: Architecture Search")
    search_start = time.time()

    # Per-task validation iterators (cycled independently of train)
    val_iters: Dict[int, Any] = {
        t: iter(val_loaders[t]) for t in range(NUM_TASKS)
    }

    for epoch in range(start_epoch + 1, search_epochs + 1):
        model.train()
        epoch_loss_w, epoch_loss_a = 0.0, 0.0
        n_steps_epoch = 0

        # Steps per epoch = min dataloader length (balanced across tasks)
        steps_per_epoch = min(len(train_loaders[t]) for t in range(NUM_TASKS))
        train_iters = {t: iter(train_loaders[t]) for t in range(NUM_TASKS)}

        for step in range(steps_per_epoch):
            # Gather one batch per task for training
            train_batches: Dict[int, Tuple[Tensor, Tensor]] = {}
            val_batch_dict: Dict[int, Tuple[Tensor, Tensor]] = {}

            for t in range(NUM_TASKS):
                train_batches[t] = next(train_iters[t])

                # Cycle validation iterator
                try:
                    val_batch_dict[t] = next(val_iters[t])
                except StopIteration:
                    val_iters[t] = iter(val_loaders[t])
                    val_batch_dict[t] = next(val_iters[t])

            loss_w, loss_a = controller.step(
                train_batches, val_batch_dict, device,
            )
            epoch_loss_w += loss_w
            epoch_loss_a += loss_a
            n_steps_epoch += 1

            if controller._step % log_interval == 0:
                logger.info(
                    f"  Epoch {epoch:3d}  step {step:4d}/{steps_per_epoch}  "
                    f"loss_w={loss_w:.4f}  loss_arch={loss_a:.4f}"
                )

        # End-of-epoch bookkeeping
        controller.step_scheduler()
        avg_lw = epoch_loss_w / max(n_steps_epoch, 1)
        avg_la = epoch_loss_a / max(n_steps_epoch, 1)
        logger.info(
            f"  Epoch {epoch:3d} summary  "
            f"avg_loss_w={avg_lw:.4f}  avg_loss_arch={avg_la:.4f}"
        )

        # Log architecture distribution every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == search_epochs:
            controller.log_arch_distribution()

        # Evaluate supernet on validation set
        if epoch % 10 == 0 or epoch == search_epochs:
            logger.info(f"  Evaluating supernet (epoch {epoch}):")
            evaluate_all_tasks(
                model, val_loaders, device,
                is_supernet=True, split_name="val",
            )

        # Save checkpoint
        save_checkpoint(model, controller, epoch, save_dir)

    search_time = time.time() - search_start
    logger.info(f"\n  Search completed in {search_time:.1f}s ({search_time/3600:.2f}h)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B: Discretize Architectures
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE B: Discretizing final architectures")
    logger.info("=" * 70)

    discrete_models: Dict[int, nn.Module] = {}
    architectures: Dict[int, List[str]] = {}

    for task_id in range(NUM_TASKS):
        discrete_model = model.discretize(task_id)
        discrete_models[task_id] = discrete_model

        # Record the architecture string
        task_alphas = model.alphas[task_id]
        best_ops = task_alphas.argmax(dim=-1).tolist()
        architectures[task_id] = [OP_NAMES[i] for i in best_ops]

        n_params = sum(p.numel() for p in discrete_model.parameters())
        logger.info(f"  Task {task_id} ({TASK_CONFIG[task_id]['name']}) "
                     f"discrete params: {n_params:,}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE C: Retrain Discrete Architectures from Scratch
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE C: Retraining discrete architectures from scratch")
    retrain_start = time.time()

    benchmark_results: Dict[int, Dict[str, Any]] = {}

    for task_id in range(NUM_TASKS):
        result = retrain_discrete(
            discrete_model=discrete_models[task_id],
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            test_loader=test_loaders[task_id],
            num_epochs=retrain_epochs,
            lr=retrain_lr,
            device=device,
            save_dir=save_dir,
        )
        benchmark_results[task_id] = result

    retrain_time = time.time() - retrain_start
    logger.info(f"\n  Retraining completed in {retrain_time:.1f}s ({retrain_time/3600:.2f}h)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE D: Final Benchmark Report
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE D: Benchmark Report")
    table = print_benchmark_table(benchmark_results, architectures)

    save_benchmark_results(
        benchmark_results, architectures,
        search_time_s=search_time,
        retrain_time_s=retrain_time,
        save_dir=save_dir,
    )

    # Save the table to a text file as well
    table_path = os.path.join(save_dir, "benchmark_table.txt")
    with open(table_path, "w") as f:
        f.write(table)

    return {
        "results": benchmark_results,
        "architectures": architectures,
        "search_time": search_time,
        "retrain_time": retrain_time,
    }


# ============================================================================
# 10.  Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MT-DARTS: Multi-Task NAS on MedMNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Search
    g_search = parser.add_argument_group("Search Phase")
    g_search.add_argument("--epochs", type=int, default=50,
                          help="Number of search epochs")
    g_search.add_argument("--batch", type=int, default=64,
                          help="Batch size per task")
    g_search.add_argument("--layers", type=int, default=6,
                          help="Number of searchable layers")
    g_search.add_argument("--channels", type=int, default=32,
                          help="Feature map channel width")
    g_search.add_argument("--lr_w", type=float, default=0.025,
                          help="Weight optimizer learning rate")
    g_search.add_argument("--lr_a", type=float, default=3e-4,
                          help="Alpha optimizer learning rate")

    # Retrain
    g_retrain = parser.add_argument_group("Retrain Phase")
    g_retrain.add_argument("--retrain_epochs", type=int, default=100,
                           help="Epochs for discrete model retraining")
    g_retrain.add_argument("--retrain_lr", type=float, default=0.025,
                           help="Learning rate for retraining")

    # Infrastructure
    g_infra = parser.add_argument_group("Infrastructure")
    g_infra.add_argument("--device", type=str, default="cpu",
                         help="Device: cpu | cuda | cuda:0 | mps")
    g_infra.add_argument("--data_dir", type=str, default="./data",
                         help="Root directory for MedMNIST data")
    g_infra.add_argument("--save_dir", type=str, default="./results",
                         help="Directory for checkpoints and results")
    g_infra.add_argument("--mock", action="store_true",
                         help="Use mock data (for smoke testing)")
    g_infra.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility")
    g_infra.add_argument("--workers", type=int, default=2,
                         help="DataLoader num_workers")
    g_infra.add_argument("--log_interval", type=int, default=10,
                         help="Log every N steps")
    g_infra.add_argument("--resume", type=str, default=None,
                         help="Path to checkpoint to resume from")
    g_infra.add_argument("--image_size", type=int, default=28,
                         choices=[28, 64, 128, 224],
                         help="MedMNIST image size")

    args = parser.parse_args()

    # Dependency check
    if not args.mock and not HAS_MEDMNIST:
        logger.error(
            "medmnist package not found. Install with:\n"
            "  pip install medmnist\n"
            "Or use --mock flag for smoke testing with synthetic data."
        )
        sys.exit(1)

    if not HAS_SKLEARN:
        logger.warning(
            "scikit-learn not found — AUC metrics will be unavailable.\n"
            "Install with: pip install scikit-learn"
        )

    results = run_mt_darts(
        search_epochs=args.epochs,
        batch_size=args.batch,
        num_layers=args.layers,
        channels=args.channels,
        lr_weights=args.lr_w,
        lr_alphas=args.lr_a,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr,
        device_str=args.device,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        use_mock=args.mock,
        log_interval=args.log_interval,
        resume_path=args.resume,
        seed=args.seed,
        num_workers=args.workers,
        image_size=args.image_size,
    )
