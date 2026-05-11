"""
TaskAwareSupernet — the multi-task DARTS supernet.

Architecture:  stem  →  [MixedOp × num_layers]  →  head_k

Gradient disentanglement:
  self.alphas[task_id] slices a (L, O) sub-tensor.  Only row [task_id]
  of ∂L/∂alphas receives non-zero gradients, so each task's architecture
  parameters evolve independently.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .normalizers import annealed_sparsemax
from .ops import MixedOp, NUM_OPS, OP_NAMES

# Local mapping avoids importing data.py here (would create a circular dep).
_TASK_NAMES: Dict[int, str] = {0: "PathMNIST", 1: "ChestMNIST", 2: "DermaMNIST"}


class TaskAwareSupernet(nn.Module):
    """
    Multi-task supernet shared across architecture search.

    ChestMNIST (task 1) logits are raw — BCEWithLogitsLoss in the
    controller applies sigmoid numerically stably.
    """

    # Tasks that use multi-label (BCEWithLogitsLoss) instead of CE.
    MULTILABEL_TASKS = {1}   # ChestMNIST

    def __init__(
        self,
        num_tasks:            int            = 3,
        num_layers:           int            = 6,
        channels:             int            = 64,
        num_classes_per_task: Optional[List[int]] = None,
        img_size:             int            = 64,
    ) -> None:
        super().__init__()
        self.num_tasks  = num_tasks
        self.num_layers = num_layers
        self.channels   = channels
        self.img_size   = img_size

        if num_classes_per_task is None:
            num_classes_per_task = [9, 14, 7]
        assert len(num_classes_per_task) == num_tasks, (
            f"Expected {num_tasks} class counts, got {len(num_classes_per_task)}"
        )
        self.num_classes = num_classes_per_task

        # ── Per-task adaptive stems ────────────────────────────────────────
        # Each task gets its own private stem rather than sharing one.
        # Gradient disentanglement (Contrib. [A]) applies at the cell level;
        # independent stems let each task learn its own low-level feature
        # extractor without cross-task interference at the pixel level.
        def _make_stem() -> nn.Sequential:
            if img_size > 32:
                return nn.Sequential(
                    nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels // 2),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU6(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(3, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU6(inplace=True),
                )

        self.stems = nn.ModuleList([_make_stem() for _ in range(num_tasks)])

        self.cells = nn.ModuleList(
            [MixedOp(channels) for _ in range(num_layers)]
        )

        # ── Task-specific classification heads ────────────────────────────
        # PathMNIST / ChestMNIST: GAP → Dropout(0.3) → FC(4C) → ReLU → Dropout(0.2) → FC(nc)
        # DermaMNIST (task 2): deeper head with BN + extra hidden layer to
        #   improve calibration on the 7-class imbalanced 10k dataset.
        hidden_dim = max(channels * 4, 256)
        heads: list = []
        for t, nc in enumerate(num_classes_per_task):
            if t == 2:
                # Deeper head: GAP → D(0.3) → FC(4C) → LN → ReLU → D(0.2) → FC(2C) → ReLU → D(0.1) → FC(7)
                # LayerNorm instead of BatchNorm1d: BN crashes on single-sample
                # micro-batches (common when task_id==2 has 1 DermaMNIST image
                # in a given batch).  LN normalises per-feature and works with
                # any batch size.
                hidden2 = max(channels * 2, 128)
                heads.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(channels, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden_dim, hidden2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                    nn.Linear(hidden2, nc),
                ))
            else:
                heads.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(channels, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden_dim, nc),
                ))
        self.heads = nn.ModuleList(heads)

        # Architecture parameters — shape (T, L, O).
        # Near-zero init → sparsemax starts close to uniform distribution.
        self.alphas = nn.Parameter(
            torch.zeros(num_tasks, num_layers, NUM_OPS),
            requires_grad=True,
        )
        nn.init.normal_(self.alphas, mean=0.0, std=1e-3)
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor, task_id: int, tau: float = 1.0) -> Tensor:
        """
        Forward pass for one task.

        Contrib. [A][B]: Uses annealed_sparsemax (not softmax) to produce
        sparse, temperature-sharpened operation weights.

        Args:
            x       : Input images, shape (B, 3, H, W).
            task_id : Which task head + alpha row to use.
            tau     : Current sparsemax temperature (decreasing over epochs).
        """
        task_alphas       = self.alphas[task_id]                      # (L, O)
        weights_per_layer = annealed_sparsemax(task_alphas, tau=tau)  # (L, O)

        x = self.stems[task_id](x)
        for i, cell in enumerate(self.cells):
            x = cell(x, weights_per_layer[i])
        return self.heads[task_id](x)

    # ── Parameter groups ──────────────────────────────────────────────────────

    def arch_parameters(self) -> List[nn.Parameter]:
        return [self.alphas]

    def weight_parameters(self) -> List[nn.Parameter]:
        alpha_id = id(self.alphas)
        return [p for p in self.parameters() if id(p) != alpha_id]

    # ── Discretisation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def discretize(self, task_id: int) -> nn.Sequential:
        """
        Extract a discrete model for ``task_id`` via argmax(alpha[k]).

        Returns a deep-copied nn.Sequential fully detached from the supernet,
        ready for stand-alone retraining.
        """
        best_indices = self.alphas[task_id].argmax(dim=-1).tolist()
        chosen: List[nn.Module] = [
            copy.deepcopy(self.cells[l].ops[idx])
            for l, idx in enumerate(best_indices)
        ]
        discrete = nn.Sequential(
            copy.deepcopy(self.stems[task_id]),
            *chosen,
            copy.deepcopy(self.heads[task_id]),
        )
        name = _TASK_NAMES.get(task_id, str(task_id))
        print(f"\n[discretize] Task {task_id} ({name}):")
        for l, idx in enumerate(best_indices):
            print(f"  Layer {l:2d} → {OP_NAMES[idx]}")
        n_params = sum(p.numel() for p in discrete.parameters())
        print(f"  Total params: {n_params:,}\n")
        return discrete
