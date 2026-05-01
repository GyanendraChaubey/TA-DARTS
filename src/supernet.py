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
        channels:             int            = 32,
        num_classes_per_task: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_tasks  = num_tasks
        self.num_layers = num_layers
        self.channels   = channels

        if num_classes_per_task is None:
            num_classes_per_task = [9, 14, 7]
        assert len(num_classes_per_task) == num_tasks, (
            f"Expected {num_tasks} class counts, got {len(num_classes_per_task)}"
        )
        self.num_classes = num_classes_per_task

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )
        self.cells = nn.ModuleList(
            [MixedOp(channels) for _ in range(num_layers)]
        )
        # Per-task heads with Dropout for regularisation.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(channels, nc),
            )
            for nc in num_classes_per_task
        ])

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
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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

        x = self.stem(x)
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
            copy.deepcopy(self.stem),
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
