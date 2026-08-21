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
# TASK_REGISTRY imported lazily inside methods to avoid circular imports.


class TaskAwareSupernet(nn.Module):
    """
    Multi-task supernet shared across architecture search.

    ChestMNIST (task 1) logits are raw — BCEWithLogitsLoss in the
    controller applies sigmoid numerically stably.
    """

    def __init__(
        self,
        num_tasks:            int            = 3,
        num_layers:           int            = 6,
        channels:             int            = 64,
        num_classes_per_task: Optional[List[int]] = None,
        img_size:             int            = 64,
        task_ids:             Optional[List[int]] = None,
        multilabel_tasks:     Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        task_ids        : Ordered list of registry task IDs used by this supernet.
                          Length must equal num_tasks.  Defaults to [0,1,2].
        multilabel_tasks: Registry task IDs that use BCEWithLogitsLoss.
                          Derived from TASK_REGISTRY when None.
        """
        super().__init__()
        self.num_tasks  = num_tasks
        self.num_layers = num_layers
        self.channels   = channels
        self.img_size   = img_size

        # Resolve task ordering (index into supernet == position in task_ids).
        from .data import TASK_REGISTRY, DEFAULT_TASK_IDS
        _ids = task_ids if task_ids is not None else DEFAULT_TASK_IDS
        assert len(_ids) == num_tasks, (
            f"len(task_ids)={len(_ids)} must equal num_tasks={num_tasks}"
        )
        self.task_ids: List[int] = list(_ids)

        # Which positions in self.task_ids use multi-label loss.
        if multilabel_tasks is not None:
            self.MULTILABEL_TASKS: set = set(multilabel_tasks)
        else:
            self.MULTILABEL_TASKS = {
                pos for pos, tid in enumerate(self.task_ids)
                if TASK_REGISTRY[tid][4]   # is_multilabel flag
            }

        if num_classes_per_task is None:
            num_classes_per_task = [TASK_REGISTRY[tid][2] for tid in self.task_ids]
        assert len(num_classes_per_task) == num_tasks, (
            f"Expected {num_tasks} class counts, got {len(num_classes_per_task)}"
        )
        self.num_classes = num_classes_per_task

        # ── Per-task adaptive stems ────────────────────────────────────────
        # Each task gets its own private stem rather than sharing one.
        # Gradient disentanglement (Contrib. [A]) applies at the cell level;
        # independent stems let each task learn its own low-level feature
        # extractor without cross-task interference at the pixel level.
        #
        # Stem design by resolution:
        #   28px  → single 3×3 conv, no stride  (spatial kept at 28×28)
        #   33–127px → two strided convs, ÷4    (e.g. 64 → 16px)
        #   128px+ → three strided convs, ÷8    (e.g. 224 → 28px)
        #
        # The 224px path reduces spatial to ~28×28 before the cells so that
        # cell memory cost is identical to the 28px baseline — only the stem
        # forward pass is larger (handled with gradient checkpointing if needed).
        def _make_stem() -> nn.Sequential:
            if img_size >= 128:
                # 224px → 112 → 56 → 28  (three stride-2 convs)
                return nn.Sequential(
                    nn.Conv2d(3, channels // 4, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels // 4),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(channels // 4, channels // 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels // 2),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU6(inplace=True),
                )
            elif img_size > 32:
                # 64px → 32 → 16  (two stride-2 convs)
                return nn.Sequential(
                    nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels // 2),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU6(inplace=True),
                )
            else:
                # 28px → 28  (single conv, no stride)
                return nn.Sequential(
                    nn.Conv2d(3, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU6(inplace=True),
                )

        self.stems = nn.ModuleList([_make_stem() for _ in range(num_tasks)])

        self.downsample_layers = [num_layers // 3, 2 * num_layers // 3]
        self.cells = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        c = channels
        for i in range(num_layers):
            if i in self.downsample_layers:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c * 2),
                    nn.ReLU6(inplace=True)
                ))
                c *= 2
            else:
                self.downsamples.append(nn.Identity())
            self.cells.append(MixedOp(c))

        # ── Task-specific classification heads ────────────────────────────
        # Small datasets (< ~10k train samples) get a deeper head with
        # LayerNorm + extra hidden layer for better calibration.
        # This is determined from TASK_REGISTRY rather than hardcoded to task 2.
        # LayerNorm is used instead of BatchNorm1d because BN crashes on
        # single-sample micro-batches; LN works with any batch size.
        from .data import TASK_REGISTRY
        SMALL_DATASET_TASKS = {2, 5, 6}   # DermaMNIST, RetinaMNIST, BreastMNIST

        final_channels = c
        hidden_dim = max(final_channels * 4, 256)
        heads: list = []
        for pos, (tid, nc) in enumerate(zip(self.task_ids, num_classes_per_task)):
            if tid in SMALL_DATASET_TASKS:
                hidden2 = max(final_channels * 2, 128)
                heads.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(final_channels, hidden_dim),
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
                    nn.Linear(final_channels, hidden_dim),
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
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
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
            x = self.downsamples[i](x)
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
        # task_id here is a supernet position index (0..num_tasks-1).
        from .data import TASK_REGISTRY
        registry_id = self.task_ids[task_id]
        name = TASK_REGISTRY[registry_id][0]

        best_indices = self.alphas[task_id].argmax(dim=-1).tolist()
        discrete = nn.Sequential()
        discrete.add_module("stem", copy.deepcopy(self.stems[task_id]))
        for l, idx in enumerate(best_indices):
            if l in self.downsample_layers:
                discrete.add_module(f"downsample_{l}", copy.deepcopy(self.downsamples[l]))
            discrete.add_module(f"cell_{l}", copy.deepcopy(self.cells[l].ops[idx]))
        discrete.add_module("head", copy.deepcopy(self.heads[task_id]))
        print(f"\n[discretize] Task {task_id} ({name}):")
        for l, idx in enumerate(best_indices):
            print(f"  Layer {l:2d} → {OP_NAMES[idx]}")
        n_params = sum(p.numel() for p in discrete.parameters())
        print(f"  Total params: {n_params:,}\n")
        return discrete
