"""
Task-aware loss routing.

  task_loss() — dispatches to CrossEntropyLoss or BCEWithLogitsLoss based
                on task_id, matching the multi-label nature of ChestMNIST.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .supernet import TaskAwareSupernet


def task_loss(
    logits:  Tensor,
    labels,          # int scalar | float Tensor [num_classes] | list thereof
    task_id: int,
    device:  torch.device,
) -> Tensor:
    """
    Route to the correct loss function per task.

    Multi-label tasks (ChestMNIST, task_id in MULTILABEL_TASKS):
        BCEWithLogitsLoss — sigmoid applied internally for numerical stability.
        Labels must be float multi-hot vectors.

    Single-label tasks (PathMNIST, DermaMNIST):
        CrossEntropyLoss — standard multi-class classification.

    Args:
        logits  : Model output, shape (B, num_classes).
        labels  : Ground-truth — a scalar int, a tensor, or a list thereof.
        task_id : Determines loss function and label dtype.
        device  : Target device for label tensors.
    """
    if task_id in TaskAwareSupernet.MULTILABEL_TASKS:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        elif isinstance(labels, list):
            lbl_t = torch.stack(labels).to(device)
        else:
            lbl_t = labels.to(device)
        return F.binary_cross_entropy_with_logits(logits, lbl_t.float())
    else:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        else:
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)
        return F.cross_entropy(logits, lbl_t)
