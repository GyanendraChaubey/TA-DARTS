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

# ── Per-task class weights ────────────────────────────────────────────────────
# DermaMNIST (task 2) class distribution from MedMNIST v2 official train split
# (7,007 samples total):
#   0: actinic keratoses      (327)  → 21.4×
#   1: basal cell carcinoma   (514)  → 13.6×
#   2: benign keratosis       (1099) →  6.4×
#   3: dermatofibroma         (115)  → 60.9×
#   4: melanoma               (1113) →  6.3×
#   5: melanocytic nevi       (6705) →  1.0×  ← majority class, weight 1.0
#   6: vascular lesions       (142)  → 47.2×
# Weights = max_count / class_count, clipped at 25 to avoid gradient explosion.
_DERMA_CLASS_WEIGHTS = torch.tensor(
    [21.4, 13.6, 6.4, 25.0, 6.3, 1.0, 25.0],
    dtype=torch.float32,
)

# ChestMNIST (task 1) — 14 binary labels, mostly negative.
# pos_weight reweights the positive class in BCEWithLogitsLoss.
_CHEST_POS_WEIGHT = torch.ones(14) * 10.0


def task_loss(
    logits:          Tensor,
    labels,          # int scalar | float Tensor [num_classes] | list thereof
    task_id:         int,
    device:          torch.device,
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Route to the correct loss function per task.

    Multi-label tasks (ChestMNIST, task_id in MULTILABEL_TASKS):
        BCEWithLogitsLoss — sigmoid applied internally for numerical stability.
        Labels must be float multi-hot vectors. label_smoothing not applied.
        pos_weight applied to reweight sparse positive labels.

    Single-label tasks (PathMNIST, DermaMNIST):
        CrossEntropyLoss — standard multi-class classification.
        label_smoothing applied when > 0.
        DermaMNIST uses class weights to handle imbalance.

    Args:
        logits          : Model output, shape (B, num_classes).
        labels          : Ground-truth — a scalar int, a tensor, or a list.
        task_id         : Determines loss function and label dtype.
        device          : Target device for label tensors.
        label_smoothing : Smoothing factor for CE tasks (0 = disabled).
    """
    if task_id in TaskAwareSupernet.MULTILABEL_TASKS:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        elif isinstance(labels, list):
            lbl_t = torch.stack(labels).to(device)
        else:
            lbl_t = labels.to(device)
        return F.binary_cross_entropy_with_logits(
            logits,
            lbl_t.float(),
            pos_weight=_CHEST_POS_WEIGHT.to(device),
        )
    else:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        else:
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)

        # DermaMNIST (task 2): apply class weights to handle imbalance.
        weight = None
        if task_id == 2:
            weight = _DERMA_CLASS_WEIGHTS.to(device)

        return F.cross_entropy(
            logits, lbl_t,
            weight=weight,
            label_smoothing=label_smoothing,
        )
