"""
Task-aware loss routing.

  task_loss() — dispatches to CrossEntropyLoss or BCEWithLogitsLoss based
                on task_id, matching the multi-label nature of ChestMNIST.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

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
# This static fallback is replaced at retrain time by per-label computed weights.
_CHEST_POS_WEIGHT = torch.ones(14) * 10.0

# Focal loss gamma for DermaMNIST — γ=2 is standard; higher values increase
# focus on hard examples but can destabilise early training.
_DERMA_FOCAL_GAMMA: float = 2.0


def _multiclass_focal_loss(
    logits:          Tensor,
    targets:         Tensor,
    gamma:           float            = 2.0,
    weight:          Optional[Tensor] = None,
    label_smoothing: float            = 0.0,
) -> Tensor:
    """
    Multiclass focal loss (Lin et al., ICCV 2017).

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Adaptively down-weights well-classified examples so training concentrates
    on hard, misclassified samples — more effective than fixed class weights
    for DermaMNIST's severe long-tail distribution.

    Args:
        logits          : Raw logits (B, C).
        targets         : Class indices (B,).
        gamma           : Focusing parameter (0 = standard CE).
        weight          : Optional per-class weights (C,).
        label_smoothing : Label smoothing for the base CE loss.
    """
    ce = F.cross_entropy(
        logits, targets,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Clamp away from 1.0 to avoid 0^0 NaN when γ > 0.
        focal_weight = (1.0 - p_t.clamp(max=1.0 - 1e-6)) ** gamma
    return (focal_weight * ce).mean()


def task_loss(
    logits:            Tensor,
    labels,            # int scalar | float Tensor [num_classes] | list thereof
    task_id:           int,
    device:            torch.device,
    label_smoothing:   float            = 0.0,
    use_class_weights: bool             = True,
    use_focal:         bool             = False,
    chest_pos_weight:  Optional[Tensor] = None,
    is_multilabel:     bool             = False,
) -> Tensor:
    """
    Route to the correct loss function per task.

    Multi-label tasks (ChestMNIST, is_multilabel=True):
        BCEWithLogitsLoss — sigmoid applied internally for numerical stability.
        Labels must be float multi-hot vectors. label_smoothing not applied.
        pos_weight applied to reweight sparse positive labels.
        ``chest_pos_weight`` overrides the static fallback when provided
        (pass per-label weights computed from the actual training split).

    Single-label tasks (PathMNIST, DermaMNIST):
        CrossEntropyLoss — standard multi-class classification.
        label_smoothing applied when > 0.
        DermaMNIST uses class weights to handle imbalance unless
        ``use_class_weights=False`` (set to False when a WeightedRandomSampler
        already balances the data, to avoid double-correcting imbalance).
        When ``use_focal=True`` and task_id==2, uses Focal Loss instead of CE
        to adaptively focus on hard examples.

    Args:
        logits             : Model output, shape (B, num_classes).
        labels             : Ground-truth — a scalar int, a tensor, or a list.
        task_id            : The task's *registry* id (see TASK_REGISTRY in
                             src/data.py) — NOT its position in a custom
                             --tasks selection.  Only gates the DermaMNIST-
                             specific class weights/focal loss below
                             (registry id 2).
        device             : Target device for label tensors.
        label_smoothing    : Smoothing factor for CE tasks (0 = disabled).
        use_class_weights  : If False, disable per-class weights even for
                             DermaMNIST (use when the DataLoader already
                             up-samples minority classes via WeightedRandomSampler).
        use_focal          : If True and task_id==2, use Focal Loss instead of
                             plain CE for DermaMNIST hard-example mining.
        chest_pos_weight   : Per-label pos_weight tensor for ChestMNIST (14,).
                             Overrides the static _CHEST_POS_WEIGHT when given.
        is_multilabel      : Whether this task uses multi-label (BCE) loss
                             instead of single-label (CE).  Caller resolves
                             this from TASK_REGISTRY / model.MULTILABEL_TASKS.
    """
    if is_multilabel:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        elif isinstance(labels, list):
            lbl_t = torch.stack(labels).to(device)
        else:
            lbl_t = labels.to(device)
        pw = (chest_pos_weight if chest_pos_weight is not None
              else _CHEST_POS_WEIGHT)
        return F.binary_cross_entropy_with_logits(
            logits,
            lbl_t.float(),
            pos_weight=pw.to(device),
        )
    else:
        if isinstance(labels, Tensor):
            lbl_t = labels.to(device)
        else:
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)

        # DermaMNIST (task 2): apply class weights to handle imbalance.
        # Disabled when a WeightedRandomSampler already balances the data
        # (use_class_weights=False) to avoid double-correcting imbalance.
        weight = None
        if task_id == 2 and use_class_weights:
            weight = _DERMA_CLASS_WEIGHTS.to(device)

        # DermaMNIST focal loss: adaptively down-weights easy examples so
        # training focuses on hard, misclassified samples.
        if use_focal and task_id == 2:
            return _multiclass_focal_loss(
                logits, lbl_t,
                gamma=_DERMA_FOCAL_GAMMA,
                weight=weight,
                label_smoothing=label_smoothing,
            )

        return F.cross_entropy(
            logits, lbl_t,
            weight=weight,
            label_smoothing=label_smoothing,
        )
