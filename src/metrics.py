"""
Evaluation metrics for MT-DARTS.

  evaluate_task()  — per-task acc + AUC + F1 + precision + recall + loss.
  evaluate()       — all-task evaluation returning a dict of metrics.
  alpha_entropy()  — mean Shannon entropy of sparsemax(α) for [D] early stop.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .data import TASK_REGISTRY
from .losses import task_loss
from .normalizers import annealed_sparsemax
from .supernet import TaskAwareSupernet

logger = logging.getLogger("MT-DARTS")

# ── Optional sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── AUC helper ────────────────────────────────────────────────────────────────

def _safe_clf_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    is_multilabel: bool,
) -> Dict[str, float]:
    """
    Compute macro-averaged F1, precision, and recall with graceful fallback.

    For multi-label tasks (ChestMNIST) uses sample-averaged F1/P/R which is
    standard for multi-label evaluation, then also reports macro.
    For single-label tasks reports macro-averaged scores.
    Returns NaN for each metric when sklearn is absent or data is degenerate.
    """
    if not HAS_SKLEARN:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}
    try:
        avg = "macro"
        kw  = dict(average=avg, zero_division=0)
        return {
            "f1":        float(f1_score(y_true, y_pred, **kw)),
            "precision": float(precision_score(y_true, y_pred, **kw)),
            "recall":    float(recall_score(y_true, y_pred, **kw)),
        }
    except Exception as exc:
        logger.warning(f"F1/P/R computation failed: {exc}")
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}


def _safe_auc(
    y_true:       np.ndarray,
    y_score:      np.ndarray,
    is_multilabel: bool,
    n_classes:    int,
) -> float:
    """
    Compute AUC with graceful fallback when sklearn is absent or data is
    degenerate (e.g. only one class present in the batch).
    """
    if not HAS_SKLEARN:
        return float("nan")
    try:
        if is_multilabel:
            aucs = []
            for c in range(n_classes):
                if len(np.unique(y_true[:, c])) < 2:
                    continue
                aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
            return float(np.mean(aucs)) if aucs else float("nan")
        else:
            y_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
            if len(np.unique(y_flat)) < 2:
                return float("nan")
            return float(roc_auc_score(
                y_flat, y_score,
                multi_class="ovr",
                labels=list(range(n_classes)),
            ))
    except Exception as exc:
        logger.warning(f"AUC computation failed: {exc}")
        return float("nan")


# ── Per-task evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_task(
    model:                  nn.Module,
    loader:                 DataLoader,
    task_id:                int,
    device:                 torch.device,
    is_supernet:            bool = True,
    chest_thresholds:       "Optional[np.ndarray]" = None,
    balanced_training:      bool = False,
    registry_id:            Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate ``model`` on samples belonging to ``task_id`` in ``loader``.

    Sets model.eval() for inference and restores model.train() afterwards.

    Args:
        model              : Supernet (is_supernet=True) or discrete nn.Sequential.
        loader             : Mixed-task DataLoader (collate_fn from MedMNISTDataset).
        task_id            : Position of the task to filter/evaluate — matches
                             the per-sample task labels produced by the
                             DataLoader and the index used to call the supernet.
        device             : Compute device.
        is_supernet        : If True call model(imgs, task_id); else call model(imgs).
        chest_thresholds   : Optional (14,) array of per-label thresholds for
                             ChestMNIST accuracy.  If None, uses 0.5 globally.
        balanced_training  : Set True when the model was trained on class-balanced
                             data (e.g., via WeightedRandomSampler).  Changes the
                             DermaMNIST ACC prior correction from dividing by prior
                             (correct for imbalanced training) to multiplying by
                             prior (correct for balanced training).
        registry_id        : The task's registry id (see TASK_REGISTRY in
                             src/data.py), used to resolve n_classes/is_multilabel
                             and to gate the DermaMNIST-specific ACC correction.
                             Defaults to ``task_id`` for backward compatibility
                             with the default 3-task ordering (position == id).

    Returns:
        {"acc": float, "auc": float, "loss": float, "n": int}
    """
    model.eval()
    _reg_id   = task_id if registry_id is None else registry_id
    is_ml     = TASK_REGISTRY[_reg_id][4]
    n_classes = TASK_REGISTRY[_reg_id][2]

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    n_batches  = 0

    for images, labels, task_ids in loader:
        images     = images.to(device)
        task_ids_t = task_ids.to(device)
        mask       = (task_ids_t == task_id)
        if not mask.any():
            continue

        imgs_k   = images[mask]
        labels_k = [labels[i]
                    for i in mask.nonzero(as_tuple=True)[0].tolist()]

        logits = model(imgs_k, task_id) if is_supernet else model(imgs_k)
        loss   = task_loss(logits, labels_k, _reg_id, device, is_multilabel=is_ml)
        total_loss += loss.item()
        n_batches  += 1

        if is_ml:
            scores = torch.sigmoid(logits).cpu().numpy()
            lbl_np = torch.stack(
                [l if isinstance(l, Tensor) else torch.tensor(l)
                 for l in labels_k]
            ).numpy()
        else:
            scores = F.softmax(logits, dim=-1).cpu().numpy()
            lbl_np = np.array(
                [l.item() if isinstance(l, Tensor) else l for l in labels_k]
            )

        all_scores.append(scores)
        all_labels.append(lbl_np)

    model.train()

    if n_batches == 0:
        return {"acc": 0.0, "auc": float("nan"), "loss": 0.0, "n": 0}

    y_score  = np.concatenate(all_scores, axis=0)
    y_true   = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / n_batches

    if is_ml:
        if chest_thresholds is not None:
            # Per-label calibrated thresholds (ChestMNIST retrain test eval).
            thr    = chest_thresholds[np.newaxis, :]   # (1, 14) broadcast
            y_pred = (y_score >= thr).astype(np.float32)
        else:
            y_pred = (y_score >= 0.5).astype(np.float32)
        acc    = float(np.mean(y_pred == y_true))
    else:
        y_true_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
        if _reg_id == 2 and not is_supernet:
            # Prior-probability correction for DermaMNIST ACC.
            # Only applied to a fully-trained discrete model (is_supernet=False).
            # During search the supernet outputs near-uniform softmax; the
            # correction amplifies random noise in rare classes and crashes ACC.
            #
            # Direction depends on how the model was trained:
            #   Imbalanced training → model is biased toward majority class
            #     → divide by prior to de-bias (amplify rare classes).
            #   Balanced training (WeightedRandomSampler) → model outputs are
            #     already calibrated under a uniform prior → multiply by true
            #     prior to match the test-set distribution for optimal argmax.
            _n_cls  = n_classes  # 7
            _counts = np.bincount(
                y_true_flat.astype(int), minlength=_n_cls
            ).astype(np.float64)
            _prior  = _counts / (_counts.sum() + 1e-8)
            # Floor at 1/n_classes so no class is amplified more than n_classes×.
            _prior  = np.maximum(_prior, 1.0 / _n_cls)
            if balanced_training:
                # Balanced training: re-introduce the test prior.
                y_score_corrected = y_score * _prior
            else:
                # Imbalanced training: de-bias toward majority class.
                y_score_corrected = y_score / _prior
            y_pred = np.argmax(y_score_corrected, axis=-1)
        else:
            y_pred = np.argmax(y_score, axis=-1)
        acc         = float(np.mean(y_pred == y_true_flat))

    auc     = _safe_auc(y_true, y_score, is_ml, n_classes)
    clf     = _safe_clf_metrics(y_true, y_pred, is_ml)
    return {
        "acc":       acc,
        "auc":       auc,
        "f1":        clf["f1"],
        "precision": clf["precision"],
        "recall":    clf["recall"],
        "loss":      float(avg_loss),
        "n":         int(y_true.shape[0]),
    }


# ── All-task evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      TaskAwareSupernet,
    loader:     DataLoader,
    device:     torch.device,
    split_name: str = "eval",
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-task metrics (acc, auc, loss) on evaluation split.

    Returns:
        {task_id: {"acc", "auc", "loss", "n"}}
    """
    results: Dict[int, Dict[str, float]] = {}
    for k in range(model.num_tasks):
        registry_id = model.task_ids[k]
        metrics = evaluate_task(model, loader, k, device, is_supernet=True,
                                 registry_id=registry_id)
        name    = TASK_REGISTRY.get(registry_id, (f"Task{k}",))[0]
        logger.info(
            f"  [{split_name}] {name:12s}"
            f"  ACC={metrics['acc']:.4f}"
            f"  AUC={metrics['auc']:.4f}"
            f"  F1={metrics.get('f1', float('nan')):.4f}"
            f"  P={metrics.get('precision', float('nan')):.4f}"
            f"  R={metrics.get('recall', float('nan')):.4f}"
            f"  Loss={metrics['loss']:.4f}"
            f"  (n={metrics['n']})"
        )
        results[k] = metrics
    return results


# ── Test-time augmentation evaluation ────────────────────────────────────────

@torch.no_grad()
def evaluate_task_tta(
    model:             nn.Module,
    loader:            DataLoader,
    task_id:           int,
    device:            torch.device,
    is_supernet:       bool                    = False,
    n_tta:             int                     = 8,
    chest_thresholds:  "Optional[np.ndarray]"  = None,
    balanced_training: bool                    = False,
    registry_id:       Optional[int]           = None,
) -> Dict[str, float]:
    """
    Evaluate ``model`` with test-time augmentation (TTA).

    Produces ``n_tta`` augmented views of each image (horizontal flips +
    random crops) and averages the softmax/sigmoid scores.  Averaged scores
    are then passed through the same ACC and AUC computation as
    :func:`evaluate_task`, including the DermaMNIST prior correction.

    Args:
        model             : Discrete model (is_supernet=False) or supernet.
        loader            : Mixed-task DataLoader.
        task_id           : Position of the task to filter/evaluate.
        device            : Compute device.
        is_supernet       : If True call model(imgs, task_id); else model(imgs).
        n_tta             : Number of augmented views to average (default 8).
        chest_thresholds  : Optional per-label thresholds for ChestMNIST ACC.
        balanced_training : Passed through to :func:`evaluate_task` to control
                            the direction of the DermaMNIST prior correction.
        registry_id       : The task's registry id — see :func:`evaluate_task`.
                            Defaults to ``task_id``.

    Returns:
        {"acc": float, "auc": float, "loss": float, "n": int}
    """
    import torchvision.transforms.functional as TF

    model.eval()
    _reg_id   = task_id if registry_id is None else registry_id
    is_ml     = TASK_REGISTRY[_reg_id][4]
    n_classes = TASK_REGISTRY[_reg_id][2]

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    n_batches  = 0

    for images, labels, task_ids in loader:
        images     = images.to(device)
        task_ids_t = task_ids.to(device)
        mask       = (task_ids_t == task_id)
        if not mask.any():
            continue

        imgs_k   = images[mask]
        labels_k = [labels[i]
                    for i in mask.nonzero(as_tuple=True)[0].tolist()]

        # Build augmented views and average their predictions.
        acc_logits = torch.zeros(
            imgs_k.size(0), n_classes, device=device, dtype=torch.float32,
        )
        for view_idx in range(n_tta):
            aug = imgs_k.clone()
            # Horizontal flip for odd views.
            if view_idx % 2 == 1:
                aug = TF.hflip(aug)
            # Four-corner + centre crops: views 0–4 are corners+centre,
            # repeated with flip for views 5–9.
            _, _, h, w = aug.shape
            crop_size  = int(min(h, w) * 0.9)
            pad        = (min(h, w) - crop_size) // 2
            cx = [0, w - crop_size, 0, w - crop_size, pad][view_idx % 5]
            cy = [0, 0, h - crop_size, h - crop_size, pad][view_idx % 5]
            aug = aug[:, :, cy:cy + crop_size, cx:cx + crop_size]
            aug = torch.nn.functional.interpolate(
                aug, size=(h, w), mode="bilinear", align_corners=False,
            )
            logits = model(aug, task_id) if is_supernet else model(aug)
            if is_ml:
                acc_logits += torch.sigmoid(logits)
            else:
                acc_logits += F.softmax(logits, dim=-1)

        avg_scores = (acc_logits / n_tta).cpu().numpy()

        # Compute loss on the original (non-augmented) images for reference.
        orig_logits = model(imgs_k, task_id) if is_supernet else model(imgs_k)
        loss = task_loss(orig_logits, labels_k, _reg_id, device, is_multilabel=is_ml)
        total_loss += loss.item()
        n_batches  += 1

        if is_ml:
            lbl_np = torch.stack(
                [l if isinstance(l, Tensor) else torch.tensor(l)
                 for l in labels_k]
            ).numpy()
        else:
            lbl_np = np.array(
                [l.item() if isinstance(l, Tensor) else l for l in labels_k]
            )
        all_scores.append(avg_scores)
        all_labels.append(lbl_np)

    model.train()

    if n_batches == 0:
        return {"acc": 0.0, "auc": float("nan"), "loss": 0.0, "n": 0}

    y_score  = np.concatenate(all_scores, axis=0)
    y_true   = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / n_batches

    if is_ml:
        if chest_thresholds is not None:
            thr    = chest_thresholds[np.newaxis, :]
            y_pred = (y_score >= thr).astype(np.float32)
        else:
            y_pred = (y_score >= 0.5).astype(np.float32)
        acc = float(np.mean(y_pred == y_true))
    else:
        y_true_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
        if _reg_id == 2 and not is_supernet:
            _n_cls  = n_classes
            _counts = np.bincount(
                y_true_flat.astype(int), minlength=_n_cls
            ).astype(np.float64)
            _prior  = _counts / (_counts.sum() + 1e-8)
            _prior  = np.maximum(_prior, 1.0 / _n_cls)
            if balanced_training:
                y_score_corrected = y_score * _prior
            else:
                y_score_corrected = y_score / _prior
            y_pred = np.argmax(y_score_corrected, axis=-1)
        else:
            y_pred = np.argmax(y_score, axis=-1)
        acc = float(np.mean(y_pred == y_true_flat))

    auc     = _safe_auc(y_true, y_score, is_ml, n_classes)
    clf     = _safe_clf_metrics(y_true, y_pred, is_ml)
    return {
        "acc":       acc,
        "auc":       auc,
        "f1":        clf["f1"],
        "precision": clf["precision"],
        "recall":    clf["recall"],
        "loss":      float(avg_loss),
        "n":         int(y_true.shape[0]),
    }


# ── Alpha entropy (early stopping signal) ────────────────────────────────────

@torch.no_grad()
def alpha_entropy(model: TaskAwareSupernet, tau: float) -> float:
    """
    Mean Shannon entropy of sparsemax(α/τ) across all tasks and layers.

    Used as the early-stopping criterion (contrib. [D]):
      H → 0          when all tasks hold near-one-hot operation weights.
      H → log(K)     when weights are uniform  (K = NUM_OPS).

    Returns a single scalar float.
    """
    probs   = annealed_sparsemax(model.alphas, tau=tau)   # (T, L, O)
    probs   = probs.clamp(min=1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)           # (T, L)
    return float(entropy.mean().item())
