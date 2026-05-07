"""
Evaluation metrics for MT-DARTS v2.

  evaluate_task()  — per-task acc + AUC + loss (supernet or discrete model).
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

from .data import MedMNISTDataset
from .losses import task_loss
from .normalizers import annealed_sparsemax
from .supernet import TaskAwareSupernet

logger = logging.getLogger("MT-DARTS")

# ── Optional sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── AUC helper ────────────────────────────────────────────────────────────────

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
    model:       nn.Module,
    loader:      DataLoader,
    task_id:     int,
    device:      torch.device,
    is_supernet: bool = True,
    chest_thresholds: "Optional[np.ndarray]" = None,
) -> Dict[str, float]:
    """
    Evaluate ``model`` on samples belonging to ``task_id`` in ``loader``.

    Sets model.eval() for inference and restores model.train() afterwards.

    Args:
        model              : Supernet (is_supernet=True) or discrete nn.Sequential.
        loader             : Mixed-task DataLoader (collate_fn from MedMNISTDataset).
        task_id            : Which task to filter and evaluate.
        device             : Compute device.
        is_supernet        : If True call model(imgs, task_id); else call model(imgs).
        chest_thresholds   : Optional (14,) array of per-label thresholds for
                             ChestMNIST accuracy.  If None, uses 0.5 globally.

    Returns:
        {"acc": float, "auc": float, "loss": float, "n": int}
    """
    model.eval()
    is_ml     = task_id in TaskAwareSupernet.MULTILABEL_TASKS
    n_classes = MedMNISTDataset.NUM_CLASSES[task_id]

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
        loss   = task_loss(logits, labels_k, task_id, device)
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
        y_pred      = np.argmax(y_score, axis=-1)
        y_true_flat = y_true.squeeze() if y_true.ndim > 1 else y_true
        acc         = float(np.mean(y_pred == y_true_flat))

    auc = _safe_auc(y_true, y_score, is_ml, n_classes)
    return {"acc": acc, "auc": auc, "loss": float(avg_loss),
            "n": int(y_true.shape[0])}


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
        metrics = evaluate_task(model, loader, k, device, is_supernet=True)
        name    = MedMNISTDataset.TASK_NAMES.get(k, f"Task{k}")
        logger.info(
            f"  [{split_name}] {name:12s}"
            f"  ACC={metrics['acc']:.4f}"
            f"  AUC={metrics['auc']:.4f}"
            f"  Loss={metrics['loss']:.4f}"
            f"  (n={metrics['n']})"
        )
        results[k] = metrics
    return results


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
