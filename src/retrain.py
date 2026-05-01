"""
Post-search retraining of discrete architectures.

After architecture search, ``retrain_discrete()`` trains the selected
per-task model from scratch following the standard DARTS evaluation protocol:

  1. Re-initialise all weights.
  2. Train with SGD + CosineAnnealingLR.
  3. Track best validation AUC; restore best weights before final test eval.
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MedMNISTDataset
from .losses import task_loss
from .metrics import evaluate_task
from .supernet import TaskAwareSupernet

logger = logging.getLogger("MT-DARTS")


def _reinit_weights(model: nn.Module) -> None:
    """Re-initialise all learnable tensors from scratch (DARTS convention)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def retrain_discrete(
    discrete_model: nn.Module,
    task_id:        int,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    test_loader:    DataLoader,
    num_epochs:     int              = 200,
    lr:             float            = 0.025,
    weight_decay:   float            = 3e-4,
    grad_clip:      float            = 5.0,
    device:         torch.device     = torch.device("cpu"),
    save_dir:       Optional[str]    = None,
    mixup_alpha:    float            = 0.2,
    label_smoothing: float           = 0.1,
) -> Dict[str, Any]:
    """
    Retrain a discrete architecture from random initialisation.

    Improvements over baseline:
      - Mixup augmentation (alpha=0.2) for better generalisation.
      - Linear LR warmup then CosineAnnealingLR.
      - Label smoothing for CE tasks (PathMNIST, DermaMNIST).
      - Val AUC checkpoint selection every epoch.

    Args:
        discrete_model  : Output of :meth:`TaskAwareSupernet.discretize`.
        task_id         : Task index (determines loss and dataset metadata).
        train_loader    : Mixed-task training DataLoader.
        val_loader      : Mixed-task validation DataLoader (best-model selection).
        test_loader     : Mixed-task test DataLoader (final evaluation).
        num_epochs      : Number of retraining epochs.
        lr              : Initial SGD learning rate.
        weight_decay    : L2 regularisation.
        grad_clip       : Gradient clipping norm.
        device          : Compute device.
        save_dir        : If provided, saves the best checkpoint here.
        mixup_alpha     : Beta distribution alpha for Mixup (0 = disabled).
        label_smoothing : Label smoothing for CE tasks (0 = disabled).

    Returns:
        {"test_acc", "test_auc", "test_loss", "best_val_auc", "n_params"}
    """
    task_name     = MedMNISTDataset.TASK_NAMES[task_id]
    is_multilabel = task_id in TaskAwareSupernet.MULTILABEL_TASKS
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Retraining discrete model — {task_name} (task {task_id})")
    logger.info(f"  mixup_alpha={mixup_alpha}  label_smoothing={label_smoothing}")
    logger.info(f"{'─' * 60}")

    _reinit_weights(discrete_model)
    discrete_model = discrete_model.to(device)

    n_params = sum(p.numel() for p in discrete_model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.SGD(
        discrete_model.parameters(),
        lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True,
    )

    # Warmup for first ~5% of epochs, then cosine anneal.
    warmup_epochs = max(5, num_epochs // 20)
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    best_val_auc: float = -1.0
    best_state         = None

    for epoch in range(1, num_epochs + 1):
        discrete_model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, labels, task_ids in train_loader:
            mask = (task_ids == task_id)
            if not mask.any():
                continue

            idx_list = mask.nonzero(as_tuple=True)[0].tolist()
            images_k = images[mask].to(device)
            labels_k = [labels[i] for i in idx_list]

            # ── Mixup ────────────────────────────────────────────────────────
            use_mixup = mixup_alpha > 0.0 and images_k.size(0) >= 2
            if use_mixup:
                lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam  = max(lam, 1.0 - lam)   # keep dominant sample
                shuf = torch.randperm(images_k.size(0), device=device)
                images_k_mix   = lam * images_k + (1.0 - lam) * images_k[shuf]
                labels_k_shuf  = [labels_k[i] for i in shuf.tolist()]
            else:
                images_k_mix  = images_k
                labels_k_shuf = labels_k
                lam           = 1.0

            optimizer.zero_grad()
            logits = discrete_model(images_k_mix)

            if use_mixup and is_multilabel:
                # Mix float multi-hot label vectors directly for BCE.
                lbl_a = torch.stack(labels_k).to(device)
                lbl_b = torch.stack(labels_k_shuf).to(device)
                loss  = F.binary_cross_entropy_with_logits(
                    logits, lam * lbl_a + (1.0 - lam) * lbl_b,
                )
            elif use_mixup:
                # Weighted sum of CE losses for the two mixed classes.
                loss = (
                    lam * task_loss(logits, labels_k, task_id, device,
                                    label_smoothing)
                    + (1.0 - lam) * task_loss(logits, labels_k_shuf, task_id,
                                              device, label_smoothing)
                )
            else:
                loss = task_loss(logits, labels_k, task_id, device,
                                 label_smoothing)

            loss.backward()
            nn.utils.clip_grad_norm_(discrete_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Evaluate every epoch for fine-grained best-checkpoint selection.
        val_metrics = evaluate_task(
            discrete_model, val_loader, task_id, device, is_supernet=False,
        )
        if epoch % 10 == 0 or epoch == num_epochs:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"  [{task_name}] Epoch {epoch:3d}/{num_epochs}"
                f"  train_loss={avg_loss:.4f}"
                f"  val_ACC={val_metrics['acc']:.4f}"
                f"  val_AUC={val_metrics['auc']:.4f}"
            )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state   = copy.deepcopy(discrete_model.state_dict())

    if best_state is not None:
        discrete_model.load_state_dict(best_state)

    test_metrics = evaluate_task(
        discrete_model, test_loader, task_id, device, is_supernet=False,
    )
    logger.info(
        f"  [{task_name}] FINAL TEST"
        f"  ACC={test_metrics['acc']:.4f}"
        f"  AUC={test_metrics['auc']:.4f}"
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"discrete_{task_name.lower()}_best.pt")
        torch.save(discrete_model.state_dict(), path)
        logger.info(f"  Saved best model → {path}")

    return {
        "test_acc":     test_metrics["acc"],
        "test_auc":     test_metrics["auc"],
        "test_loss":    test_metrics["loss"],
        "best_val_auc": best_val_auc,
        "n_params":     n_params,
    }
