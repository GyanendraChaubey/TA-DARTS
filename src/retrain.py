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
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import MedMNISTDataset
from .losses import task_loss
from .metrics import evaluate_task

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
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


def retrain_discrete(
    discrete_model: nn.Module,
    task_id:        int,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    test_loader:    DataLoader,
    num_epochs:     int              = 100,
    lr:             float            = 0.025,
    weight_decay:   float            = 3e-4,
    grad_clip:      float            = 5.0,
    device:         torch.device     = torch.device("cpu"),
    save_dir:       Optional[str]    = None,
) -> Dict[str, Any]:
    """
    Retrain a discrete architecture from random initialisation.

    Args:
        discrete_model : Output of :meth:`TaskAwareSupernet.discretize`.
        task_id        : Task index (determines loss and dataset metadata).
        train_loader   : Mixed-task training DataLoader.
        val_loader     : Mixed-task validation DataLoader (for best-model selection).
        test_loader    : Mixed-task test DataLoader (for final evaluation).
        num_epochs     : Number of retraining epochs.
        lr             : Initial SGD learning rate.
        weight_decay   : L2 regularisation.
        grad_clip      : Gradient clipping norm.
        device         : Compute device.
        save_dir       : If provided, saves the best checkpoint here.

    Returns:
        {"test_acc", "test_auc", "test_loss", "best_val_auc", "n_params"}
    """
    task_name = MedMNISTDataset.TASK_NAMES[task_id]
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Retraining discrete model — {task_name} (task {task_id})")
    logger.info(f"{'─' * 60}")

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

    best_val_auc: float = -1.0
    best_state         = None

    for epoch in range(1, num_epochs + 1):
        discrete_model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, labels, task_ids in train_loader:
            # Keep mask on CPU because DataLoader tensors are on CPU here.
            mask = (task_ids == task_id)
            if not mask.any():
                continue

            images_k = images[mask].to(device)
            labels_k = [labels[i]
                        for i in mask.nonzero(as_tuple=True)[0].tolist()]

            optimizer.zero_grad()
            logits = discrete_model(images_k)
            loss   = task_loss(logits, labels_k, task_id, device)
            loss.backward()
            nn.utils.clip_grad_norm_(discrete_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs:
            val_metrics = evaluate_task(
                discrete_model, val_loader, task_id, device, is_supernet=False,
            )
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                f"  [{task_name}] Epoch {epoch:3d}/{num_epochs}"
                f"  train_loss={avg_loss:.4f}"
                f"  val_ACC={val_metrics['acc']:.4f}"
                f"  val_AUC={val_metrics['auc']:.4f}"
            )
            if val_metrics["auc"] > best_val_auc or math.isnan(best_val_auc):
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
