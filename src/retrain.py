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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import DataLoader as _DL, WeightedRandomSampler

from .data import TASK_REGISTRY, build_weighted_sampler, compute_chest_pos_weights
from .losses import task_loss
from .metrics import evaluate_task

logger = logging.getLogger("MT-DARTS")


@torch.no_grad()
def _calibrate_chest_thresholds(
    model:      nn.Module,
    val_loader: DataLoader,
    device:     torch.device,
    task_id:    int,
    n_labels:   int = 14,
) -> np.ndarray:
    """
    Find per-label decision thresholds that maximise per-label F1 on the
    validation set for ChestMNIST.

    Sweeps 49 evenly-spaced thresholds in [0.02, 0.98] per label and picks
    the one with the highest F1.  Returns an array of shape (n_labels,) used
    at test-time instead of the fixed 0.5.

    Accuracy is *not* used as the calibration objective: with label
    prevalence as low as ~0.2%, the accuracy-maximising threshold degenerates
    to "always predict negative" (trivially ~99.8% accurate, but recall = 0).
    F1 penalises that collapse directly.  Labels with zero positives in the
    validation split are left at the default 0.5 — every threshold is
    equally uninformative for them (F1 = 0 everywhere), so there is no
    signal to calibrate against.

    Args:
        task_id : Position of ChestMNIST within the DataLoader's active task
                  list (matches the per-sample task labels it yields).
    """
    model.eval()
    all_scores, all_labels = [], []
    for images, labels, task_ids in val_loader:
        mask = (task_ids == task_id)
        if not mask.any():
            continue
        imgs_k   = images[mask].to(device)
        labels_k = [labels[i] for i in mask.nonzero(as_tuple=True)[0].tolist()]
        logits   = model(imgs_k)
        scores   = torch.sigmoid(logits).cpu().numpy()
        lbl_np   = torch.stack(
            [l if isinstance(l, torch.Tensor) else torch.tensor(l)
             for l in labels_k]
        ).numpy()
        all_scores.append(scores)
        all_labels.append(lbl_np)

    if not all_scores:
        return np.full(n_labels, 0.5)

    y_score = np.concatenate(all_scores, axis=0)  # (N, 14)
    y_true  = np.concatenate(all_labels, axis=0)  # (N, 14)
    best_thresholds = np.full(n_labels, 0.5)

    for c in range(n_labels):
        n_pos = y_true[:, c].sum()
        if n_pos == 0:
            continue  # no positive examples to calibrate against — keep 0.5
        best_f1 = -1.0
        for thr in np.linspace(0.02, 0.98, 49):
            pred = y_score[:, c] >= thr
            tp = np.logical_and(pred, y_true[:, c] == 1).sum()
            fp = np.logical_and(pred, y_true[:, c] == 0).sum()
            fn = np.logical_and(~pred, y_true[:, c] == 1).sum()
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = thr

    logger.info(
        f"  [ChestMNIST] Calibrated thresholds (min={best_thresholds.min():.2f}"
        f"  max={best_thresholds.max():.2f}"
        f"  mean={best_thresholds.mean():.2f})"
    )
    model.train()
    return best_thresholds


def _cutmix_batch(
    images: torch.Tensor,
    labels: list,
    alpha: float = 1.0,
):
    """
    CutMix augmentation (Yun et al., 2019).
    Returns (mixed_images, labels_a, labels_b, lam).
    ``lam`` is the recomputed pixel-area ratio after clipping the bounding box.
    """
    lam_beta = float(np.random.beta(alpha, alpha))
    lam_beta = max(lam_beta, 1.0 - lam_beta)   # keep dominant sample
    b, c, h, w = images.shape
    shuf = torch.randperm(b)
    cut_ratio = np.sqrt(1.0 - lam_beta)
    cut_h = max(1, int(h * cut_ratio))
    cut_w = max(1, int(w * cut_ratio))
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[shuf, :, y1:y2, x1:x2]
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(h * w)
    labels_b = [labels[i] for i in shuf.tolist()]
    return mixed, labels, labels_b, lam


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
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def retrain_discrete(
    discrete_model:       nn.Module,
    task_id:              int,
    registry_id:          int,
    train_loader:         DataLoader,
    val_loader:           DataLoader,
    test_loader:          DataLoader,
    num_epochs:           int              = 200,
    lr:                   float            = 0.025,
    weight_decay:         float            = 3e-4,
    grad_clip:            float            = 5.0,
    device:               torch.device     = torch.device("cpu"),
    save_dir:             Optional[str]    = None,
    mixup_alpha:          float            = 0.2,
    label_smoothing:      float            = 0.1,
    use_weighted_sampler: bool             = True,
    use_tta:              bool             = False,
) -> Dict[str, Any]:
    """
    Retrain a discrete architecture from random initialisation.

    Improvements over baseline:
      - Mixup + CutMix augmentation (50/50 random choice) for DermaMNIST.
      - WeightedRandomSampler for class-balanced batches on DermaMNIST.
      - Linear LR warmup then CosineAnnealingLR (or WarmRestarts for DermaMNIST).
      - Label smoothing for CE tasks (PathMNIST, DermaMNIST).
      - Val AUC checkpoint selection every epoch.
      - Optional test-time augmentation (8-view ensemble) at final eval.

    Args:
        discrete_model        : Output of :meth:`TaskAwareSupernet.discretize`.
        task_id               : Position of this task within the active task
                                list — matches the per-sample task labels
                                produced by the DataLoaders.
        registry_id           : This task's id in TASK_REGISTRY (src/data.py).
                                Used to resolve its name/class-count/multilabel
                                flag and to gate the DermaMNIST/ChestMNIST-
                                specific tuning below (registry ids 2 and 1),
                                independent of where the task lands in a
                                custom --tasks selection.
        train_loader          : Mixed-task training DataLoader.
        val_loader            : Mixed-task validation DataLoader (best-model selection).
        test_loader           : Mixed-task test DataLoader (final evaluation).
        num_epochs            : Number of retraining epochs.
        lr                    : Initial SGD learning rate.
        weight_decay          : L2 regularisation.
        grad_clip             : Gradient clipping norm.
        device                : Compute device.
        save_dir              : If provided, saves the best checkpoint here.
        mixup_alpha           : Beta distribution alpha for Mixup (0 = disabled).
        label_smoothing       : Label smoothing for CE tasks (0 = disabled).
        use_weighted_sampler  : If True and this is DermaMNIST, build a
                                class-balanced DataLoader for retraining.
        use_tta               : If True, use 8-view TTA for final test evaluation.

    Returns:
        {"test_acc", "test_auc", "test_loss", "best_val_auc", "n_params"}
    """
    task_name     = TASK_REGISTRY[registry_id][0]
    is_multilabel = TASK_REGISTRY[registry_id][4]
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Retraining discrete model — {task_name} (task {task_id})")
    logger.info(f"  mixup_alpha={mixup_alpha}  label_smoothing={label_smoothing}")
    logger.info(f"{'─' * 60}")

    _reinit_weights(discrete_model)
    discrete_model = discrete_model.to(device)

    # ── DermaMNIST: build class-balanced DataLoader ────────────────────────
    # Wraps the underlying dataset in a WeightedRandomSampler so each class
    # appears with equal expected frequency, addressing the 67% majority bias.
    # When the sampler is active, class weights in task_loss are disabled to
    # avoid double-correcting imbalance (sampler already balances classes).
    _sampler_active = False
    if registry_id == 2 and use_weighted_sampler:
        try:
            sampler = build_weighted_sampler(
                task_id=task_id,
                dataset=train_loader.dataset,
            )
            active_train_loader = _DL(
                train_loader.dataset,
                batch_size=train_loader.batch_size or 64,
                sampler=sampler,
                num_workers=train_loader.num_workers,
                collate_fn=train_loader.collate_fn,
                pin_memory=train_loader.pin_memory,
            )
            _sampler_active = True
            logger.info("  [DermaMNIST] Using WeightedRandomSampler for class-balanced batches.")
            logger.info("  [DermaMNIST] Class weights in loss disabled (sampler already balances).")
        except Exception as exc:
            logger.warning(f"  [DermaMNIST] WeightedRandomSampler failed ({exc}); falling back to original loader.")
            active_train_loader = train_loader
    else:
        active_train_loader = train_loader

    # When the sampler is active, class weights would double-correct the
    # imbalance that the sampler already handles → disable them.
    _use_class_weights = not _sampler_active

    # ── ChestMNIST: per-label positive weights ────────────────────────────
    # The static _CHEST_POS_WEIGHT = 10 × ones(14) is a rough approximation.
    # Computing exact neg/pos ratios from the training split is far more
    # accurate given ChestMNIST label prevalence ranges from ~0.2% to ~19%.
    if registry_id == 1:
        _chest_pw = compute_chest_pos_weights(train_loader.dataset, task_id=task_id,
                                               n_labels=TASK_REGISTRY[registry_id][2])
        logger.info(
            f"  [ChestMNIST] Per-label pos_weights computed from training data"
            f"  (min={_chest_pw.min():.1f}  max={_chest_pw.max():.1f}"
            f"  mean={_chest_pw.mean():.1f})"
        )
    else:
        _chest_pw = None

    # ── DermaMNIST: focal loss for hard-example mining ────────────────────
    # When the sampler balances classes, class weights are disabled.  Focal
    # loss fills the gap by adaptively down-weighting easy examples so
    # gradients concentrate on hard, misclassified samples — more effective
    # than static class weights for a severe long-tail distribution.
    _use_focal = (registry_id == 2)

    n_params = sum(p.numel() for p in discrete_model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    # ── FLOPs (multiply-accumulate operations) ────────────────────────────────
    # Uses thop if available; falls back gracefully so training still runs
    # without it.  Install with: pip install thop
    flops_m: float = float("nan")
    try:
        from thop import profile as thop_profile
        # Infer spatial size from the first stem conv's expected input.
        # We use img_size if accessible, otherwise default to 28.
        _img_size = getattr(discrete_model, "_img_size", 28)
        _dummy    = torch.zeros(1, 3, _img_size, _img_size).to(device)
        _flops, _ = thop_profile(discrete_model, inputs=(_dummy,), verbose=False)
        flops_m   = _flops / 1e6
        logger.info(f"  FLOPs: {flops_m:.1f} M")
    except Exception as _exc:
        logger.debug(f"  FLOPs not computed (thop unavailable or failed: {_exc})")

    optimizer = torch.optim.SGD(
        discrete_model.parameters(),
        lr=lr, momentum=0.9,
        # DermaMNIST needs stronger L2 regularisation (small 7k dataset).
        weight_decay=weight_decay * 10 if registry_id == 2 else weight_decay,
        nesterov=True,
    )

    # ── Learning-rate scheduler ────────────────────────────────────────────
    # DermaMNIST: CosineAnnealingWarmRestarts after a short warmup.
    #   Warm restarts help escape the narrow, imbalance-induced local minima
    #   that a simple cosine anneal tends to get trapped in on the 7k set.
    # PathMNIST / ChestMNIST: standard warmup → CosineAnnealingLR.
    if registry_id == 2:
        warmup_epochs = 10
        warmup_sched  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        restart_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, restart_sched],
            milestones=[warmup_epochs],
        )
    else:
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

    best_val_auc: float  = -1.0
    best_state           = None
    # DermaMNIST is small (7k samples) — allow more patience so retraining
    # does not stop prematurely before augmentation-driven gains appear.
    retrain_patience     = 40 if registry_id == 2 else 20
    retrain_no_improve   = 0

    # ── SWA (Stochastic Weight Averaging) ────────────────────────────────────
    # Averages weights in the last 25% of training (or last 50 epochs) to
    # find a flatter, better-generalising minimum without extra training time.
    # Replaces the single best-checkpoint restore at the end of training.
    _swa_start   = max(num_epochs * 3 // 4, num_epochs - 50)
    _swa_model   = torch.optim.swa_utils.AveragedModel(discrete_model)
    _swa_updated = False

    for epoch in range(1, num_epochs + 1):
        # Scale drop_prob linearly from 0.0 to 0.2
        drop_path_prob = 0.2 * (epoch - 1) / (num_epochs - 1) if num_epochs > 1 else 0.0
        for m in discrete_model.modules():
            if hasattr(m, 'drop_prob'):
                m.drop_prob = drop_path_prob

        discrete_model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, labels, task_ids in active_train_loader:
            mask = (task_ids == task_id)
            if not mask.any():
                continue

            idx_list = mask.nonzero(as_tuple=True)[0].tolist()
            images_k = images[mask].to(device)
            labels_k = [labels[i] for i in idx_list]

            # ── Augmentation ─────────────────────────────────────────────────
            # DermaMNIST: 50/50 random choice of CutMix or Mixup.
            #   CutMix preserves local textures; Mixup regularises globally.
            #   Combined they improve calibration on the 7-class imbalanced set.
            # Other tasks: standard Mixup only.
            effective_alpha = (mixup_alpha * 2.0) if (registry_id == 2 and mixup_alpha > 0) else mixup_alpha
            use_aug = effective_alpha > 0.0 and images_k.size(0) >= 2

            if use_aug and registry_id == 2 and np.random.rand() < 0.5:
                # CutMix branch
                images_k_mix, labels_k, labels_k_shuf, lam = _cutmix_batch(
                    images_k, labels_k, alpha=effective_alpha,
                )
                images_k_mix = images_k_mix.to(device)
            elif use_aug:
                # Mixup branch
                lam  = float(np.random.beta(effective_alpha, effective_alpha))
                lam  = max(lam, 1.0 - lam)
                shuf = torch.randperm(images_k.size(0), device=device)
                images_k_mix   = lam * images_k + (1.0 - lam) * images_k[shuf]
                labels_k_shuf  = [labels_k[i] for i in shuf.tolist()]
            else:
                images_k_mix  = images_k
                labels_k_shuf = labels_k
                lam           = 1.0

            optimizer.zero_grad()
            logits = discrete_model(images_k_mix)

            if use_aug and is_multilabel:
                # Mix float multi-hot label vectors directly for BCE.
                lbl_a = torch.stack(labels_k).to(device)
                lbl_b = torch.stack(labels_k_shuf).to(device)
                pw    = _chest_pw if _chest_pw is not None else None
                from .losses import _CHEST_POS_WEIGHT
                pw = pw if pw is not None else _CHEST_POS_WEIGHT
                loss  = F.binary_cross_entropy_with_logits(
                    logits, lam * lbl_a + (1.0 - lam) * lbl_b,
                    pos_weight=pw.to(device),
                )
            elif use_aug:
                # Weighted sum of CE losses for the two mixed classes.
                loss = (
                    lam * task_loss(logits, labels_k, registry_id, device,
                                    label_smoothing,
                                    use_class_weights=_use_class_weights,
                                    use_focal=_use_focal,
                                    chest_pos_weight=_chest_pw,
                                    is_multilabel=is_multilabel)
                    + (1.0 - lam) * task_loss(logits, labels_k_shuf, registry_id,
                                              device, label_smoothing,
                                              use_class_weights=_use_class_weights,
                                              use_focal=_use_focal,
                                              chest_pos_weight=_chest_pw,
                                              is_multilabel=is_multilabel)
                )
            else:
                # DermaMNIST: use minimal label smoothing so decision
                # boundaries stay sharp — critical for accuracy on a
                # 7-class imbalanced dataset.
                effective_ls = 0.02 if registry_id == 2 else label_smoothing
                loss = task_loss(logits, labels_k, registry_id, device,
                                 effective_ls,
                                 use_class_weights=_use_class_weights,
                                 use_focal=_use_focal,
                                 chest_pos_weight=_chest_pw,
                                 is_multilabel=is_multilabel)

            loss.backward()
            nn.utils.clip_grad_norm_(discrete_model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Accumulate SWA snapshot once we enter the averaging window.
        if epoch >= _swa_start:
            _swa_model.update_parameters(discrete_model)
            _swa_updated = True

        # Evaluate discrete_model every epoch for fine-grained best-checkpoint selection.
        val_metrics = evaluate_task(
            discrete_model, val_loader, task_id, device, is_supernet=False,
            balanced_training=_sampler_active, registry_id=registry_id,
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
            best_val_auc       = val_metrics["auc"]
            best_state         = copy.deepcopy(discrete_model.state_dict())
            retrain_no_improve = 0
        else:
            retrain_no_improve += 1
            if retrain_no_improve >= retrain_patience:
                logger.info(
                    f"  [{task_name}] Early stopping at epoch {epoch}"
                    f" — no val AUC improvement for {retrain_patience} epochs"
                    f" (best={best_val_auc:.4f})."
                )
                break

    if _swa_updated:
        # Copy averaged parameters into discrete_model.
        _swa_sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in _swa_model.state_dict().items()
            if k != "n_averaged"
        }
        discrete_model.load_state_dict(_swa_sd)
        # Refresh BN running stats using task-filtered images (cumulative MA).
        for _m in discrete_model.modules():
            if isinstance(_m, nn.BatchNorm2d):
                _m.reset_running_stats()
                _m.momentum = None   # triggers 1/N cumulative moving average
        discrete_model.train()
        with torch.no_grad():
            for _imgs, _lbls, _tids in active_train_loader:
                _mask = (_tids == task_id)
                if not _mask.any():
                    continue
                discrete_model(_imgs[_mask].to(device))
        logger.info(
            f"  [{task_name}] SWA applied"
            f" (averaged from epoch {_swa_start}/{num_epochs})."
        )
    elif best_state is not None:
        discrete_model.load_state_dict(best_state)

    # ChestMNIST: calibrate per-label thresholds on val set before test eval
    # to maximise per-label F1 (fixed 0.5 is suboptimal with calibrated pos_weight,
    # and accuracy-maximising thresholds collapse to all-negative on rare labels).
    if registry_id == 1:
        chest_thresholds = _calibrate_chest_thresholds(
            discrete_model, val_loader, device, task_id,
            n_labels=TASK_REGISTRY[registry_id][2],
        )
    else:
        chest_thresholds = None

    if use_tta:
        from .metrics import evaluate_task_tta
        test_metrics = evaluate_task_tta(
            discrete_model, test_loader, task_id, device, is_supernet=False,
            chest_thresholds=chest_thresholds,
            balanced_training=_sampler_active, registry_id=registry_id,
        )
    else:
        test_metrics = evaluate_task(
            discrete_model, test_loader, task_id, device, is_supernet=False,
            chest_thresholds=chest_thresholds,
            balanced_training=_sampler_active, registry_id=registry_id,
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
        "test_acc":       test_metrics["acc"],
        "test_auc":       test_metrics["auc"],
        "test_f1":        test_metrics.get("f1",        float("nan")),
        "test_precision": test_metrics.get("precision", float("nan")),
        "test_recall":    test_metrics.get("recall",    float("nan")),
        "test_loss":      test_metrics["loss"],
        "best_val_auc":   best_val_auc,
        "n_params":       n_params,
        "flops_m":        flops_m,
    }
