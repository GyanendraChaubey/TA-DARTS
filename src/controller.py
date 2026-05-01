"""
SearchController — bilevel optimisation engine for MT-DARTS v2.

Design:
  Phase 1  — update weight parameters (alphas frozen).
  Phase 2  — update architecture parameters α every ``alpha_update_freq``
              weight steps (delayed update, contrib. [C]).
  Scheduler — CosineAnnealingLR per epoch + sparsemax temperature annealing
              (contrib. [B]).
  Checkpoint — full state save/resume including current tau.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .data import MedMNISTDataset
from .losses import task_loss
from .normalizers import annealed_sparsemax
from .supernet import TaskAwareSupernet

logger = logging.getLogger("MT-DARTS")


class SearchController:
    """
    Manages the bilevel optimisation loop.

    Parameters
    ----------
    model : TaskAwareSupernet
    epochs : int
        Total search epochs (used by cosine LR scheduler).
    lr_weights / lr_alphas : float
        Learning rates for weight and alpha optimisers.
    tau_init / anneal_factor / anneal_interval : float / float / int
        Sparsemax temperature schedule: τ = τ_init × a^(epoch // m).
    alpha_update_freq : int
        Run Phase 2 every this many weight steps (contrib. [C]).
    """

    def __init__(
        self,
        model:             TaskAwareSupernet,
        epochs:            int,
        lr_weights:        float = 0.025,
        lr_alphas:         float = 3e-4,
        momentum:          float = 0.9,
        weight_decay_w:    float = 3e-4,
        weight_decay_a:    float = 1e-3,
        grad_clip:         float = 5.0,
        eta_min:           float = 1e-4,
        tau_init:          float = 1.5,
        anneal_factor:     float = 0.75,
        anneal_interval:   int   = 5,
        alpha_update_freq: int   = 10,
        label_smoothing:   float = 0.0,
    ) -> None:
        self.model             = model
        self.grad_clip         = grad_clip
        self.num_tasks         = model.num_tasks
        self._step             = 0

        # Contrib. [B] — temperature annealing
        self.tau_init          = tau_init
        self.anneal_factor     = anneal_factor
        self.anneal_interval   = anneal_interval
        self._current_tau: float = tau_init

        # Contrib. [C] — delayed alpha updates
        self.alpha_update_freq = alpha_update_freq

        # Label smoothing for CE tasks during search
        self.label_smoothing   = label_smoothing

        self.opt_weights = torch.optim.SGD(
            model.weight_parameters(),
            lr=lr_weights,
            momentum=momentum,
            weight_decay=weight_decay_w,
            nesterov=True,
        )
        self.opt_arch = torch.optim.Adam(
            model.arch_parameters(),
            lr=lr_alphas,
            betas=(0.5, 0.999),
            weight_decay=weight_decay_a,
        )
        # CosineAnnealingLR called once per epoch (standard DARTS convention)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_weights,
            T_max=epochs,
            eta_min=eta_min,
        )

    # ── Internal: loss computation ────────────────────────────────────────────

    def _compute_loss(
        self,
        images:   torch.Tensor,
        labels:   list,
        task_ids: torch.Tensor,
        device:   torch.device,
    ) -> torch.Tensor:
        """Weighted-average loss across tasks present in this mini-batch."""
        total_loss = torch.tensor(0.0, device=device)
        n_samples  = 0

        for k in range(self.num_tasks):
            mask = (task_ids == k)
            if not mask.any():
                continue
            imgs_k   = images[mask]
            labels_k = [labels[i]
                        for i in mask.nonzero(as_tuple=True)[0].tolist()]
            # Contrib. [A][B]: forward with current annealed tau
            logits_k = self.model(imgs_k, k, tau=self._current_tau)
            loss_k   = task_loss(logits_k, labels_k, k, device,
                                 self.label_smoothing)
            total_loss = total_loss + loss_k * mask.sum()
            n_samples += mask.sum().item()

        if n_samples > 0:
            total_loss = total_loss / n_samples
        return total_loss

    # ── Bilevel step ──────────────────────────────────────────────────────────

    def step(
        self,
        train_batch: Tuple,
        val_batch:   Tuple,
        device:      torch.device,
    ) -> Tuple[float, float]:
        """
        Execute one bilevel optimisation step.

        Returns ``(loss_w, loss_a)`` as Python floats.
        ``loss_a`` is 0.0 on steps where Phase 2 is skipped (contrib. [C]).
        """
        images_tr, labels_tr, tids_tr = train_batch
        images_va, labels_va, tids_va = val_batch
        images_tr = images_tr.to(device)
        images_va = images_va.to(device)
        tids_tr   = tids_tr.to(device)
        tids_va   = tids_va.to(device)

        # ── Phase 1: weight update (alphas frozen) ───────────────────────────
        self.model.alphas.requires_grad_(False)
        self.opt_weights.zero_grad()
        loss_w = self._compute_loss(images_tr, labels_tr, tids_tr, device)
        loss_w.backward()
        nn.utils.clip_grad_norm_(self.model.weight_parameters(), self.grad_clip)
        self.opt_weights.step()

        # ── Phase 2: alpha update (every alpha_update_freq steps) [C] ────────
        loss_a_val = 0.0
        if (self._step + 1) % self.alpha_update_freq == 0:
            self.model.alphas.requires_grad_(True)
            self.opt_arch.zero_grad(set_to_none=True)
            for p in self.model.weight_parameters():
                p.grad = None

            loss_a = self._compute_loss(images_va, labels_va, tids_va, device)
            loss_a.backward()
            self.opt_arch.step()
            loss_a_val = loss_a.item()
        else:
            # Re-enable grad for alphas so eval code can query them
            self.model.alphas.requires_grad_(True)

        self._step += 1
        return loss_w.item(), loss_a_val

    # ── Scheduler (call once per epoch) ──────────────────────────────────────

    def step_scheduler(self, epoch: int) -> None:
        """
        Advance cosine LR and sparsemax temperature (contrib. [B]).

        τ_epoch = τ_init × anneal_factor^(epoch // anneal_interval), ≥ 1e-2.
        """
        self.scheduler.step()
        self._current_tau = max(
            self.tau_init * (self.anneal_factor ** (epoch // self.anneal_interval)),
            1e-2,
        )
        logger.info(
            f"  LR → {self.scheduler.get_last_lr()[0]:.6f}"
            f"  |  τ → {self._current_tau:.4f}"
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch:    int,
        ckpt_dir: str = "checkpoints",
        tag:      str = "latest",
    ) -> str:
        """Save full training state; returns path written."""
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"mt_darts_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "step":        self._step,
            "model_state": self.model.state_dict(),
            "opt_weights": self.opt_weights.state_dict(),
            "opt_arch":    self.opt_arch.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "current_tau": self._current_tau,
        }, path)
        logger.info(f"  [ckpt] Saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> int:
        """Load training state; returns epoch to resume from."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt_weights.load_state_dict(ckpt["opt_weights"])
        self.opt_arch.load_state_dict(ckpt["opt_arch"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self._step        = ckpt["step"]
        self._current_tau = ckpt.get("current_tau", self.tau_init)
        logger.info(
            f"  [ckpt] Resumed from {path} "
            f"(epoch {ckpt['epoch']}, τ={self._current_tau:.4f})"
        )
        return ckpt["epoch"]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def log_arch_distribution(self) -> None:
        """Log the current sparsemax architecture weights for all tasks."""
        from .ops import OP_NAMES  # local import avoids top-level cycle

        logger.info(
            f"\n[step {self._step}] Architecture distribution "
            f"(sparsemax α, τ={self._current_tau:.4f})"
            f"  LR={self.scheduler.get_last_lr()[0]:.5f}"
        )
        soft = annealed_sparsemax(self.model.alphas, tau=self._current_tau)
        for t in range(self.num_tasks):
            name = MedMNISTDataset.TASK_NAMES.get(t, f"Task{t}")
            logger.info(f"  {name}:")
            for lay in range(self.model.num_layers):
                probs    = soft[t, lay].tolist()
                best_idx = int(soft[t, lay].argmax())
                bar      = "  ".join(
                    f"{OP_NAMES[i][:10]:10s}={p:.3f}"
                    for i, p in enumerate(probs)
                )
                logger.info(f"    Layer {lay}: {bar}  ← {OP_NAMES[best_idx]}")
