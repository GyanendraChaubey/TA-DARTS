"""
SearchController — bilevel optimisation engine for MT-DARTS.

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

from .losses import task_loss
from .normalizers import annealed_sparsemax, annealed_softmax
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
        lr_weights:        float = 1e-3,
        lr_alphas:         float = 3e-4,
        momentum:          float = 0.9,
        weight_decay_w:    float = 3e-4,
        weight_decay_a:    float = 1e-3,
        grad_clip:         float = 5.0,
        eta_min:           float = 1e-4,
        tau_init:          float = 1.5,
        anneal_factor:     float = 0.95,   # fixed: 0.90 caused premature collapse
        anneal_interval:   int   = 5,      # fixed: 3 dropped tau too fast
        tau_min:           float = 0.30,
        alpha_update_freq: int   = 10,
        search_micro_batch:int   = 0,
        label_smoothing:   float = 0.0,
        arch_reg_lambda:   float = 0.01,   # entropy regularisation strength
        use_sparsemax:     bool  = True,   # ablation: sparsemax vs softmax
        task_normalize:    bool  = True,   # ablation: task-equal vs sample-weighted loss
    ) -> None:
        self.model             = model
        self.grad_clip         = grad_clip
        self.num_tasks         = model.num_tasks
        self._step             = 0

        # Ablation: alpha normaliser used for the entropy term and diagnostics
        # (the forward-pass normaliser lives on the model itself, see
        # TaskAwareSupernet.use_sparsemax — kept in sync by the caller so both
        # halves of the search use the same normaliser).
        self.use_sparsemax     = use_sparsemax
        self._alpha_normalizer = annealed_sparsemax if use_sparsemax else annealed_softmax

        # Ablation: task-normalised (equal-weight per task) vs sample-weighted
        # loss averaging — see docstring on _compute_loss below.
        self.task_normalize    = task_normalize

        # Contrib. [B] — temperature annealing
        self.tau_init          = tau_init
        self.anneal_factor     = anneal_factor
        self.anneal_interval   = anneal_interval
        self.tau_min           = tau_min
        self._current_tau: float = tau_init

        # Contrib. [C] — delayed alpha updates
        self.alpha_update_freq = alpha_update_freq
        # 0 means "use full batch". If >0, search updates run in micro-batches.
        self.search_micro_batch = max(int(search_micro_batch), 0)

        # Label smoothing for CE tasks during search
        self.label_smoothing   = label_smoothing

        # Architecture entropy regularisation — penalises low-entropy alpha
        # distributions to prevent any single op (including ResidualBN)
        # from dominating all layers.  λ=0.01 is mild; increase to 0.05 if
        # ResidualBN dominance is still observed after a full run.
        self.arch_reg_lambda   = arch_reg_lambda

        self.opt_weights = torch.optim.AdamW(
            model.weight_parameters(),
            lr=lr_weights,
            weight_decay=weight_decay_w,
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
        """Task-normalised loss: equal weight per task regardless of sample count.

        When ``self.task_normalize`` is False (ablation), falls back to the
        previous sample-weighted averaging behaviour, which let PathMNIST
        (90k samples) dominate over DermaMNIST (7k). The default (True) makes
        each task's mean loss contribute equally so architecture search is
        not biased toward the largest dataset.
        """
        task_losses = []
        task_counts = []

        for k in range(self.num_tasks):
            mask = (task_ids == k)
            if not mask.any():
                continue
            imgs_k   = images[mask]
            labels_k = [labels[i]
                        for i in mask.nonzero(as_tuple=True)[0].tolist()]
            # Contrib. [A][B]: forward with current annealed tau
            logits_k = self.model(imgs_k, k, tau=self._current_tau)
            registry_id = self.model.task_ids[k]
            loss_k   = task_loss(logits_k, labels_k, registry_id, device,
                                 self.label_smoothing,
                                 is_multilabel=(k in self.model.MULTILABEL_TASKS))
            task_losses.append(loss_k)
            task_counts.append(int(mask.sum().item()))

        if not task_losses:
            return torch.tensor(0.0, device=device)

        if self.task_normalize:
            # Equal-weight mean across tasks (not across samples).
            task_loss_mean = sum(task_losses) / len(task_losses)
        else:
            # Sample-count-weighted mean — recovers the pre-task-normalisation
            # behaviour (ablation). Each task_loss is already a mean over its
            # own subset, so weighting by subset size and dividing by the
            # total batch size gives the true global per-sample mean loss.
            total_n = sum(task_counts)
            task_loss_mean = sum(
                l * n for l, n in zip(task_losses, task_counts)
            ) / total_n

        # ── Architecture entropy regularisation ───────────────────────────────
        # Maximise entropy of sparsemax(α) across all tasks and layers.
        # This penalises low-entropy (op-dominated) distributions, preventing
        # any single op — including ResidualBN — from collapsing the search.
        #
        #   L_total = L_task  −  λ · H(sparsemax(α))
        #
        # The minus sign is because we MINIMISE the total loss, so subtracting
        # entropy is equivalent to maximising it.
        # Only applied when lambda > 0 to allow easy ablation (set to 0.0).
        if self.arch_reg_lambda > 0.0:
            soft = self._alpha_normalizer(self.model.alphas, tau=self._current_tau)
            # Clamp to avoid log(0); sparsemax can produce exact zeros.
            entropy = -(soft * soft.clamp(min=1e-9).log()).sum(dim=-1).mean()
            arch_reg = -self.arch_reg_lambda * entropy
            return task_loss_mean + arch_reg

        return task_loss_mean

    def _run_microbatched_backward(
        self,
        images: torch.Tensor,
        labels: list,
        task_ids: torch.Tensor,
        device: torch.device,
        micro_batch_size: int,
    ) -> float:
        """Backpropagate averaged batch loss by chunking to reduce peak memory."""
        batch_size = images.size(0)
        if batch_size == 0:
            return 0.0

        if micro_batch_size <= 0 or micro_batch_size >= batch_size:
            loss = self._compute_loss(images, labels, task_ids, device)
            loss.backward()
            return float(loss.item())

        loss_value = 0.0
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            chunk_loss = self._compute_loss(
                images[start:end], labels[start:end], task_ids[start:end], device
            )
            scale = float(end - start) / float(batch_size)
            (chunk_loss * scale).backward()
            loss_value += float(chunk_loss.item()) * scale
        return loss_value

    def _step_impl(
        self,
        images_tr: torch.Tensor,
        labels_tr: list,
        tids_tr: torch.Tensor,
        images_va: torch.Tensor,
        labels_va: list,
        tids_va: torch.Tensor,
        device: torch.device,
        micro_batch_size: int,
    ) -> Tuple[float, float]:
        # Phase 1: update weights, freeze alphas.
        self.model.alphas.requires_grad_(False)
        self.opt_weights.zero_grad(set_to_none=True)
        loss_w = self._run_microbatched_backward(
            images_tr, labels_tr, tids_tr, device, micro_batch_size
        )
        nn.utils.clip_grad_norm_(list(self.model.weight_parameters()), self.grad_clip)
        self.opt_weights.step()

        # Phase 2: update alphas on schedule.
        loss_a_val = 0.0
        if (self._step + 1) % self.alpha_update_freq == 0:
            self.model.alphas.requires_grad_(True)
            self.opt_arch.zero_grad(set_to_none=True)
            for p in self.model.weight_parameters():
                p.grad = None

            loss_a_val = self._run_microbatched_backward(
                images_va, labels_va, tids_va, device, micro_batch_size
            )
            self.opt_arch.step()
        else:
            self.model.alphas.requires_grad_(True)

        return loss_w, loss_a_val

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
        images_tr = images_tr.to(device, non_blocking=True)
        images_va = images_va.to(device, non_blocking=True)
        tids_tr   = tids_tr.to(device, non_blocking=True)
        tids_va   = tids_va.to(device, non_blocking=True)

        preferred_micro_batch = self.search_micro_batch or images_tr.size(0)

        try:
            loss_w, loss_a_val = self._step_impl(
                images_tr,
                labels_tr,
                tids_tr,
                images_va,
                labels_va,
                tids_va,
                device,
                preferred_micro_batch,
            )
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise

            torch.cuda.empty_cache()
            fallback = min(images_tr.size(0), max(1, preferred_micro_batch // 2))
            recovered = False
            while fallback >= 1:
                try:
                    logger.warning(
                        "CUDA OOM at step %d. Retrying with search_micro_batch=%d",
                        self._step,
                        fallback,
                    )
                    loss_w, loss_a_val = self._step_impl(
                        images_tr,
                        labels_tr,
                        tids_tr,
                        images_va,
                        labels_va,
                        tids_va,
                        device,
                        fallback,
                    )
                    recovered = True
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if fallback == 1:
                        break
                    fallback = max(1, fallback // 2)

            if not recovered:
                raise

        self._step += 1
        return float(loss_w), float(loss_a_val)

    # ── Scheduler (call once per epoch) ──────────────────────────────────────

    def step_scheduler(self, epoch: int) -> None:
        """
        Advance cosine LR and sparsemax temperature (contrib. [B]).

        τ_epoch = τ_init × anneal_factor^(epoch // anneal_interval), ≥ 1e-2.
        """
        self.scheduler.step()
        self._current_tau = max(
            self.tau_init * (self.anneal_factor ** (epoch // self.anneal_interval)),
            self.tau_min,
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
            "epoch":            epoch,
            "step":             self._step,
            "model_state":      self.model.state_dict(),
            "opt_weights":      self.opt_weights.state_dict(),
            "opt_arch":         self.opt_arch.state_dict(),
            "scheduler":        self.scheduler.state_dict(),
            "arch_reg_lambda":  self.arch_reg_lambda,
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
        self._step             = ckpt["step"]
        self._current_tau      = ckpt.get("current_tau", self.tau_init)
        self.arch_reg_lambda   = ckpt.get("arch_reg_lambda", self.arch_reg_lambda)
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
        from .data import TASK_REGISTRY

        _normalizer_name = "sparsemax" if self.use_sparsemax else "softmax"
        logger.info(
            f"\n[step {self._step}] Architecture distribution "
            f"({_normalizer_name} α, τ={self._current_tau:.4f})"
            f"  LR={self.scheduler.get_last_lr()[0]:.5f}"
        )
        soft = self._alpha_normalizer(self.model.alphas, tau=self._current_tau)
        for t in range(self.num_tasks):
            registry_id = self.model.task_ids[t]
            name = TASK_REGISTRY.get(registry_id, (f"Task{t}",))[0]
            logger.info(f"  {name}:")
            for lay in range(self.model.num_layers):
                probs    = soft[t, lay].tolist()
                best_idx = int(soft[t, lay].argmax())
                bar      = "  ".join(
                    f"{OP_NAMES[i][:10]:10s}={p:.3f}"
                    for i, p in enumerate(probs)
                )
                logger.info(f"    Layer {lay}: {bar}  ← {OP_NAMES[best_idx]}")
