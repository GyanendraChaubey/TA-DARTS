"""
run_search() — full MT-DARTS v2 pipeline.

Phases:
  A — Architecture search (bilevel optimisation with early stopping).
  B — Discretise: extract per-task argmax architectures from the supernet.
  C — Retrain: train each discrete model from scratch.
  D — Report: print ASCII table and write JSON benchmark results.

Entry points:
  - Imported directly:  ``from train import run_search``
  - CLI:                ``python main.py --epochs 50 ...``
"""
from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch

from src.controller import SearchController
from src.data import MedMNISTDataset, build_dataloaders
from src.metrics import alpha_entropy, evaluate
from src.ops import OP_NAMES
from src.reporting import print_benchmark_table, save_benchmark_results
from src.retrain import retrain_discrete
from src.supernet import TaskAwareSupernet
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MT-DARTS")


def run_search(
    # ── Search ──────────────────────────────────────────────────────────────
    num_epochs:        int   = 50,
    batch_size:        int   = 64,
    num_layers:        int   = 6,
    channels:          int   = 32,
    lr_weights:        float = 0.025,
    lr_alphas:         float = 3e-4,
    log_interval:      int   = 25,
    eval_interval:     int   = 1,
    ckpt_interval:     int   = 5,
    # ── Retrain ─────────────────────────────────────────────────────────────
    retrain_epochs:    int   = 100,
    retrain_lr:        float = 0.025,
    # ── Contrib. [B] temperature annealing ──────────────────────────────────
    tau_init:          float = 1.5,
    anneal_factor:     float = 0.75,
    anneal_interval:   int   = 5,
    # ── Contrib. [C] delayed alpha updates ──────────────────────────────────
    alpha_update_freq: int   = 10,
    # ── Contrib. [D] early stopping ─────────────────────────────────────────
    entropy_threshold: float = 0.05,
    # ── Infrastructure ──────────────────────────────────────────────────────
    ckpt_dir:          str          = "checkpoints",
    save_dir:          str          = "./results",
    resume_from:       Optional[str] = None,
    use_real_data:     bool         = True,
    device_str:        str          = "cpu",
    seed:              int          = 42,
    num_workers:       int          = 0,
    img_size:          int          = 64,
    mixup_alpha:       float        = 0.2,
    label_smoothing:   float        = 0.1,
) -> Dict[str, Any]:
    """
    Execute the full MT-DARTS v2 pipeline and return benchmark results.

    Returns
    -------
    dict with keys: "results", "architectures", "search_time", "retrain_time".
    """
    set_seed(seed)
    device = torch.device(device_str)
    os.makedirs(save_dir, exist_ok=True)

    # ── CSV writer for search training curves (Fix 2+5) ──────────────────────
    csv_path = os.path.join(save_dir, "search_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "PathMNIST_auc", "ChestMNIST_auc",
                         "DermaMNIST_auc", "alpha_entropy", "tau"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader_bilevel, eval_loader = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        use_real_data=use_real_data,
        seed=seed,
        img_size=img_size,
    )

    # ── Model & controller ────────────────────────────────────────────────────
    model = TaskAwareSupernet(
        num_tasks=3,
        num_layers=num_layers,
        channels=channels,
        num_classes_per_task=[9, 14, 7],
        img_size=img_size,
    ).to(device)

    controller = SearchController(
        model,
        epochs=num_epochs,
        lr_weights=lr_weights,
        lr_alphas=lr_alphas,
        tau_init=tau_init,
        anneal_factor=anneal_factor,
        anneal_interval=anneal_interval,
        alpha_update_freq=alpha_update_freq,
        label_smoothing=label_smoothing,
    )

    start_epoch = 1
    if resume_from:
        start_epoch = controller.load_checkpoint(resume_from) + 1

    logger.info("=" * 72)
    logger.info(
        f"MT-DARTS v2  |  device={device}  |  epochs={num_epochs}"
        f"  |  seed={seed}"
    )
    logger.info(
        f"  supernet params : {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info(f"  alpha params    : {model.alphas.numel()}")
    logger.info(f"  train batches   : {len(train_loader)}")
    logger.info(
        f"  τ_init={tau_init}  anneal_factor={anneal_factor}"
        f"  anneal_interval={anneal_interval}"
    )
    logger.info(
        f"  alpha_update_freq={alpha_update_freq}"
        f"  entropy_threshold={entropy_threshold}"
    )
    logger.info("=" * 72)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: Architecture Search
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE A: Architecture Search")
    search_start        = time.time()
    val_iter            = iter(val_loader_bilevel)
    early_stopped_epoch = num_epochs

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss_w = epoch_loss_a = 0.0
        alpha_steps  = 0
        t0           = time.time()

        for train_batch in train_loader:
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter  = iter(val_loader_bilevel)
                val_batch = next(val_iter)

            loss_w, loss_a = controller.step(train_batch, val_batch, device)
            epoch_loss_w  += loss_w
            if loss_a > 0.0:
                epoch_loss_a += loss_a
                alpha_steps  += 1

            if controller._step % log_interval == 0:
                lr = controller.scheduler.get_last_lr()[0]
                logger.info(
                    f"  Ep {epoch:3d}  step {controller._step:5d}"
                    f"  loss_w={loss_w:.4f}  loss_α={loss_a:.4f}"
                    f"  lr={lr:.5f}  τ={controller._current_tau:.3f}"
                )

        # Advance LR + temperature annealing once per epoch
        controller.step_scheduler(epoch)

        n_batches  = len(train_loader)
        elapsed    = time.time() - t0
        avg_loss_a = epoch_loss_a / alpha_steps if alpha_steps > 0 else 0.0
        logger.info(
            f"\nEpoch {epoch:3d}/{num_epochs}"
            f"  avg_loss_w={epoch_loss_w / n_batches:.4f}"
            f"  avg_loss_α={avg_loss_a:.4f}"
            f"  τ={controller._current_tau:.4f}"
            f"  time={elapsed:.1f}s"
        )

        if epoch % eval_interval == 0:
            eval_results = evaluate(
                model, eval_loader, device, split_name=f"epoch{epoch}"
            )
        else:
            eval_results = None

        if epoch % ckpt_interval == 0:
            controller.save_checkpoint(
                epoch, ckpt_dir=ckpt_dir, tag=f"epoch{epoch:03d}"
            )
        controller.save_checkpoint(epoch, ckpt_dir=ckpt_dir, tag="latest")

        # Contrib. [D] — early stopping on alpha convergence
        ent = alpha_entropy(model, controller._current_tau)
        logger.info(
            f"  Mean alpha entropy: {ent:.4f} (threshold={entropy_threshold})"
        )

        # ── Write training curves to CSV (Fix 2+5) ───────────────────────────
        if eval_results is not None:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{eval_results.get(0, {}).get('auc', float('nan')):.4f}",
                    f"{eval_results.get(1, {}).get('auc', float('nan')):.4f}",
                    f"{eval_results.get(2, {}).get('auc', float('nan')):.4f}",
                    f"{ent:.6f}",
                    f"{controller._current_tau:.4f}",
                ])

        if ent < entropy_threshold:
            logger.info(
                f"  ▶ Early stopping: entropy {ent:.4f} < {entropy_threshold}"
                f" at epoch {epoch}. Architectures have converged."
            )
            early_stopped_epoch = epoch
            break

    search_time = time.time() - search_start
    logger.info(
        f"\n  Search completed in {search_time:.1f}s ({search_time / 3600:.2f}h)"
        f"  |  stopped at epoch {early_stopped_epoch}/{num_epochs}"
    )
    controller.log_arch_distribution()

    # ── Save per-task architecture snapshot to file (Fix 6) ───────────────────
    arch_snapshot_path = os.path.join(save_dir, "architecture_snapshot.txt")
    with open(arch_snapshot_path, "w") as f:
        from src.normalizers import annealed_sparsemax as _sp
        _soft = _sp(model.alphas, tau=controller._current_tau)
        for t in range(model.num_tasks):
            tname = MedMNISTDataset.TASK_NAMES.get(t, f"Task{t}")
            best  = model.alphas[t].argmax(dim=-1).tolist()
            arch  = [OP_NAMES[i] for i in best]
            f.write(f"{tname}: {arch}\n")
            for lay in range(model.num_layers):
                probs = _soft[t, lay].tolist()
                bar   = "  ".join(f"{OP_NAMES[i][:10]:10s}={p:.3f}"
                                  for i, p in enumerate(probs))
                f.write(f"  Layer {lay}: {bar}\n")
            f.write("\n")
    logger.info(f"  Architecture snapshot saved → {arch_snapshot_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: Discretise
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE B: Discretising final architectures")
    logger.info("=" * 72)

    discrete_models: Dict[int, torch.nn.Module] = {}
    architectures:   Dict[int, List[str]]       = {}

    for k in range(model.num_tasks):
        discrete_models[k] = model.discretize(k)
        best_indices        = model.alphas[k].argmax(dim=-1).tolist()
        architectures[k]   = [OP_NAMES[i] for i in best_indices]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE C: Retrain discrete architectures
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE C: Retraining discrete architectures from scratch")
    retrain_start: float = time.time()

    benchmark_results: Dict[int, Dict[str, Any]] = {}
    for k in range(model.num_tasks):
        result = retrain_discrete(
            discrete_model=discrete_models[k],
            task_id=k,
            train_loader=train_loader,
            val_loader=val_loader_bilevel,
            test_loader=eval_loader,
            num_epochs=retrain_epochs,
            lr=retrain_lr,
            device=device,
            save_dir=save_dir,
            mixup_alpha=mixup_alpha,
            label_smoothing=label_smoothing,
        )
        result["architecture"] = architectures[k]
        benchmark_results[k]   = result

    retrain_time = time.time() - retrain_start
    logger.info(
        f"\n  Retraining completed in {retrain_time:.1f}s "
        f"({retrain_time / 3600:.2f}h)"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE D: Benchmark report
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n▶ PHASE D: Benchmark Report")
    table = print_benchmark_table(benchmark_results, architectures)

    save_benchmark_results(
        benchmark_results,
        architectures,
        search_time_s=search_time,
        retrain_time_s=retrain_time,
        save_dir=save_dir,
    )

    table_path = os.path.join(save_dir, "benchmark_table.txt")
    with open(table_path, "w") as fh:
        fh.write(table + "\n")

    return {
        "results":       benchmark_results,
        "architectures": architectures,
        "search_time":   search_time,
        "retrain_time":  retrain_time,
    }
