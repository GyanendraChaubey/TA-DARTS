"""
Regenerate results/benchmark_results.json and results/benchmark_table.txt
from the already-trained discrete checkpoints, without re-running the
multi-hour retrain phase.

Why this exists
----------------
`_calibrate_chest_thresholds` (src/retrain.py) previously picked ChestMNIST's
per-label decision thresholds by maximising raw accuracy, which collapses to
"always predict negative" on rare labels (prevalence down to ~0.2%). That bug
was fixed to calibrate on F1 instead. This script re-evaluates the *existing*
trained weights (results/discrete_*_best.pt) with the corrected calibration
and with real FLOPs profiling (previously silently null — thop wasn't
installed), so the benchmark report reflects the actual fix rather than a
fabricated number. Only ChestMNIST's numbers change from this; PathMNIST and
DermaMNIST are single-label (argmax, no threshold calibration involved) and
are re-scored here only to fill in real FLOPs.

Usage: python scripts/regenerate_report.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from thop import profile as thop_profile
from torch.utils.data import DataLoader

from src.data import TASK_REGISTRY, MedMNISTDataset
from src.metrics import evaluate_task
from src.ops import OP_NAMES
from src.reporting import print_benchmark_table, save_benchmark_results
from src.retrain import _calibrate_chest_thresholds
from src.supernet import TaskAwareSupernet

CHANNELS  = 128
IMG_SIZE  = 64
LAYERS    = 8
TASK_IDS  = [0, 1, 2]
BATCH     = 64
SEED      = 42
DEVICE    = torch.device("cpu")

CKPT_PATHS = {
    0: "results/discrete_pathmnist_best.pt",
    1: "results/discrete_chestmnist_best.pt",
    2: "results/discrete_dermamnist_best.pt",
}

ARCHITECTURES = {
    0: ["ResidualBN", "MaxPool3x3", "SepConv3x3", "MBConvSE", "MBConv3x3",
        "SepConv3x3", "SepConv5x5", "SepConv5x5"],
    1: ["MBConv3x3", "MaxPool3x3", "SepConv5x5", "MBConv3x3", "SepConv3x3",
        "ResidualBN", "ResidualBN", "MBConvSE"],
    2: ["MBConv5x5", "DilatedConv3x3", "MBConv5x5", "MBConv5x5", "MBConv5x5",
        "MaxPool3x3", "MaxPool3x3", "MaxPool3x3"],
}


def build_discrete_model(task_id: int, arch: list) -> torch.nn.Module:
    """Reconstruct the exact discrete nn.Sequential produced by
    TaskAwareSupernet.discretize() for a known architecture, so the saved
    state_dict loads with strict=True."""
    supernet = TaskAwareSupernet(
        num_tasks=len(TASK_IDS), num_layers=LAYERS, channels=CHANNELS,
        img_size=IMG_SIZE, task_ids=TASK_IDS,
    )
    with torch.no_grad():
        for l, op_name in enumerate(arch):
            idx = OP_NAMES.index(op_name)
            supernet.alphas[task_id, l, :] = -10.0
            supernet.alphas[task_id, l, idx] = 10.0
    discrete = supernet.discretize(task_id)
    return discrete


def main() -> None:
    # Only val/test are needed for rescoring — skip building the (unused)
    # 175k-image train split that build_dataloaders() would otherwise load.
    val_ds  = MedMNISTDataset("val",  seed=SEED + 1, use_real=True,
                               img_size=IMG_SIZE, task_ids=TASK_IDS)
    test_ds = MedMNISTDataset("test", seed=SEED + 2, use_real=True,
                               img_size=IMG_SIZE, task_ids=TASK_IDS)
    kw = dict(collate_fn=MedMNISTDataset.collate_fn, num_workers=0, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH, shuffle=True,  drop_last=True,  **kw)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, drop_last=False, **kw)

    # best_val_auc reflects the training-time checkpoint-selection criterion,
    # and search/retrain timing reflect actual compute time — this script
    # doesn't re-run training, so carry both forward from the original run
    # rather than fabricating them.
    prior_best_val_auc = {}
    search_time_s, retrain_time_s = 0.0, 0.0
    prior_path = "results/benchmark_results.json"
    if os.path.exists(prior_path):
        with open(prior_path) as fh:
            _prior = json.load(fh)
        search_time_s  = _prior.get("search_time_seconds", 0.0)
        retrain_time_s = _prior.get("retrain_time_seconds", 0.0)
        for tid in TASK_IDS:
            _name = TASK_REGISTRY[tid][0]
            if _name in _prior.get("tasks", {}):
                prior_best_val_auc[tid] = _prior["tasks"][_name].get("best_val_auc")

    results:       dict = {}
    architectures: dict = {}
    for task_id in TASK_IDS:
        registry_id = task_id  # default 3-task ordering: position == registry id
        name = TASK_REGISTRY[registry_id][0]
        arch = ARCHITECTURES[task_id]

        model = build_discrete_model(task_id, arch)
        state_dict = torch.load(CKPT_PATHS[task_id], map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)  # CPU — thop hooks aren't MPS-safe
        flops_m = flops / 1e6

        model.to(DEVICE)

        chest_thresholds = None
        if registry_id == 1:
            chest_thresholds = _calibrate_chest_thresholds(
                model, val_loader, DEVICE, task_id,
                n_labels=TASK_REGISTRY[registry_id][2],
            )

        metrics = evaluate_task(
            model, test_loader, task_id, DEVICE, is_supernet=False,
            chest_thresholds=chest_thresholds,
            balanced_training=(registry_id == 2), registry_id=registry_id,
        )

        print(f"[{name}] params={n_params:,}  flops_m={flops_m:.1f}  "
              f"acc={metrics['acc']:.4f}  auc={metrics['auc']:.4f}  "
              f"f1={metrics['f1']:.4f}  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}")

        results[task_id] = {
            "test_acc":       metrics["acc"],
            "test_auc":       metrics["auc"],
            "test_f1":        metrics["f1"],
            "test_precision": metrics["precision"],
            "test_recall":    metrics["recall"],
            "test_loss":      metrics["loss"],
            "best_val_auc":   prior_best_val_auc.get(task_id),
            "n_params":       n_params,
            "flops_m":        flops_m,
        }
        architectures[task_id] = arch

    task_names = {tid: TASK_REGISTRY[tid][0] for tid in TASK_IDS}

    table = print_benchmark_table(results, architectures, task_names=task_names)

    mean_auc = float(np.mean([r["test_auc"] for r in results.values()]))
    search_efficiency = mean_auc / (search_time_s / 3600.0) if search_time_s > 0 else 0.0

    save_benchmark_results(
        results, architectures,
        search_time_s=search_time_s, retrain_time_s=retrain_time_s,
        save_dir="results", search_efficiency=search_efficiency,
        task_names=task_names,
    )
    with open("results/benchmark_table.txt", "w") as fh:
        fh.write(table + "\n")

    print("\nWrote results/benchmark_results.json and results/benchmark_table.txt")


if __name__ == "__main__":
    main()
