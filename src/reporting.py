"""
Benchmark reporting: ASCII table + JSON result persistence.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .data import MedMNISTDataset

logger = logging.getLogger("MT-DARTS")


def print_benchmark_table(
    results:       Dict[int, Dict[str, Any]],
    architectures: Dict[int, List[str]],
) -> str:
    """
    Pretty-print final benchmark results as an ASCII table.

    Returns the formatted table string (also printed to stdout).
    """
    sep    = "═" * 72
    header = (
        f"{'Task':<14s} │ {'ACC (%)':>8s} │ {'AUC':>8s} │ "
        f"{'Params':>10s} │ Architecture"
    )
    lines = [
        "",
        sep,
        "  MT-DARTS v2 — Benchmark Results on MedMNIST",
        sep,
        header,
        "─" * 72,
    ]

    accs: List[float] = []
    aucs: List[float] = []

    for task_id in sorted(results.keys()):
        r           = results[task_id]
        name        = MedMNISTDataset.TASK_NAMES[task_id]
        acc_pct     = r["test_acc"] * 100
        auc_val     = r["test_auc"]
        params      = r["n_params"]
        arch        = architectures.get(task_id, [])
        op_counts   = Counter(arch)
        arch_summary = ", ".join(
            f"{op}×{cnt}" for op, cnt in op_counts.most_common(3)
        )
        lines.append(
            f"  {name:<12s} │ {acc_pct:>7.2f}% │ {auc_val:>8.4f} │ "
            f"{params:>10,d} │ {arch_summary}"
        )
        accs.append(acc_pct)
        aucs.append(auc_val)

    lines.append("─" * 72)
    lines.append(
        f"  {'Macro Avg':<12s} │ {float(np.mean(accs)):>7.2f}% │ "
        f"{float(np.mean(aucs)):>8.4f} │ {'—':>10s} │"
    )
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table


def save_benchmark_results(
    results:        Dict[int, Dict[str, Any]],
    architectures:  Dict[int, List[str]],
    search_time_s:  float,
    retrain_time_s: float,
    save_dir:       str,
) -> None:
    """
    Persist benchmark results as ``{save_dir}/benchmark_results.json``.
    """
    output: Dict[str, Any] = {
        "search_time_seconds":  search_time_s,
        "retrain_time_seconds": retrain_time_s,
        "tasks": {},
    }
    for task_id in sorted(results.keys()):
        name = MedMNISTDataset.TASK_NAMES[task_id]
        output["tasks"][name] = {
            **results[task_id],
            "architecture": architectures.get(task_id, []),
        }

    accs = [r["test_acc"] for r in results.values()]
    aucs = [r["test_auc"] for r in results.values()]
    output["macro_avg_acc"] = float(np.mean(accs))
    output["macro_avg_auc"] = float(np.mean(aucs))

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "benchmark_results.json")
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Benchmark results saved → {path}")
