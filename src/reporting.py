"""
Benchmark reporting: ASCII table + JSON result persistence.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .data import TASK_REGISTRY, DEFAULT_TASK_IDS

logger = logging.getLogger("MT-DARTS")

# ResNet-18 reference values (MedMNIST v2 paper, 28px)
RESNET18_PARAMS = 11_200_000
RESNET18_FLOPS_M = 1_820.0   # ~1.82 GFLOPs at 28px


def _fmt(v: float, fmt: str = ".4f") -> str:
    """Format float, returning '—' for NaN."""
    return ("—" if math.isnan(v) else f"{v:{fmt}}")


def print_benchmark_table(
    results:       Dict[int, Dict[str, Any]],
    architectures: Dict[int, List[str]],
    task_names:    Dict[int, str] = None,
) -> str:
    """
    Pretty-print final benchmark results as an ASCII table.

    Columns: Task | ACC | AUC | F1 | Params | FLOPs(M) | AUC/MParam | vs RN18 | Architecture
    Returns the formatted table string (also printed to stdout).
    """
    # Build name lookup: prefer caller-supplied task_names, fall back to registry.
    def _name(task_id: int) -> str:
        if task_names and task_id in task_names:
            return task_names[task_id]
        return TASK_REGISTRY.get(task_id, (f"Task{task_id}",))[0]

    sep    = "═" * 110
    header = (
        f"{'Task':<14s} │ {'ACC (%)':>7s} │ {'AUC':>6s} │ {'F1':>6s} │"
        f" {'Params':>10s} │ {'FLOPs(M)':>9s} │ {'AUC/MP':>7s} │ {'vs RN18':>7s} │ Architecture"
    )
    lines = [
        "",
        sep,
        "  MT-DARTS v2 — Benchmark Results on MedMNIST",
        sep,
        header,
        "─" * 110,
    ]

    accs: List[float] = []
    aucs: List[float] = []
    f1s:  List[float] = []

    for task_id in sorted(results.keys()):
        r          = results[task_id]
        name       = _name(task_id)
        acc_pct    = r["test_acc"] * 100
        auc_val    = r["test_auc"]
        f1_val     = r.get("test_f1", float("nan"))
        params     = r["n_params"]
        flops_m    = r.get("flops_m", float("nan"))
        ratio_pct  = 100.0 * params / RESNET18_PARAMS
        # Accuracy density: AUC per million parameters (higher = more efficient)
        auc_per_mp = auc_val / (params / 1e6) if params > 0 else float("nan")

        arch        = architectures.get(task_id, [])
        op_counts   = Counter(arch)
        arch_summary = ", ".join(
            f"{op}×{cnt}" for op, cnt in op_counts.most_common(3)
        )
        lines.append(
            f"  {name:<12s} │ {acc_pct:>6.2f}% │ {_fmt(auc_val)} │ {_fmt(f1_val)} │"
            f" {params:>10,d} │ {_fmt(flops_m, '.1f'):>9s} │ {_fmt(auc_per_mp, '.4f'):>7s} │"
            f" {ratio_pct:>6.1f}% │ {arch_summary}"
        )
        accs.append(acc_pct)
        aucs.append(auc_val)
        if not math.isnan(f1_val):
            f1s.append(f1_val)

    lines.append("─" * 110)
    f1_avg = float(np.mean(f1s)) if f1s else float("nan")
    lines.append(
        f"  {'Macro Avg':<12s} │ {float(np.mean(accs)):>6.2f}% │ "
        f"{float(np.mean(aucs)):.4f} │ {_fmt(f1_avg)} │"
        f" {'—':>10s} │ {'—':>9s} │ {'—':>7s} │ {'—':>7s} │"
    )
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table


def save_benchmark_results(
    results:           Dict[int, Dict[str, Any]],
    architectures:     Dict[int, List[str]],
    search_time_s:     float,
    retrain_time_s:    float,
    save_dir:          str,
    search_efficiency: float       = 0.0,
    task_names:        Dict[int, str] = None,
) -> None:
    """
    Persist benchmark results as ``{save_dir}/benchmark_results.json``.
    """
    def _name(task_id: int) -> str:
        if task_names and task_id in task_names:
            return task_names[task_id]
        return TASK_REGISTRY.get(task_id, (f"Task{task_id}",))[0]

    output: Dict[str, Any] = {
        "search_time_seconds":            search_time_s,
        "retrain_time_seconds":           retrain_time_s,
        "search_efficiency_auc_per_hour": search_efficiency,
        "tasks": {},
    }
    for task_id in sorted(results.keys()):
        name     = _name(task_id)
        n_params = results[task_id].get("n_params", 0)
        flops_m  = results[task_id].get("flops_m", float("nan"))
        auc_val  = results[task_id].get("test_auc", float("nan"))
        output["tasks"][name] = {
            **results[task_id],
            "architecture":          architectures.get(task_id, []),
            "params_vs_resnet18_pct": round(100.0 * n_params / RESNET18_PARAMS, 2),
            "flops_m":               None if math.isnan(flops_m) else round(flops_m, 2),
            "auc_per_million_params": None if (math.isnan(auc_val) or n_params == 0)
                                      else round(auc_val / (n_params / 1e6), 6),
        }

    accs = [r["test_acc"] for r in results.values()]
    aucs = [r["test_auc"] for r in results.values()]
    f1s  = [r["test_f1"]  for r in results.values()
            if not math.isnan(r.get("test_f1", float("nan")))]

    output["macro_avg_acc"] = float(np.mean(accs))
    output["macro_avg_auc"] = float(np.mean(aucs))
    output["macro_avg_f1"]  = float(np.mean(f1s)) if f1s else None

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "benchmark_results.json")
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    logger.info(f"Benchmark results saved → {path}")
