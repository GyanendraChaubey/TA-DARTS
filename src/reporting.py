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

# ResNet-18 reference values (MedMNIST v2 paper's ResNet-18(28) baseline: a
# small-image adaptation — 3x3 stride-1 stem, no initial maxpool — NOT the
# standard ImageNet-resolution ResNet-18). Verified by profiling that exact
# architecture with the same thop profiler used on our own models
# (scripts/profile_resnet18_baseline.py): 11,173,449-11,176,014 params
# (num_classes-dependent, negligible) and 458.63 MFLOPs at 28x28.
# The previous 1,820.0 MFLOPs figure was the well-known *ImageNet-resolution*
# (224px) ResNet-18 FLOPs count, silently mislabeled as the 28px baseline —
# ~4x too high for this comparison.
RESNET18_PARAMS = 11_200_000
RESNET18_FLOPS_M = 458.63   # verified at 28px, matching the paper's ResNet-18(28) baseline


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

    Columns: Task | ACC | AUC | F1 | Params | FLOPs(M) | AUC/MParam | AUC/GFLOP | Params%RN18 | FLOPs%RN18 | Architecture

    The RN18 columns compare against the paper's actual ResNet-18(28) baseline
    (RESNET18_PARAMS / RESNET18_FLOPS_M above) at its own native resolution —
    not a same-resolution comparison, since our models run at whatever
    --img-size the pipeline was configured with.
    Returns the formatted table string (also printed to stdout).
    """
    # Build name lookup: prefer caller-supplied task_names, fall back to registry.
    def _name(task_id: int) -> str:
        if task_names and task_id in task_names:
            return task_names[task_id]
        return TASK_REGISTRY.get(task_id, (f"Task{task_id}",))[0]

    sep    = "═" * 138
    header = (
        f"{'Task':<14s} │ {'ACC (%)':>7s} │ {'AUC':>6s} │ {'F1':>6s} │"
        f" {'Params':>10s} │ {'FLOPs(M)':>9s} │ {'AUC/MP':>7s} │ {'AUC/GF':>7s} │"
        f" {'Params%RN18':>11s} │ {'FLOPs%RN18':>10s} │ Architecture"
    )
    lines = [
        "",
        sep,
        "  MT-DARTS v2 — Benchmark Results on MedMNIST",
        sep,
        header,
        "─" * 138,
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
        params_pct = 100.0 * params / RESNET18_PARAMS
        flops_pct  = (100.0 * flops_m / RESNET18_FLOPS_M
                      if not math.isnan(flops_m) else float("nan"))
        # Accuracy density: AUC per million parameters / per GFLOP (higher = more efficient)
        auc_per_mp = auc_val / (params / 1e6) if params > 0 else float("nan")
        auc_per_gf = (auc_val / (flops_m / 1e3)
                      if (not math.isnan(flops_m) and flops_m > 0) else float("nan"))

        arch        = architectures.get(task_id, [])
        op_counts   = Counter(arch)
        arch_summary = ", ".join(
            f"{op}×{cnt}" for op, cnt in op_counts.most_common(3)
        )
        flops_pct_s = (f"{flops_pct:.1f}%" if not math.isnan(flops_pct) else "—")
        lines.append(
            f"  {name:<12s} │ {acc_pct:>6.2f}% │ {_fmt(auc_val)} │ {_fmt(f1_val)} │"
            f" {params:>10,d} │ {_fmt(flops_m, '.1f'):>9s} │ {_fmt(auc_per_mp, '.4f'):>7s} │"
            f" {_fmt(auc_per_gf, '.3f'):>7s} │ {params_pct:>10.1f}% │ {flops_pct_s:>10s} │ {arch_summary}"
        )
        accs.append(acc_pct)
        aucs.append(auc_val)
        if not math.isnan(f1_val):
            f1s.append(f1_val)

    lines.append("─" * 138)
    f1_avg = float(np.mean(f1s)) if f1s else float("nan")
    lines.append(
        f"  {'Macro Avg':<12s} │ {float(np.mean(accs)):>6.2f}% │ "
        f"{float(np.mean(aucs)):.4f} │ {_fmt(f1_avg)} │"
        f" {'—':>10s} │ {'—':>9s} │ {'—':>7s} │ {'—':>7s} │ {'—':>11s} │ {'—':>10s} │"
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
            "flops_vs_resnet18_pct": None if math.isnan(flops_m)
                                      else round(100.0 * flops_m / RESNET18_FLOPS_M, 2),
            "auc_per_million_params": None if (math.isnan(auc_val) or n_params == 0)
                                      else round(auc_val / (n_params / 1e6), 6),
            "auc_per_gflop":         None if (math.isnan(auc_val) or math.isnan(flops_m) or flops_m == 0)
                                      else round(auc_val / (flops_m / 1e3), 4),
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


# ── Ablation comparison ──────────────────────────────────────────────────────

def summarize_for_ablation(benchmark_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Roll up one ablation variant's ``benchmark_results.json`` (as produced by
    :func:`save_benchmark_results`) into the headline numbers used for
    cross-variant comparison in :func:`print_ablation_table` /
    :func:`save_ablation_summary`.
    """
    tasks          = benchmark_json.get("tasks", {})
    n_params_total = sum(t.get("n_params", 0) or 0 for t in tasks.values())
    flops_vals     = [t["flops_m"] for t in tasks.values() if t.get("flops_m") is not None]

    return {
        "macro_avg_acc":                  benchmark_json.get("macro_avg_acc"),
        "macro_avg_auc":                  benchmark_json.get("macro_avg_auc"),
        "macro_avg_f1":                   benchmark_json.get("macro_avg_f1"),
        "search_time_seconds":            benchmark_json.get("search_time_seconds"),
        "retrain_time_seconds":           benchmark_json.get("retrain_time_seconds"),
        "search_efficiency_auc_per_hour": benchmark_json.get("search_efficiency_auc_per_hour"),
        "n_params_total":                 n_params_total,
        "mean_flops_m":                   (float(np.mean(flops_vals)) if flops_vals else None),
        "per_task": {
            name: {
                "test_acc":     t.get("test_acc"),
                "test_auc":     t.get("test_auc"),
                "test_f1":      t.get("test_f1"),
                "n_params":     t.get("n_params"),
                "flops_m":      t.get("flops_m"),
                "architecture": t.get("architecture"),
            }
            for name, t in tasks.items()
        },
    }


def print_ablation_table(
    variant_summaries: Dict[str, Dict[str, Any]],
    baseline:          str = "baseline",
) -> str:
    """
    Pretty-print a cross-variant ablation comparison as an ASCII table.

    Columns: Variant | Status | ACC | AUC | F1 | ΔAUC vs baseline | Params | Search(s) | Retrain(s) | AUC/hr

    Each entry in ``variant_summaries`` is expected to be the rollup produced
    by :func:`summarize_for_ablation`, plus a ``"status"`` key
    (``"ok" | "failed" | "skipped_existing"``). Returns the formatted table
    string (also printed to stdout), matching :func:`print_benchmark_table`.
    """
    def _f(v: Any, fmt: str = ".4f") -> str:
        if v is None:
            return "—"
        if isinstance(v, float) and math.isnan(v):
            return "—"
        return f"{v:{fmt}}"

    sep    = "═" * 118
    header = (
        f"{'Variant':<26s} │ {'Status':<8s} │ {'ACC (%)':>7s} │ {'AUC':>6s} │"
        f" {'F1':>6s} │ {'ΔAUC':>7s} │ {'Params':>10s} │ {'Search(s)':>9s} │ {'AUC/hr':>7s}"
    )
    lines = [
        "",
        sep,
        "  MT-DARTS v2 — Ablation Comparison",
        sep,
        header,
        "─" * 118,
    ]

    base = variant_summaries.get(baseline)
    base_auc = base.get("macro_avg_auc") if (base and base.get("status") == "ok") else None

    for name, v in variant_summaries.items():
        status  = v.get("status", "ok")
        acc     = v.get("macro_avg_acc")
        auc     = v.get("macro_avg_auc")
        f1      = v.get("macro_avg_f1")
        params  = v.get("n_params_total")
        search_s = v.get("search_time_seconds")
        eff     = v.get("search_efficiency_auc_per_hour")

        delta_s = "—"
        if status == "ok" and auc is not None and base_auc is not None and name != baseline:
            delta_s = f"{auc - base_auc:+.4f}"

        acc_s   = f"{acc * 100:.2f}%" if acc is not None else "—"
        params_s = f"{params:,d}" if params else "—"
        lines.append(
            f"  {name:<24s} │ {status:<8s} │ {acc_s:>7s} │ {_f(auc)} │"
            f" {_f(f1)} │ {delta_s:>7s} │ {params_s:>10s} │ {_f(search_s, '.1f'):>9s} │ {_f(eff, '.3f')}"
        )

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


def save_ablation_summary(
    variant_summaries: Dict[str, Dict[str, Any]],
    save_dir:          str,
    baseline:          str = "baseline",
    config:            Dict[str, Any] = None,
) -> None:
    """
    Persist ``{save_dir}/ablation_summary.json``: per-variant rollups (from
    :func:`summarize_for_ablation`), the shared run config, and
    ``deltas_vs_baseline`` (ΔAUC, ΔACC, Δsearch-time) for every non-baseline
    variant with ``status == "ok"``.
    """
    output: Dict[str, Any] = {
        "config":           config or {},
        "baseline_variant": baseline,
        "variants":         variant_summaries,
    }

    base   = variant_summaries.get(baseline)
    deltas: Dict[str, Any] = {}
    if base is not None and base.get("status") == "ok":
        for name, v in variant_summaries.items():
            if name == baseline or v.get("status") != "ok":
                continue

            def _delta(key: str):
                a, b = v.get(key), base.get(key)
                return (a - b) if (a is not None and b is not None) else None

            deltas[name] = {
                "delta_macro_avg_auc":       _delta("macro_avg_auc"),
                "delta_macro_avg_acc":       _delta("macro_avg_acc"),
                "delta_search_time_seconds": _delta("search_time_seconds"),
            }
    output["deltas_vs_baseline"] = deltas

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "ablation_summary.json")
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    logger.info(f"Ablation summary saved → {path}")
