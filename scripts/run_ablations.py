"""
Run the MT-DARTS ablation suite end to end.

Runs the *real* Phase A->D pipeline (train.run_search) once per ablation
variant, writing each variant's full output (benchmark_results.json,
benchmark_table.txt, search_curves.csv, architecture snapshots, discrete
checkpoints) into its own results/ablations/<variant_name>/ subfolder, then
rolls all variants up into one comparison table and JSON summary.

Why this exists
----------------
The paper previously had an "Ablation Study" section that was removed because
there was no real data backing it. This script produces that data: for each
of the six ablated contributions (sparsemax vs softmax, temperature
annealing, alpha-update frequency, entropy early stopping, task-balanced
sampling, architecture entropy regularisation) plus the already-free
mixup/label-smoothing/task-normalisation toggles, it runs a full search +
retrain and records the resulting AUC/ACC/params/timing exactly like a
normal run — nothing here is mocked or estimated.

Usage
-----
Quick smoke test (mock data, tiny epochs, CPU):
    python scripts/run_ablations.py --no-real --epochs 2 --retrain-epochs 2 \\
        --batch 8 --device cpu --save-root /tmp/ablations_smoke

Full paper-matching sweep (real data, GPU):
    python scripts/run_ablations.py --epochs 50 --retrain-epochs 200 \\
        --device cuda

Re-run only specific variants:
    python scripts/run_ablations.py --variants baseline,softmax_normalizer

Force re-running variants whose results already exist:
    python scripts/run_ablations.py --force-rerun
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import apply_ablation_overrides
from src.reporting import (
    print_ablation_table,
    save_ablation_summary,
    summarize_for_ablation,
)
from train import run_search

logger = logging.getLogger("MT-DARTS")


def _build_variants() -> "OrderedDict[str, Dict[str, Any]]":
    """
    Return an ordered {variant_name: run_search kwarg overrides} table.

    B/D-style variants reuse main.apply_ablation_overrides so this script's
    "disabled" semantics can never drift from main.py's --no-contrib-b/c/d.
    """
    b_off = apply_ablation_overrides(
        tau_init=1.5, anneal_factor=0.95, anneal_interval=5, tau_min=0.30,
        alpha_freq=10, entropy_thresh=0.05, auc_patience=15,
        no_contrib_b=True, no_contrib_c=False, no_contrib_d=False,
    )
    d_off = apply_ablation_overrides(
        tau_init=1.5, anneal_factor=0.95, anneal_interval=5, tau_min=0.30,
        alpha_freq=10, entropy_thresh=0.05, auc_patience=15,
        no_contrib_b=False, no_contrib_c=False, no_contrib_d=True,
    )

    return OrderedDict([
        ("baseline",                  {}),
        ("no_contrib_b_temp_anneal",  {k: b_off[k] for k in
                                        ("anneal_factor", "anneal_interval", "tau_min")}),
        ("alpha_freq_1",              {"alpha_update_freq": 1}),
        ("alpha_freq_5",              {"alpha_update_freq": 5}),
        ("alpha_freq_10",             {"alpha_update_freq": 10}),
        ("alpha_freq_20",             {"alpha_update_freq": 20}),
        ("no_contrib_d_early_stop",   {k: d_off[k] for k in
                                        ("entropy_threshold", "auc_patience")}),
        ("no_task_balanced_sampling", {"balance_tasks": False}),
        ("no_arch_reg",               {"arch_reg_lambda": 0.0}),
        ("softmax_normalizer",        {"use_sparsemax": False}),
        # Already fully supported by run_search() at zero extra implementation
        # cost — wired in alongside the six confirmed ablations.
        ("no_mixup",                  {"mixup_alpha": 0.0}),
        ("no_label_smoothing",        {"label_smoothing": 0.0}),
        ("sample_weighted_loss",      {"task_normalize": False}),
    ])


# Heavy, compute-costly — opt-in only via --with-imgsize-sweep, not run by default.
_IMGSIZE_VARIANTS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("img_size_64", {"img_size": 64}),
])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MT-DARTS ablation suite end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",          type=int, default=20,
                        help="Search epochs per variant (full paper run: 50)")
    parser.add_argument("--retrain-epochs",  type=int, default=50,
                        help="Retrain epochs per variant (full paper run: 200)")
    parser.add_argument("--device",          type=str, default="cpu",
                        help="Device: cpu | cuda | mps")
    parser.add_argument("--tasks",           type=str, default="0,1,2",
                        help="Comma-separated task IDs, e.g. '0,1,2'")
    parser.add_argument("--batch",           type=int, default=64)
    parser.add_argument("--layers",          type=int, default=8)
    parser.add_argument("--channels",        type=int, default=128)
    parser.add_argument("--img-size",        type=int, default=28)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--workers",         type=int, default=0)
    parser.add_argument("--no-real",         action="store_true",
                        help="Force mock data even if medmnist is installed")
    parser.add_argument("--save-root",       type=str, default="results/ablations")
    parser.add_argument("--ckpt-root",       type=str, default="checkpoints/ablations")
    parser.add_argument("--variants",        type=str, default="all",
                        help="Comma-separated variant names to run, or 'all'")
    parser.add_argument("--skip",            type=str, default="",
                        help="Comma-separated variant names to exclude from --variants all")
    parser.add_argument("--with-imgsize-sweep", action="store_true",
                        help="Also run the img_size_64 variant (compute-heavy, opt-in)")
    parser.add_argument("--force-rerun",     action="store_true",
                        help="Re-run variants even if their results already exist")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    task_ids = sorted(int(t.strip()) for t in args.tasks.split(",") if t.strip())

    all_variants = _build_variants()
    if args.with_imgsize_sweep:
        all_variants.update(_IMGSIZE_VARIANTS)

    if args.variants == "all":
        selected = list(all_variants.keys())
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        selected = [v for v in selected if v not in skip]
    else:
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in selected if v not in all_variants]
        if unknown:
            raise ValueError(
                f"Unknown variant(s) {unknown}. Available: {list(all_variants.keys())}"
            )

    logger.info("=" * 72)
    logger.info(f"MT-DARTS ablation suite — {len(selected)} variant(s): {selected}")
    logger.info(
        f"  epochs={args.epochs}  retrain_epochs={args.retrain_epochs}"
        f"  device={args.device}  tasks={task_ids}"
    )
    logger.info("=" * 72)

    variant_summaries: Dict[str, Dict[str, Any]] = {}

    for name in selected:
        overrides = all_variants[name]
        variant_save_dir = os.path.join(args.save_root, name)
        variant_ckpt_dir  = os.path.join(args.ckpt_root, name)
        result_path = os.path.join(variant_save_dir, "benchmark_results.json")

        if os.path.exists(result_path) and not args.force_rerun:
            logger.info(f"[{name}] results already exist — skipping (use --force-rerun to redo)")
            status = "skipped_existing"
        else:
            logger.info(f"\n▶▶▶ ABLATION VARIANT: {name}  (overrides={overrides})")
            kwargs: Dict[str, Any] = dict(
                num_epochs=args.epochs,
                retrain_epochs=args.retrain_epochs,
                batch_size=args.batch,
                num_layers=args.layers,
                channels=args.channels,
                device_str=args.device,
                use_real_data=not args.no_real,
                seed=args.seed,
                num_workers=args.workers,
                img_size=args.img_size,
                task_ids=task_ids,
                save_dir=variant_save_dir,
                ckpt_dir=variant_ckpt_dir,
                **overrides,
            )
            try:
                run_search(**kwargs)
                status = "ok"
            except Exception:
                logger.exception(f"[{name}] FAILED")
                status = "failed"

        if os.path.exists(result_path):
            with open(result_path) as fh:
                bench_json = json.load(fh)
            summary = summarize_for_ablation(bench_json)
        else:
            summary = {}
        summary["status"]    = status
        summary["overrides"] = overrides
        variant_summaries[name] = summary

    table = print_ablation_table(variant_summaries, baseline="baseline")
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "ablation_summary.txt"), "w") as fh:
        fh.write(table + "\n")

    save_ablation_summary(
        variant_summaries,
        save_dir=args.save_root,
        baseline="baseline",
        config=dict(
            epochs=args.epochs, retrain_epochs=args.retrain_epochs,
            device=args.device, tasks=args.tasks, seed=args.seed,
            img_size=args.img_size, batch_size=args.batch,
        ),
    )

    logger.info(f"\nWrote {args.save_root}/ablation_summary.json and ablation_summary.txt")


if __name__ == "__main__":
    main()
