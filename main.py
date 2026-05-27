"""
CLI entry point for MT-DARTS v2.

Examples
--------
Full run on real MedMNIST data::

    python main.py --epochs 50 --batch 64 --device cuda

Quick smoke test with mock data::

    python main.py --no-real --epochs 2 --batch 32 --device cpu
"""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("MT-DARTS")

try:
    from sklearn.metrics import roc_auc_score  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MT-DARTS v2: Multi-Task NAS on MedMNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Search Phase")
    g.add_argument("--epochs",          type=int,   default=50,
                   help="Number of search epochs")
    g.add_argument("--batch",           type=int,   default=64,
                   help="Batch size")
    g.add_argument("--layers",          type=int,   default=8,
                   help="Number of searchable layers")
    g.add_argument("--channels",        type=int,   default=128,
                   help="Feature map channel width")
    g.add_argument("--lr_w",            type=float, default=1e-3,
                   help="Weight optimiser learning rate")
    g.add_argument("--lr_a",            type=float, default=3e-4,
                   help="Alpha optimiser learning rate")
    g.add_argument("--log",             type=int,   default=25,
                   help="Log every N gradient steps")
    g.add_argument("--eval-every",      type=int,   default=1,
                   help="Evaluate accuracy every N epochs")
    g.add_argument("--ckpt-every",      type=int,   default=5,
                   help="Save checkpoint every N epochs")
    # Contrib. [B]
    g.add_argument("--tau_init",        type=float, default=1.5,
                   help="[B] Initial sparsemax temperature")
    g.add_argument("--anneal_factor",   type=float, default=0.95,
                   help="[B] Temperature annealing factor per interval")
    g.add_argument("--anneal_interval", type=int,   default=5,
                   help="[B] Epochs between temperature decay steps")
    g.add_argument("--tau_min",         type=float, default=0.30,
                   help="[B] Minimum sparsemax temperature floor (prevents collapse)")
    # Contrib. [C]
    g.add_argument("--alpha_freq",      type=int,   default=10,
                   help="[C] Update alpha every N weight steps")
    # Contrib. [D]
    g.add_argument("--entropy_thresh",  type=float, default=0.05,
                   help="[D] Early-stop threshold on mean alpha entropy")
    g.add_argument("--auc_patience",    type=int,   default=15,
                   help="[D] Stop if mean AUC does not improve for this many epochs")
    g.add_argument("--rewind_thresh",   type=float, default=0.10,
                   help="[D] Rewind alphas to best if mean AUC drops by this fraction")

    g2 = parser.add_argument_group("Retrain Phase")
    g2.add_argument("--retrain_epochs", type=int,   default=200,
                    help="Epochs for discrete model retraining")
    g2.add_argument("--retrain_lr",     type=float, default=0.025,
                    help="Learning rate for retraining")
    g2.add_argument("--mixup-alpha",    type=float, default=0.2,
                    help="Mixup Beta distribution alpha (0 = disabled)")
    g2.add_argument("--label-smoothing",type=float, default=0.1,
                    help="Label smoothing for CE tasks (0 = disabled)")
    g2.add_argument("--tta",            action="store_true",
                    help="Enable test-time augmentation (8-view ensemble) at final benchmark eval")
    # Ablation flags — disable individual contributions for experimental comparison
    g2.add_argument("--no-contrib-b",   action="store_true",
                    help="[Ablation] Disable [B] temperature annealing: use fixed tau=tau_init throughout")
    g2.add_argument("--no-contrib-c",   action="store_true",
                    help="[Ablation] Disable [C] delayed alpha updates: update alphas every weight step")
    g2.add_argument("--no-contrib-d",   action="store_true",
                    help="[Ablation] Disable [D] entropy early stopping: run all epochs unconditionally")

    g3 = parser.add_argument_group("Infrastructure")
    g3.add_argument("--device",         type=str,   default="cpu",
                    help="Device: cpu | cuda | mps")
    g3.add_argument("--ckpt-dir",       type=str,   default="checkpoints")
    g3.add_argument("--save-dir",       type=str,   default="./results",
                    help="Directory for results and saved models")
    g3.add_argument("--resume",         type=str,   default=None,
                    help="Path to checkpoint to resume from")
    g3.add_argument("--no-real",        action="store_true",
                    help="Force mock data even if medmnist is installed")
    g3.add_argument("--seed",           type=int,   default=42)
    g3.add_argument("--workers",        type=int,   default=0,
                    help="DataLoader num_workers")
    g3.add_argument("--img-size",       type=int,   default=28,
                    help="Input resolution: 28 | 64 | 128 (64 recommended)")
    g3.add_argument("--search-micro-batch", type=int, default=0,
                    help="Micro-batch size used only during search steps (0 = full batch)")

    return parser


if __name__ == "__main__":
    from train import run_search

    args = build_parser().parse_args()

    if not _HAS_SKLEARN:
        logger.warning(
            "scikit-learn not installed — AUC metrics will be NaN. "
            "Install with: pip install scikit-learn"
        )

    # Apply ablation overrides before passing to run_search.
    tau_init_eff      = args.tau_init
    anneal_factor_eff = args.anneal_factor if not args.no_contrib_b else 1.0
    anneal_int_eff    = args.anneal_interval if not args.no_contrib_b else 9999
    tau_min_eff       = args.tau_min if not args.no_contrib_b else args.tau_init
    alpha_freq_eff    = args.alpha_freq if not args.no_contrib_c else 1
    entropy_eff       = args.entropy_thresh if not args.no_contrib_d else 0.0
    auc_patience_eff  = args.auc_patience if not args.no_contrib_d else 99999

    run_search(
        num_epochs        = args.epochs,
        batch_size        = args.batch,
        num_layers        = args.layers,
        channels          = args.channels,
        lr_weights        = args.lr_w,
        lr_alphas         = args.lr_a,
        retrain_epochs    = args.retrain_epochs,
        retrain_lr        = args.retrain_lr,
        log_interval      = args.log,
        eval_interval     = args.eval_every,
        ckpt_interval     = args.ckpt_every,
        ckpt_dir          = args.ckpt_dir,
        save_dir          = args.save_dir,
        resume_from       = args.resume,
        use_real_data     = not args.no_real,
        device_str        = args.device,
        seed              = args.seed,
        num_workers       = args.workers,
        tau_init          = tau_init_eff,
        anneal_factor     = anneal_factor_eff,
        anneal_interval   = anneal_int_eff,
        tau_min           = tau_min_eff,
        alpha_update_freq = alpha_freq_eff,
        entropy_threshold = entropy_eff,
        auc_patience      = auc_patience_eff,
        rewind_thresh     = args.rewind_thresh,
        img_size          = args.img_size,
        mixup_alpha       = args.mixup_alpha,
        label_smoothing   = args.label_smoothing,
        search_micro_batch= args.search_micro_batch,
        use_tta           = args.tta,
    )
