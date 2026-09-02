"""
Utility helpers shared across MT-DARTS.

  set_seed()         — reproducible RNG initialisation.
  _make_divisible()  — channel-width rounding used by MBConv.
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("MT-DARTS")


def set_seed(seed: int = 42) -> None:
    """
    Set all relevant RNG seeds for fully reproducible runs.
    Call this before constructing any model or dataloader.

    Determinism stack:
      - PYTHONHASHSEED         : Python dict/set ordering
      - numpy / random seeds   : data augmentation RNG
      - torch manual seed      : weight init, dropout masks
      - cudnn.deterministic    : disables non-deterministic cuDNN kernels
      - cudnn.benchmark=False  : disables auto-tuner (picks same kernel every run)
      - use_deterministic_algorithms : forces deterministic CUDA atomics
                                       (BatchNorm, AdaptiveAvgPool, scatter, etc.)
      - CUBLAS_WORKSPACE_CONFIG: required by cublas deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # warn_only=True lets ops without a deterministic implementation fall back
    # gracefully instead of raising an error — important for TTA crop/resize ops.
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(f"Random seed set to {seed}")


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
