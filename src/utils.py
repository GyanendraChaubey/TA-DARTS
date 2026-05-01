"""
Utility helpers shared across MT-DARTS v2.

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
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # disable auto-tuner for reproducibility
    logger.info(f"Random seed set to {seed}")


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
