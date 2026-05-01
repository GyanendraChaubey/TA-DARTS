"""
MT-DARTS v2 — modular package.

Quick imports::

    from src.supernet import TaskAwareSupernet
    from src.controller import SearchController
    from src.data import MedMNISTDataset, build_dataloaders
    from src.normalizers import sparsemax, annealed_sparsemax
    from src.metrics import evaluate, evaluate_task, alpha_entropy
    from src.losses import task_loss
    from src.retrain import retrain_discrete
    from src.reporting import print_benchmark_table, save_benchmark_results
    from src.utils import set_seed

Full pipeline entry-point::

    from train import run_search
"""

from .controller import SearchController
from .data import MedMNISTDataset, build_dataloaders
from .losses import task_loss
from .metrics import alpha_entropy, evaluate, evaluate_task
from .normalizers import annealed_sparsemax, sparsemax
from .reporting import print_benchmark_table, save_benchmark_results
from .retrain import retrain_discrete
from .supernet import TaskAwareSupernet
from .utils import set_seed

__all__ = [
    "TaskAwareSupernet",
    "SearchController",
    "MedMNISTDataset",
    "build_dataloaders",
    "sparsemax",
    "annealed_sparsemax",
    "evaluate",
    "evaluate_task",
    "alpha_entropy",
    "task_loss",
    "retrain_discrete",
    "print_benchmark_table",
    "save_benchmark_results",
    "set_seed",
]
