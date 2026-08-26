"""
Data pipeline for multi-task MedMNIST.

  TASK_REGISTRY     — full 12-task registry; subset via task_ids argument.
  MedMNISTDataset   — unified PyTorch Dataset for any subset of MedMNIST tasks;
                       falls back to synthetic mock data when the medmnist
                       package is unavailable.
  build_dataloaders — convenience factory returning
                       (train_loader, val_bilevel_loader, eval_loader).

Adding tasks later
------------------
Pass ``task_ids=[0, 1, 2, 3, ...]`` to ``MedMNISTDataset`` and
``build_dataloaders``.  Everything else — supernet heads, alpha tensors,
loss routing — scales automatically from TASK_REGISTRY.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Full 12-task registry ─────────────────────────────────────────────────────
# Each entry: (name, medmnist_class_str, n_classes, is_grayscale, is_multilabel)
# medmnist_class_str is resolved lazily so importing this file never fails when
# the medmnist package is absent.
TASK_REGISTRY: Dict[int, Tuple[str, str, int, bool, bool]] = {
    0:  ("PathMNIST",      "PathMNIST",      9,  False, False),
    1:  ("ChestMNIST",     "ChestMNIST",     14, False, True ),
    2:  ("DermaMNIST",     "DermaMNIST",     7,  False, False),
    3:  ("OCTMNIST",       "OCTMNIST",        4,  True,  False),
    4:  ("PneumoniaMNIST", "PneumoniaMNIST",  2,  True,  False),
    5:  ("RetinaMNIST",    "RetinaMNIST",     5,  False, False),
    6:  ("BreastMNIST",    "BreastMNIST",     2,  True,  False),
    7:  ("BloodMNIST",     "BloodMNIST",      8,  False, False),
    8:  ("TissueMNIST",    "TissueMNIST",     8,  True,  False),
    9:  ("OrganAMNIST",    "OrganAMNIST",     11, True,  False),
    10: ("OrganCMNIST",    "OrganCMNIST",     11, True,  False),
    11: ("OrganSMNIST",    "OrganSMNIST",     11, True,  False),
}

# Default 3-task subset (preserves existing behaviour).
DEFAULT_TASK_IDS: List[int] = [0, 1, 2]

# ── Per-task normalisation stats ──────────────────────────────────────────────
_TASK_MEAN: Dict[int, List[float]] = {
    0:  [0.7406, 0.5330, 0.7059],   # PathMNIST
    1:  [0.4914, 0.4914, 0.4914],   # ChestMNIST (grayscale → 3ch)
    2:  [0.7632, 0.5380, 0.5614],   # DermaMNIST
    3:  [0.1248, 0.1248, 0.1248],   # OCTMNIST
    4:  [0.5623, 0.5623, 0.5623],   # PneumoniaMNIST
    5:  [0.4366, 0.2152, 0.1237],   # RetinaMNIST
    6:  [0.3257, 0.3257, 0.3257],   # BreastMNIST
    7:  [0.7943, 0.6597, 0.7174],   # BloodMNIST
    8:  [0.1015, 0.1015, 0.1015],   # TissueMNIST
    9:  [0.4753, 0.4753, 0.4753],   # OrganAMNIST
    10: [0.4686, 0.4686, 0.4686],   # OrganCMNIST
    11: [0.4706, 0.4706, 0.4706],   # OrganSMNIST
}
_TASK_STD: Dict[int, List[float]] = {
    0:  [0.1735, 0.2069, 0.1571],
    1:  [0.2023, 0.2023, 0.2023],
    2:  [0.1409, 0.1526, 0.1686],
    3:  [0.2260, 0.2260, 0.2260],
    4:  [0.2479, 0.2479, 0.2479],
    5:  [0.2954, 0.1824, 0.1007],
    6:  [0.3284, 0.3284, 0.3284],
    7:  [0.2138, 0.2345, 0.1900],
    8:  [0.1403, 0.1403, 0.1403],
    9:  [0.2910, 0.2910, 0.2910],
    10: [0.2824, 0.2824, 0.2824],
    11: [0.2859, 0.2859, 0.2859],
}


def _get_transforms(task_id: int, train: bool, img_size: int = 64):
    """
    Return a callable torchvision transform for one task.
    Gracefully degrades to a no-op if torchvision is absent.

    Small datasets (< 10k train samples) get heavier augmentation.
    Currently: DermaMNIST (task 2), BreastMNIST (task 6), RetinaMNIST (task 5).
    All other tasks get standard augmentation.
    """
    # Tasks with small training sets that benefit from heavier augmentation.
    HEAVY_AUG_TASKS = {2, 5, 6}

    try:
        import torchvision.transforms as T
        mean   = _TASK_MEAN.get(task_id, [0.5, 0.5, 0.5])
        std    = _TASK_STD.get(task_id, [0.5, 0.5, 0.5])
        resize = [T.Resize((img_size, img_size))] if img_size != 28 else []
        if train:
            if task_id in HEAVY_AUG_TASKS:
                return T.Compose([
                    *resize,
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(degrees=180),
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                   scale=(0.85, 1.15), shear=10),
                    T.ColorJitter(brightness=0.5, contrast=0.5,
                                  saturation=0.4, hue=0.15),
                    T.ConvertImageDtype(torch.uint8),
                    T.RandAugment(num_ops=3, magnitude=12),
                    T.ConvertImageDtype(torch.float32),
                    T.RandomErasing(p=0.35, scale=(0.02, 0.15)),
                    T.Normalize(mean=mean, std=std),
                ])
            return T.Compose([
                *resize,
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.2, hue=0.05),
                T.ConvertImageDtype(torch.uint8),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ConvertImageDtype(torch.float32),
                T.RandomErasing(p=0.2, scale=(0.02, 0.10)),
                T.Normalize(mean=mean, std=std),
            ])
        return T.Compose([*resize, T.Normalize(mean=mean, std=std)])
    except ImportError:
        return lambda x: x   # no-op fallback


class MedMNISTDataset(Dataset):
    """
    Multi-task MedMNIST dataset.

    Yields tuples ``(image [3,H,W], label, task_id)`` where:
      - label is a ``long`` scalar for single-label tasks.
      - label is a ``float`` multi-hot tensor ``[num_classes]`` for
        multi-label tasks (e.g. ChestMNIST).

    Task selection
    --------------
    Pass ``task_ids=[0, 1, 2]`` to use the default 3 tasks, or any subset/
    superset of the 12 tasks defined in ``TASK_REGISTRY``.  The class-level
    dicts (NUM_CLASSES, TASK_NAMES, etc.) are built dynamically from the
    registry so all downstream code scales automatically.
    """

    # Populated dynamically from TASK_REGISTRY in __init__.
    NUM_CLASSES:   Dict[int, int]  = {}
    TASK_NAMES:    Dict[int, str]  = {}
    IS_GRAYSCALE:  Dict[int, bool] = {}
    IS_MULTILABEL: Dict[int, bool] = {}

    def __init__(
        self,
        split:            str            = "train",
        num_mock_samples: int            = 1200,
        seed:             int            = 42,
        use_real:         bool           = True,
        img_size:         int            = 64,
        task_ids:         Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.split    = split
        self.is_train = (split == "train")
        self.img_size = img_size

        # Resolve active task set from registry.
        _ids = task_ids if task_ids is not None else DEFAULT_TASK_IDS
        self.active_task_ids: List[int] = sorted(_ids)

        # Instance-level dicts (shadow the class-level placeholders so state
        # from one MedMNISTDataset instance never leaks into another).
        # Keyed by registry id (matches TASK_REGISTRY), same as before.
        self.NUM_CLASSES:   Dict[int, int]  = {}
        self.TASK_NAMES:    Dict[int, str]  = {}
        self.IS_GRAYSCALE:  Dict[int, bool] = {}
        self.IS_MULTILABEL: Dict[int, bool] = {}
        for tid in self.active_task_ids:
            if tid not in TASK_REGISTRY:
                raise ValueError(
                    f"task_id={tid} not found in TASK_REGISTRY. "
                    f"Valid ids: {sorted(TASK_REGISTRY)}"
                )
            name, _, nc, gray, ml = TASK_REGISTRY[tid]
            self.NUM_CLASSES[tid]   = nc
            self.TASK_NAMES[tid]    = name
            self.IS_GRAYSCALE[tid]  = gray
            self.IS_MULTILABEL[tid] = ml

        # Per-sample task labels stored below use the *position* within
        # active_task_ids (0..len-1), matching how the supernet indexes its
        # stems/heads/alphas — NOT the raw registry id.  self.transforms is
        # keyed the same way; _get_transforms() itself still takes the
        # registry id so it can look up the right augmentation/normalisation.
        self._pos_of_tid = {tid: pos for pos, tid in enumerate(self.active_task_ids)}
        self.transforms = {
            pos: _get_transforms(tid, train=self.is_train, img_size=img_size)
            for pos, tid in enumerate(self.active_task_ids)
        }
        self.images:   List[Tensor] = []
        self.labels:   list         = []
        self.task_ids: List[int]    = []

        loaded_real = False
        if use_real:
            try:
                loaded_real = self._load_real(split)
            except Exception as exc:
                print(f"[MedMNIST] Real data unavailable ({exc}). "
                      f"Using synthetic mock data.")

        if not loaded_real:
            self._load_mock(num_mock_samples, seed)

    # ── Real data loader ──────────────────────────────────────────────────────

    def _load_real(self, split: str) -> bool:
        import medmnist  # noqa: F401

        sources = []
        for tid in self.active_task_ids:
            name, cls_str, _, _, _ = TASK_REGISTRY[tid]
            cls = getattr(medmnist, cls_str)
            sources.append((cls, tid, split))

        for cls, task_id, s in sources:
            ds  = cls(split=s, download=True, as_rgb=True, size=self.img_size)
            pos = self._pos_of_tid[task_id]
            for img_np, lbl_np in zip(ds.imgs, ds.labels):
                t = torch.from_numpy(img_np).float() / 255.0
                if t.dim() == 2:          # grayscale (H, W) → (1, H, W)
                    t = t.unsqueeze(0)
                else:                     # (H, W, C) → (C, H, W)
                    t = t.permute(2, 0, 1)
                img = t
                # `as_rgb=True` above only affects the dataset's own
                # __getitem__ (PIL path) — it does NOT touch the raw
                # `ds.imgs` array we read here, so natively single-channel
                # sources (e.g. ChestMNIST's X-rays) still arrive as (1,H,W)
                # even though TASK_REGISTRY's is_grayscale flag says False
                # for them. Expand to 3 channels whenever the array actually
                # came out single-channel, rather than trusting that flag.
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)

                if self.IS_MULTILABEL[task_id]:
                    lbl = torch.from_numpy(lbl_np.flatten().astype("float32"))
                else:
                    lbl = int(lbl_np[0])

                self.images.append(img)
                self.labels.append(lbl)
                # Stored as *position* (not registry id) — see __init__ note.
                self.task_ids.append(pos)

        print(f"[MedMNIST] Loaded real data: {len(self.images)} samples "
              f"(split={split})")
        return True

    # ── Mock / synthetic data loader ──────────────────────────────────────────

    def _load_mock(self, n: int, seed: int) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)

        for i in range(n):
            pos     = i % len(self.active_task_ids)
            task_id = self.active_task_ids[pos]
            if self.IS_GRAYSCALE[task_id]:
                gray = torch.rand(1, self.img_size, self.img_size, generator=rng)
                img  = gray.repeat(3, 1, 1)
            else:
                img  = torch.rand(3, self.img_size, self.img_size, generator=rng)

            if self.IS_MULTILABEL[task_id]:
                nc       = self.NUM_CLASSES[task_id]
                lbl      = torch.zeros(nc, dtype=torch.float32)
                n_active = int(torch.randint(1, 4, (1,), generator=rng).item())
                indices  = torch.randperm(nc, generator=rng)[:n_active]
                lbl[indices] = 1.0
            else:
                lbl = int(
                    torch.randint(0, self.NUM_CLASSES[task_id], (1,),
                                  generator=rng).item()
                )

            self.images.append(img)
            self.labels.append(lbl)
            # Stored as *position* (not registry id) — see __init__ note.
            self.task_ids.append(pos)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        task_id = self.task_ids[idx]
        img = self.transforms[task_id](img)
        return img, self.labels[idx], task_id

    @staticmethod
    def collate_fn(batch):
        """
        Custom collation that handles mixed scalar / tensor labels.
        Returns (images_tensor, labels_list, task_ids_tensor).
        """
        images, labels, task_ids = zip(*batch)
        images_t   = torch.stack(images)
        task_ids_t = torch.tensor(task_ids, dtype=torch.long)
        return images_t, list(labels), task_ids_t


# ── Weighted sampler for imbalanced tasks ─────────────────────────────────────

def build_weighted_sampler(
    task_id: int,
    dataset: "MedMNISTDataset",
    oversample_factor: float = 2.0,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives each class equal expected
    frequency per epoch for the given task.

    Only samples belonging to ``task_id`` are considered; other-task
    samples receive weight 0 so they are never drawn.  The sampler
    targets ``oversample_factor × n_task_samples`` draws per epoch,
    which oversamples minority classes to compensate for imbalance.

    Args:
        task_id           : Position of the task to balance within
                            ``dataset`` (matches ``dataset.task_ids``, i.e.
                            the index into the active task list — not the
                            registry id).
        dataset           : A MedMNISTDataset instance (any split).
        oversample_factor : Multiplier on the number of task samples for
                            the sampler length.  Default 2.0 doubles the
                            effective epoch length for the task.

    Returns:
        WeightedRandomSampler ready to pass to DataLoader.
    """
    # Collect per-class counts for this task.
    class_counts: dict = {}
    task_indices: List[int] = []
    for idx in range(len(dataset)):
        if dataset.task_ids[idx] != task_id:
            continue
        lbl = dataset.labels[idx]
        cls = int(lbl.item() if isinstance(lbl, Tensor) else lbl)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        task_indices.append(idx)

    if not task_indices:
        raise ValueError(
            f"No samples found for task_id={task_id} in the dataset."
        )

    # Weight per sample = 1 / class_frequency (uniform class distribution).
    sample_weights = torch.zeros(len(dataset), dtype=torch.float64)
    for idx in task_indices:
        lbl = dataset.labels[idx]
        cls = int(lbl.item() if isinstance(lbl, Tensor) else lbl)
        sample_weights[idx] = 1.0 / class_counts[cls]

    num_samples = int(len(task_indices) * oversample_factor)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def compute_chest_pos_weights(
    dataset:    "MedMNISTDataset",
    task_id:    int,
    n_labels:   int   = 14,
    max_weight: float = 50.0,
) -> "torch.Tensor":
    """
    Compute per-label BCEWithLogitsLoss ``pos_weight`` for ChestMNIST.

    pos_weight[i] = n_negative[i] / n_positive[i], clipped at ``max_weight``.

    The static fallback ``_CHEST_POS_WEIGHT = 10×1`` is a rough approximation.
    ChestMNIST's 14 disease labels have prevalence ranging from ~0.2% (Hernia)
    to ~19% (Infiltration), so per-label weights derived from the actual training
    split are far more accurate and improve AUC on rare disease labels.

    Args:
        task_id  : Position of ChestMNIST within ``dataset`` (matches the
                   per-sample task labels in ``dataset.task_ids``, i.e. the
                   index into the active task list — not the registry id).
        n_labels : Number of ChestMNIST labels (14).

    Falls back to the static weights if no ChestMNIST samples are found
    (e.g., when using mock data).
    """
    from .losses import _CHEST_POS_WEIGHT  # avoid circular import at module level

    pos_counts = torch.zeros(n_labels, dtype=torch.float64)
    total      = 0

    for idx in range(len(dataset)):
        if dataset.task_ids[idx] != task_id:
            continue
        lbl = dataset.labels[idx]
        if isinstance(lbl, Tensor):
            pos_counts += lbl.double()
        else:
            pos_counts += torch.tensor(lbl, dtype=torch.float64)
        total += 1

    if total == 0 or pos_counts.sum() == 0:
        return _CHEST_POS_WEIGHT.clone()   # fallback: no real ChestMNIST data

    neg_counts = float(total) - pos_counts
    pw = (neg_counts / pos_counts.clamp(min=1.0)).clamp(max=max_weight)
    return pw.float()


# ── DataLoader factory ────────────────────────────────────────────────────────

def _build_task_balanced_sampler(dataset: "MedMNISTDataset") -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives each *task* equal expected
    frequency per epoch.

    Without balancing, DermaMNIST (~7k samples) is only ~4% of the
    combined 175k-sample train set, so each batch of 64 contains on
    average only ~2.5 DermaMNIST images — severely limiting what the
    architecture search can learn for that task.  This sampler ensures
    each task contributes equally to every mini-batch.
    """
    task_counts: dict = {}
    for tid in dataset.task_ids:
        task_counts[tid] = task_counts.get(tid, 0) + 1

    sample_weights = torch.zeros(len(dataset), dtype=torch.float64)
    for idx, tid in enumerate(dataset.task_ids):
        sample_weights[idx] = 1.0 / task_counts[tid]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(
    batch_size:    int                   = 64,
    num_workers:   int                   = 0,
    use_real_data: bool                  = True,
    seed:          int                   = 42,
    img_size:      int                   = 64,
    balance_tasks: bool                  = True,
    task_ids:      Optional[List[int]]   = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct and return ``(train_loader, val_bilevel_loader, eval_loader)``.

    - ``train_loader``        — task-balanced sampler (balance_tasks=True) or
                                shuffled; drop_last=True  (weight updates).
    - ``val_bilevel_loader``  — shuffled, drop_last=True  (alpha updates).
    - ``eval_loader``         — no shuffle, drop_last=False (metric evaluation).

    ``task_ids`` selects which tasks to include (default: [0,1,2]).
    Pass e.g. ``task_ids=list(range(12))`` to run all 12 MedMNIST tasks.
    """
    _tids = task_ids if task_ids is not None else DEFAULT_TASK_IDS
    train_ds = MedMNISTDataset("train", num_mock_samples=1600,
                               seed=seed,     use_real=use_real_data,
                               img_size=img_size, task_ids=_tids)
    val_ds   = MedMNISTDataset("val",   num_mock_samples=400,
                               seed=seed + 1, use_real=use_real_data,
                               img_size=img_size, task_ids=_tids)
    test_ds  = MedMNISTDataset("test",  num_mock_samples=400,
                               seed=seed + 2, use_real=use_real_data,
                               img_size=img_size, task_ids=_tids)

    kw = dict(collate_fn=MedMNISTDataset.collate_fn, num_workers=num_workers,
              pin_memory=True)

    if balance_tasks:
        task_sampler = _build_task_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=task_sampler,
            drop_last=True, **kw,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            drop_last=True, **kw,
        )
    val_loader_bilevel = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, **kw,
    )
    eval_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, **kw,
    )
    return train_loader, val_loader_bilevel, eval_loader
