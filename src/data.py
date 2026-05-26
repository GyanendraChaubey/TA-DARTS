"""
Data pipeline for multi-task MedMNIST.

  MedMNISTDataset   — unified PyTorch Dataset for PathMNIST, ChestMNIST,
                       DermaMNIST; falls back to synthetic mock data when
                       the medmnist package is unavailable.
  build_dataloaders — convenience factory returning
                       (train_loader, val_bilevel_loader, eval_loader).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Per-task normalisation stats ──────────────────────────────────────────────
# Replace with dataset-specific values computed from real MedMNIST splits.
_TASK_MEAN: dict = {
    0: [0.7406, 0.5330, 0.7059],   # PathMNIST  (RGB histology)
    1: [0.4914, 0.4914, 0.4914],   # ChestMNIST (grayscale → 3ch)
    2: [0.7632, 0.5380, 0.5614],   # DermaMNIST (RGB dermoscopy)
}
_TASK_STD: dict = {
    0: [0.1735, 0.2069, 0.1571],
    1: [0.2023, 0.2023, 0.2023],
    2: [0.1409, 0.1526, 0.1686],
}


def _get_transforms(task_id: int, train: bool, img_size: int = 64):
    """
    Return a callable torchvision transform for one task.
    Gracefully degrades to a no-op if torchvision is absent.

    Train: resize (if needed) + flips + strong colour jitter
           + RandAugment + RandomErasing + normalise.
    Eval : resize (if needed) + normalise only.
    """
    try:
        import torchvision.transforms as T
        mean   = _TASK_MEAN[task_id]
        std    = _TASK_STD[task_id]
        resize = [T.Resize((img_size, img_size))] if img_size != 28 else []
        if train:
            # DermaMNIST (task 2) gets heavier augmentation: it has only 7k
            # train samples and severe class imbalance.  Strong colour and
            # geometric perturbations act as an implicit regulariser and
            # improve minority-class generalisation.
            if task_id == 2:
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
                # RandAugment requires uint8; convert, augment, convert back.
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

    Yields tuples ``(image [3,28,28], label, task_id)`` where:
      - label is a ``long`` scalar for single-label tasks
        (PathMNIST, DermaMNIST).
      - label is a ``float`` multi-hot tensor ``[num_classes]`` for
        multi-label tasks (ChestMNIST).
    """

    NUM_CLASSES:   dict = {0: 9,  1: 14, 2: 7}
    TASK_NAMES:    dict = {0: "PathMNIST", 1: "ChestMNIST", 2: "DermaMNIST"}
    IS_GRAYSCALE:  dict = {0: False, 1: True,  2: False}
    IS_MULTILABEL: dict = {0: False, 1: True,  2: False}

    def __init__(
        self,
        split:            str  = "train",
        num_mock_samples: int  = 1200,
        seed:             int  = 42,
        use_real:         bool = True,
        img_size:         int  = 64,
    ) -> None:
        super().__init__()
        self.split    = split
        self.is_train = (split == "train")
        self.img_size = img_size

        self.transforms = {
            k: _get_transforms(k, train=self.is_train, img_size=img_size)
            for k in self.NUM_CLASSES
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
        from medmnist import ChestMNIST, DermaMNIST, PathMNIST  # noqa: F401

        sources = [
            (PathMNIST,  0, split),
            (ChestMNIST, 1, split),
            (DermaMNIST, 2, split),
        ]
        for cls, task_id, s in sources:
            ds = cls(split=s, download=True, as_rgb=True, size=self.img_size)
            for img_np, lbl_np in zip(ds.imgs, ds.labels):
                t = torch.from_numpy(img_np).float() / 255.0
                if t.dim() == 2:          # grayscale (H, W) → (1, H, W)
                    t = t.unsqueeze(0)
                else:                     # (H, W, C) → (C, H, W)
                    t = t.permute(2, 0, 1)
                img = t
                if self.IS_GRAYSCALE[task_id]:
                    img = img.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                img = self.transforms[task_id](img)

                if self.IS_MULTILABEL[task_id]:
                    lbl = torch.from_numpy(lbl_np.flatten().astype("float32"))
                else:
                    lbl = int(lbl_np[0])

                self.images.append(img)
                self.labels.append(lbl)
                self.task_ids.append(task_id)

        print(f"[MedMNIST] Loaded real data: {len(self.images)} samples "
              f"(split={split})")
        return True

    # ── Mock / synthetic data loader ──────────────────────────────────────────

    def _load_mock(self, n: int, seed: int) -> None:
        rng       = torch.Generator()
        rng.manual_seed(seed)
        num_tasks = len(self.NUM_CLASSES)

        for i in range(n):
            task_id = i % num_tasks
            if self.IS_GRAYSCALE[task_id]:
                gray = torch.rand(1, self.img_size, self.img_size, generator=rng)
                img  = gray.repeat(3, 1, 1)
            else:
                img  = torch.rand(3, self.img_size, self.img_size, generator=rng)

            img = self.transforms[task_id](img)

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
            self.task_ids.append(task_id)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], self.task_ids[idx]

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
        task_id           : Task to balance (e.g. 2 = DermaMNIST).
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
    max_weight: float = 50.0,
) -> "torch.Tensor":
    """
    Compute per-label BCEWithLogitsLoss ``pos_weight`` for ChestMNIST (task 1).

    pos_weight[i] = n_negative[i] / n_positive[i], clipped at ``max_weight``.

    The static fallback ``_CHEST_POS_WEIGHT = 10×1`` is a rough approximation.
    ChestMNIST's 14 disease labels have prevalence ranging from ~0.2% (Hernia)
    to ~19% (Infiltration), so per-label weights derived from the actual training
    split are far more accurate and improve AUC on rare disease labels.

    Falls back to the static weights if no ChestMNIST samples are found
    (e.g., when using mock data).
    """
    from .losses import _CHEST_POS_WEIGHT  # avoid circular import at module level

    n_labels   = MedMNISTDataset.NUM_CLASSES[1]  # 14
    pos_counts = torch.zeros(n_labels, dtype=torch.float64)
    total      = 0

    for idx in range(len(dataset)):
        if dataset.task_ids[idx] != 1:
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
    batch_size:    int  = 64,
    num_workers:   int  = 0,
    use_real_data: bool = True,
    seed:          int  = 42,
    img_size:      int  = 64,
    balance_tasks: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct and return ``(train_loader, val_bilevel_loader, eval_loader)``.

    - ``train_loader``        — task-balanced sampler (balance_tasks=True) or
                                shuffled; drop_last=True  (weight updates).
    - ``val_bilevel_loader``  — shuffled, drop_last=True  (alpha updates).
    - ``eval_loader``         — no shuffle, drop_last=False (metric evaluation).

    ``balance_tasks`` (default True) applies a task-balanced WeightedRandomSampler
    to ``train_loader`` so each task (PathMNIST, ChestMNIST, DermaMNIST) has
    equal expected frequency per batch, preventing the minority task DermaMNIST
    from being starved during architecture search.
    """
    train_ds = MedMNISTDataset("train", num_mock_samples=1600,
                               seed=seed,     use_real=use_real_data,
                               img_size=img_size)
    val_ds   = MedMNISTDataset("val",   num_mock_samples=400,
                               seed=seed + 1, use_real=use_real_data,
                               img_size=img_size)
    test_ds  = MedMNISTDataset("test",  num_mock_samples=400,
                               seed=seed + 2, use_real=use_real_data,
                               img_size=img_size)

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
