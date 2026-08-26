"""Per-label F1/support breakdown for ChestMNIST — shows which labels drag
down the macro-F1 average, to support reporting per-label results alongside
the single macro number."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data import MedMNISTDataset, TASK_REGISTRY
from src.ops import OP_NAMES
from src.retrain import _calibrate_chest_thresholds
from src.supernet import TaskAwareSupernet

CHANNELS, IMG_SIZE, LAYERS, TASK_IDS = 128, 64, 8, [0, 1, 2]
arch = ["MBConv3x3", "MaxPool3x3", "SepConv5x5", "MBConv3x3", "SepConv3x3",
        "ResidualBN", "ResidualBN", "MBConvSE"]

supernet = TaskAwareSupernet(num_tasks=3, num_layers=LAYERS, channels=CHANNELS,
                              img_size=IMG_SIZE, task_ids=TASK_IDS)
with torch.no_grad():
    for l, op_name in enumerate(arch):
        idx = OP_NAMES.index(op_name)
        supernet.alphas[1, l, :] = -10.0
        supernet.alphas[1, l, idx] = 10.0
model = supernet.discretize(1)
model.load_state_dict(torch.load("results/discrete_chestmnist_best.pt", map_location="cpu"))
model.eval()

val_ds  = MedMNISTDataset("val",  seed=43, use_real=True, img_size=IMG_SIZE, task_ids=TASK_IDS)
test_ds = MedMNISTDataset("test", seed=44, use_real=True, img_size=IMG_SIZE, task_ids=TASK_IDS)
kw = dict(collate_fn=MedMNISTDataset.collate_fn, num_workers=0)
val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=True,  drop_last=True,  **kw)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False, **kw)

device = torch.device("cpu")
thresholds = _calibrate_chest_thresholds(model, val_loader, device, 1, n_labels=14)

all_scores, all_labels = [], []
with torch.no_grad():
    for images, labels, task_ids in test_loader:
        mask = (task_ids == 1)
        if not mask.any():
            continue
        imgs_k = images[mask]
        lbl_k  = torch.stack([labels[i] for i in mask.nonzero(as_tuple=True)[0].tolist()])
        scores = torch.sigmoid(model(imgs_k)).numpy()
        all_scores.append(scores)
        all_labels.append(lbl_k.numpy())

y_score = np.concatenate(all_scores, axis=0)
y_true  = np.concatenate(all_labels, axis=0)
y_pred  = (y_score >= thresholds[np.newaxis, :]).astype(np.float32)

names = medmnist_labels = __import__("medmnist").INFO["chestmnist"]["label"]
print(f"{'label':<14s} {'n_pos (test)':>12s} {'prevalence':>11s} {'F1':>7s} {'thr':>6s}")
f1s = []
for c in range(14):
    n_pos = int(y_true[:, c].sum())
    prev  = n_pos / y_true.shape[0]
    f1 = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
    f1s.append(f1)
    print(f"{names[str(c)]:<14s} {n_pos:>12d} {prev*100:>10.2f}% {f1:>7.3f} {thresholds[c]:>6.2f}")
print(f"\nMacro F1 (mean of above): {np.mean(f1s):.4f}")
