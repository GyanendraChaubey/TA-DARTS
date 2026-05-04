# MT-DARTS v2 — Multi-Task Neural Architecture Search on MedMNIST

MT-DARTS v2 jointly searches for per-task neural architectures across three
medical image classification benchmarks in a single differentiable pass,
then retrains each discovered architecture from scratch.

---

## Tasks

| # | Dataset     | Type        | Classes | Train samples |
|---|-------------|-------------|---------|---------------|
| 0 | PathMNIST   | Single-label| 9       | ~89,996       |
| 1 | ChestMNIST  | Multi-label | 14      | ~78,468       |
| 2 | DermaMNIST  | Single-label| 7       | ~7,007        |

---

## Quick start

**Smoke test** (CPU, mock data, ~30 seconds):

```bash
python main.py --no-real --epochs 2 --batch 32 --device cpu \
               --img-size 64 --channels 32 --layers 4 --retrain_epochs 3 \
               --tau_min 0.25
```

**Recommended run** (Kaggle GPU, ~7–8 hours total):

```bash
python main.py \
  --epochs 20 \
  --retrain_epochs 200 \
  --img-size 64 \
  --anneal_factor 0.85 \
  --anneal_interval 5 \
  --tau_min 0.25 \
  --batch 128 \
  --device cuda \
  --seed 42 \
  --save-dir /kaggle/working/results \
  --ckpt-dir /kaggle/working/checkpoints \
  --workers 4
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline

```
Phase A  Architecture search (bilevel optimisation, best-alpha tracked)
Phase B  Discretize per-task architecture from best-AUC alpha snapshot
Phase C  Retrain each discrete model from scratch (200 epochs)
Phase D  Benchmark report (ASCII table + JSON)
```

---

## Key design choices

**Shared supernet, per-task alpha.** A single backbone with 8 searchable
layers is shared across all tasks. Each task has its own architecture
parameter tensor `alpha_t` of shape `(L=8, O=7)`, so the search finds
task-specific operation sequences within the shared feature extractor.

**Annealed sparsemax.** Operation weights are computed as
`sparsemax(alpha / tau)`. Unlike softmax, sparsemax produces exact zeros,
so bad operations are completely suppressed. Temperature `tau` is annealed
from 1.5 downwards with a configurable floor (`--tau_min`, default 0.1)
that prevents gradient collapse at low temperatures.

**Best-alpha discretization.** During search, the alpha tensor is
snapshotted whenever mean AUC across all tasks improves. Phase B uses this
best-AUC snapshot — not the final epoch — to choose operations. This makes
the discovered architecture robust to late-search instability.

---

## Search space (7 operations)

| Op             | Description                                          |
|----------------|------------------------------------------------------|
| MBConv3x3      | Inverted residual, 3×3 depthwise, expansion 4        |
| MBConv5x5      | Inverted residual, 5×5 depthwise, expansion 6        |
| MBConvSE       | MBConv3×3 + Squeeze-and-Excitation (reduction r=4)   |
| DilatedConv3x3 | Dilated depthwise 3×3 (dilation=2) + pointwise 1×1  |
| SepConv5x5     | Depthwise-separable 5×5 + residual                   |
| SkipConnect    | Identity pass-through                                |
| Zero           | Zero tensor (layer dropped)                          |

---

## Important hyperparameters

| Flag                | Default | Effect                                        |
|---------------------|---------|-----------------------------------------------|
| `--epochs`          | 50      | Search epochs                                 |
| `--tau_init`        | 1.5     | Starting sparsemax temperature                |
| `--anneal_factor`   | 0.75    | Temperature decay per interval (0.85 = safer) |
| `--anneal_interval` | 5       | Epochs between decay steps                    |
| `--tau_min`         | 0.1     | Temperature floor — prevents collapse         |
| `--retrain_epochs`  | 200     | Epochs for discrete model retraining          |
| `--alpha_freq`      | 10      | Update alpha every N weight steps             |
| `--entropy_thresh`  | 0.05    | Early-stop when architecture entropy < this  |

> **Note on `--tau_min`:** with `anneal_factor=0.75` and 60 search epochs,
> tau decays to ~0.047 without a floor. Below ~0.2, sparsemax gradients
> degrade and architecture weights collapse. Set `--tau_min 0.25` for runs
> over 20 epochs.

---

## Output files

| Path                              | Contents                          |
|-----------------------------------|-----------------------------------|
| `results/benchmark_table.txt`     | Human-readable results table      |
| `results/benchmark_results.json`  | Machine-readable results + archs  |
| `results/architecture_snapshot.txt` | Per-task op distributions       |
| `results/discrete_*_best.pt`      | Best retrained model per task     |
| `checkpoints/mt_darts_latest.pt`  | Latest search checkpoint          |
| `checkpoints/mt_darts_epoch*.pt`  | Periodic search checkpoints       |
| `results/search_curves.csv`       | Per-epoch AUC, entropy, tau       |

---

## Project structure

```
main.py              CLI entry point
train.py             Four-phase pipeline orchestrator
src/
  supernet.py        TaskAwareSupernet (shared stem + MixedOps + 3 heads)
  controller.py      SearchController (bilevel optimiser, tau annealing)
  ops.py             7 candidate operations + MixedOp
  normalizers.py     sparsemax / annealed_sparsemax
  data.py            MedMNISTDataset + DataLoader builder
  losses.py          CE (PathMNIST, DermaMNIST) / BCE (ChestMNIST)
  retrain.py         Discrete model retraining with Mixup + warmup LR
  metrics.py         ACC, AUC, alpha entropy
  reporting.py       ASCII table + JSON report writer
  utils.py           set_seed, _make_divisible
```

---

## Documentation

- [architecture.md](architecture.md) — full technical reference (ops, supernet,
  bilevel search, retrain protocol, hyperparameters)
- [explainer.md](explainer.md) — plain-English walkthrough with no equations
