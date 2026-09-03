# MT-DARTS: Multi-Task Differentiable Architecture Search for Multi-Modal Medical Image Classification

**[Project page](https://gyanendrachaubey.github.io/MT-DARTS/)**

> Paper is still under correction — not yet public. It'll be posted to
> arXiv and linked here once finalised.

MT-DARTS jointly searches for per-task neural architectures across the
MedMNIST benchmark suite in a single differentiable pass, then retrains each
discovered architecture from scratch. It runs 3 tasks by default; any subset
of the 12-task MedMNIST registry can be selected with `--tasks`.

---

## Tasks

Default (`--tasks 0,1,2`):

| # | Dataset     | Type        | Classes | Train samples |
|---|-------------|-------------|---------|---------------|
| 0 | PathMNIST   | Single-label| 9       | ~89,996       |
| 1 | ChestMNIST  | Multi-label | 14      | ~78,468       |
| 2 | DermaMNIST  | Single-label| 7       | ~7,007        |

Any subset of all 12 MedMNIST tasks can be run instead — pass a comma-separated
list of ids, e.g. `--tasks 0,1,2,3,4,5,6,7,8,9,10,11` for the full suite. The
full registry (ids, class counts, grayscale/multi-label flags) lives in
`TASK_REGISTRY` in `src/data.py`. Dataset-specific tuning (DermaMNIST's
class-weighted/focal loss + CutMix, ChestMNIST's per-label threshold
calibration) automatically follows those two tasks wherever they appear in a
custom selection — it's not tied to task position.

---

## Quick start

**Smoke test** (CPU, mock data, ~30 seconds):

```bash
python main.py --no-real --epochs 2 --batch 32 --device cpu \
               --img-size 64 --channels 32 --layers 4 --retrain_epochs 3 \
               --tau_min 0.25
```

**Recommended run** (Kaggle T4 GPU, benchmark-beating settings, ~5–7 hours):

```bash
python main.py \
  --epochs 20 \
  --retrain_epochs 200 \
  --channels 128 \
  --layers 8 \
  --batch 64 \
  --img-size 64 \
  --tau_min 0.25 \
  --device cuda \
  --seed 42 \
  --save-dir /kaggle/working/results \
  --ckpt-dir /kaggle/working/checkpoints \
  --workers 2
```

> Omits `--anneal_factor`/`--anneal_interval` to use the current defaults
> (`0.95`/`5`) — the earlier `0.80`/`3` values annealed tau too fast and
> caused premature architecture collapse. See the note under
> [Important hyperparameters](#important-hyperparameters).

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
layers is shared across all active tasks (3 by default, up to 12 via
`--tasks`). Each task has its own architecture parameter tensor `alpha_t` of
shape `(L=8, O=10)` and its own classification head, so the search finds
task-specific operation sequences within the shared feature extractor.

**Annealed sparsemax.** Operation weights are computed as
`sparsemax(alpha / tau)`. Unlike softmax, sparsemax produces exact zeros,
so bad operations are completely suppressed. Temperature `tau` is annealed
from 1.5 downwards with a configurable floor (`--tau_min`, default 0.30)
that prevents gradient collapse at low temperatures.

**Best-alpha discretization.** During search, the alpha tensor is
snapshotted whenever mean AUC across all tasks improves. Phase B uses this
best-AUC snapshot — not the final epoch — to choose operations. This makes
the discovered architecture robust to late-search instability.

**Task-balanced search batches.** With the default 3 tasks, DermaMNIST is
only ~4% of the combined 175k dataset, giving it ~2–3 images per batch of 64
without correction. `build_dataloaders` uses `WeightedRandomSampler`
(task-level inverse frequency) to deliver roughly equal images per task per
batch, so small tasks meaningfully influence architecture parameters
throughout search regardless of how many tasks are active.

**Equal-weight task loss + entropy regularisation.** `SearchController`
averages each active task's mean loss with equal weight (not sample-weighted),
so a large dataset like PathMNIST can't dominate the gradient signal, and
clips the combined gradient norm (`--grad-clip`, not currently exposed via
CLI) before each weight step. An entropy penalty on `sparsemax(alpha)`
(`arch_reg_lambda`, default `0.01`) discourages any single operation —
including `ResidualBN` — from collapsing the search at every layer.

**Imbalance-aware retraining.**
- *DermaMNIST*: WeightedRandomSampler (class-balanced batches) + Focal Loss
  (γ=2, concentrates gradients on hard examples) + 50%/50% CutMix/Mixup.
  Class weights in the loss are disabled when the sampler is active to
  avoid double-correcting the same imbalance.
- *ChestMNIST*: per-label `pos_weight` computed from actual training
  prevalence (label range 0.2%–19%), replacing the static ×10 fallback.
- These apply automatically whenever DermaMNIST/ChestMNIST are part of the
  active `--tasks` selection, regardless of their position in it.

**Stochastic Weight Averaging (SWA).** In the last 25% of retraining
(or last 50 epochs), model weights are averaged rather than selecting a
single best checkpoint. The averaged model sits in a flatter loss basin and
generalises better — typical gain of 0.5–1.0% AUC at zero extra compute.

---

## Search space (10 operations)

| Op             | Description                                                   |
|----------------|---------------------------------------------------------------|
| MBConv3x3      | Inverted residual, 3×3 depthwise, expansion 4                 |
| MBConv5x5      | Inverted residual, 5×5 depthwise, expansion 6                 |
| MBConvSE       | MBConv3×3 + Squeeze-and-Excitation (reduction r=4)            |
| DilatedConv3x3 | Dilated depthwise 3×3 (dilation=2, RF≈5×5) + pointwise 1×1   |
| DilatedConv5x5 | Dilated depthwise 5×5 (dilation=2, RF≈9×9) + pointwise 1×1   |
| SepConv3x3     | Depthwise-separable 3×3 + residual                            |
| SepConv5x5     | Depthwise-separable 5×5 + residual                            |
| ResidualBN     | Identity shortcut + BatchNorm (`BN(x) + x`)                    |
| AvgPool3x3     | 3×3 average pooling — smooth spatial aggregation              |
| MaxPool3x3     | 3×3 max pooling — preserves dominant activations              |

> **Why no Zero?** In the original graph-based DARTS, Zero represents the
> absence of an edge and is safe because other paths can route around it.
> In a sequential chain, Zero at any layer kills all downstream gradient
> flow during retraining, guaranteed AUC=0.5. AvgPool3x3 is the replacement:
> it provides a low-cost "do little" option without dead layers.

> **Why ResidualBN instead of a pure skip?** A pure identity op has zero
> training cost, so sparsemax always collapsed toward it regardless of task.
> Adding BatchNorm keeps the near-skip shortcut path but gives the op
> learnable parameters (γ, β) and running statistics the search has to
> actually optimise, removing the free-lunch incentive that drove skip
> dominance.

---

## Important hyperparameters

| Flag                | Default | Effect                                        |
|---------------------|---------|-----------------------------------------------|
| `--tasks`           | `0,1,2` | Comma-separated task ids from `TASK_REGISTRY` |
| `--epochs`          | 50      | Search epochs                                 |
| `--lr_a`            | 3e-4    | Alpha (architecture) learning rate            |
| `--tau_init`        | 1.5     | Starting sparsemax temperature                |
| `--anneal_factor`   | 0.95    | Temperature decay per interval                |
| `--anneal_interval` | 5       | Epochs between decay steps                    |
| `--tau_min`         | 0.30    | Temperature floor — prevents collapse         |
| `--auc_patience`    | 15      | Early-stop if mean AUC stalls for N epochs    |
| `--rewind_thresh`   | 0.10    | Rewind alphas if AUC drops >10%               |
| `--retrain_epochs`  | 200     | Max epochs for discrete model retraining      |
| `--alpha_freq`      | 10      | Update alpha every N weight steps             |
| `--entropy_thresh`  | 0.05    | Early-stop when architecture entropy < this   |
| `--tta`             | off     | 8-view test-time augmentation ensemble — measured *worse* than plain eval on this benchmark (see note below); leave off unless re-validated |
| `--channels`        | 128     | Feature channel width                         |
| `--img-size`        | 28      | Input resolution: 28 \| 64 \| 128 (64 recommended) |

> **Note on `--tta`:** every one of the 8 TTA views in `evaluate_task_tta`
> (src/metrics.py) applies a 90%-scale crop before resizing back up — none of
> the 8 views evaluates the clean, uncropped image. Retraining never used a
> matching random-resized-crop augmentation, so all 8 views are
> out-of-distribution relative to training and TTA measurably *hurts*
> results (PathMNIST: 90.33%→86.06% ACC, 0.8732→0.8188 F1, confirmed by
> re-running both paths against the same checkpoint). Benchmark numbers in
> `results/` are from plain (non-TTA) eval.

> **Note on `--anneal_factor`/`--anneal_interval`:** earlier values of
> `0.90`/`3` decayed tau too fast and collapsed the architecture search
> prematurely. The current defaults (`0.95`/`5`) anneal more gradually;
> avoid going more aggressive than that without also raising `--tau_min`.

> **Note on `--tau_min`:** with `anneal_factor=0.75` over 60+ search epochs,
> tau decays below 0.05 without a floor. Below ~0.2, sparsemax gradients
> degrade and architecture weights collapse. The default 0.30 is safe for
> 20-epoch searches; raise to 0.40 for searches over 50 epochs.

---

## Output files

| Path                              | Contents                          |
|-----------------------------------|-----------------------------------|
| `results/benchmark_table.txt`     | Human-readable results table (ACC, AUC, F1, params, FLOPs, AUC/MParam) |
| `results/benchmark_results.json`  | Machine-readable results + archs  |
| `results/architecture_snapshot.txt` | Per-task op distributions       |
| `results/discrete_*_best.pt`      | Best retrained model per task     |
| `checkpoints/mt_darts_latest.pt`  | Latest search checkpoint          |
| `checkpoints/mt_darts_epoch*.pt`  | Periodic search checkpoints       |
| `results/search_curves.csv`       | Per-epoch AUC/ACC per task, entropy, tau |

---

## Ablation studies

`scripts/run_ablations.py` runs the real Phase A→D pipeline once per
ablation variant — sparsemax vs softmax, temperature annealing on/off,
alpha-update frequency `{1,5,10,20}`, entropy early stopping on/off,
task-balanced sampling on/off, architecture entropy regularisation on/off,
plus Mixup/label-smoothing/task-normalisation on/off — and saves everything
under `results/ablations/`. One command runs and saves the whole suite:

```bash
python scripts/run_ablations.py --epochs 50 --retrain-epochs 200 --device cuda
```

(For a quick local check instead of a full paper-matching run, use
`--no-real --epochs 2 --retrain-epochs 2 --device cpu`.)

`results/ablations/` and `checkpoints/ablations/` are created automatically
and are separate from the main `results/`/`checkpoints/` — nothing from a
normal run is touched. Each variant gets its own
`results/ablations/<variant_name>/` (full output, same schema as a normal
run), and the sweep as a whole produces:

| Path                                          | Contents                                              |
|------------------------------------------------|--------------------------------------------------------|
| `results/ablations/ablation_summary.json`     | Per-variant AUC/ACC/F1/params/timing + deltas vs. baseline |
| `results/ablations/ablation_summary.txt`      | Human-readable comparison table                       |

Re-running the same command later skips any variant whose results already
exist (pass `--force-rerun` to redo them). Search-space topology variants
and shared-vs-per-task alpha tensors aren't covered — both need real
architecture changes rather than a parameter toggle.

---

## Project structure

```
main.py              CLI entry point
train.py             Four-phase pipeline orchestrator
src/
  supernet.py        TaskAwareSupernet (shared stem + MixedOps + per-task heads,
                      task count/order set via --tasks)
  controller.py      SearchController (bilevel optimiser, tau annealing,
                      entropy regularisation)
  ops.py             10 candidate operations + MixedOp
  normalizers.py     sparsemax / annealed_sparsemax / annealed_softmax
  data.py            TASK_REGISTRY (12 MedMNIST tasks) + MedMNISTDataset +
                      DataLoader builder
  losses.py          CE/Focal (single-label) / BCE+per-label-weights (multi-label)
  retrain.py         Discrete model retraining with Mixup/CutMix, SWA, warm
                      restarts, FLOPs profiling
  metrics.py         ACC, AUC, F1, precision, recall, alpha entropy
  reporting.py       ASCII table + JSON report writer; ablation comparison
                      table/summary writer
  utils.py           set_seed, _make_divisible
scripts/
  run_ablations.py            Ablation-suite orchestrator (see Ablation studies)
  regenerate_report.py        Re-score existing checkpoints without retraining
  profile_resnet18_baseline.py  Independent FLOPs profiling for baselines
  chest_per_label_f1.py       ChestMNIST per-label F1 breakdown
```

---

## Documentation

Two deeper reference docs (`paper/docs/architecture.md` — full technical
reference; `paper/docs/explainer.md` — plain-English walkthrough) exist
locally but `paper/` is git-ignored, so they aren't part of this repository
and won't be present after a fresh clone.
