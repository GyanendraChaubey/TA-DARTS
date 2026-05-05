# MT-DARTS v2 — Architecture Reference

## 1. Scope

This document describes the architecture, optimization pipeline, and
experimental protocol of MT-DARTS v2.  Every section maps directly to
source files.

Codebase entry points:
- main.py
- train.py
- src/supernet.py
- src/controller.py
- src/data.py
- src/retrain.py

---

## 2. Problem Setup

Three medical image classification tasks run jointly:

    Task 0 - PathMNIST    single-label   9 classes
    Task 1 - ChestMNIST   multi-label   14 classes
    Task 2 - DermaMNIST   single-label   7 classes

A shared supernet provides feature extraction.  Each task owns:
- an independent classification head,
- an independent architecture parameter tensor alpha_t over L layers.

Goal: learn per-task discrete architectures in one differentiable search
pass, then retrain each selected architecture from scratch.

---

## 3. Four-Phase Pipeline  (train.py :: run_search)

```
+------------------------------------------------------------------+
|  Phase A   Bilevel architecture search on the supernet           |
|            Best alpha weights tracked throughout by mean AUC     |
+------------------------------------------------------------------+
|  Phase B   Restore best-AUC alphas; discretize per-task arch     |
+------------------------------------------------------------------+
|  Phase C   Retrain each discrete model from random init          |
+------------------------------------------------------------------+
|  Phase D   Evaluate and emit benchmark report                    |
+------------------------------------------------------------------+
```

---

## 4. Search Space  (src/ops.py)

Ten candidate operations, O = 10:

```
+---+----------------+----------------------------------------------+
| # | Name           | Description                                  |
+---+----------------+----------------------------------------------+
| 0 | MBConv3x3      | Inverted-residual, 3x3 dw, expansion 4       |
| 1 | MBConv5x5      | Inverted-residual, 5x5 dw, expansion 6       |
| 2 | MBConvSE       | MBConv3x3 + Squeeze-and-Excitation (r=4)     |
| 3 | DilatedConv3x3 | Dilated dw 3x3 (dilation=2, RF≈5×5) + pw 1x1|
| 4 | DilatedConv5x5 | Dilated dw 5x5 (dilation=2, RF≈9×9) + pw 1x1|
| 5 | SepConv3x3     | Depthwise-sep 3x3 + residual                 |
| 6 | SepConv5x5     | Depthwise-sep 5x5 + residual                 |
| 7 | SkipConnect    | Identity (residual pass-through)             |
| 8 | AvgPool3x3     | 3×3 average pooling (stride=1, pad=1)        |
| 9 | MaxPool3x3     | 3×3 max pooling (stride=1, pad=1)            |
+---+----------------+----------------------------------------------+
```

Design rationale:
- Zero was removed from the search space.  In a sequential chain architecture
  a Zero op kills all downstream gradient flow during retraining; AvgPool3x3
  is the direct replacement providing a "do little" option without dead layers.
- DilatedConv5x5 (RF≈9×9) complements DilatedConv3x3 (RF≈5×5), covering
  diffuse spatial patterns in ChestMNIST at 16×16 feature maps.
- SepConv3x3 complements SepConv5x5: local vs. wider efficient kernels.
- MaxPool3x3 and AvgPool3x3 are complementary: max preserves dominant
  activations (sharp edges, lesion boundaries); avg smooths spatial features.

SEBlock inside MBConvSE:
  global-avg-pool -> Linear(C, C//4) -> ReLU -> Linear(C//4, C) -> Sigmoid
  -> channel-wise multiply.

MixedOp (during search):

    h_out = sum_{o=1}^{10}  pi_{t,l,o} * op_o(h_in)

where pi is computed by annealed sparsemax (see Section 6).

---

## 5. Supernet Design  (src/supernet.py)

### 5.1 Adaptive input stem

```
img_size > 32  (default: 64x64 input)
  +------------------------------------------+
  | Conv2d(3, C/2, 3x3, stride=2, pad=1)    |
  | BatchNorm2d  +  ReLU6                    |
  | Conv2d(C/2, C, 3x3, stride=2, pad=1)    |
  | BatchNorm2d  +  ReLU6                    |
  +------------------------------------------+
  Output: (B, C, H/4, W/4)   -- e.g. 64x64 -> 16x16

img_size <= 32  (28x28 smoke test mode)
  +------------------------------------------+
  | Conv2d(3, C, 3x3, stride=1, pad=1)      |
  | BatchNorm2d  +  ReLU6                    |
  +------------------------------------------+
  Output: (B, C, H, W)
```

Default C = 64 channels.

### 5.2 Searchable body

L MixedOp cells in sequence (default L = 8):

```
  stem_out
     |
     v
  +----------+    +----------+    +----------+    +----------+
  | MixedOp  |--->| MixedOp  |--->|   ...    |--->| MixedOp  |
  |  cell 0  |    |  cell 1  |    |          |    | cell L-1 |
  +----------+    +----------+    +----------+    +----------+
      ^                ^                               ^
      |                |                               |
  alpha_t[0]       alpha_t[1]                    alpha_t[L-1]
  (task-specific weights per layer, shape (7,) after sparsemax)
```

### 5.3 Task-specific classification heads

After AdaptiveAvgPool2d(1) + Flatten  ->  (B, C):

```
  +--------------------------------------------------------------+
  | Dropout(p=0.3)                                               |
  | Linear(C, max(4*C, 256))                                     |
  | ReLU                                                         |
  | Dropout(p=0.2)                                               |
  | Linear(max(4*C, 256), nc_k)                                  |
  +--------------------------------------------------------------+

  nc_0 = 9   PathMNIST
  nc_1 = 14  ChestMNIST
  nc_2 = 7   DermaMNIST
```

### 5.4 Architecture parameters

```
  alpha   shape: (T=3, L=8, O=10)
  numel:  240
  init:   zeros
  optim:  Adam  (separate from network weights)
```

### 5.5 Complete forward path (single task t)

```
  Input (B, 3, 64, 64)
         |
         v
  +======================+
  |   Adaptive Stem      |
  |  2x stride-2 conv    |
  +======================+
         | (B, 64, 16, 16)
         v
  sparsemax(alpha_t / tau)  ->  pi_t  shape (L, 10)
         |
  +------+------+--  ...  --+------+
  |             |            |      |
  v             v            v      v
  MixedOp_0  MixedOp_1  ...  MixedOp_{L-1}
         |
         v  (B, 64, 16, 16)
  AdaptiveAvgPool2d(1)  ->  Flatten  ->  (B, 64)
         |
  +-----------+-----------+-----------+
  |           |           |           |
  v           v           v           |
  Head_0    Head_1    Head_2           |
  (B,9)    (B,14)     (B,7)           |
  PathMNIST ChestMNIST DermaMNIST     |
  CE+smooth BCE        CE+smooth  <---+
```

---

## 6. Bilevel Search Controller  (src/controller.py)

### 6.1 Optimizers

```
  +-----------+------+-------------------------------------------+
  | Param set | Opt  | Settings                                  |
  +-----------+------+-------------------------------------------+
  | weights w | SGD  | lr=0.025, momentum=0.9, wd=3e-4, nesterov|
  |           |      | + CosineAnnealingLR(T_max=E, eta_min=1e-4)|
  +-----------+------+-------------------------------------------+
  | alphas a  | Adam | lr=3e-4, weight_decay=1e-3                |
  +-----------+------+-------------------------------------------+
```

### 6.2 Per-step update rule

```
  Every step — weight update with per-task gradient normalization:

  For each task k present in the train mini-batch:
  |   loss_k  = task_loss( supernet(train_imgs_k, k, tau), labels_k )
  |   g_k     = grad_w( loss_k )             -- gradient on shared weights
  |   g_k    /= ||g_k||_2                    -- normalize to unit norm
  |   accumulate g_k into w.grad

  After all tasks accumulated:
  |   clip_grad_norm_(w, max_norm=5)
  |   w  <--  w - eta_w * w.grad

  Gradient normalization ensures all tasks contribute equally to the shared
  weight update regardless of dataset size differences (PathMNIST ~89k vs
  DermaMNIST ~7k samples).

  Every alpha_update_freq steps  (default = 10):
  |   loss_a = task_loss( supernet(val_imgs, task, tau), val_labels )
  |   alpha  <--  alpha - eta_a * grad_a( loss_a )
```

### 6.3 Temperature annealing

    tau_e = max( tau_0 * a ^ floor(e / m),  tau_min )

    tau_0    = 1.5    --tau_init
    a        = 0.85   --anneal_factor   (default; 0.75 collapses by epoch 25)
    m        = 10     --anneal_interval
    tau_min  = 0.1    --tau_min         (floor prevents sparsemax collapse)

Operation weights:

    pi_{t,l} = sparsemax( alpha_{t,l} / tau_e )

Sparsemax produces exact zeros; many ops are fully dropped during search.

**Warning:** with anneal_factor=0.75 over 60 epochs, tau decays to ~0.047
without the floor, causing sparsemax to behave as a hard argmax and making
gradient flow collapse.  Keep tau_min >= 0.1 for runs longer than 30 epochs.
For stable 20-epoch searches, tau_min=0.25 is recommended.

### 6.4 Label smoothing during search

task_loss() passes label_smoothing=0.1 to CrossEntropyLoss for tasks 0 and 2.
ChestMNIST (task 1, BCEWithLogitsLoss) is unaffected.

### 6.5 Entropy-based early stopping

    H = -(1 / T*L) * sum_{t,l,o}  pi_{t,l,o} * log(pi_{t,l,o})

Search halts early when H < 0.05  (--entropy_thresh).

---

## 7. Data Pipeline  (src/data.py)

### 7.1 Dataset

MedMNISTDataset: unified mixed-task dataset.
Each sample: (image tensor, label, task_id).

Image source:
- Real  : medmnist API, size=img_size, as_rgb=True, split={train,val,test}.
- Mock  : torch.rand(3, img_size, img_size)  -- used with --no-real flag.

### 7.2 Training transforms

```
  +----------------------------------------------------+
  | Resize((img_size, img_size))  if img_size != 28    |
  | RandomHorizontalFlip()                             |
  | RandomVerticalFlip()                               |
  | ColorJitter(b=0.3, c=0.3, s=0.2, h=0.05)          |
  | ConvertImageDtype(uint8)    -- required by RandAug |
  | RandAugment(num_ops=2, magnitude=9)                |
  | ConvertImageDtype(float32)                         |
  | RandomErasing(p=0.2, scale=(0.02, 0.10))           |
  | Normalize(task_mean, task_std)                     |
  +----------------------------------------------------+
```

Eval transforms: Resize (if needed) + Normalize only.

### 7.3 Label formats

```
  +------------+--------+------+------------------------------+
  | Task       | id     | Type | Shape / Range                |
  +------------+--------+------+------------------------------+
  | PathMNIST  | 0      | long | scalar, [0, 8]               |
  | ChestMNIST | 1      | float| tensor (14,), multi-hot      |
  | DermaMNIST | 2      | long | scalar, [0, 6]               |
  +------------+--------+------+------------------------------+
```

### 7.4 DataLoaders

```
  build_dataloaders() produces three loaders:

  train_loader         -- shuffled, mixed-task batches
  val_loader_bilevel   -- for alpha updates during search
  eval_loader          -- test set, final evaluation only

  Default: batch_size=64, pin_memory=True, num_workers=0
```

---

## 8. Loss Routing  (src/losses.py)

```
  +---------------------------+-----------------------------------+
  | Task                      | Loss                              |
  +---------------------------+-----------------------------------+
  | PathMNIST   (task_id=0)   | CrossEntropyLoss(smooth=0.1)     |
  | ChestMNIST  (task_id=1)   | BCEWithLogitsLoss (no smoothing) |
  | DermaMNIST  (task_id=2)   | CrossEntropyLoss(smooth=0.1)     |
  +---------------------------+-----------------------------------+
```

Class-imbalance weighting:

```
  DermaMNIST  -- CrossEntropyLoss(weight=_DERMA_CLASS_WEIGHTS)
                 [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 2.0]
                 (melanoma class upweighted 5x, dermatofibroma 2x)

  ChestMNIST  -- BCEWithLogitsLoss(pos_weight=ones(14)*10.0)
                 (all 14 conditions are rare; positives upweighted 10x)

  PathMNIST   -- no extra weighting (classes are roughly balanced)
```

Combined with per-task gradient normalization in SearchController (Section
6.2), task contributions are balanced at both the loss level (class weights)
and the gradient level (unit-norm scaling per task).

Class-imbalance weighting (defined in losses.py):

```
  DermaMNIST  -- CrossEntropyLoss(weight=_DERMA_CLASS_WEIGHTS)
                 _DERMA_CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 2.0]
                 (class 4 = melanoma upweighted 5x, class 6 = dermatofibroma 2x)

  ChestMNIST  -- BCEWithLogitsLoss(pos_weight=_CHEST_POS_WEIGHT)
                 _CHEST_POS_WEIGHT = ones(14) * 10.0
                 (all 14 conditions are rare; positive samples upweighted 10x)

  PathMNIST   -- no extra weighting (classes are roughly balanced)
```

Combined with per-task gradient normalization in SearchController, the
effective task contribution during weight updates is equalised both at the
loss level (class weights) and at the gradient level (unit-norm scaling).

---

## 9. Retraining Protocol  (src/retrain.py)

### 9.1 Weight reinitialization

```
  Conv2d   : kaiming_normal(fan_out, relu)
  BN       : weight=1, bias=0  (guards against bias=None)
  Linear   : normal(0, 0.01),  bias=0
```

### 9.2 Optimizer and LR schedule

```
  SGD  lr=0.025  momentum=0.9  wd=3e-4  nesterov=True

  SequentialLR:
  +----------------------------------------------+
  | warmup_epochs = max(5, num_epochs // 20)     |
  |                                              |
  | [0 .. warmup_epochs)                         |
  |   LinearLR: start_factor=0.1 -> 1.0          |
  |                                              |
  | [warmup_epochs .. num_epochs)                |
  |   CosineAnnealingLR: eta_min=1e-5            |
  +----------------------------------------------+

  Default num_epochs = 200
```

### 9.3 Mixup augmentation  (alpha = 0.2)

```
  lam ~ Beta(0.2, 0.2),  lam = max(lam, 1-lam)   (keep dominant)
  x_mix = lam * x_a  +  (1-lam) * x_b

  CE tasks:
    loss = lam * CE(logits, labels_a)  +  (1-lam) * CE(logits, labels_b)

  BCE task (ChestMNIST):
    label_mix = lam * label_a  +  (1-lam) * label_b
    loss = BCE(logits, label_mix)
```

### 9.4 Checkpoint selection

Val AUC is evaluated every epoch.
Best-AUC state is restored before test evaluation.

### 9.5 Early stopping (patience = 20)

If val AUC does not improve for 20 consecutive epochs, retraining halts
early and the best checkpoint is used.  This prevents wasting compute
when a discrete architecture has converged or is degenerate:

```
  retrain_patience   = 20
  retrain_no_improve = 0

  each epoch:
    if val_auc > best_val_auc:
        save best state
        retrain_no_improve = 0
    else:
        retrain_no_improve += 1
        if retrain_no_improve >= retrain_patience:
            break
```

---

## 10. Phase B Architecture Sanity Checks  (train.py)

After discretization (Phase B) and before retraining (Phase C), each
per-task architecture is validated:

```
  For each task k:
    1. Log chosen architecture to console.

    2. Hard block — if "Zero" appears anywhere in the architecture
       (possible when loading a checkpoint from a pre-v2 run):
         -> log ERROR and skip retraining for that task entirely.
         Zero at any layer kills all downstream gradient flow.

    3. Soft warning — if >= L/2 layers are SkipConnect:
         -> log WARNING (degenerate all-identity chain).
         Retraining still proceeds.
```

With Zero removed from the search space (Section 4) this guard acts as a
safety net for old checkpoints loaded via --resume-from.

---

## 11. Discretization  (src/supernet.py :: discretize)

For each task t, each layer l:

    op*(t, l) = argmax_o  best_alphas[t, l, o]

where `best_alphas` is the alpha snapshot saved at the epoch with the
highest mean AUC across all tasks — NOT the final epoch.  This prevents
the discretization from being corrupted by a late-search collapse.

Builds:  DiscreteModel = stem + [ op*(t,0), ..., op*(t,L-1) ] + head_t
Passed directly to retrain_discrete().

---

## 12. Metrics and Reporting

```
  src/metrics.py
  +---------------------------------------------------------+
  | ACC   argmax comparison for single-label tasks          |
  |       threshold=0.5 for multi-label (ChestMNIST)        |
  | AUC   sklearn roc_auc_score; fallback to ACC on error   |
  | Loss  mean task loss over the split                     |
  +---------------------------------------------------------+

  src/reporting.py
  +---------------------------------------------------------+
  | ASCII table   -> results/benchmark_table.txt            |
  | JSON summary  -> results/benchmark_results.json         |
  |   includes: per-task arch string, ACC, AUC, params,     |
  |             macro averages                              |
  +---------------------------------------------------------+
```

---

## 13. Default Hyperparameters

```
+---------------------+----------+----------------------------------+
| Parameter           | Default  | CLI flag                         |
+---------------------+----------+----------------------------------+
| Search epochs       |       50 | --epochs                         |
| Batch size          |       64 | --batch                          |
| Layers L            |        8 | --layers                         |
| Channels C          |       64 | --channels                       |
| Image size          |       64 | --img-size                       |
| LR weights          |    0.025 | --lr_w                           |
| LR alphas           |    3e-4  | --lr_a                           |
| tau_init            |      1.5 | --tau_init                       |
| anneal_factor       |     0.85 | --anneal_factor                  |
| anneal_interval     |       10 | --anneal_interval                |
| tau_min             |      0.1 | --tau_min  (collapse floor)      |
| auc_patience        |       10 | --auc_patience                   |
| rewind_thresh       |     0.10 | --rewind_thresh                  |
| alpha_update_freq   |       10 | --alpha_freq                     |
| entropy_threshold   |     0.05 | --entropy_thresh                 |
| Retrain epochs      |      200 | --retrain_epochs                 |
| Mixup alpha         |      0.2 | --mixup-alpha                    |
| Label smoothing     |      0.1 | --label-smoothing                |
+---------------------+----------+----------------------------------+
```

Recommended settings for a stable 20-epoch search on Kaggle:

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

## 14. System Architecture Diagram

```
  +================================================================+
  |                     MT-DARTS v2 System                         |
  +================================================================+

  [User / CLI]
       |
       |  python main.py <args>
       v
  +--------------------+
  |      main.py       |  argparse  ->  run_search(...)
  +--------------------+
       |
       v
  +--------------------+
  |      train.py      |  run_search()  -- 4-phase orchestrator
  +--------------------+
       |
       +------------------+------------------+------------------+
       |                  |                  |                  |
       v                  v                  v                  v
  +----------+     +------------+     +-----------+     +----------+
  |  src/    |     |   src/     |     |   src/    |     |  src/    |
  |  data.py |     | supernet   |     | controller|     | retrain  |
  |          |     |    .py     |     |    .py    |     |   .py    |
  | build_   |     | TaskAware  |     | Search    |     | retrain_ |
  | data-    |     | Supernet   |     | Controller|     | discrete |
  | loaders  |     |            |     |           |     |          |
  +----------+     +------+-----+     +-----+-----+     +----+-----+
       |                  |                 |                 |
       |           +------+          +------+------+     +----+-----+
       |           |      |          |             |     |          |
       v           v      v          v             v     v          v
  +--------+  +------+  +------+  +------+  +-------+ +------+ +------+
  | mednist|  | ops  |  | norm |  | loss |  | metric| | loss | |metric|
  |  API   |  | .py  |  | .py  |  | .py  |  |  .py  | |  .py | | .py  |
  |        |  |      |  |      |  |      |  |       | |      | |      |
  | 3 tasks|  | 10ops|  |sparse|  |CE /  |  |entropy| |CE /  | |ACC / |
  |        |  | Mixed|  | max  |  |BCE+LS|  |alpha_H| |BCE+LS| | AUC  |
  +--------+  | Op   |  |anneal|  +------+  +-------+ +------+ +------+
              +------+  +------+
                                                    |
                                                    v
                                             +------------+
                                             |   src/     |
                                             | reporting  |
                                             |    .py     |
                                             |            |
                                             |benchmark   |
                                             |_table.txt  |
                                             |_results    |
                                             |  .json     |
                                             +------------+
```

---

## 15. Supernet Internal Dataflow

```
  Input (B, 3, 64, 64)
         |
         v
  +==============================+
  |       Adaptive Stem          |
  |  Conv(3->32, s=2) BN ReLU6   |
  |  Conv(32->64, s=2) BN ReLU6  |
  +==============================+
         | (B, 64, 16, 16)
         |
  alpha_t (L, 10) --sparsemax(./tau)--> pi_t (L, 10)
         |
         +-------------+-- ... --+-----------+
         |             |                     |
         v             v                     v
    +----------+  +----------+         +----------+
    | MixedOp  |  | MixedOp  |  . . .  | MixedOp  |
    |  cell 0  |  |  cell 1  |         | cell L-1 |
    |          |  |          |         |          |
    | sum_o    |  | sum_o    |         | sum_o    |
    | pi[0,o]  |  | pi[1,o]  |         | pi[L,o]  |
    | * op_o   |  | * op_o   |         | * op_o   |
    +----+-----+  +----+-----+         +----+-----+
         |              |                   |
         +--------------+-- ... ------------+
                                            |
                                            v  (B, 64, 16, 16)
                              AdaptiveAvgPool2d(1) -> Flatten
                                            |  (B, 64)
                                            |
                 +-------------+------------+-----------+
                 |             |                        |
                 v             v                        v
          +----------+  +----------+            +----------+
          | Head  t=0|  | Head  t=1|            | Head  t=2|
          | Drop 0.3 |  | Drop 0.3 |            | Drop 0.3 |
          | Lin->256 |  | Lin->256 |            | Lin->256 |
          | ReLU     |  | ReLU     |            | ReLU     |
          | Drop 0.2 |  | Drop 0.2 |            | Drop 0.2 |
          | Lin->9   |  | Lin->14  |            | Lin->7   |
          +----+-----+  +----+-----+            +----+-----+
               |             |                       |
               v             v                       v
         Logits(B,9)   Logits(B,14)           Logits(B,7)
         PathMNIST     ChestMNIST             DermaMNIST
         CE+smooth     BCE                    CE+smooth
```

---

## 16. Bilevel Optimization and Retraining Flow

```
  +==============================================================+
  |                   PHASE A: Bilevel Search                    |
  +==============================================================+

  for epoch e in 1 .. E:
  |
  |   tau_e = max( 1.5 * 0.75^floor(e/5),  0.01 )
  |
  |   for (train_batch, val_batch):
  |   |
  |   |   ---- WEIGHT UPDATE (every step) ----------------------
  |   |   logits = supernet(imgs, task_id, tau=tau_e)
  |   |   loss_w = task_loss(logits, labels, label_smooth=0.1)
  |   |   w  <--  w - eta_w * grad_w(loss_w)     [SGD + cosine]
  |   |
  |   |   ---- ALPHA UPDATE (every 10 steps) -------------------
  |   |   if step % 10 == 0:
  |   |     logits_v = supernet(val_imgs, task_id, tau=tau_e)
  |   |     loss_a  = task_loss(logits_v, val_labels)
  |   |     alpha  <--  alpha - eta_a * grad_a(loss_a)   [Adam]
  |   |
  |   +----------------------------------------------------------
  |
  |   ---- EARLY STOP CHECK ------------------------------------
  |   H = mean entropy of  sparsemax(alpha / tau_e)
  |   if H < 0.05 : break
  |
  +--------------------------------------------------------------

  +==============================================================+
  |                   PHASE B: Discretize                        |
  +==============================================================+

  for task t in {0, 1, 2}:
      for layer l in 0 .. L-1:
          op*(t,l) = argmax_o  alpha[t, l, o]
      discrete_t = stem + [op*(t,0) .. op*(t,L-1)] + head_t

  +==============================================================+
  |                   PHASE C: Retrain                           |
  +==============================================================+

  for task t in {0, 1, 2}:
  |
  |   reinit_weights(discrete_t)
  |                                          warmup  |  cosine
  |   LR schedule:  LinearLR(0.1->1.0) -----+--------+--------->
  |                 (max(5, 200//20) ep)     10ep      190ep
  |
  |   for epoch e in 1 .. 200:
  |   |   lam ~ Beta(0.2, 0.2),  lam = max(lam, 1-lam)
  |   |   x_mix = lam*x_a + (1-lam)*x_b
  |   |   compute mixed loss (CE or BCE per task)
  |   |   clip grads at 5.0
  |   |   eval val AUC every epoch  -->  save best checkpoint
  |   +--
  |
  |   restore best checkpoint
  |   evaluate on test set -> ACC, AUC, loss
  |
  +--------------------------------------------------------------

  +==============================================================+
  |                   PHASE D: Report                            |
  +==============================================================+

  print ASCII table  +  write JSON  -->  results/
```

---

## 17. Module Dependency Map

```
  main.py
     |
     +---> train.py :: run_search
               |
               +---> src/data.py :: build_dataloaders
               |         |
               |         +---> MedMNISTDataset
               |                  (real: medmnist API | mock: rand)
               |
               +---> src/supernet.py :: TaskAwareSupernet
               |         |
               |         +---> src/ops.py :: MixedOp, 10 primitives
               |         +---> src/normalizers.py :: annealed_sparsemax
               |
               +---> src/controller.py :: SearchController
               |         |
               |         +---> src/losses.py   :: task_loss
               |         +---> src/metrics.py  :: alpha_entropy
               |         +---> src/normalizers.py
               |
               +---> src/retrain.py :: retrain_discrete
               |         |
               |         +---> src/losses.py  :: task_loss
               |         +---> src/metrics.py :: evaluate_task
               |
               +---> src/metrics.py   :: evaluate
               +---> src/reporting.py :: print_benchmark / save_benchmark
                              |
                              +---> results/benchmark_table.txt
                              +---> results/benchmark_results.json
```

---

## 18. Reproducibility Checklist

1. Seeds      -- --seed 42 (default).  Applied to numpy, torch, random.
2. Data       -- official MedMNIST train/val/test splits, no leakage.
3. Normaliz.  -- per-task mean/std constants fixed in src/data.py.
4. Compute    -- report device, memory, wall-clock time (search + retrain).
5. Params     -- reported per task in benchmark table (n_params column).
6. Multi-seed -- run 3-5 seeds, report mean +/- std for AUC and ACC.
7. Artifacts  -- checkpoints -> ckpt-dir;  JSON -> save-dir.

---

## 19. Suggested Ablations

1. Sparsemax vs softmax for alpha normalization.
2. Temperature annealing on/off; different anneal factors {0.5, 0.75, 0.9}.
3. Alpha update frequency sweep: {1, 5, 10, 20}.
4. Entropy early stopping on/off.
5. Search space: drop DilatedConv5x5, drop SE branch, add StandaloneSE.
6. Shared vs task-specific alpha parameters.
7. Mixup on/off; label smoothing on/off.
8. img_size: 28 vs 64 vs 128.

---

## 20. Paper Paragraph (Drop-in Draft)

"We propose MT-DARTS v2, a task-aware differentiable NAS framework for
multi-task MedMNIST classification.  The method uses a shared supernet
with task-specific architecture logits over a ten-operator search space
(MBConv3x3, MBConv5x5, MBConvSE, DilatedConv3x3, DilatedConv5x5, SepConv3x3,
SepConv5x5, SkipConnect, AvgPool3x3, MaxPool3x3).  Zero was excluded from
the search space: in a sequential chain architecture a Zero operation kills
all downstream gradient flow during retraining, making it unsuitable for
non-DAG supernets.  Architecture weights are computed via annealed sparsemax,
yielding sparse, progressively sharper operator distributions.  Network
weights and architecture parameters are optimized in a bilevel loop with
delayed architecture updates (every 10 steps) to stabilize search.  Label
smoothing (epsilon=0.1) is applied to CE tasks during both search and
retraining.  Convergence is monitored via mean architecture entropy, enabling
principled early stopping.  After search, each task architecture is
discretized by argmax selection and retrained for up to 200 epochs with
early stopping (patience=20), Mixup (alpha=0.2), and a linear-warmup cosine
schedule, reporting ACC and AUC per task and macro averages."

---

## 21. Known Implementation Notes

1. Mixed-task DataLoader emits (image, label, task_id) tuples; all modules
   filter by task_id before computing losses or metrics.
2. ChestMNIST is loaded with as_rgb=True; grayscale mock images are
   expanded to 3 channels with .repeat(3,1,1).
3. AUC is undefined when only one class is present in a batch; the
   implementation falls back to ACC gracefully.
4. RandAugment requires uint8 input; the transform pipeline does
   float->uint8 before RandAugment and uint8->float32 after.
5. SEBlock uses bias=False Linear layers; _init_weights in both
   supernet.py and retrain.py guard against None bias with explicit checks.
