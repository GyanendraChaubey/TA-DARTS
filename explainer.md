# MT-DARTS v2 — Plain English Explainer

This document explains the project to someone with no machine learning background.
No equations. No jargon. Just analogies and plain English.

---

## What is this project trying to do?

Doctors look at medical images every day — skin lesion photos, chest X-rays,
tissue samples under a microscope — and try to spot diseases.  This project
teaches a computer to do the same thing, automatically, across three different
types of medical image at once.

The three tasks are:

- **PathMNIST**   — classify tissue samples into 9 types of cancer/tissue.
- **ChestMNIST**  — look at chest X-rays and flag up to 14 different conditions
                    at the same time (a patient can have more than one).
- **DermaMNIST**  — classify skin lesion photos into 7 categories.

The twist: instead of a human expert designing the neural network by hand, the
computer **designs its own network** automatically.  That is what "Neural
Architecture Search" (NAS) means.

---

## What is a neural network, briefly?

Think of a neural network like an assembly line in a factory.

Raw material (an image) enters at one end.  It passes through a series of
machines (layers), each of which transforms it slightly.  At the other end
a result pops out: "this looks like melanoma" or "this X-ray shows pneumonia".

Each machine on the line can be one of several types — some good at spotting
fine details, some good at spotting broad patterns, some that just pass the
material through unchanged.  Choosing which machines to put on the line, and
in what order, is the architecture design problem.

---

## The big idea: let the computer design its own assembly line

Normally a human researcher spends weeks trying different assembly-line
configurations.  MT-DARTS automates this.

Here is the trick: instead of committing to one machine at each position, the
system starts with **all possible machines running in parallel** at every spot
on the line.  Each machine gets a "vote weight" — how much its output counts.

During training, the system learns:
1. Which machine at each position actually helps get the right answer.
2. How to classify images correctly.

Both are learned simultaneously.  By the end, the vote weights become very
lopsided: usually one machine gets almost all the votes and the rest get zero.
That is the computer "deciding" what to put at that position.

This process is called **DARTS** (Differentiable Architecture Search).
The "MT" stands for Multi-Task — it runs this process for all three medical
image tasks at the same time, sharing most of the work.

---

## The ten machines (operations) to choose from

At each position on the assembly line, one of ten possible processing steps
can be chosen:

```
+----+----------------+--------------------------------------------------+
| #  | Name           | What it does (plain English)                     |
+----+----------------+--------------------------------------------------+
|  1 | MBConv3x3      | Looks at small 3x3 patches, finds patterns       |
|  2 | MBConv5x5      | Looks at slightly larger 5x5 patches             |
|  3 | MBConvSE       | Like MBConv3x3 but also decides which features   |
|    |                | are most important and amplifies them            |
|  4 | DilatedConv3x3 | Looks at a wider area with gaps in between --    |
|    |                | captures bigger-picture context (moderate range) |
|  5 | DilatedConv5x5 | Same idea but covers an even wider area --       |
|    |                | useful for diffuse patterns across a whole X-ray |
|  6 | SepConv3x3     | An efficient small-patch scanner using fewer     |
|    |                | calculations than a standard 3x3                 |
|  7 | SepConv5x5     | An efficient larger-patch scanner                |
|  8 | SkipConnect    | Passes the image through unchanged (a shortcut)  |
|  9 | AvgPool3x3     | Smooths out each local area by averaging --      |
|    |                | good for diffuse, low-contrast regions           |
| 10 | MaxPool3x3     | Picks the strongest signal in each patch --      |
|    |                | good for sharp edges and lesion boundaries       |
+----+----------------+--------------------------------------------------+
```

The "SE" in MBConvSE stands for Squeeze-and-Excitation — it is a small
attention mechanism that asks "of all the features I found, which ones matter
most for this image?" and dials the rest down.

> **Why no "Zero" (turn the layer off) option?**
> Earlier versions included a Zero option. In theory this sounds useful.
> In practice, when the computer chose Zero for any layer on the sequential
> assembly line, it broke every layer after it too — nothing could flow
> through. The computer worked around this during the search phase (when all
> machines run in parallel), but after committing to the winning architecture
> and retraining from scratch there was no workaround. The result was
> guaranteed random guessing no matter how long retraining ran. AvgPool3x3
> replaces Zero as a cheap low-impact option that does not kill the network.

---

## How the system is structured

Think of the system as having three parts stacked on top of each other.

```
  +-------------------------------------------------------+
  |  Part 1: Shared feature extractor                     |
  |                                                       |
  |  Every image passes through this.  It is shared by   |
  |  all three tasks.  Like a common pre-processing       |
  |  department that all three production lines use.      |
  |                                                       |
  |  Input image (64x64 pixels, colour)                   |
  |     |                                                 |
  |     v                                                 |
  |  [Stem] -- shrinks the image from 64x64 to 16x16     |
  |     |      while deepening the feature description    |
  |     v                                                 |
  |  [8 searchable layers] -- each picks one of 7 ops    |
  +-------------------------------------------------------+
          |           |           |
          v           v           v
  +----------+  +----------+  +----------+
  |  Part 2  |  |  Part 2  |  |  Part 2  |
  | Head for |  | Head for |  | Head for |
  | PathMNIST|  | ChestMNI |  | DermaMNI |
  |  (9 out) |  | (14 out) |  |  (7 out) |
  +----------+  +----------+  +----------+
       |              |             |
       v              v             v
  "tissue type"  "14 disease   "skin lesion
   1 of 9        yes/no flags"  type 1 of 7"

  Part 3: separate answer for each task
```

The "Head" is the final decision-making step.  It takes the shared features
and converts them into task-specific answers.

---

## Four stages of the experiment

The whole training run has four phases, done in order.

### Stage 1 — Search (the competition)

The system runs for up to 50 rounds (epochs).  In each round it sees thousands of
medical images and does two things alternately:

- **Update the image-reading skill** (the machine weights) — make the network
  better at reading images given the current assembly-line choices.
- **Update the assembly-line votes** (the architecture weights) — change which
  machine gets more/less vote weight at each position, based on which
  configuration gave better answers.

By the end, the vote weights converge: at each position, usually one machine
has nearly all the votes.

Throughout the search, the system silently records which set of vote weights
gave the best average accuracy across all three tasks.  This "best snapshot"
is what is used in Stage 2 — not the final epoch's weights, which may have
drifted if training ran too long.

There is also a temperature control on the vote weights.  Early in training
the temperature is high, so votes are spread out and many machines contribute.
Over time the temperature falls, making the votes sharper.  A minimum
temperature floor prevents votes from becoming so extreme that the gradient
signal breaks down (a failure mode called sparsemax collapse).

The three tasks are very unequal in size — the tissue-sample task has about
89,000 training images while the skin-lesion task has only 7,000.  Without
correction, the large task would dominate the learning and the small task
would barely influence the shared network.  To prevent this, the system
normalises each task's "teaching signal" to the same size before combining
them.  Think of it as giving every subject teacher the same number of votes
in a curriculum meeting, regardless of class size.

There is a clever stopping rule: the system measures how "decided" the votes
are.  Once the votes are decisive enough (low entropy), it stops early — no
point running longer when the choice is already clear.  There is also a
patience rule: if the best combined accuracy across all three tasks has not
improved for 10 rounds in a row, training stops.  And if accuracy suddenly
drops sharply, the vote weights are automatically rolled back to the last
good snapshot (the "rewind" safety net).

### Stage 2 — Commit (pick the winner and verify it)

At each of the 8 positions, simply pick the machine that got the most votes
in the best-snapshot recorded during Stage 1.
Now we have a fixed, lean assembly line for each of the three tasks.

```
  Example result for one task:
  Layer 0 -> MBConv5x5
  Layer 1 -> SkipConnect
  Layer 2 -> MBConvSE
  ...
```

Before moving on, the system checks each chosen assembly line for obvious
problems.  If a line somehow contains a broken step (which can happen if
an old saved file from a previous version is loaded), it logs a clear error
message and skips retraining for that task rather than wasting hours training
a broken network.  It also warns if too many steps are plain pass-throughs
(SkipConnect), which would mean the line is barely doing any work.

### Stage 3 — Retrain (train the winner from scratch)

The voted-on architecture is now trained from the beginning, with all weights
reset to zero-knowledge.  This is important: during the search the weights were
shared across all possible machines, which makes them slightly sub-optimal.
Retraining from scratch gives the final architecture its full performance.

This retraining uses a few extra tricks to improve accuracy:
- **Mixup** — two images are blended together (like a double-exposure photo)
  and the network has to predict a blend of both answers.  This stops the
  network from being overconfident.
- **Label smoothing** — instead of telling the network "this is 100% melanoma",
  we say "this is 90% melanoma and a little bit of everything else."  Again,
  prevents overconfidence.
- **Warmup then cosine cooldown** — the learning speed starts slow, ramps up,
  then gradually slows down to a gentle finish.  Like a car accelerating
  smoothly and braking gradually.
- **Early stopping** — if the network’s accuracy on held-out images stops
  improving for 20 rounds in a row, training stops automatically.  This saves
  several hours of wasted compute if the architecture has already converged
  (or is broken — a healthy network shows clear improvement well within 20
  rounds).

### Stage 4 — Report

The final models are tested on images they have never seen before.  Results
are saved as a table and a JSON file showing accuracy and AUC for each task.

AUC (Area Under the Curve) is a standard medical-AI metric.  A value of 1.0
means perfect; 0.5 means random guessing.

---

## What the code files do

```
  +------------------+------------------------------------------------+
  | File             | Role                                           |
  +------------------+------------------------------------------------+
  | main.py          | Front door. Run this to start everything.      |
  |                  | Handles command-line settings (how long to      |
  |                  | train, which GPU to use, etc.)                 |
  +------------------+------------------------------------------------+
  | train.py         | The manager. Runs Stage 1 -> 2 -> 3 -> 4 in   |
  |                  | order and coordinates all other files.          |
  +------------------+------------------------------------------------+
  | src/data.py      | The data loader. Fetches the medical images,   |
  |                  | mixes the three datasets together, applies      |
  |                  | random flips, colour shifts, and other tricks  |
  |                  | to make the network more robust.               |
  +------------------+------------------------------------------------+
  | src/ops.py       | Defines the 10 candidate machines and how       |
  |                  | MixedOp runs all of them in parallel during     |
  |                  | the search phase.  Zero was removed — in a     |
  |                  | sequential line it breaks every layer after it. |
  +------------------+------------------------------------------------+
  | src/supernet.py  | Builds the full assembly line (stem + 8 layers |
  |                  | + 3 heads) and manages the vote weights.        |
  +------------------+------------------------------------------------+
  | src/controller.py| The search engine. Runs Stage 1: alternates    |
  |                  | between updating image-reading skill and        |
  |                  | updating vote weights.                         |
  +------------------+------------------------------------------------+
  | src/losses.py    | Decides how to measure mistakes:               |
  |                  | - CrossEntropy for PathMNIST and DermaMNIST    |
  |                  | - Binary cross-entropy for ChestMNIST          |
  |                  |   (because multiple diseases can be present)   |
  +------------------+------------------------------------------------+
  | src/retrain.py   | Runs Stage 3: trains the winning architecture  |
  |                  | from scratch with Mixup and label smoothing.   |
  +------------------+------------------------------------------------+
  | src/metrics.py   | Measures performance: accuracy, AUC, and the  |
  |                  | "how decided are the votes?" entropy score.    |
  +------------------+------------------------------------------------+
  | src/normalizers.py| The math behind converting raw vote numbers   |
  |                  | into proper weights that sum correctly.         |
  |                  | Uses "sparsemax" which can produce exact zeros  |
  |                  | (unlike softmax which always gives some small  |
  |                  | weight to every option).                       |
  +------------------+------------------------------------------------+
  | src/reporting.py | Saves results to a text table and JSON file.  |
  +------------------+------------------------------------------------+
```

---

## Why sparsemax instead of the more common softmax?

When converting vote numbers into weights, the standard approach (softmax)
always gives every option a small non-zero weight, even terrible ones.  During
search this means every machine is always running a little bit, wasting compute.

Sparsemax can output exact zeros — it completely ignores the bad options.
This makes the search cleaner and the final committed architecture easier to
predict from the vote weights.

---

## Why share most of the network across tasks?

Medical images from different modalities (skin, X-ray, tissue) share
low-level features — edges, textures, colour gradients.  By sharing the
stem and searchable layers, the network builds these general skills once
and reuses them, rather than learning them three times.

Only the final classification head is task-specific, because what
"tissue type" means is completely different from what "skin lesion type" means.

---

## What makes this better than just training three separate networks?

1. **Efficiency** — one search run finds three architectures simultaneously.
2. **Shared knowledge** — low-level features learned for PathMNIST help
   DermaMNIST and ChestMNIST too.
3. **Architecture quality** — DARTS finds compact, well-suited architectures
   automatically rather than relying on a human's intuition.
4. **Modern tricks** — Mixup, label smoothing, SE attention, RandAugment,
   and warmup scheduling all push accuracy higher.

---

## How to run it

Smoke test (fast, fake data, checks nothing crashes):

    python main.py --no-real --epochs 2 --batch 32 --device cpu \
                   --img-size 64 --channels 32 --layers 4 --retrain_epochs 3 \
                   --tau_min 0.25

Recommended run on a GPU (Kaggle, stable 20-epoch search):

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

Results appear in:
- results/benchmark_table.txt  (human-readable)
- results/benchmark_results.json  (machine-readable, for further analysis)
- results/discrete_*_best.pt  (the trained model files)
