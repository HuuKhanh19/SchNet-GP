# CONAN-SchNet (SchNet-GP)

> **Interpretable molecular property prediction** — a frozen SchNet encoder followed by an
> energy-routed mixture of *symbolic* (EvoGP) experts with Boltzmann conformer aggregation.

CONAN-SchNet is a **two-stage** pipeline. Stage 1 trains a standard SchNet graph neural
network end-to-end on 3D conformers. Stage 2 **freezes** that encoder and builds an
*interpretable* read-out head: a delta-learning descriptor baseline, an energy gate that
routes conformers to experts, a mixture of **Multiple-Feature-Construction (MFC)** experts
whose features are evolved by **genetic programming**, and a learned Boltzmann aggregation
over conformers. The goal is to replace an opaque MLP head with **closed-form, inspectable
symbolic features** while still exploiting the 3D conformer ensemble and conformer energies.

Supported regression datasets: **ESOL**, **FreeSolv**, **Lipophilicity**.
(BACE / classification scaffolding exists, but Step 2 is currently regression-only.)

---

## Table of contents

1. [How it works](#1-how-it-works)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Configuration (Hydra)](#5-configuration-hydra)
6. [Usage](#6-usage)
7. [The Step 2 pipeline in detail](#7-the-step-2-pipeline-in-detail)
8. [Outputs](#8-outputs)
9. [Tuning guide](#9-tuning-guide)
10. [Reproducibility & determinism](#10-reproducibility--determinism)
11. [Troubleshooting / known issues](#11-troubleshooting--known-issues)
12. [Interpreting results & fair comparison](#12-interpreting-results--fair-comparison)
13. [`cf_expressions.txt` format](#13-cf_expressionstxt-format)

---

## 1. How it works

### Step 1 — SchNet baseline (representation learner)
A SchNet GNN is trained end-to-end on multi-conformer 3D molecules. The model uses a
**hierarchical read-out**: atom embeddings → per-conformer pooling → per-molecule prediction
(uniform aggregation over conformers). The best checkpoint (selected on the validation set)
is saved and later reused by Step 2.

### Step 2 — frozen encoder + interpretable energy-MoE head
Step 2 loads Step 1's frozen encoder and runs six phases:

| Phase | What it does |
|-------|--------------|
| **0 — Embeddings** | Forward the frozen encoder over every conformer; mean-pool atoms → one 128-d vector per conformer. Standardize (z-score) on train. |
| **A — Delta baseline** | `RidgeCV` on RDKit 2D (or Morgan) descriptors predicts a molecule-level `y_base`. The experts only learn the residual `Δ = y − y_base`. |
| **B — Energy gate** | Split conformers into energy bins by quantiles of the relative conformer energy `dE`; route each conformer to one expert. |
| **1 — MFC experts** | For each bin, evolve a population of single-output expression trees (EvoGP) to maximize \|Pearson corr\| with `Δ`, keep `q` mutually-decorrelated features, fit a ridge combiner. Fallback to plain ridge if a bin is too small or GP fails. |
| **2 — Aggregation τ** | Freeze experts; learn a single scalar temperature `τ` for Boltzmann weights `softmax(−dE/τ)` over conformers (Adam). |
| **EVAL** | Aggregate per-conformer `Δ̂` → `ŷ = y_base + Δ̂` per molecule → RMSE / MAE on train / valid / test. |

**Why this design:** the GP head produces actual math formulas over the learned embedding
dimensions (see [§13](#13-cf_expressionstxt-format)), so the read-out is inspectable and fast
at inference (no GP needed once the trees are fixed). The energy gate + Boltzmann aggregation
inject conformer-energy physics that Step 1's uniform read-out ignores.

> **Important framing.** Step 2's final number is `y_base + Δ̂`. When `delta_learning=true`, a
> large share of the accuracy can come from the **descriptor baseline**, not the GP head. The
> fair test of the *head's* contribution is `delta_learning=false` (experts predict `y`
> directly from SchNet embeddings) vs Step 1, on the **same split**. See [§12](#12-interpreting-results--fair-comparison).

---

## 2. Repository structure

```
SchNet-GP/
├── configs/
│   └── base.yaml                  # Hydra config: datasets, split, conformer, schnet, training (+ optional step2)
├── data/
│   ├── raw/                       # input CSVs: refined_ESOL.csv, refined_FreeSolv.csv, refined_Lipophilicity.csv, refined_BACE.csv
│   └── processed/                 # auto-generated cache
│       └── {dataset}/seed_{k}/
│           ├── train.csv valid.csv test.csv
│           └── {N}_conformers/{train,valid,test}.pkl   # cached 3D conformers
├── experiments/                   # auto-generated outputs
│   ├── step1/{dataset}/seed_{k}/{timestamp}/   # best_model.pt, results.json
│   └── step2/{dataset}/
│       ├── seed_{k}/{timestamp}/   # results.json, conan_head.pt, cf_expressions.txt, history.json
│       └── multiseed_summary.json  # mean ± std across seeds (multi-seed runs)
├── scripts/
│   ├── preprocess_data.py         # raw CSV -> split -> data/processed
│   ├── run_step1.py               # train the SchNet baseline (Step 1)
│   └── run_step2.py               # train the interpretable head (Step 2)
└── src/
    ├── data/
    │   ├── data_loader.py         # prepare_dataset, create_dataloaders, SchNetMolDataset, collate_multi_conformer
    │   ├── splitter.py            # random_split / random_scaffold_split (Murcko)
    │   └── conformer.py           # RDKit ETKDG 3D conformers + MMFF optimization + relative energies (dE)
    ├── models/
    │   ├── schnet.py              # SchNet GNN (encoder + hierarchical atom→conf→mol read-out)
    │   ├── embedding_extractor.py # resolve_checkpoint, load_frozen_encoder, extract_conf_embeddings
    │   ├── delta_baseline.py      # DeltaBaseline: RDKit2D/Morgan descriptors + RidgeCV (y_base)
    │   ├── energy_gate.py         # EnergyGate: quantile energy bins + routing
    │   ├── mfc_expert.py          # MFCExpert: EvoGP symbolic features + ridge combiner
    │   └── conan_head.py          # ConanHead, predict_per_conf_delta, aggregate
    ├── trainers/
    │   ├── step1_trainer.py       # Step-1 training loop (early stopping, LR schedule)
    │   └── step2_trainer.py       # Step-2 six-phase pipeline (PHASE 0/A/B/1/2/EVAL)
    └── utils/
        ├── utils.py               # seed_everything, set_determinism
        ├── standardize.py         # Standardizer (train-fit z-score)
        └── scatter.py             # scatter_add, scatter_softmax
```

### Call order when you run `run_step2.py`

```
run_step2.py (main)
  └─ configs/base.yaml                      # config
  └─ utils.py: seed_everything/set_determinism
  └─ data_loader.py: prepare_dataset        # → splitter.py (scaffold/random split)
  └─ data_loader.py: create_dataloaders     # → SchNetMolDataset → conformer.py (3D + dE)
  └─ embedding_extractor.py: resolve_checkpoint / load_frozen_encoder  # → schnet.py
  └─ step2_trainer.py: Step2Trainer.train
        PHASE 0  embedding_extractor.extract_conf_embeddings ; standardize.py
        PHASE A  delta_baseline.py (RDKit/Morgan + RidgeCV)
        PHASE B  energy_gate.py (quantile bins)
        PHASE 1  mfc_expert.py (EvoGP) ; conan_head.predict_per_conf_delta
        PHASE 2  scatter.py (scatter_add/scatter_softmax)
        EVAL     conan_head.ConanHead.predict / aggregate
        _save    cf_expressions.txt, conan_head.pt, history.json, results.json
```

---

## 3. Installation

A CUDA GPU is **required** for Step 2 (EvoGP runs custom CUDA kernels).

```bash
conda create -n conan_es python=3.10
conda activate conan_es

# core
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install hydra-core omegaconf pandas numpy scikit-learn

# chemistry
pip install rdkit            # or: conda install -c conda-forge rdkit

# genetic programming (GPU) — install per upstream; it compiles CUDA kernels
# and needs a CUDA toolkit matching your PyTorch build.
#   see the EvoGP project for exact instructions
```

> **Notes**
> - Pin the PyTorch CUDA build to your driver/toolkit. EvoGP's kernels must match it.
> - For reproducible cuBLAS, export `CUBLAS_WORKSPACE_CONFIG=":4096:8"` before the first GPU call
>   (the scripts also `setdefault` this). See [§10](#10-reproducibility--determinism).
> - To reduce VRAM fragmentation: `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.

---

## 4. Data

Put the raw CSVs in `data/raw/` (filenames are configured per dataset in `base.yaml`):

| Dataset | File | SMILES col | Target col | Task |
|---------|------|-----------|-----------|------|
| esol | `refined_ESOL.csv` | `smiles` | `measured` | regression (logS) |
| freesolv | `refined_FreeSolv.csv` | `smiles` | `measured` | regression (kcal/mol) |
| lipo | `refined_Lipophilicity.csv` | `smiles` | `measured` | regression (logD) |
| bace | `refined_BACE.csv` | `smiles` | `class` | classification (AUC) |

`scripts/preprocess_data.py` reads a CSV, detects/uses the SMILES & target columns,
preprocesses, splits into train/valid/test, and writes `data/processed/{dataset}/seed_{k}/`.
3D conformers are generated lazily on first use and cached under `{N}_conformers/*.pkl`.

The default split is **`random_scaffold`** (Bemis–Murcko scaffold split): train/valid/test
contain *disjoint scaffolds*. This is a hard, realistic generalization test — expect higher
error and noticeable seed-to-seed variance compared to a random split.

---

## 5. Configuration (Hydra)

All configuration lives in `configs/base.yaml`. Key sections (defaults shown):

```yaml
dataset_name: esol
random_seed_train: 0          # seeds model/GP training (fixed across split seeds)
gpu: 1                        # GPU index; <0 forces CPU

data:
  split_ratio: [0.8, 0.1, 0.1]
  split_method: "random_scaffold"   # random_scaffold | random
  random_seed_split: 3              # which split to use (overridable)

conformer:
  num_conformers: 10
  optimize_mmff: true
  random_seed_gen: 42

schnet:
  n_atom_basis: 128
  n_interactions: 6
  n_rbf: 50
  cutoff: 5.0
  n_filters: 128

training:                      # Step 1
  epochs: 300
  batch_size: 32
  learning_rate: 0.001
  scheduler: reduce_on_plateau
  early_stopping_patience: 100
  gradient_clip: 1.0
```

### Step 2 parameters

Step 2 has its own block of defaults. **By default these live in `DEFAULT_STEP2` inside
`scripts/run_step2.py`** (not in `base.yaml`). Because Hydra runs CLI overrides against the
*yaml schema*, overriding `step2.*` from the CLI when `base.yaml` has no `step2:` block fails
with `Key 'step2' is not in struct`. You have two options:

- **Quick:** prefix the override with `+` (Hydra creates the key), e.g.
  `+step2.expert.log_gp_every=10`.
- **Permanent (recommended):** paste the full `step2:` block (and `split_seeds: null`) into
  `configs/base.yaml`. Then every `step2.*` override works *without* `+`.

The Step-2 defaults:

```yaml
split_seeds: null              # CLI: split_seeds=[0,1,2,3,4] -> multi-seed run + averaging

step2:
  encoder_checkpoint: auto     # auto = newest Step-1 best_model.pt for this dataset/seed
  emb_pool: mean               # mean | add  (atom -> conformer pooling)
  standardize_emb: true
  delta_learning: true         # true: experts learn residual; false: experts predict y directly
  descriptors: rdkit2d         # rdkit2d | morgan | none
  gate:
    num_experts: 3             # number of energy bins / experts
    binning: quantile
    energy_clip: train_max
    min_conf_per_bin: 100      # bins smaller than this fall back to ridge (no GP)
  expert:
    q: 6                       # final # of constructed features (after decorrelation)
    n_best: 64                 # top-N by |corr| before decorrelation
    pop_size: 1000             # GP population
    generation_limit: 100      # GP generations
    max_tree_len: 128          # max nodes per tree (storage / memory)
    max_layer_cnt: 5           # max tree depth at generation (expressivity & memory)
    mutation_max_layer_cnt: 3
    mutation_rate: 0.2
    survival_rate: 0.3
    elite_rate: 0.01
    using_funcs: ['+', '-', '*', '/']
    const_range: [-3.0, 3.0]   # matches standardized embeddings (~±3σ)
    ridge_alphas: [0.001, 0.01, 0.1, 1.0, 10.0]
    gp_seed: 0
    log_gp_every: 0            # 0 = off; N = print GP convergence every N generations
    gp_max_samples: 4000       # 0 = use all; >0 = subsample per bin to bound GPU memory
  agg: learned_softmax         # mean | boltzmann | learned_softmax
  tau_init: 0.593              # RT (kcal/mol)
  tau_steps: 300               # Phase-2 Adam steps
  tau_log_every: 50            # log train/val RMSE every N tau steps
```

> `gp_max_samples`, `log_gp_every`, `tau_steps`, `tau_log_every`, and `set_determinism` are
> optional additions; see [§9](#9-tuning-guide)–[§11](#11-troubleshooting--known-issues).

---

## 6. Usage

### 0) Preprocess (optional — splits are also created on first train)
```bash
python scripts/preprocess_data.py dataset_name=esol      # or dataset_name=all
```

### 1) Train Step 1 (required before Step 2 — produces the frozen encoder)
```bash
python scripts/run_step1.py dataset_name=esol
# for multiple splits, train each seed so Step 2 can find the matching checkpoint:
python scripts/run_step1.py dataset_name=lipo data.random_seed_split=0
python scripts/run_step1.py dataset_name=lipo data.random_seed_split=1
# ... seeds 2,3,4
```

### 2) Train Step 2
```bash
# single split (uses data.random_seed_split from config)
python scripts/run_step2.py dataset_name=esol

# multi-seed in ONE command -> mean ± std (needs a Step-1 checkpoint per seed)
python scripts/run_step2.py dataset_name=esol +split_seeds=[0,1,2,3,4]
```

### Common overrides
```bash
# per-generation GP logging (PHASE 1)
python scripts/run_step2.py dataset_name=esol +step2.expert.log_gp_every=10

# bound GPU memory on large datasets (see Troubleshooting)
python scripts/run_step2.py dataset_name=lipo +split_seeds=[2] \
    +step2.expert.gp_max_samples=3000 +step2.expert.max_layer_cnt=3

# ABLATIONS (for the fair comparison vs Step 1)
python scripts/run_step2.py dataset_name=lipo +step2.delta_learning=false   # head only, no descriptors
python scripts/run_step2.py dataset_name=lipo +step2.agg=mean               # uniform conformer aggregation
python scripts/run_step2.py dataset_name=lipo +step2.descriptors=morgan     # swap descriptor set
```

> If you added the `step2:` block to `base.yaml`, drop the leading `+` from `step2.*` overrides.
> `split_seeds` still needs `+` unless you add `split_seeds: null` to the yaml.

---

## 7. The Step 2 pipeline in detail

**PHASE 0 — embeddings.** Forward the frozen encoder over all conformers (`shuffle=False`, so
`conf→mol` alignment is preserved). Atom embeddings are pooled (`emb_pool`) to one 128-d vector
per conformer. A `Standardizer` (z-score) is fit on **train** embeddings and applied everywhere.

**PHASE A — delta baseline.** `DeltaBaseline` featurizes molecules with RDKit 2D descriptors
(or Morgan) and fits `RidgeCV` on **train** to predict `y_base` at the molecule level. The
per-conformer residual `Δ = (y − y_base)[conf2mol]` is the target for the experts. With
`delta_learning=false`, `y_base = 0` and the experts predict `y` directly.

**PHASE B — energy gate.** `EnergyGate` computes quantile boundaries on the **train** relative
energies `dE` and routes each conformer into one bin. Each bin gets its own expert
(specialization by energy regime).

**PHASE 1 — MFC experts (EvoGP).** Per bin:
1. Randomly initialize a population of `pop_size` single-output expression trees
   (operators `+ − × ÷`, constants in `const_range`).
2. Evolve for `generation_limit` generations; fitness = \|Pearson corr(tree output, Δ)\| on the
   bin's **train** conformers (selection / crossover / mutation each generation).
3. Rank the final population by \|corr\|, keep the top `n_best`, then greedily drop the most
   mutually-correlated until `q` remain (informative **and** complementary features).
4. Fit a `RidgeCV` combiner mapping the `q` features → `Δ`.
   *Fallback:* if the bin has fewer than `min_conf_per_bin` samples, or GP errors out, fit
   ridge directly on the 128-d embedding (`mode=ridge`).

**PHASE 2 — aggregation τ.** With experts frozen, optimize a single scalar `τ` (Adam,
`tau_steps`) for Boltzmann weights `w = softmax(−dE/τ)` over a molecule's conformers, minimizing
train MSE of `y_base + Σ w·Δ̂`. (`agg=mean` → uniform; `agg=boltzmann` → fixed `τ = tau_init`.)

**EVAL.** Aggregate per-conformer `Δ̂` to per-molecule, add `y_base`, report RMSE/MAE on
train/valid/test, plus a decomposition line (see below).

---

## 8. Outputs

Each Step-2 run writes to `experiments/step2/{dataset}/seed_{k}/{timestamp}/`:

| File | Contents |
|------|----------|
| `results.json` | config, `tau`, per-split metrics, expert modes, full `history` |
| `conan_head.pt` | the full head (`db`, standardizer, gate, experts, `tau`) — loadable where EvoGP is installed |
| `cf_expressions.txt` | sympy strings of the `q` constructed features per expert (see [§13](#13-cf_expressionstxt-format)) |
| `history.json` | per-expert fit quality, the post-PHASE-1 checkpoint, the per-step τ trace, and the final eval |

Multi-seed runs also write `experiments/step2/{dataset}/multiseed_summary.json` and print a
mean ± std table.

### Reading the console
```
[decomp] train-mean floor = 1.0983  | 2D-baseline-alone = 0.7865  | full Step2 = 0.7495
```
- **train-mean floor** — RMSE of predicting the train mean on test (naive baseline). A useful
  model must be well below this.
- **2D-baseline-alone** — `y_base` (descriptors only). If this ≈ full Step2, the GP head added
  little and the result is driven by descriptors.
- **full Step2** — `y_base + Δ̂`.

Per-generation GP log (`log_gp_every>0`):
```
gen  10 | best|corr|=0.58 | mean|corr|=0.12 | alive=971/1000
```
`alive` = individuals that aren't degenerate (constant output); if it collapses toward 0, the
population is degenerating (raise `elite_rate` / lower `mutation_rate`).

---

## 9. Tuning guide

### Memory / OOM (Step 2, PHASE 1)
GP peak VRAM scales roughly as:
```
peak ≈ k · pop_size · (tree size, bounded by max_layer_cnt / max_tree_len) · n_samples_per_bin · 4 bytes   (k ≈ 2–3)
```
Levers, best first:

| Knob | Effect | Cost |
|------|--------|------|
| `gp_max_samples` | subsample conformers per bin; memory ∝ n_samples | almost none (a few thousand samples estimate \|corr\| tightly) |
| `max_layer_cnt` | tree depth ⇒ size ~2^depth | shorter, more interpretable formulas; less expressivity |
| `max_tree_len` | node-storage cap | similar |
| `pop_size` | population size | reduces search breadth (quality) |
| `gate.num_experts` | more bins ⇒ smaller bins | modeling change; tiny bins fall back to ridge |

**Recommended for Lipophilicity:** `gp_max_samples=3000`, `max_layer_cnt=3`, keep `pop_size=1000`.

### GP quality (after OOM is fixed)
- `generation_limit` — raise to 150–200 if the per-gen `best|corr|` is still climbing (no extra peak memory).
- `mutation_rate` (0.1–0.4), `survival_rate` (0.2–0.5), `elite_rate` (0.01–0.05) — exploration vs exploitation.
- `q` — features tend to be redundant; 3–4 often loses nothing vs 6.
- `using_funcs` — adding `sin/cos/exp/log` increases expressivity but risks overflow/NaN.

### Modeling (affects the result, not GP)
`delta_learning` (the fair-comparison ablation), `descriptors`, `agg`, `tau_init`, `gate.num_experts`.

### Leave alone
`standardize_emb` (keep `true`; `const_range` is matched to it), `const_prob/out_prob/layer_leaf_prob/sample_cnt`,
`ridge_alphas` (CV handles it). Vary `gp_seed` only to report mean ± std.

---

## 10. Reproducibility & determinism

Step 2's main source of run-to-run variation is **EvoGP** (random init + stochastic operators
on GPU). The discrete top-`n_best` selection amplifies tiny float differences into different
feature sets. To pin everything PyTorch can pin, add `set_determinism(seed)` to `utils.py` and
call it first in `run_step2.py`'s `main`:

```python
def set_determinism(seed: int, strict: bool = False) -> None:
    import os, random
    import numpy as np, torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except Exception as e:
        print(f"  [determinism] use_deterministic_algorithms unavailable: {e!r}")
```

Caveat: EvoGP's CUDA kernels may keep small residual variance regardless. If exact
reproducibility matters, run GP on CPU (slow) or — the recommended scientific approach —
report **mean ± std over seeds** (`+split_seeds=[0,1,2,3,4]`). A difference only in the last
2–3 decimals is harmless float-ordering noise; a large difference means GP landed in a
different basin.

---

## 11. Troubleshooting / known issues

**`Could not override 'step2.…'. Key 'step2' is not in struct`.**
`base.yaml` has no `step2:` block. Prefix the override with `+` (e.g.
`+step2.expert.log_gp_every=10`), or paste the `step2:` block into `base.yaml` (see [§5](#5-configuration-hydra)).

**`CUDA out of memory` in PHASE 1.** The bin × population × tree-depth evaluation buffer is too
big (common on Lipophilicity, ~11k conformers/bin). Fix with the memory knobs in [§9](#9-tuning-guide):
`+step2.expert.gp_max_samples=3000 +step2.expert.max_layer_cnt=3`, and
`export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`. Verify the log shows `mode=gp`
(not `ridge fallback`) afterward — otherwise GP still didn't run. Note: descriptor choice
(`rdkit2d` vs `morgan`) does **not** affect this; the GP runs on SchNet embeddings.

**Test RMSE explodes to millions (e.g. `Test RMSE=3405407`).** An *unbounded* RDKit descriptor
(notoriously `Ipc`) takes an extreme value on a test molecule out of the train range; the
train-fit `StandardScaler` maps it to a gigantic z-score (which `nan_to_num` does **not** catch
— it is finite, just huge), and a weakly-regularized ridge multiplies it into a huge
prediction. The tell-tale sign is `RMSE/MAE ≈ √(n_test)` (one outlier dominates). **Fix:** clip
standardized features in `DeltaBaseline.fit` **and** `.predict`:
```python
Xs = self.scaler.transform(X)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)   # predict() only
Xs = np.clip(Xs, -10.0, 10.0)                              # <-- add in BOTH fit and predict
```
Optionally drop `Ipc` or use `Descriptors.Ipc(mol, avg=True)`.

**Baseline worse than the train-mean floor.** A linear `RidgeCV` cannot be worse than the floor
by being "too simple" (large `alpha` → intercept ≈ mean ≈ floor). Worse-than-floor means
**overfitting** (check the printed `alpha`; widen `ridge_alphas` upward for Morgan's 2048 dims),
**distribution shift** (scaffold split → test scaffolds are unseen; Morgan suffers most,
`rdkit2d` degrades more gracefully), or a scaling bug. It is *not* a reason to use a more
complex baseline model.

**Results differ run-to-run.** See [§10](#10-reproducibility--determinism).

---

## 12. Interpreting results & fair comparison

Step 2 (`delta_learning=true`) = `descriptor baseline` + `GP/SchNet head`. The descriptor
baseline is an extra feature source Step 1 never sees, and for ESOL/Lipo/FreeSolv it is a
strong predictor on its own. Therefore **"Step 2 (full) vs Step 1" is not a clean test of the
GP head** — it conflates "descriptors help" with "the head helps". The same change can flip the
outcome entirely (e.g., on Lipophilicity, a Morgan baseline can drag Step 2 *below* Step 1).

To attribute contributions, run on the **same split(s)** and compare:

| Configuration | Isolates |
|---------------|----------|
| Step 1 (SchNet) | baseline representation + readout |
| `delta_learning=true`, experts off (≈ `2D-baseline-alone`) | descriptors only |
| `delta_learning=false` | **head only** (SchNet embeddings + GP + energy aggregation) |
| `delta_learning=true` (full) | descriptors + head |

The honest "did the head help?" comparison is **`delta_learning=false` vs Step 1**. Report
mean ± std across the 5 seeds; a single scaffold split is not conclusive.

---

## 13. `cf_expressions.txt` format

Each block is one expert (one energy bin); `CF0…CF{q-1}` are its constructed features:

```
=== Expert 0 (mode=gp) ===
  CF0: x105 - x58 + x6 - x70 - x87 + 7.376 - (...)/(x120 + 2.640)
  ...
```

- **Variables `x0…x127` are the 128 standardized SchNet embedding dimensions** — *not* named
  chemical descriptors. (`export_expressions()` is called without `symbol_names`, so sympy uses
  index names.)
- Each `CFk` is one single-output tree, simplified by sympy; operators are limited to `+ − × ÷`.
- At inference, plug a conformer's standardized 128-d embedding into the `q` formulas → `q`
  numbers → the expert's ridge combines them → `Δ̂`. **The ridge weights are not in this file**
  — they live in `conan_head.pt`. This file is the feature *definitions* only.
- Near-duplicate `CFk` within an expert indicate GP population convergence; the effective number
  of independent features is often `< q` (a hint that `q` can be lowered).

---

*CONAN-SchNet — frozen SchNet + energy-routed mixture of EvoGP experts. Regression: ESOL, FreeSolv, Lipophilicity.*