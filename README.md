# litfit

**litfit** /lɪt fɪt/ — the shortest path from someone else's embedding to your task.

## Installation

```bash
pip install -e .
```

For faster statistics computation on CUDA GPUs:

```bash
pip install triton
```

## Usage

```python
from litfit import (
    load_askubuntu, encode_texts, split_data,
    compute_stats, generate_all_projections, evaluate_projections,
)

all_ids, all_texts, id_to_group = load_askubuntu(max_groups=1000)
embs = encode_texts("intfloat/e5-base-v2", all_texts)
data = split_data(all_ids, all_texts, embs, id_to_group)
train_ids, _, train_embs, _ = data["train"]
val_ids, _, val_embs, _ = data["val"]
test_ids, _, test_embs, _ = data["test"]

st = compute_stats(train_embs, train_ids, id_to_group)
all_W = generate_all_projections(st, neg=None, include_neg_methods=False)
results, summary = evaluate_projections(
    all_W, val_embs, val_ids, id_to_group,
    test_embs=test_embs, test_ids=test_ids,
    dim_fractions=(0.1, 0.2, 0.5, 1.0),
)
```

### Streaming + fast + dim search (low memory)

Combine streaming statistics, fast projections (~40 configs), lazy evaluation, and
automatic dimension search for a memory-efficient pipeline:

```python
from litfit import (
    compute_stats_streaming, generate_fast_projections,
    find_dim_range, evaluate_projections,
)

def pair_batches():
    for i in range(0, len(X_pairs_memmap), 1024):
        yield X_pairs_memmap[i:i+1024], Y_pairs_memmap[i:i+1024]

st = compute_stats_streaming(pair_batches())
dim_fractions = find_dim_range(st, val_embs, val_ids, id_to_group)
all_W = generate_fast_projections(st, lazy=True)
results, summary = evaluate_projections(
    all_W, test_embs, test_ids, id_to_group,
    dim_fractions=dim_fractions,
)
```

<details>
<summary><strong>Full walkthrough: data concepts, splitting, extracting the best projection</strong></summary>

litfit operates on three data structures:

- **`ids`** — a list of unique identifiers, one per embedding (strings, ints, anything hashable).
- **`id_to_group`** — a dict mapping each id to a group label. Items that share a group are treated as positives (duplicates / paraphrases / relevant matches). Everything else is a negative.
- **`embs`** — a numpy array or torch tensor of shape `(n, d)`, one row per id.

For example, if questions 0, 1, 2 are duplicates and 3, 4 are duplicates:
```python
ids = [0, 1, 2, 3, 4]
id_to_group = {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B'}
```

Here is a complete pipeline — from loading data to exporting a `torch.nn.Linear`:

```python
import torch
import torch.nn as nn
from litfit import (
    load_askubuntu, encode_texts, split_data,
    compute_stats, generate_fast_projections,
    find_dim_range, evaluate_projections,
)

# --- 1. Load & encode ---
# load_askubuntu returns (ids, texts, id_to_group).
# max_groups limits how many duplicate-groups to keep (for speed).
all_ids, all_texts, id_to_group = load_askubuntu(max_groups=1500)
embs = encode_texts("intfloat/e5-base-v2", all_texts)

# --- 2. Split into train / val / test ---
# split_data does a group-aware split: all items in a group stay together,
# so no group leaks across splits. Default: 60/20/20.
data = split_data(all_ids, all_texts, embs, id_to_group)
train_ids, _, train_embs, _ = data["train"]
val_ids,   _, val_embs,   _ = data["val"]
test_ids,  _, test_embs,  _ = data["test"]

# --- 3. Compute sufficient statistics from training pairs ---
# compute_stats builds covariance matrices (Sigma_XX, Sigma_XY, etc.)
# from all positive pairs implied by id_to_group.
st = compute_stats(train_embs, train_ids, id_to_group)

# --- 4. Find useful dimension range ---
# Scans Rayleigh projections at many dims to find where performance peaks.
# Returns dim_fractions focused on the useful range.
dim_fractions = find_dim_range(st, val_embs, val_ids, id_to_group)

# --- 5. Generate & evaluate projections ---
# generate_fast_projections returns ~40 (method, hyperparams) configs.
# evaluate_projections uses explore-exploit scheduling on the val set.
all_W = generate_fast_projections(st)
results, summary = evaluate_projections(
    all_W, val_embs, val_ids, id_to_group,
    test_embs=test_embs, test_ids=test_ids,
    dim_fractions=dim_fractions,
)

# --- 6. Extract the best projection ---
# results keys are tuples like ('m_rayleigh', 'reg=0.1').
# Each value is {n_dims: {'MAP@50': ..., 'R@1': ..., ...}}.
# n_dims=None means full-dimensional.
best_key = max(results, key=lambda k: results[k][None]['MAP@50'])
W = all_W[best_key]                # shape (d, d) or (d, k)

# Optionally truncate to the best reduced dimension:
best_dim = 128
projected = test_embs @ W[:, :best_dim]  # shape (n, best_dim)

# --- 7. (Optional) Recompute stats on ALL data for best performance ---
# The train split was used for fitting and val/test for model selection.
# Once you've picked the best config, recompute stats on all available
# embeddings so the final projection sees the most signal.
full_st = compute_stats(embs, all_ids, id_to_group)
all_W_full = generate_fast_projections(full_st, verbose=False)
W = all_W_full[best_key]

# --- 8. Export as torch.nn.Linear for inference ---
out_dim = best_dim             # or W.shape[1] for full
layer = nn.Linear(W.shape[0], out_dim, bias=False)
layer.weight = nn.Parameter(W[:, :out_dim].T.cpu().float())
# Use it: projected = layer(input_embs)
```
</details>

See the [docs](docs/) for [more examples](docs/examples.md), [architecture diagrams](docs/architecture.md), and streaming scripts.

## Device Support

- **CUDA**: Full support with optional Triton acceleration
- **CPU**: Full support
- **MPS**: Not supported (missing linalg ops)

## Development

```bash
pip install -e ".[dev]"
pytest
mypy litfit
black litfit tests
```
