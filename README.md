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

See [more examples](docs/examples.md) for contrastive methods, individual projections, and more.

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
