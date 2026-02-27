# Examples

## Quick Start

```python
from litfit import (
    compute_stats, generate_all_projections, evaluate_projections,
)

st = compute_stats(embs, ids, id_to_group)
all_W = generate_all_projections(st, neg=None, include_neg_methods=False)
results, summary = evaluate_projections(
    all_W, val_embs, val_ids, id_to_group,
    test_embs=test_embs, test_ids=test_ids,
)
```

The library works with any embeddings — you just need:
- `embs`: a numpy array or torch tensor of shape `(n, d)`
- `ids`: a list of unique identifiers for each embedding
- `id_to_group`: a dict mapping each id to a group label (items in the same group are treated as positives)

## Fast mode (~40 configs instead of ~1400)

`generate_fast_projections` runs only the methods and hyperparameter combinations
that consistently perform best across datasets and models. It produces roughly 30x
fewer projections with minimal quality loss — ideal for quick experiments or when
compute time matters.

```python
from litfit import (
    compute_stats, generate_fast_projections, evaluate_projections,
)

st = compute_stats(train_embs, train_ids, id_to_group)
all_W = generate_fast_projections(st)
results, summary = evaluate_projections(
    all_W, val_embs, val_ids, id_to_group,
    test_embs=test_embs, test_ids=test_ids,
)
```

The fast subset covers: Rayleigh, Ray→AsymRef, Ray→AsymRef→MSE, Ray→MSE→AsymRef,
Ray→MSE, and SplitRankRay. When negative stats are passed, it also includes
RayContr→MSE and RayContr→MSE+neg.

## Finding the optimal dimension range

Projection performance varies heavily with the number of output dimensions.
`find_dim_range` uses the cheapest method (Rayleigh) to quickly scan many dimension
counts and returns `dim_fractions` concentrated in the useful range — so the full
evaluation doesn't waste time on dimensions that clearly hurt or don't help.

```python
from litfit import (
    load_askubuntu, encode_texts, split_data,
    compute_stats, generate_fast_projections,
    find_dim_range, evaluate_projections,
)

# 1. Load data and encode
all_ids, all_texts, id_to_group = load_askubuntu(max_groups=1500)
embs = encode_texts("intfloat/e5-base-v2", all_texts)
data = split_data(all_ids, all_texts, embs, id_to_group)
train_ids, _, train_embs, _ = data["train"]
val_ids, _, val_embs, _ = data["val"]
test_ids, _, test_embs, _ = data["test"]

# 2. Compute statistics
st = compute_stats(train_embs, train_ids, id_to_group)

# 3. Scan dimensions with Rayleigh to find where projections help
dim_fractions = find_dim_range(st, val_embs, val_ids, id_to_group)
# Prints a table like:
#   dims      MAP@50   vs base
#     38      0.5332   -0.0498
#    190      0.6397   +0.0567  <-- peak
#    768      0.6046   +0.0216
# Useful range: dims 190-380 (fracs 0.25-0.49)
# Returned dim_fractions: (0.05, 0.15, 0.25, 0.31, 0.37, 0.43, 0.49, 1.0)

# 4. Generate and evaluate with focused dim_fractions
all_W = generate_fast_projections(st)
results, summary = evaluate_projections(
    all_W, val_embs, val_ids, id_to_group,
    test_embs=test_embs, test_ids=test_ids,
    dim_fractions=dim_fractions,
)
```

## Contrastive methods with negative pairs

```python
from litfit import compute_all_stats, generate_all_projections

st, neg = compute_all_stats(train_embs, train_ids, id_to_group, n_neg=3)
all_W = generate_all_projections(st, neg=neg, include_neg_methods=True)
```

## Applying a learned projection

```python
best_key = max(results, key=lambda k: results[k][None]["MAP@50"])
W = all_W[best_key]

projected = new_embs @ W
projected_reduced = new_embs @ W[:, :128]
```

## Evaluating a single projection

```python
from litfit import evaluate_retrieval_fast

baseline = evaluate_retrieval_fast(val_embs, val_ids, id_to_group)
scores = evaluate_retrieval_fast(val_embs @ W, val_ids, id_to_group)
```

## Individual projection methods

```python
from litfit import m_rayleigh, m_mse, m_ray_mse

W_ray = m_rayleigh(st, reg=0.1)
W_mse = m_mse(st, reg=0.1)
W_combined = m_ray_mse(st, reg=0.1, reg_mse=1.0)
```

## End-to-end streaming example

Two complete scripts that stream a HuggingFace dataset, compute statistics via
`compute_stats_streaming`, save val/test embeddings as numpy memmaps, and run
the full `find_dim_range` → `generate_fast_projections` → `evaluate_projections`
pipeline:

- [`streaming_example.py`](streaming_example.py) — AG News (news topic classification, 4 classes)
- [`streaming_askubuntu.py`](streaming_askubuntu.py) — AskUbuntu (duplicate question pairs, union-find grouping)

```bash
python docs/streaming_example.py              # AG News, full run
python docs/streaming_example.py --cached     # reload memmaps, skip encoding

python docs/streaming_askubuntu.py            # AskUbuntu, full run
python docs/streaming_askubuntu.py --cached   # reload memmaps
```

## Streaming + lazy for low memory usage

When your dataset is too large to hold all pairs in memory, combine streaming statistics with lazy projection generation. Streaming accumulates covariance matrices incrementally, and lazy mode computes each projection on demand rather than storing them all at once.

```python
from litfit import compute_stats_streaming, generate_all_projections

def pair_batches():
    for i in range(0, len(X_pairs), 1024):
        yield X_pairs[i:i+1024], Y_pairs[i:i+1024]

st = compute_stats_streaming(pair_batches())
all_W = generate_all_projections(st, neg=None, include_neg_methods=False, lazy=True)

for key in all_W:
    W = all_W[key]        # computed on access, not cached
    scores = val_embs @ W
    # W is garbage-collected after this iteration
```
