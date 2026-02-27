#!/usr/bin/env python
"""
Streaming pipeline: learn embedding projections from AG News via HuggingFace.

AG News (news topic classification) is unlikely to appear in E5's contrastive
training data, which is built from sentence-pair datasets (NLI, QA, forums).
The 4 topic classes (World / Sports / Business / Sci-Tech) give a clean
grouping signal for learning projections.

Workflow:
  1. Sample val/test rows from the test split, encode, save as numpy memmaps
  2. Stream training split in chunks → encode → form same-label pairs →
     feed to compute_stats_streaming (constant memory)
  3. generate_fast_projections  (~40 configs)
  4. find_dim_range             (quick scan for useful dimensions)
  5. evaluate_projections       (explore-exploit over configs × dims)

Requirements (beyond litfit):
  pip install datasets sentence-transformers

Usage:
  python docs/streaming_example.py              # full run
  python docs/streaming_example.py --cached     # skip encoding, load memmaps
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import litfit as el

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = 'intfloat/e5-base-v2'
DATASET_NAME = 'fancyzhx/ag_news'
TRAIN_CHUNK_SIZE = 4096       # rows per streaming chunk
MAX_TRAIN_ROWS = 80_000       # stop streaming after this many
VAL_SIZE = 3_000
TEST_SIZE = 3_000
CACHE_DIR = Path('data/ag_news_memmaps')
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────

def save_memmap(path: Path, arr: np.ndarray) -> None:
    """Write a numpy array as a memory-mapped file."""
    mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mm[:] = arr
    mm.flush()
    # Save shape metadata so we can reload without knowing it in advance
    np.save(str(path) + '.meta.npy', np.array(arr.shape))
    print(f'  Saved {path}  shape={arr.shape}  dtype={arr.dtype}')


def load_memmap(path: Path) -> np.ndarray:
    """Reload a memmap saved with save_memmap."""
    shape = tuple(np.load(str(path) + '.meta.npy'))
    return np.memmap(path, dtype=np.float32, mode='r', shape=shape)


def pairs_from_chunk(
    model: SentenceTransformer,
    texts: list[str],
    labels: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encode a chunk of texts and return (X, Y) positive-pair batches."""
    embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True)

    # Group indices by label
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    # Pair up consecutive items within each group
    x_idx, y_idx = [], []
    for indices in groups.values():
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            x_idx.append(indices[i])
            y_idx.append(indices[i + 1])

    if not x_idx:
        return []

    X = torch.tensor(embs[x_idx], dtype=el.DTYPE, device=el.DEVICE)
    Y = torch.tensor(embs[y_idx], dtype=el.DTYPE, device=el.DEVICE)
    return [(X, Y)]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true',
                        help='Skip encoding; load val/test from memmaps')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # ── 1. Val / test embeddings ────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.cached and (CACHE_DIR / 'val_embs.dat').exists():
        print('Loading cached val/test memmaps...')
        val_np = load_memmap(CACHE_DIR / 'val_embs.dat')
        test_np = load_memmap(CACHE_DIR / 'test_embs.dat')
        val_labels = np.load(CACHE_DIR / 'val_labels.npy').tolist()
        test_labels = np.load(CACHE_DIR / 'test_labels.npy').tolist()
    else:
        print('Sampling val/test from AG News test split...')
        ds_test = load_dataset(DATASET_NAME, split='test').shuffle(seed=SEED)

        val_rows = ds_test.select(range(VAL_SIZE))
        test_rows = ds_test.select(range(VAL_SIZE, VAL_SIZE + TEST_SIZE))

        # E5 models expect a task prefix
        val_texts = [f"query: {r['text']}" for r in val_rows]
        test_texts = [f"query: {r['text']}" for r in test_rows]
        val_labels = [r['label'] for r in val_rows]
        test_labels = [r['label'] for r in test_rows]

        print(f'Encoding {len(val_texts)} val texts...')
        val_np = model.encode(val_texts, batch_size=64,
                              show_progress_bar=True, convert_to_numpy=True)
        print(f'Encoding {len(test_texts)} test texts...')
        test_np = model.encode(test_texts, batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)

        save_memmap(CACHE_DIR / 'val_embs.dat', val_np)
        save_memmap(CACHE_DIR / 'test_embs.dat', test_np)
        np.save(CACHE_DIR / 'val_labels.npy', np.array(val_labels))
        np.save(CACHE_DIR / 'test_labels.npy', np.array(test_labels))

    val_ids = list(range(len(val_labels)))
    val_id_to_group = dict(enumerate(val_labels))
    test_ids = list(range(len(test_labels)))
    test_id_to_group = dict(enumerate(test_labels))

    val_embs = el.to_torch(val_np)
    test_embs = el.to_torch(test_np)

    # ── 2. Stream training pairs ────────────────────────────────────
    def stream_pairs():
        """Yield (X_batch, Y_batch) from the AG News training split."""
        ds = load_dataset(DATASET_NAME, split='train', streaming=True)
        chunk_texts: list[str] = []
        chunk_labels: list[int] = []
        rows_seen = 0

        for row in ds:
            chunk_texts.append(f"query: {row['text']}")
            chunk_labels.append(row['label'])

            if len(chunk_texts) >= TRAIN_CHUNK_SIZE:
                yield from pairs_from_chunk(model, chunk_texts, chunk_labels)
                rows_seen += len(chunk_texts)
                print(f'  Streamed {rows_seen:,}/{MAX_TRAIN_ROWS:,} rows')
                chunk_texts, chunk_labels = [], []
                if rows_seen >= MAX_TRAIN_ROWS:
                    break

        # Flush remaining rows
        if chunk_texts:
            yield from pairs_from_chunk(model, chunk_texts, chunk_labels)

    print(f'\nStreaming up to {MAX_TRAIN_ROWS:,} training rows...')
    st = el.compute_stats_streaming(stream_pairs())
    print(f'Statistics computed from {st["n_pairs"]:,} pairs  (d={st["X_mean"].shape[0]})')

    # ── 3. Fast projections ─────────────────────────────────────────
    print('\nGenerating fast projections...')
    all_W = el.generate_fast_projections(st)
    print(f'{len(all_W)} projection configs generated')

    # ── 4. Dimension range ──────────────────────────────────────────
    print('\nScanning useful dimension range...')
    dim_fracs = el.find_dim_range(st, val_embs, val_ids, val_id_to_group)
    print(f'Selected dim_fractions: {dim_fracs}')

    # ── 5. Evaluate ─────────────────────────────────────────────────
    print('\nEvaluating projections (Ctrl-C for early stop)...')
    results, summary = el.evaluate_projections(
        all_W,
        val_embs, val_ids, val_id_to_group,
        test_embs=test_embs, test_ids=test_ids,
        test_id_to_group=test_id_to_group,
        dim_fractions=dim_fracs,
    )

    # ── 6. Show best projection (select on val, report on test) ────
    # Find the (key, n_dims) pair with the highest val score across all dims
    best_key, best_ndims, best_val = None, None, -1.0
    for key, dims_dict in results.items():
        for n_dims, scores in dims_dict.items():
            if scores['MAP@50'] > best_val:
                best_key, best_ndims, best_val = key, n_dims, scores['MAP@50']

    W = all_W[best_key]
    proj = test_embs @ (W if best_ndims is None else W[:, :best_ndims])
    test_score = el.evaluate_retrieval_fast(proj, test_ids, test_id_to_group)
    baseline = el.evaluate_retrieval_fast(test_embs, test_ids, test_id_to_group)
    dims_label = f'full({W.shape[1]})' if best_ndims is None else f'{best_ndims}'
    print(f'\nBest projection (selected on val): {best_key}  dims={dims_label}')
    print(f'Val  MAP@50:           {best_val:.4f}')
    print(f'Test MAP@50:           {test_score["MAP@50"]:.4f}')
    print(f'Test baseline MAP@50:  {baseline["MAP@50"]:.4f}')
    print(f'Test improvement:      {test_score["MAP@50"] - baseline["MAP@50"]:+.4f}')


if __name__ == '__main__':
    main()
