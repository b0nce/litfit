#!/usr/bin/env python
"""
Streaming pipeline: learn embedding projections from AskUbuntu duplicate questions.

sentence-transformers/askubuntu contains query-positive-negative triplets from
AskUbuntu.  Each row has a query string, a list of positive (duplicate) strings,
and a list of hard negatives.  Positive pairs (query <-> duplicate) provide a
natural signal for learning projections.

Workflow:
  1. Sample rows for val/test, deduplicate with union-find, encode, save memmaps
  2. Stream remaining rows -> encode query-positive pairs ->
     feed to compute_stats_streaming (constant memory)
  3. generate_fast_projections  (~40 configs)
  4. find_dim_range             (quick scan for useful dimensions)
  5. evaluate_projections       (explore-exploit over configs x dims)

Requirements (beyond litfit):
  pip install datasets sentence-transformers

Usage:
  python docs/streaming_askubuntu.py              # full run
  python docs/streaming_askubuntu.py --cached     # reload memmaps, skip encoding
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
DATASET_NAME = 'sentence-transformers/askubuntu'
TRAIN_CHUNK_SIZE = 2048       # rows per streaming chunk
MAX_TRAIN_ROWS = 50_000       # stop streaming after this many rows
VAL_ROWS = 1_500              # dataset rows sampled for val
TEST_ROWS = 1_500             # dataset rows sampled for test
CACHE_DIR = Path('data/askubuntu_memmaps')
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────────

class UnionFind:
    """Union-find for grouping duplicate questions."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def save_memmap(path: Path, arr: np.ndarray) -> None:
    """Write a numpy array as a memory-mapped file."""
    mm = np.memmap(path, dtype=arr.dtype, mode='w+', shape=arr.shape)
    mm[:] = arr
    mm.flush()
    np.save(str(path) + '.meta.npy', np.array(arr.shape))
    print(f'  Saved {path}  shape={arr.shape}  dtype={arr.dtype}')


def load_memmap(path: Path) -> np.ndarray:
    """Reload a memmap saved with save_memmap."""
    shape = tuple(np.load(str(path) + '.meta.npy'))
    return np.memmap(path, dtype=np.float32, mode='r', shape=shape)


def rows_to_eval_data(rows: list[dict]) -> tuple[list[str], list[int], dict[int, int]]:
    """Build unique text list with union-find grouping from dataset rows.

    Returns (texts, ids, id_to_group).  Texts sharing a query-positive
    relationship are assigned to the same group via transitive closure.
    """
    uf = UnionFind()
    texts_seen: dict[str, int] = {}
    next_id = 0

    for row in rows:
        query = row['query'].strip()
        if query not in texts_seen:
            texts_seen[query] = next_id
            next_id += 1

        for pos in (row['positive'] or []):
            pos = pos.strip()
            if not pos:
                continue
            if pos not in texts_seen:
                texts_seen[pos] = next_id
                next_id += 1
            uf.union(query, pos)

    # Map union-find roots to contiguous group ids
    texts = [''] * len(texts_seen)
    for t, idx in texts_seen.items():
        texts[idx] = t

    root_to_gid: dict[str, int] = {}
    gid_counter = 0
    id_to_group: dict[int, int] = {}

    for t, idx in texts_seen.items():
        root = uf.find(t)
        if root not in root_to_gid:
            root_to_gid[root] = gid_counter
            gid_counter += 1
        id_to_group[idx] = root_to_gid[root]

    # Keep only groups with >= 2 members (singletons are useless for retrieval)
    group_sizes: dict[int, int] = defaultdict(int)
    for gid in id_to_group.values():
        group_sizes[gid] += 1

    keep = {idx for idx, gid in id_to_group.items() if group_sizes[gid] >= 2}
    texts = [texts[i] for i in sorted(keep)]
    remap = {old: new for new, old in enumerate(sorted(keep))}
    ids = list(range(len(texts)))
    id_to_group = {remap[old]: gid for old, gid in id_to_group.items() if old in keep}

    return texts, ids, id_to_group


def pairs_from_chunk(
    model: SentenceTransformer,
    rows: list[dict],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encode a chunk of rows and return (X, Y) positive-pair batches."""
    # Collect unique texts and explicit pairs
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []

    def _idx(t: str) -> int:
        if t not in text_to_idx:
            text_to_idx[t] = len(all_texts)
            all_texts.append(t)
        return text_to_idx[t]

    for row in rows:
        query = row['query'].strip()
        if not query:
            continue
        qi = _idx(query)
        for pos in (row['positive'] or []):
            pos = pos.strip()
            if pos:
                pairs.append((qi, _idx(pos)))

    if not pairs:
        return []

    # Encode all unique texts at once
    prefixed = [f'query: {t}' for t in all_texts]
    embs = model.encode(prefixed, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True)

    x_idx = [q for q, _ in pairs]
    y_idx = [p for _, p in pairs]
    X = torch.tensor(embs[x_idx], dtype=el.DTYPE, device=el.DEVICE)
    Y = torch.tensor(embs[y_idx], dtype=el.DTYPE, device=el.DEVICE)
    return [(X, Y)]


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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
        val_ids = np.load(CACHE_DIR / 'val_ids.npy').tolist()
        val_groups = np.load(CACHE_DIR / 'val_groups.npy').tolist()
        test_ids = np.load(CACHE_DIR / 'test_ids.npy').tolist()
        test_groups = np.load(CACHE_DIR / 'test_groups.npy').tolist()
        val_id_to_group = dict(zip(val_ids, val_groups))
        test_id_to_group = dict(zip(test_ids, test_groups))
    else:
        # Load full dataset, shuffle, split into val / test / train
        print('Loading dataset for val/test sampling...')
        ds_full = load_dataset(DATASET_NAME, split='train',
                               trust_remote_code=True).shuffle(seed=SEED)

        split_point = VAL_ROWS + TEST_ROWS
        val_rows = [ds_full[i] for i in range(VAL_ROWS)]
        test_rows = [ds_full[i] for i in range(VAL_ROWS, split_point)]

        print('Building val groups (union-find)...')
        val_texts, val_ids, val_id_to_group = rows_to_eval_data(val_rows)
        n_val_groups = len(set(val_id_to_group.values()))
        print(f'  {len(val_texts)} unique texts, {n_val_groups} groups')

        print('Building test groups (union-find)...')
        test_texts, test_ids, test_id_to_group = rows_to_eval_data(test_rows)
        n_test_groups = len(set(test_id_to_group.values()))
        print(f'  {len(test_texts)} unique texts, {n_test_groups} groups')

        # Encode with E5 query prefix
        print(f'Encoding {len(val_texts)} val texts...')
        val_np = model.encode([f'query: {t}' for t in val_texts],
                              batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)
        print(f'Encoding {len(test_texts)} test texts...')
        test_np = model.encode([f'query: {t}' for t in test_texts],
                               batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True)

        # Save memmaps
        save_memmap(CACHE_DIR / 'val_embs.dat', val_np)
        save_memmap(CACHE_DIR / 'test_embs.dat', test_np)
        np.save(CACHE_DIR / 'val_ids.npy', np.array(val_ids))
        np.save(CACHE_DIR / 'val_groups.npy',
                np.array([val_id_to_group[i] for i in val_ids]))
        np.save(CACHE_DIR / 'test_ids.npy', np.array(test_ids))
        np.save(CACHE_DIR / 'test_groups.npy',
                np.array([test_id_to_group[i] for i in test_ids]))

    val_embs = el.to_torch(val_np)
    test_embs = el.to_torch(test_np)

    # ── 2. Stream training pairs ────────────────────────────────────
    # Stream from the full dataset.  There is minor overlap with val/test
    # rows (which were sampled from the same split), but this is negligible
    # for learning covariance statistics and keeps the example simple.
    def stream_pairs():
        """Yield (X_batch, Y_batch) from query-positive pairs."""
        ds = load_dataset(DATASET_NAME, split='train', streaming=True,
                          trust_remote_code=True)
        chunk: list[dict] = []
        rows_seen = 0

        for row in ds:
            chunk.append(row)
            if len(chunk) >= TRAIN_CHUNK_SIZE:
                yield from pairs_from_chunk(model, chunk)
                rows_seen += len(chunk)
                print(f'  Streamed {rows_seen:,}/{MAX_TRAIN_ROWS:,} rows')
                chunk = []
                if rows_seen >= MAX_TRAIN_ROWS:
                    break

        if chunk:
            yield from pairs_from_chunk(model, chunk)

    print(f'\nStreaming up to {MAX_TRAIN_ROWS:,} training rows...')
    st = el.compute_stats_streaming(stream_pairs())
    print(f'Statistics computed from {st["n_pairs"]:,} pairs  '
          f'(d={st["X_mean"].shape[0]})')

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
