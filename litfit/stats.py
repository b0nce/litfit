import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch

from .device import DEVICE, DTYPE, _normalize, to_torch


def _try_use_triton(n: int) -> bool:
    """Return True if Triton kernels should be used (CUDA + large enough batch)."""
    if DEVICE.type != "cuda" or n < 2**18:
        return False
    try:
        from .triton_kernels import weighted_outer_products  # noqa: F401

        return True
    except (ImportError, Exception):
        return False


def _flush_triton(
    X: torch.Tensor, Y: torch.Tensor, sum_XX: torch.Tensor, sum_XY: torch.Tensor, sum_YY: torch.Tensor
) -> None:
    """Accumulate outer products using Triton kernel."""
    from .triton_kernels import weighted_outer_products

    n = X.shape[0]
    w = torch.ones(n, device=DEVICE, dtype=DTYPE)
    weighted_outer_products(X.T.contiguous(), Y.T.contiguous(), w, sum_XX, sum_XY, sum_YY, accumulate=True)


_REQUIRED_KEYS = {'Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean'}
_REQUIRED_NEG_KEYS = {'Sigma_cross_neg'}


def _check_st(st: dict) -> None:
    """Validate that st contains all required statistics keys."""
    if not _REQUIRED_KEYS.issubset(st.keys()):
        raise ValueError(f"Missing keys: {_REQUIRED_KEYS - st.keys()}")


def _check_neg(neg: dict | None) -> None:
    """Validate that neg is provided and contains required keys."""
    if neg is None:
        raise ValueError("This method requires negative pair statistics")
    if not _REQUIRED_NEG_KEYS.issubset(neg.keys()):
        raise ValueError(f"Missing neg keys: {_REQUIRED_NEG_KEYS - neg.keys()}")


def compute_stats(
    embs: torch.Tensor,
    ids: list,
    id_to_group: dict,
    symmetrize: bool = True,
) -> dict[str, Any]:
    """Compute all sufficient statistics from embeddings and group labels.

    Uses chunked streaming accumulation with fixed-size buffers to avoid
    materializing all pairs at once. Peak memory is O(chunk_size * d) for
    pair embeddings plus O(max_group_size^2) for cached pair indices.
    """
    embs_t = to_torch(embs)
    d = embs_t.shape[1]

    group_to_indices = defaultdict(list)
    for i, qid in enumerate(ids):
        group_to_indices[id_to_group[qid]].append(i)

    # Find max group size for triu_indices caching
    max_k = max((len(v) for v in group_to_indices.values()), default=0)
    if max_k < 2:
        # No pairs possible — return zeros
        zero_d = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
        return {
            'Sigma_XX': zero_d,
            'Sigma_YY': zero_d.clone(),
            'Sigma_XY': zero_d.clone(),
            'Sigma_total': zero_d.clone(),
            'Sigma_cross': zero_d.clone(),
            'X_mean': torch.zeros(d, device=DEVICE, dtype=DTYPE),
            'n_pairs': 0,
        }

    # Cache triu_indices for the max group size, mask for smaller groups
    max_rows, max_cols = torch.triu_indices(max_k, max_k, offset=1, device=DEVICE)
    _idx_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_pair_indices(k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if k not in _idx_cache:
            mask = (max_rows < k) & (max_cols < k)
            _idx_cache[k] = (max_rows[mask], max_cols[mask])
        return _idx_cache[k]

    # Streaming accumulation with fixed-size buffer
    chunk_size = 1 << 16  # 65536 pairs per flush
    sum_X = torch.zeros(d, device=DEVICE, dtype=DTYPE)
    sum_Y = torch.zeros(d, device=DEVICE, dtype=DTYPE)
    sum_XX = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    sum_YY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    sum_XY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    n_pairs = 0

    # Buffers for batching (global index tensors)
    xs_buf: list[torch.Tensor] = []
    ys_buf: list[torch.Tensor] = []
    buf_len = 0

    use_triton = _try_use_triton(chunk_size)

    def flush() -> None:
        nonlocal buf_len
        if buf_len == 0:
            return
        xs_t = torch.cat(xs_buf)
        ys_t = torch.cat(ys_buf)
        X = embs_t[xs_t]
        Y = embs_t[ys_t]
        sum_X.add_(X.sum(0))
        sum_Y.add_(Y.sum(0))
        if use_triton:
            _flush_triton(X, Y, sum_XX, sum_XY, sum_YY)
        else:
            sum_XX.addmm_(X.T, X)
            sum_YY.addmm_(Y.T, Y)
            sum_XY.addmm_(X.T, Y)
        xs_buf.clear()
        ys_buf.clear()
        buf_len = 0

    for indices in group_to_indices.values():
        k = len(indices)
        if k < 2:
            continue
        rows, cols = get_pair_indices(k)
        indices_t = torch.tensor(indices, device=DEVICE, dtype=torch.long)
        xs_buf.append(indices_t[rows])
        ys_buf.append(indices_t[cols])
        buf_len += len(rows)
        n_pairs += len(rows)
        if buf_len >= chunk_size:
            flush()
    flush()  # remaining buffer

    # Derive covariances from sums (same math as compute_stats_streaming)
    if symmetrize:
        n = 2 * n_pairs
        sum_all = sum_X + sum_Y
        mu = sum_all / n
        SXX = (sum_XX + sum_YY) / n - mu.unsqueeze(1) * mu.unsqueeze(0)
        SYY = SXX
        SXY = (sum_XY + sum_XY.T) / n - mu.unsqueeze(1) * mu.unsqueeze(0)
    else:
        n = n_pairs
        mu_x = sum_X / n
        mu_y = sum_Y / n
        SXX = sum_XX / n - mu_x.unsqueeze(1) * mu_x.unsqueeze(0)
        SYY = (
            sum_YY / n
            - mu_x.unsqueeze(1) * mu_y.unsqueeze(0)
            - mu_y.unsqueeze(1) * mu_x.unsqueeze(0)
            + mu_x.unsqueeze(1) * mu_x.unsqueeze(0)
        )
        SXY = sum_XY / n - mu_x.unsqueeze(1) * mu_y.unsqueeze(0)
        mu = mu_x

    return {
        'Sigma_XX': SXX,
        'Sigma_YY': SYY,
        'Sigma_XY': SXY,
        'Sigma_total': SXX + SYY,
        'Sigma_cross': SXY + SXY.T,
        'X_mean': mu,
        'n_pairs': n,
    }


def _needs_fp64(n: int, dtype: torch.dtype) -> bool:
    """Return True if 1/n underflows to 0 in the given dtype."""
    return bool(torch.tensor(1.0, dtype=dtype) / torch.tensor(float(n), dtype=dtype) == 0.0)


def compute_stats_streaming(
    pair_iterator: Iterable[tuple[torch.Tensor, torch.Tensor]],
    symmetrize: bool = True,
) -> dict[str, Any]:
    """Compute sufficient statistics incrementally from an iterator of (X_batch, Y_batch) pairs.

    Accumulates raw sums in DTYPE (typically float32). Before the final
    normalization, checks whether 1/n would underflow; if so, upcasts to
    float64 for the division step and warns.
    """
    it = iter(pair_iterator)
    first = next(it, None)
    if first is None:
        raise ValueError("pair_iterator yielded no data")

    X_batch = to_torch(first[0])
    Y_batch = to_torch(first[1])
    d = X_batch.shape[1]
    sum_X = torch.zeros(d, device=DEVICE, dtype=DTYPE)
    sum_Y = torch.zeros(d, device=DEVICE, dtype=DTYPE)
    sum_XX = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    sum_YY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    sum_XY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
    n_total = 0

    from itertools import chain

    for X_batch, Y_batch in chain([(X_batch, Y_batch)], it):
        X_batch = to_torch(X_batch)
        Y_batch = to_torch(Y_batch)
        batch_n = X_batch.shape[0]

        sum_X += X_batch.sum(dim=0)
        sum_Y += Y_batch.sum(dim=0)
        sum_XX += X_batch.T @ X_batch
        sum_YY += Y_batch.T @ Y_batch
        sum_XY += X_batch.T @ Y_batch
        n_total += batch_n

    n = 2 * n_total if symmetrize else n_total

    # Upcast to float64 only if 1/n would underflow in DTYPE
    if _needs_fp64(n, DTYPE):
        warnings.warn(
            f"n={n} is large enough that 1/n underflows to 0.0 in {DTYPE}. "
            "Upcasting accumulators to float64 for normalization."
        )
        sum_X = sum_X.to(torch.float64)
        sum_Y = sum_Y.to(torch.float64)
        sum_XX = sum_XX.to(torch.float64)
        sum_YY = sum_YY.to(torch.float64)
        sum_XY = sum_XY.to(torch.float64)

    if symmetrize:
        sum_all = sum_X + sum_Y
        mu = sum_all / n
        SXX = (sum_XX + sum_YY) / n - mu.unsqueeze(1) * mu.unsqueeze(0)
        SYY = SXX
        SXY = (sum_XY + sum_XY.T) / n - mu.unsqueeze(1) * mu.unsqueeze(0)
        X_mean = mu
    else:
        mu_x = sum_X / n
        mu_y = sum_Y / n
        # Match compute_stats: both X and Y are centered around X_mean (mu_x)
        SXX = sum_XX / n - mu_x.unsqueeze(1) * mu_x.unsqueeze(0)
        SYY = (
            sum_YY / n
            - mu_x.unsqueeze(1) * mu_y.unsqueeze(0)
            - mu_y.unsqueeze(1) * mu_x.unsqueeze(0)
            + mu_x.unsqueeze(1) * mu_x.unsqueeze(0)
        )
        SXY = sum_XY / n - mu_x.unsqueeze(1) * mu_y.unsqueeze(0)
        X_mean = mu_x

    # Convert back to working dtype (no-op if already DTYPE)
    SXX = SXX.to(dtype=DTYPE)
    SYY = SYY.to(dtype=DTYPE)
    SXY = SXY.to(dtype=DTYPE)
    X_mean = X_mean.to(dtype=DTYPE)

    return {
        'Sigma_XX': SXX,
        'Sigma_YY': SYY,
        'Sigma_XY': SXY,
        'Sigma_total': SXX + SYY,
        'Sigma_cross': SXY + SXY.T,
        'X_mean': X_mean,
        'n_pairs': n,
    }


def _compute_neg_stats(
    embs: torch.Tensor,
    ids: list,
    id_to_group: dict,
    n_neg: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hard negative pair statistics.

    Uses row-batched similarity + topk instead of materializing the full N×N
    similarity matrix. Memory: O(batch_size × N) instead of O(N²).
    """
    embs_t = to_torch(embs)
    N = len(ids)

    g2i = defaultdict(list)
    for i, qid in enumerate(ids):
        g2i[id_to_group[qid]].append(i)

    embs_norm = _normalize(embs_t)

    # Build group mask: for each row i, mask out same-group indices
    # We process in batches to avoid O(N²) memory
    batch_size = min(512, N)
    # We need n_neg + max_group_size candidates to ensure enough after masking
    max_group_size = max(len(v) for v in g2i.values())
    topk_k = min(n_neg + max_group_size, N)

    xs_idx: list[int] = []
    ys_idx: list[int] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # (batch, N) similarity
        sim_batch = embs_norm[start:end] @ embs_norm.T

        # Mask out same-group items
        for local_i, global_i in enumerate(range(start, end)):
            grp_indices = g2i[id_to_group[ids[global_i]]]
            sim_batch[local_i, grp_indices] = -float('inf')

        # topk for partial sort instead of full argsort
        _, top_indices = torch.topk(sim_batch, topk_k, dim=1)
        # Take first n_neg (already sorted descending by topk)
        top_neg = top_indices[:, :n_neg]

        for local_i in range(end - start):
            global_i = start + local_i
            for j in range(n_neg):
                xs_idx.append(global_i)
                ys_idx.append(int(top_neg[local_i, j]))

    X_neg = embs_t[xs_idx]
    Y_neg = embs_t[ys_idx]
    return X_neg, Y_neg


def compute_all_stats(
    embs: torch.Tensor,
    ids: list,
    id_to_group: dict,
    symmetrize: bool = True,
    n_neg: int = 3,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Compute positive and negative pair statistics."""
    st = compute_stats(embs, ids, id_to_group, symmetrize=symmetrize)
    X_neg, Y_neg = _compute_neg_stats(embs, ids, id_to_group, n_neg=n_neg, seed=seed)
    Xnc = X_neg - st['X_mean']
    Ync = Y_neg - st['X_mean']
    neg = {'Sigma_cross_neg': (Xnc.T @ Ync + Ync.T @ Xnc) / Xnc.shape[0]}
    return st, neg
