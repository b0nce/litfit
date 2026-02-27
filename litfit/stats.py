import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .device import DEVICE, DTYPE, _normalize, to_torch


def _try_triton_stats(
    Xc: torch.Tensor, Yc: torch.Tensor, n: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Try fused Triton kernel for covariance computation. Returns (SXX, SYY, SXY) or None if unavailable or on CPU."""
    if DEVICE.type != "cuda" or n < 2**18:
        # `n < 2 ** 18` avoids triton compilation overhead for small datasets
        return None
    try:
        from .triton_kernels import weighted_outer_products
    except ImportError:
        return None
    try:
        d = Xc.shape[1]
        # Triton kernel: X @ diag(w) @ X^T
        # We pass Xc.T (d, n) so that result is Xc.T @ diag(1/n) @ Xc = SXX
        Xt = Xc.T.contiguous()
        Yt = Yc.T.contiguous()
        w = torch.full((n,), 1.0 / n, device=DEVICE, dtype=DTYPE)
        SXX = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
        SXY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
        SYY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
        weighted_outer_products(Xt, Yt, w, SXX, SXY, SYY, accumulate=False)
        return SXX, SYY, SXY
    except Exception:
        return None


_REQUIRED_KEYS = {'Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean'}
_REQUIRED_NEG_KEYS = {'Sigma_cross_neg'}


def _check_st(st: Dict) -> None:
    """Validate that st contains all required statistics keys."""
    if not _REQUIRED_KEYS.issubset(st.keys()):
        raise ValueError(f"Missing keys: {_REQUIRED_KEYS - st.keys()}")


def _check_neg(neg: Optional[Dict]) -> None:
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
) -> Dict[str, Any]:
    """Compute all sufficient statistics from embeddings and group labels."""
    embs_t = to_torch(embs)

    group_to_indices = defaultdict(list)
    for i, qid in enumerate(ids):
        group_to_indices[id_to_group[qid]].append(i)

    xs, ys = [], []
    for gid, indices in group_to_indices.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                xs.append(indices[i])
                ys.append(indices[j])

    X = embs_t[xs]
    Y = embs_t[ys]

    if symmetrize:
        X, Y = torch.cat([X, Y], dim=0), torch.cat([Y, X], dim=0)

    X_mean = X.mean(dim=0)
    Xc = X - X_mean
    Yc = Y - X_mean
    n = X.shape[0]

    triton_result = _try_triton_stats(Xc, Yc, n)
    if triton_result is not None:
        SXX, SYY, SXY = triton_result
    else:
        SXX = Xc.T @ Xc / n
        SYY = Yc.T @ Yc / n
        SXY = Xc.T @ Yc / n

    return {
        'Sigma_XX': SXX,
        'Sigma_YY': SYY,
        'Sigma_XY': SXY,
        'Sigma_total': SXX + SYY,
        'Sigma_cross': SXY + SXY.T,
        'X_mean': X_mean,
        'n_pairs': n,
    }


def _needs_fp64(n: int, dtype: torch.dtype) -> bool:
    """Return True if 1/n underflows to 0 in the given dtype."""
    return torch.tensor(1.0, dtype=dtype) / torch.tensor(float(n), dtype=dtype) == 0.0


def compute_stats_streaming(
    pair_iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    symmetrize: bool = True,
) -> Dict[str, Any]:
    """Compute sufficient statistics incrementally from an iterator of (X_batch, Y_batch) pairs.

    Accumulates raw sums in DTYPE (typically float32). Before the final
    normalization, checks whether 1/n would underflow; if so, upcasts to
    float64 for the division step and warns.
    """
    sum_X = None
    sum_Y = None
    sum_XX = None
    sum_YY = None
    sum_XY = None
    n_total = 0

    for X_batch, Y_batch in pair_iterator:
        X_batch = to_torch(X_batch)
        Y_batch = to_torch(Y_batch)
        batch_n = X_batch.shape[0]

        if sum_X is None:
            d = X_batch.shape[1]
            sum_X = torch.zeros(d, device=DEVICE, dtype=DTYPE)
            sum_Y = torch.zeros(d, device=DEVICE, dtype=DTYPE)
            sum_XX = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
            sum_YY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)
            sum_XY = torch.zeros(d, d, device=DEVICE, dtype=DTYPE)

        sum_X += X_batch.sum(dim=0)
        sum_Y += Y_batch.sum(dim=0)
        sum_XX += X_batch.T @ X_batch
        sum_YY += Y_batch.T @ Y_batch
        sum_XY += X_batch.T @ Y_batch
        n_total += batch_n

    if n_total == 0:
        raise ValueError("pair_iterator yielded no data")

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
    """Compute hard negative pair statistics."""
    import numpy as np

    embs_t = to_torch(embs)

    g2i = defaultdict(list)
    for i, qid in enumerate(ids):
        g2i[id_to_group[qid]].append(i)

    embs_norm = _normalize(embs_t)
    sim = embs_norm @ embs_norm.T

    xs_idx, ys_idx = [], []
    sim_cpu = sim.cpu().numpy()
    for i in range(len(ids)):
        grp = set(g2i[id_to_group[ids[i]]])
        s = sim_cpu[i].copy()
        for j in grp:
            s[j] = -np.inf
        for j in np.argsort(s)[::-1][:n_neg]:
            xs_idx.append(i)
            ys_idx.append(j)

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
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Compute positive and negative pair statistics."""
    st = compute_stats(embs, ids, id_to_group, symmetrize=symmetrize)
    X_neg, Y_neg = _compute_neg_stats(embs, ids, id_to_group, n_neg=n_neg, seed=seed)
    Xnc = X_neg - st['X_mean']
    Ync = Y_neg - st['X_mean']
    neg = {'Sigma_cross_neg': (Xnc.T @ Ync + Ync.T @ Xnc) / Xnc.shape[0]}
    return st, neg
