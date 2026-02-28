import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from tqdm.auto import tqdm

from .device import DEVICE, DTYPE, _normalize, to_torch
from .methods import m_rayleigh


@torch.no_grad()
def evaluate_retrieval_fast(
    embs: torch.Tensor,
    ids: list,
    id_to_group: dict,
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Fully batched retrieval evaluation on GPU."""
    embs_t = to_torch(embs) if not isinstance(embs, torch.Tensor) else embs.to(DEVICE, DTYPE)
    embs_norm = _normalize(embs_t)
    sim = embs_norm @ embs_norm.T

    group_labels = [id_to_group[qid] for qid in ids]
    n = len(ids)

    group_ids_unique = list(set(group_labels))
    gid_map = {g: i for i, g in enumerate(group_ids_unique)}
    gl_idx = torch.tensor([gid_map[g] for g in group_labels], device=DEVICE)
    pos_mask = gl_idx.unsqueeze(0) == gl_idx.unsqueeze(1)
    pos_mask.fill_diagonal_(False)

    n_pos = pos_mask.sum(dim=1)
    has_pos = n_pos > 0

    sim.fill_diagonal_(-float('inf'))

    max_k = max(max(ks), 50)
    _, top_idx = sim.topk(min(max_k, n - 1), dim=1)

    top_is_pos = pos_mask.gather(1, top_idx)

    recalls = {}
    for k in ks:
        hits = top_is_pos[:, :k].sum(dim=1).float()
        cap = torch.minimum(torch.tensor(k, device=DEVICE, dtype=DTYPE), n_pos.float())
        r = hits / cap.clamp(min=1)
        recalls[k] = r[has_pos].sum().item()

    top50 = top_is_pos[:, :50].float()
    actual_k = top50.shape[1]
    cum_hits = top50.cumsum(dim=1)
    ranks = torch.arange(1, actual_k + 1, device=DEVICE, dtype=DTYPE).unsqueeze(0)
    precision_at = cum_hits / ranks
    # Normalise by min(n_pos, actual_k): standard MAP@k treats the relevance window as capped at k,
    # so queries with >50 positives are capped at 50.
    ap = (precision_at * top50).sum(dim=1) / n_pos.float().clamp(min=1).clamp(max=actual_k)

    nq = has_pos.sum().item()
    res = {f'R@{k}': recalls[k] / nq for k in ks}
    res['MAP@50'] = ap[has_pos].sum().item() / nq
    return res


@torch.no_grad()
def find_dim_range(
    st: dict,
    val_embs: torch.Tensor,
    val_ids: list,
    id_to_group: dict,
    n_points: int = 20,
    regs: tuple[float, ...] = (0.005, 0.01, 0.1, 1.0),
    metric: str = 'MAP@50',
    threshold: float = 0.9,
    verbose: bool = True,
    eval_fn: Callable | None = None,
) -> tuple[float, ...]:
    """Scan Rayleigh projections across dimensions to find the useful range.

    Uses the cheapest method (Rayleigh) to quickly evaluate many dimension
    counts and identify where projected performance peaks. Returns a set of
    dim_fractions concentrated in the useful range, suitable for passing to
    ``evaluate_projections(dim_fractions=...)``.

    If ``eval_fn`` is provided, it is called as
    ``eval_fn(projected_embs, val_ids, id_to_group)`` and must return a dict
    containing ``metric`` as a key. When ``None``, ``evaluate_retrieval_fast``
    is used.
    """
    _eval = eval_fn if eval_fn is not None else evaluate_retrieval_fast
    val_embs_t = to_torch(val_embs) if not isinstance(val_embs, torch.Tensor) else val_embs.to(DEVICE, DTYPE)
    d = st['Sigma_total'].shape[0]

    # Baseline: original embeddings
    baseline = _eval(val_embs_t, val_ids, id_to_group)
    base_score = baseline[metric]

    # Compute Rayleigh W for each reg
    Ws = {r: m_rayleigh(st, reg=r) for r in regs}

    # Generate evenly spaced dim counts
    step = max(1, d // n_points)
    dims = list(range(step, d, step))
    if dims[-1] != d:
        dims.append(d)

    # Evaluate each (reg, dim) combo, pick best reg per dim
    best_scores = {}
    pbar = tqdm(dims, desc="Scanning dimensions", disable=not verbose)
    for dim in pbar:
        best = -1.0
        for r in regs:
            W = Ws[r]
            p = val_embs_t @ W[:, :dim]
            sc = _eval(p, val_ids, id_to_group)[metric]
            if sc > best:
                best = sc
        best_scores[dim] = best
        pbar.set_postfix_str(f"dim={dim}, {metric}={best:.4f}")
    pbar.close()

    peak_score = max(best_scores.values())
    delta = peak_score - base_score

    if verbose:
        print(f'\nDim range scan (baseline {metric}={base_score:.4f}, ' f'peak={peak_score:.4f}, delta={delta:.4f})')
        print(f'{"dims":>6s}  {metric:>10s}  {"vs base":>8s}')
        print('-' * 28)
        for dim in dims:
            sc = best_scores[dim]
            print(f'{dim:>6d}  {sc:>10.4f}  {sc - base_score:>+8.4f}')

    # Find the range where score >= baseline + threshold * delta
    if delta <= 0:
        # No improvement from projection; return default fractions
        if verbose:
            print('No improvement over baseline, returning default fractions.')
        return (0.05, 0.1, 0.2, 0.5, 1.0)

    cutoff = base_score + threshold * delta
    useful_dims = [dim for dim in dims if best_scores[dim] >= cutoff]
    low = min(useful_dims)
    high = max(useful_dims)

    # Build fractions: 2-3 below useful range, 3-4 in the plateau, 1.0
    fracs = set()

    # Below useful range
    below_candidates = [dim for dim in dims if dim < low]
    if len(below_candidates) >= 2:
        fracs.add(below_candidates[0] / d)
        fracs.add(below_candidates[len(below_candidates) // 2] / d)
    elif below_candidates:
        fracs.add(below_candidates[0] / d)
    # Always add a small exploration point
    fracs.add(max(1, d // n_points) / d)

    # In the plateau: spread 3-4 points
    if high > low:
        plateau_step = (high - low) / 4
        for i in range(5):
            dim_val = low + plateau_step * i
            fracs.add(round(dim_val) / d)
    else:
        fracs.add(low / d)

    # Full dim
    fracs.add(1.0)

    # Sort and clamp
    sorted_fracs = sorted(f for f in fracs if 0 < f <= 1.0)

    if verbose:
        print(f'\nUseful range: dims {low}-{high} ' f'(fracs {low / d:.2f}-{high / d:.2f})')
        print(f'Returned dim_fractions: {tuple(sorted_fracs)}')

    return tuple(sorted_fracs)


def _pick_next(
    rng: Any, all_methods: list, method_queues: dict, method_evaluated: dict, method_best: dict, explore_fraction: float
) -> tuple | None:
    """Select next (method, n_dims) pair using explore-exploit scheduling."""
    active = [m for m in all_methods if method_queues[m]]
    if not active:
        return None
    unexplored = [m for m in active if method_evaluated[m] == 0]
    if unexplored:
        method = rng.choice(unexplored)
        result: tuple = method_queues[method].pop()
        return result
    if rng.random() < explore_fraction:
        method = rng.choice(active)
    else:
        scores_arr = torch.tensor([method_best[m] for m in active], dtype=DTYPE)
        scores_arr = scores_arr - scores_arr.max()
        probs = torch.exp(scores_arr * 20)
        probs = probs / probs.sum()
        probs_list = probs.tolist()
        method = active[rng.choices(range(len(active)), weights=probs_list, k=1)[0]]
    result2: tuple = method_queues[method].pop()
    return result2


def _build_summary(results: dict, all_methods: list, dim_tests: tuple, metric: str) -> dict:
    """Aggregate results dict into a method-level summary table."""
    summary: dict[str, dict] = {}
    for method_name in all_methods:
        method_results = {k: v for k, v in results.items() if k[0] == method_name}
        if not method_results:
            continue
        summary[method_name] = {}
        for n_dims in dim_tests:
            best_key = max(method_results, key=lambda k: method_results[k][n_dims][metric])
            summary[method_name][n_dims] = (best_key, method_results[best_key][n_dims])
    return summary


def _print_table(
    title: str,
    summary: dict,
    all_W: dict,
    all_methods: list,
    dim_tests: tuple,
    dim_fractions: tuple,
    d_full: int,
    metric: str,
    method_evaluated: dict,
    method_keys: dict,
    score_fn: Callable,
) -> None:
    """Print formatted ranking table to stdout."""
    dim_labels = []
    for d, f in zip(dim_tests, dim_fractions):
        if d is None:
            dim_labels.append(f"full({d_full})")
        else:
            dim_labels.append(f"{f:.0%}({d})")

    evaluated_methods = [m for m in all_methods if m in summary]
    sorted_methods = sorted(
        evaluated_methods, key=lambda m: max(summary[m][nd][1][metric] for nd in dim_tests), reverse=True
    )

    print(f"\n{title}")
    print(f"{'Method':<28s}", end="")
    for dl in dim_labels:
        print(f"  {dl:>10s}", end="")
    print(f"  {'configs':>8s}")
    print("-" * (28 + 12 * len(dim_tests) + 10))

    for method_name in sorted_methods:
        n_eval = method_evaluated[method_name]
        n_total_m = len(method_keys[method_name])
        print(f"{method_name:<28s}", end="")
        for n_dims in dim_tests:
            score = score_fn(summary, method_name, n_dims)
            print(f"  {score:>10.4f}", end="")
        print(f"  {n_eval:>3d}/{n_total_m:<3d}")


def evaluate_projections(
    all_W: dict,
    val_embs: torch.Tensor,
    val_ids: list,
    id_to_group: dict,
    test_embs: torch.Tensor | None = None,
    test_ids: list | None = None,
    test_id_to_group: dict | None = None,
    dim_fractions: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0),
    metric: str = 'MAP@50',
    explore_fraction: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
    eval_fn: Callable | None = None,
) -> tuple[dict, dict]:
    """Evaluate projections using explore-exploit scheduling.

    Supports ``KeyboardInterrupt`` for early stopping.
    """
    _eval = eval_fn if eval_fn is not None else evaluate_retrieval_fast
    rng = random.Random(seed)
    val_embs_t = to_torch(val_embs)
    test_embs_t = to_torch(test_embs) if test_embs is not None else None

    first_W = next(iter(all_W.values()))
    d_full = first_W.shape[1]
    dim_tests_list: list[int | None] = []
    for f in dim_fractions:
        if f >= 1.0:
            dim_tests_list.append(None)
        else:
            dim_tests_list.append(max(1, round(d_full * f)))
    dim_tests: tuple[int | None, ...] = tuple(dim_tests_list)

    method_keys = defaultdict(list)
    for key in all_W:
        method_keys[key[0]].append(key)
    for keys in method_keys.values():
        rng.shuffle(keys)

    results: dict[tuple, dict[int | None, dict[str, float]]] = {}
    method_best: dict[str, float] = defaultdict(float)
    method_evaluated: dict[str, int] = defaultdict(int)
    method_queues = {name: list(keys) for name, keys in method_keys.items()}
    all_methods = list(method_keys.keys())
    total = len(all_W)

    def eval_key(key: tuple) -> dict[int | None, dict[str, float]]:
        W = all_W[key]
        res_by_dim = {}
        for n_dims in dim_tests:
            p = val_embs_t @ W
            if n_dims is not None:
                p = p[:, :n_dims]
            scores = _eval(p, val_ids, id_to_group)
            res_by_dim[n_dims] = scores
        return res_by_dim

    pbar = tqdm(total=total, desc="Evaluating projections", disable=not verbose)
    global_best_score = 0.0
    global_best_key: tuple | None = None

    try:
        while True:
            key = _pick_next(rng, all_methods, method_queues, method_evaluated, method_best, explore_fraction)
            if key is None:
                break

            method_name = key[0]
            cfg_str = ', '.join(key[1:])
            pbar.set_postfix_str(
                f"{method_name} | {cfg_str} | "
                f"best={global_best_score:.4f} ({global_best_key[0] if global_best_key else '?'})"
            )

            res_by_dim = eval_key(key)
            results[key] = res_by_dim

            best_across_dims = max(res_by_dim[nd][metric] for nd in dim_tests)
            method_evaluated[method_name] += 1
            if best_across_dims > method_best[method_name]:
                method_best[method_name] = best_across_dims
            if best_across_dims > global_best_score:
                global_best_score = best_across_dims
                global_best_key = key

            pbar.update(1)

    except KeyboardInterrupt:
        if verbose:
            print(f"\n\nInterrupted! Evaluated {len(results)}/{total} projections.")

    pbar.close()

    summary = _build_summary(results, all_methods, dim_tests, metric)

    if verbose:

        def val_score_fn(summary: dict, method_name: str, n_dims: int | None) -> float:
            return summary[method_name][n_dims][1][metric]  # type: ignore[no-any-return]

        _print_table(
            "VAL SET",
            summary,
            all_W,
            all_methods,
            dim_tests,
            dim_fractions,
            d_full,
            metric,
            method_evaluated,
            method_keys,
            val_score_fn,
        )

        if test_embs_t is not None and test_ids is not None:
            _test_id_to_group = test_id_to_group if test_id_to_group is not None else id_to_group

            def test_score_fn(summary: dict, method_name: str, n_dims: int | None) -> float:
                best_key = summary[method_name][n_dims][0]
                W = all_W[best_key]
                p = test_embs_t @ W
                if n_dims is not None:
                    p = p[:, :n_dims]
                return _eval(p, test_ids, _test_id_to_group)[metric]

            _print_table(
                "TEST SET",
                summary,
                all_W,
                all_methods,
                dim_tests,
                dim_fractions,
                d_full,
                metric,
                method_evaluated,
                method_keys,
                test_score_fn,
            )

    return results, summary
