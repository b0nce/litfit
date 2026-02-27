from collections.abc import Callable, Iterator
from typing import Any

import torch
from tqdm.auto import tqdm

from .device import DEVICE
from .methods import (
    m_asym_ray_mse,
    m_asym_rayleigh,
    m_mse,
    m_ray_asym_refine,
    m_ray_asym_refine_mse,
    m_ray_contr_mse,
    m_ray_contr_mse_neg,
    m_ray_iterate,
    m_ray_mse,
    m_ray_mse_asym_refine,
    m_rayleigh,
    m_resid_guided,
    m_split_rank_ray,
    m_split_rank_ray_iterate,
    m_split_rank_ray_mse,
    m_uber,
    m_uber_contr,
    m_uber_neg,
)
from .stats import _check_neg, _check_st

_REGS = [0.01, 0.1, 1.0, 5.0]
_REGS_SHORT = [0.01, 0.1, 1.0]


class LazyProjectionDict:
    """Dict-like container that computes projection matrices on demand.

    Stores closures instead of computed matrices so that only one projection
    is materialised at a time, keeping memory usage low.  Matrices are **not**
    cached — each access recomputes the projection so the tensor can be GC'd
    after use.
    """

    def __init__(self) -> None:
        self._closures: dict[tuple, tuple[Callable, dict[str, Any]]] = {}

    def _set(self, key: tuple, fn: Callable, st: Any, neg: Any, cfg: dict) -> None:
        self._closures[key] = (fn, {"st": st, "neg": neg, **cfg})

    def __getitem__(self, key: tuple) -> torch.Tensor:
        fn, kwargs = self._closures[key]
        out = fn(**kwargs)
        result = out[0] if isinstance(out, tuple) else out
        return result  # type: ignore[no-any-return]

    def __contains__(self, key: object) -> bool:
        return key in self._closures

    def __len__(self) -> int:
        return len(self._closures)

    def __iter__(self) -> Iterator[tuple]:
        return iter(self._closures)

    def keys(self) -> Any:
        return self._closures.keys()

    def values(self) -> Iterator[torch.Tensor]:
        for key in self._closures:
            yield self[key]

    def items(self) -> Iterator[tuple[tuple, torch.Tensor]]:
        for key in self._closures:
            yield key, self[key]


def generate_fast_projections(
    st: dict,
    neg: dict | None = None,
    verbose: bool = True,
    lazy: bool = False,
) -> dict[tuple, torch.Tensor] | LazyProjectionDict:
    """Generate a curated subset of ~40 projections from the top-performing methods.

    Covers Rayleigh, Ray→AsymRef, Ray→AsymRef→MSE, Ray→MSE→AsymRef, Ray→MSE,
    and SplitRankRay with trimmed hyperparameter grids.  When *neg* is provided,
    adds a handful of contrastive configs.  Roughly 30× fewer configs than
    ``generate_all_projections`` with minimal quality loss based on benchmarks.
    """
    _check_st(st)
    results: dict[tuple, torch.Tensor] = {}
    all_jobs: list[tuple[str, Callable, dict[str, Any]]] = []

    def _register(name: str, fn: Callable, configs: list[dict[str, Any]], needs_neg: bool = False) -> None:
        if needs_neg and neg is None:
            return
        if needs_neg:
            _check_neg(neg)
        for cfg in configs:
            all_jobs.append((name, fn, cfg))

    # --- top methods from benchmarks, trimmed grids ---

    _register('Rayleigh', m_rayleigh, [{'reg': r} for r in _REGS])

    _register(
        'Ray→AsymRef', m_ray_asym_refine, [{'reg': r, 'reg_refine': rr} for r in _REGS_SHORT for rr in [0.1, 1.0]]
    )

    _register(
        'Ray→AsymRef→MSE',
        m_ray_asym_refine_mse,
        [{'reg': r, 'reg_refine': rr, 'reg_mse': rm} for r in [0.01, 0.1] for rr in [0.1, 1.0] for rm in [0.1, 1.0]],
    )

    _register(
        'Ray→MSE→AsymRef',
        m_ray_mse_asym_refine,
        [{'reg': r, 'reg_mse': rm, 'reg_refine': rr} for r in [0.01, 0.1] for rm in [0.1, 1.0] for rr in [0.1, 1.0]],
    )

    _register('Ray→MSE', m_ray_mse, [{'reg': r, 'reg_mse': rm} for r in [0.01, 0.1] for rm in [0.1, 1.0]])

    _register(
        'SplitRankRay',
        m_split_rank_ray,
        [{'frac_cross': fc, 'frac_total': ft, 'reg': r} for fc in [0.3, 0.5] for ft in [0.3, 0.5] for r in [0.01, 0.1]],
    )

    # --- contrastive (only when neg is available) ---

    _register(
        'RayContr→MSE',
        m_ray_contr_mse,
        [{'reg': r, 'alpha': 0.1, 'beta': 0.1, 'reg_mse': rm} for r in [0.01, 0.1] for rm in [0.1, 1.0]],
        needs_neg=True,
    )

    _register(
        'RayContr→MSE+neg',
        m_ray_contr_mse_neg,
        [
            {'reg': r, 'alpha': 0.1, 'beta': 0.1, 'alpha_mse': 0.1, 'reg_mse': rm}
            for r in [0.01, 0.1]
            for rm in [0.1, 1.0]
        ],
        needs_neg=True,
    )

    if lazy:
        lazy_results = LazyProjectionDict()
        for name, fn, cfg in all_jobs:
            key = (name,) + tuple(f"{k}={v}" for k, v in sorted(cfg.items()))
            lazy_results._set(key, fn, st, neg, cfg)
        if verbose:
            print(f"Total: {len(lazy_results)} projections (lazy, fast)")
            print(f"Device: {DEVICE}")
        return lazy_results

    n_failed = 0
    pbar = tqdm(all_jobs, desc="Generating projections (fast)", disable=not verbose)
    for name, fn, cfg in pbar:
        key = (name,) + tuple(f"{k}={v}" for k, v in sorted(cfg.items()))
        cfg_short = ', '.join(f"{k}={v}" for k, v in cfg.items())
        pbar.set_postfix_str(f"{name} | {cfg_short}")
        try:
            out = fn(st, neg=neg, **cfg)
            W = out[0] if isinstance(out, tuple) else out
            results[key] = W
        except Exception as e:
            n_failed += 1
            if verbose:
                tqdm.write(f"  FAILED {key}: {e}")

    if verbose:
        print(f"Total: {len(results)} projections, {n_failed} failed")
        print(f"Device: {DEVICE}")

    return results


def generate_all_projections(
    st: dict,
    neg: dict | None = None,
    include_neg_methods: bool = True,
    verbose: bool = True,
    lazy: bool = False,
) -> dict[tuple, torch.Tensor] | LazyProjectionDict:
    _check_st(st)
    results: dict[tuple, torch.Tensor] = {}
    all_jobs: list[tuple[str, Callable, dict[str, Any]]] = []

    def _register(name: str, fn: Callable, configs: list[dict[str, Any]], needs_neg: bool = False) -> None:
        if needs_neg and (neg is None or not include_neg_methods):
            return
        if needs_neg:
            _check_neg(neg)
        for cfg in configs:
            all_jobs.append((name, fn, cfg))

    _register('Rayleigh', m_rayleigh, [{'reg': r} for r in _REGS])
    _register('MSE', m_mse, [{'reg': r} for r in _REGS])
    _register('Ray→MSE', m_ray_mse, [{'reg': r, 'reg_mse': rm} for r in _REGS for rm in _REGS])
    _register('AsymRayleigh', m_asym_rayleigh, [{'reg': r} for r in _REGS])
    _register('Asym→MSE', m_asym_ray_mse, [{'reg': r, 'reg_mse': rm} for r in _REGS for rm in _REGS])
    _register('Ray→AsymRef', m_ray_asym_refine, [{'reg': r, 'reg_refine': rr} for r in _REGS for rr in _REGS])
    _register(
        'Ray→MSE→AsymRef',
        m_ray_mse_asym_refine,
        [{'reg': r, 'reg_mse': rm, 'reg_refine': rr} for r in _REGS_SHORT for rm in _REGS for rr in _REGS],
    )
    _register(
        'Ray→AsymRef→MSE',
        m_ray_asym_refine_mse,
        [{'reg': r, 'reg_refine': rr, 'reg_mse': rm} for r in _REGS_SHORT for rr in _REGS for rm in _REGS],
    )
    _register(
        'Ray→Iterate',
        m_ray_iterate,
        [{'reg': r, 'reg_mse': rm, 'reg_refine': rr} for r in _REGS_SHORT for rm in _REGS for rr in _REGS],
    )
    _register(
        'SplitRankRay',
        m_split_rank_ray,
        [
            {'frac_cross': fc, 'frac_total': ft, 'reg': r}
            for fc in [0.15, 0.3, 0.5, 0.7, 1.0]
            for ft in [0.15, 0.3, 0.5, 0.7, 1.0]
            for r in [0.01, 0.1]
        ],
    )
    _register(
        'SplitRankRay→MSE',
        m_split_rank_ray_mse,
        [
            {'frac_cross': fc, 'frac_total': ft, 'reg': r, 'reg_mse': rm}
            for fc in [0.15, 0.3, 0.5, 0.7]
            for ft in [0.15, 0.3, 0.5, 0.7]
            for r in [0.01, 0.1]
            for rm in [0.1, 1.0, 5.0]
        ],
    )
    _register(
        'SplitRankRay→Iterate',
        m_split_rank_ray_iterate,
        [
            {'frac_cross': fc, 'frac_total': ft, 'reg': r, 'reg_mse': rm, 'reg_refine': rr}
            for fc in [0.15, 0.3, 0.5, 0.7]
            for ft in [0.15, 0.3, 0.5, 0.7]
            for r in [0.01, 0.1]
            for rm in [0.1, 1.0]
            for rr in [0.1, 1.0]
        ],
    )
    _register(
        'Uber',
        m_uber,
        [
            {'alpha': a, 'reg': r, 'frac_low': fl}
            for a in [0.0, 0.3, 0.5]
            for r in [1e-4, 1e-3, 0.01]
            for fl in [0.15, 0.3, 0.5]
        ],
    )

    _register(
        'RayContr→MSE',
        m_ray_contr_mse,
        [
            {'reg': r, 'alpha': a, 'beta': b, 'reg_mse': rm}
            for r in _REGS_SHORT
            for a in [0.05, 0.1, 0.3]
            for b in [0.05, 0.1, 0.3]
            for rm in [0.1, 1.0]
        ],
        needs_neg=True,
    )
    _register(
        'RayContr→MSE+neg',
        m_ray_contr_mse_neg,
        [
            {'reg': r, 'alpha': a, 'beta': b, 'alpha_mse': am, 'reg_mse': rm}
            for r in [0.01, 0.1]
            for a in [0.05, 0.1, 0.3]
            for b in [0.05, 0.1, 0.3]
            for am in [0.0, 0.1, 0.3]
            for rm in [0.1, 1.0]
        ],
        needs_neg=True,
    )
    _register(
        'ResidGuided',
        m_resid_guided,
        [
            {'reg': r, 'gamma': g, 'reg_mse': rm}
            for r in [0.01, 0.1]
            for g in [0.1, 0.3, 0.5, 1.0]
            for rm in [0.01, 0.1, 0.5, 1.0]
        ],
        needs_neg=True,
    )
    _register(
        'Uber+neg',
        m_uber_neg,
        [
            {'alpha': a, 'reg': r, 'frac_low': fl, 'beta_neg': bn}
            for a in [0.0, 0.3, 0.5]
            for r in [1e-4, 1e-3]
            for fl in [0.15, 0.3, 0.5]
            for bn in [0.05, 0.1, 0.3]
        ],
        needs_neg=True,
    )
    _register(
        'Uber_contr',
        m_uber_contr,
        [
            {'alpha': a, 'reg': r, 'frac_low': fl, 'alpha_neg': an, 'beta_neg': bn}
            for a in [0.0, 0.3, 0.5]
            for r in [1e-4, 1e-3]
            for fl in [0.15, 0.3, 0.5]
            for an in [0.05, 0.1, 0.3]
            for bn in [0.05, 0.1]
        ],
        needs_neg=True,
    )

    if lazy:
        lazy_results = LazyProjectionDict()
        for name, fn, cfg in all_jobs:
            key = (name,) + tuple(f"{k}={v}" for k, v in sorted(cfg.items()))
            lazy_results._set(key, fn, st, neg, cfg)
        if verbose:
            print(f"Total: {len(lazy_results)} projections (lazy)")
            print(f"Device: {DEVICE}")
        return lazy_results

    n_failed = 0
    pbar = tqdm(all_jobs, desc="Generating projections", disable=not verbose)
    for name, fn, cfg in pbar:
        key = (name,) + tuple(f"{k}={v}" for k, v in sorted(cfg.items()))
        cfg_short = ', '.join(f"{k}={v}" for k, v in cfg.items())
        pbar.set_postfix_str(f"{name} | {cfg_short}")
        try:
            out = fn(st, neg=neg, **cfg)
            W = out[0] if isinstance(out, tuple) else out
            results[key] = W
        except Exception as e:
            n_failed += 1
            if verbose:
                tqdm.write(f"  FAILED {key}: {e}")

    if verbose:
        print(f"Total: {len(results)} projections, {n_failed} failed")
        print(f"Device: {DEVICE}")

    return results
