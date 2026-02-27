"""Tests for LazyProjectionDict and dispatch functions."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.dispatch import LazyProjectionDict, generate_fast_projections
from litfit.stats import compute_stats


@pytest.fixture
def make_st():
    """Create synthetic positive pair statistics."""
    torch.manual_seed(42)
    n = 20
    d = 8
    embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
    ids = list(range(n))
    id_to_group = {i: i // (n // 4) for i in range(n)}  # 4 groups
    return compute_stats(embs, ids, id_to_group)


class TestLazyProjectionDict:
    """Test LazyProjectionDict container."""

    def test_lazy_dict_creation(self):
        """Can create an empty LazyProjectionDict."""
        lpd = LazyProjectionDict()
        assert len(lpd) == 0

    def test_lazy_dict_set_and_get(self):
        """Can set and retrieve projections."""
        lpd = LazyProjectionDict()
        key = ('m_test', 'reg=0.1')
        W_original = torch.randn(8, 8, device=DEVICE, dtype=DTYPE)

        # Mock function that returns the original W
        def dummy_fn(**kwargs):
            return W_original

        lpd._set(key, dummy_fn, {}, None, {})
        assert key in lpd

        W_retrieved = lpd[key]
        assert torch.allclose(W_retrieved, W_original)

    def test_lazy_dict_recomputes_on_access(self):
        """LazyProjectionDict recomputes on each access (doesn't cache)."""
        lpd = LazyProjectionDict()
        key = ('m_test', 'reg=0.1')
        call_count = [0]

        def counting_fn(**kwargs):
            call_count[0] += 1
            return torch.randn(8, 8, device=DEVICE, dtype=DTYPE)

        lpd._set(key, counting_fn, {}, None, {})

        # First access
        _ = lpd[key]
        assert call_count[0] == 1

        # Second access should recompute
        _ = lpd[key]
        assert call_count[0] == 2

    def test_lazy_dict_len(self):
        """len() returns number of stored closures."""
        lpd = LazyProjectionDict()
        assert len(lpd) == 0

        for i in range(3):
            key = ('m_test', f'reg={i}')
            lpd._set(key, lambda **kw: torch.randn(8, 8, device=DEVICE, dtype=DTYPE), {}, None, {})

        assert len(lpd) == 3

    def test_lazy_dict_keys(self):
        """keys() returns all stored keys."""
        lpd = LazyProjectionDict()
        keys = [('m_test', f'reg={i}') for i in range(3)]

        for key in keys:
            lpd._set(key, lambda **kw: torch.randn(8, 8, device=DEVICE, dtype=DTYPE), {}, None, {})

        assert set(lpd.keys()) == set(keys)

    def test_lazy_dict_iteration(self):
        """Can iterate over keys."""
        lpd = LazyProjectionDict()
        keys = [('m_test', f'reg={i}') for i in range(3)]

        for key in keys:
            lpd._set(key, lambda **kw: torch.randn(8, 8, device=DEVICE, dtype=DTYPE), {}, None, {})

        retrieved_keys = list(lpd)
        assert set(retrieved_keys) == set(keys)

    def test_lazy_dict_contains(self):
        """__contains__ works correctly."""
        lpd = LazyProjectionDict()
        key = ('m_test', 'reg=0.1')

        assert key not in lpd
        lpd._set(key, lambda **kw: torch.randn(8, 8, device=DEVICE, dtype=DTYPE), {}, None, {})
        assert key in lpd


class TestGenerateFastProjections:
    """Test generate_fast_projections function."""

    def test_generate_fast_projections_returns_dict(self, make_st):
        """generate_fast_projections returns a dict-like object."""
        result = generate_fast_projections(make_st, lazy=False, verbose=False)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_generate_fast_projections_returns_lazy_dict(self, make_st):
        """generate_fast_projections returns LazyProjectionDict when lazy=True."""
        result = generate_fast_projections(make_st, lazy=True, verbose=False)
        assert isinstance(result, LazyProjectionDict)
        assert len(result) > 0

    def test_generate_fast_projections_with_neg(self, make_st):
        """generate_fast_projections includes contrastive methods when neg provided."""
        from litfit.stats import compute_all_stats

        # Create st and neg
        torch.manual_seed(42)
        n = 20
        d = 8
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // (n // 4) for i in range(n)}
        st, neg = compute_all_stats(embs, ids, id_to_group)

        # Without neg: some number of configs
        result_no_neg = generate_fast_projections(st, neg=None, lazy=False, verbose=False)
        count_no_neg = len(result_no_neg)

        # With neg: should have more configs (contrastive methods added)
        result_with_neg = generate_fast_projections(st, neg=neg, lazy=False, verbose=False)
        count_with_neg = len(result_with_neg)

        assert count_with_neg > count_no_neg
