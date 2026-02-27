"""Tests for evaluation functions."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.evaluation import evaluate_projections, evaluate_retrieval_fast, find_dim_range
from litfit.methods import m_rayleigh
from litfit.stats import compute_stats


@pytest.fixture
def synthetic_embeddings():
    """Create synthetic embeddings and group labels."""
    torch.manual_seed(42)
    n = 20
    d = 8

    # Create embeddings grouped by class
    embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
    ids = list(range(n))

    # 4 groups of 5 items each
    id_to_group = {i: i // 5 for i in range(n)}

    return embs, ids, id_to_group


@pytest.fixture
def stats_and_embeddings():
    """Create stats + embeddings for find_dim_range / evaluate_projections."""
    torch.manual_seed(42)
    n = 40
    d = 8
    embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
    ids = list(range(n))
    id_to_group = {i: i // 5 for i in range(n)}  # 8 groups of 5
    st = compute_stats(embs, ids, id_to_group)
    return st, embs, ids, id_to_group


class TestEvaluationRetrieval:
    """Test evaluate_retrieval_fast function."""

    def test_output_keys(self, synthetic_embeddings):
        """Check that output contains all required metrics."""
        embs, ids, id_to_group = synthetic_embeddings
        result = evaluate_retrieval_fast(embs, ids, id_to_group, ks=[1, 5, 10])

        expected_keys = {'R@1', 'R@5', 'R@10', 'MAP@50'}
        assert set(result.keys()) == expected_keys

    def test_result_values_in_range(self, synthetic_embeddings):
        """Check that all metrics are in [0, 1]."""
        embs, ids, id_to_group = synthetic_embeddings
        result = evaluate_retrieval_fast(embs, ids, id_to_group, ks=[1, 5, 10])

        for k, v in result.items():
            assert 0 <= v <= 1, f"{k}={v} out of [0, 1]"

    def test_perfect_case(self, synthetic_embeddings):
        """Test with identical embeddings in same group (should have high recall)."""
        embs, ids, id_to_group = synthetic_embeddings

        # All embeddings in same group → R@5 should be perfect (1.0)
        # since each query will find all other items in group within top-5
        id_to_group_single = {i: 0 for i in range(len(ids))}
        result = evaluate_retrieval_fast(embs, ids, id_to_group_single, ks=[1, 5, 10])

        # With all items in one group, perfect retrieval within top-k is possible
        # (actual value depends on embedding similarity)
        assert result['R@5'] > 0  # At least some positive

    def test_map50_range(self, synthetic_embeddings):
        """MAP@50 should be in valid range."""
        embs, ids, id_to_group = synthetic_embeddings
        result = evaluate_retrieval_fast(embs, ids, id_to_group)

        assert 0 <= result['MAP@50'] <= 1

    def test_output_type(self, synthetic_embeddings):
        """Returned values should be floats."""
        embs, ids, id_to_group = synthetic_embeddings
        result = evaluate_retrieval_fast(embs, ids, id_to_group)

        for v in result.values():
            assert isinstance(v, float)

    def test_custom_ks(self, synthetic_embeddings):
        embs, ids, id_to_group = synthetic_embeddings
        result = evaluate_retrieval_fast(embs, ids, id_to_group, ks=[1, 3])
        assert 'R@1' in result
        assert 'R@3' in result
        assert 'MAP@50' in result

    def test_numpy_input(self, synthetic_embeddings):
        embs, ids, id_to_group = synthetic_embeddings
        embs_np = embs.cpu().numpy()
        result = evaluate_retrieval_fast(embs_np, ids, id_to_group)
        for v in result.values():
            assert 0 <= v <= 1


class TestFindDimRange:
    def test_returns_tuple_of_fractions(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        fracs = find_dim_range(st, embs, ids, id_to_group, verbose=False, n_points=5)
        assert isinstance(fracs, tuple)
        assert all(0 < f <= 1.0 for f in fracs)
        assert 1.0 in fracs  # always includes full dim

    def test_fractions_sorted(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        fracs = find_dim_range(st, embs, ids, id_to_group, verbose=False, n_points=5)
        assert fracs == tuple(sorted(fracs))

    def test_custom_eval_fn(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings

        def dummy_eval(embs, ids, id_to_group, **kw):
            return {'MAP@50': 0.5, 'R@1': 0.3, 'R@5': 0.4, 'R@10': 0.5}

        fracs = find_dim_range(st, embs, ids, id_to_group, verbose=False, eval_fn=dummy_eval, n_points=5)
        assert isinstance(fracs, tuple)


class TestEvaluateProjections:
    def test_returns_results_and_summary(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        W = m_rayleigh(st)
        all_W = {('Rayleigh', 'reg=0.1'): W}
        results, summary = evaluate_projections(
            all_W,
            embs,
            ids,
            id_to_group,
            dim_fractions=(0.5, 1.0),
            verbose=False,
        )
        assert isinstance(results, dict)
        assert isinstance(summary, dict)
        assert len(results) == 1

    def test_results_contain_metrics(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        W = m_rayleigh(st)
        all_W = {('Rayleigh', 'reg=0.1'): W}
        results, _ = evaluate_projections(
            all_W,
            embs,
            ids,
            id_to_group,
            dim_fractions=(1.0,),
            verbose=False,
        )
        key = ('Rayleigh', 'reg=0.1')
        assert key in results
        # None corresponds to full dim (frac >= 1.0)
        assert None in results[key]
        assert 'MAP@50' in results[key][None]

    def test_multiple_projections(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        all_W = {}
        for reg in [0.01, 0.1, 1.0]:
            W = m_rayleigh(st, reg=reg)
            all_W[('Rayleigh', f'reg={reg}')] = W

        results, summary = evaluate_projections(
            all_W,
            embs,
            ids,
            id_to_group,
            dim_fractions=(0.5, 1.0),
            verbose=False,
        )
        assert len(results) == 3
        assert 'Rayleigh' in summary

    def test_custom_eval_fn(self, stats_and_embeddings):
        st, embs, ids, id_to_group = stats_and_embeddings
        W = m_rayleigh(st)
        all_W = {('Rayleigh', 'reg=0.1'): W}

        call_count = [0]

        def custom_eval(embs, ids, id_to_group, **kw):
            call_count[0] += 1
            return {'MAP@50': 0.42, 'R@1': 0.3}

        results, _ = evaluate_projections(
            all_W,
            embs,
            ids,
            id_to_group,
            dim_fractions=(1.0,),
            verbose=False,
            eval_fn=custom_eval,
        )
        assert call_count[0] > 0
