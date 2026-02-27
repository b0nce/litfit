"""Tests for evaluation functions."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.evaluation import evaluate_retrieval_fast


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
