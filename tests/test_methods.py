"""Tests for projection methods."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.methods import (
    m_asym_rayleigh,
    m_mse,
    m_ray_contr_mse,
    m_ray_contr_mse_neg,
    m_ray_mse,
    m_rayleigh,
    m_uber,
)
from litfit.stats import compute_all_stats, compute_stats


@pytest.fixture
def make_st():
    """Create synthetic positive pair statistics."""

    def _make(d: int = 8, n: int = 20) -> dict:
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // (n // 4) for i in range(n)}  # 4 groups
        return compute_stats(embs, ids, id_to_group)

    return _make


@pytest.fixture
def make_st_with_neg():
    """Create both positive and negative pair statistics."""

    def _make(d: int = 8, n: int = 20) -> tuple:
        torch.manual_seed(42)
        embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
        ids = list(range(n))
        id_to_group = {i: i // (n // 4) for i in range(n)}  # 4 groups
        st, neg = compute_all_stats(embs, ids, id_to_group)
        return st, neg

    return _make


class TestMethodShapes:
    """Test that projection methods return correct shapes."""

    def test_m_rayleigh_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        assert W.shape == (8, 8), f"Expected (8, 8), got {W.shape}"
        assert W.dtype == DTYPE
        assert W.device == DEVICE

    def test_m_mse_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_mse(st)
        assert W.shape == (8, 8), f"Expected (8, 8), got {W.shape}"

    def test_m_ray_mse_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_ray_mse(st)
        assert W.shape == (8, 8), f"Expected (8, 8), got {W.shape}"

    def test_m_asym_rayleigh_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_asym_rayleigh(st)
        assert W.shape == (8, 8), f"Expected (8, 8), got {W.shape}"


class TestUberMethods:
    """Test that methods returning (W, bias) tuples work correctly."""

    def test_m_uber_returns_tuple(self, make_st):
        st = make_st(d=8, n=20)
        result = m_uber(st)
        assert isinstance(result, tuple)
        assert len(result) == 2
        W, bias = result
        assert W.shape == (8, 8)
        assert bias.shape == (8,)


class TestNegativeMethods:
    """Test methods that require negative pair statistics."""

    def test_m_ray_contr_mse_requires_neg(self, make_st):
        st = make_st(d=8, n=20)
        with pytest.raises(ValueError, match="negative pair statistics"):
            m_ray_contr_mse(st, neg=None)

    def test_m_ray_contr_mse_with_neg(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        W = m_ray_contr_mse(st, neg=neg)
        assert W.shape == (8, 8)

    def test_m_ray_contr_mse_neg_with_neg(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        W = m_ray_contr_mse_neg(st, neg=neg)
        assert W.shape == (8, 8)


class TestMethodProperties:
    """Test mathematical properties of projection methods."""

    def test_rayleigh_eigenvector_orthogonality(self, make_st):
        """Rayleigh eigenvectors are B-orthonormal where B = Sigma_total + reg*I."""
        reg = 0.1  # default reg in m_rayleigh
        st = make_st(d=8, n=20)
        W = m_rayleigh(st, reg=reg)
        # _eigh(Sigma_cross, B) returns evec s.t. W^T @ B @ W = I
        B = st['Sigma_total'] + reg * torch.eye(8, device=DEVICE, dtype=DTYPE)
        gram = W.T @ B @ W
        expected = torch.eye(8, device=DEVICE, dtype=DTYPE)
        assert torch.allclose(gram, expected, atol=1e-5)

    def test_methods_dont_return_nan(self, make_st):
        """All methods should return finite values."""
        st = make_st(d=8, n=20)
        methods = [m_rayleigh, m_mse, m_ray_mse, m_asym_rayleigh]
        for method in methods:
            W = method(st)
            assert torch.isfinite(W).all(), f"{method.__name__} returned NaN"
