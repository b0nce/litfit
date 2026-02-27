"""Tests for projection methods."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.methods import (
    _asym_refine,
    _eigh,
    _importance_sort,
    _mse_refine,
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestEigh:
    def test_standard_eigendecomposition(self):
        torch.manual_seed(0)
        A = torch.randn(4, 4, device=DEVICE, dtype=DTYPE)
        A = A @ A.T  # symmetric PSD
        ev, evec = _eigh(A)
        # A @ evec = evec @ diag(ev)
        recon = evec @ torch.diag(ev) @ evec.T
        assert torch.allclose(A, recon, atol=1e-5)

    def test_generalized_eigendecomposition(self):
        torch.manual_seed(0)
        A = torch.randn(4, 4, device=DEVICE, dtype=DTYPE)
        A = A @ A.T
        B = torch.eye(4, device=DEVICE, dtype=DTYPE) * 2 + 0.1 * A
        ev, evec = _eigh(A, B)
        # A @ v = lambda * B @ v for each eigenvector
        for i in range(4):
            lhs = A @ evec[:, i]
            rhs = ev[i] * B @ evec[:, i]
            assert torch.allclose(lhs, rhs, atol=1e-4)


class TestImportanceSort:
    def test_columns_reordered(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        W_sorted = _importance_sort(W, st)
        assert W_sorted.shape == W.shape
        # Importance should be decreasing
        Cf = W_sorted.T @ st['Sigma_cross'] @ W_sorted
        Sf = W_sorted.T @ st['Sigma_total'] @ W_sorted
        imp = torch.diag(Cf) / (torch.diag(Sf) + 1e-8)
        for i in range(len(imp) - 1):
            assert imp[i] >= imp[i + 1] - 1e-6


class TestMseRefine:
    def test_output_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        W2 = _mse_refine(W, st, reg_mse=0.1)
        assert W2.shape == W.shape

    def test_finite_output(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        W2 = _mse_refine(W, st, reg_mse=0.1)
        assert torch.isfinite(W2).all()


class TestAsymRefine:
    def test_output_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        W2 = _asym_refine(W, st, reg_refine=0.1)
        assert W2.shape == W.shape

    def test_finite_output(self, make_st):
        st = make_st(d=8, n=20)
        W = m_rayleigh(st)
        W2 = _asym_refine(W, st, reg_refine=0.1)
        assert torch.isfinite(W2).all()


# ---------------------------------------------------------------------------
# All projection methods: shape + finiteness
# ---------------------------------------------------------------------------


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

    def test_m_asym_ray_mse_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_asym_ray_mse(st)
        assert W.shape == (8, 8)

    def test_m_ray_asym_refine_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_ray_asym_refine(st)
        assert W.shape == (8, 8)

    def test_m_ray_mse_asym_refine_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_ray_mse_asym_refine(st)
        assert W.shape == (8, 8)

    def test_m_ray_asym_refine_mse_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_ray_asym_refine_mse(st)
        assert W.shape == (8, 8)

    def test_m_ray_iterate_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_ray_iterate(st)
        assert W.shape == (8, 8)

    def test_m_split_rank_ray_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_split_rank_ray(st)
        assert W.shape == (8, 8)

    def test_m_split_rank_ray_mse_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_split_rank_ray_mse(st)
        assert W.shape == (8, 8)

    def test_m_split_rank_ray_iterate_shape(self, make_st):
        st = make_st(d=8, n=20)
        W = m_split_rank_ray_iterate(st)
        assert W.shape == (8, 8)


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

    def test_m_uber_neg_returns_tuple(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        result = m_uber_neg(st, neg=neg)
        assert isinstance(result, tuple)
        W, bias = result
        assert W.shape == (8, 8)
        assert bias.shape == (8,)

    def test_m_uber_contr_returns_tuple(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        result = m_uber_contr(st, neg=neg)
        assert isinstance(result, tuple)
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

    def test_m_resid_guided_shape(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        W = m_resid_guided(st, neg=neg)
        assert W.shape == (8, 8)

    def test_m_resid_guided_requires_neg(self, make_st):
        st = make_st(d=8, n=20)
        with pytest.raises(ValueError, match="negative pair statistics"):
            m_resid_guided(st, neg=None)

    def test_m_uber_neg_requires_neg(self, make_st):
        st = make_st(d=8, n=20)
        with pytest.raises(ValueError, match="negative pair statistics"):
            m_uber_neg(st, neg=None)

    def test_m_uber_contr_requires_neg(self, make_st):
        st = make_st(d=8, n=20)
        with pytest.raises(ValueError, match="negative pair statistics"):
            m_uber_contr(st, neg=None)


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

    def test_all_basic_methods_finite(self, make_st):
        """All basic methods should return finite values."""
        st = make_st(d=8, n=20)
        methods = [
            m_rayleigh,
            m_mse,
            m_ray_mse,
            m_asym_rayleigh,
            m_asym_ray_mse,
            m_ray_asym_refine,
            m_ray_mse_asym_refine,
            m_ray_asym_refine_mse,
            m_ray_iterate,
            m_split_rank_ray,
            m_split_rank_ray_mse,
            m_split_rank_ray_iterate,
        ]
        for method in methods:
            W = method(st)
            assert torch.isfinite(W).all(), f"{method.__name__} returned NaN/Inf"

    def test_all_neg_methods_finite(self, make_st_with_neg):
        """All neg methods should return finite values."""
        st, neg = make_st_with_neg(d=8, n=20)
        for method in [m_ray_contr_mse, m_ray_contr_mse_neg, m_resid_guided]:
            W = method(st, neg=neg)
            assert torch.isfinite(W).all(), f"{method.__name__} returned NaN/Inf"

    def test_uber_methods_finite(self, make_st_with_neg):
        st, neg = make_st_with_neg(d=8, n=20)
        for method in [m_uber_neg, m_uber_contr]:
            W, bias = method(st, neg=neg)
            assert torch.isfinite(W).all(), f"{method.__name__} W has NaN/Inf"
            assert torch.isfinite(bias).all(), f"{method.__name__} bias has NaN/Inf"

    def test_different_reg_values(self, make_st):
        """Methods should work with various regularization values."""
        st = make_st(d=8, n=20)
        for reg in [0.001, 0.01, 0.1, 1.0, 5.0]:
            W = m_rayleigh(st, reg=reg)
            assert torch.isfinite(W).all()

    def test_split_rank_fractions(self, make_st):
        """split_rank_ray should work with various frac values."""
        st = make_st(d=8, n=20)
        for frac in [0.15, 0.3, 0.5, 0.7, 1.0]:
            W = m_split_rank_ray(st, frac_cross=frac, frac_total=frac)
            assert W.shape == (8, 8)
            assert torch.isfinite(W).all()

    def test_ray_iterate_rounds(self, make_st):
        """ray_iterate should work with different round counts."""
        st = make_st(d=8, n=20)
        for n_rounds in [1, 3, 5]:
            W = m_ray_iterate(st, n_rounds=n_rounds)
            assert W.shape == (8, 8)
            assert torch.isfinite(W).all()
