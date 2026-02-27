"""Tests for compute_stats, compute_all_stats, and validation helpers."""

import pytest
import torch

from litfit.device import DEVICE, DTYPE
from litfit.stats import _check_neg, _check_st, compute_all_stats, compute_stats


@pytest.fixture
def simple_data():
    """Simple test data with known group structure."""
    torch.manual_seed(42)
    n, d = 20, 8
    embs = torch.randn(n, d, device=DEVICE, dtype=DTYPE)
    ids = list(range(n))
    id_to_group = {i: i // 5 for i in range(n)}  # 4 groups of 5
    return embs, ids, id_to_group


# ---------------------------------------------------------------------------
# _check_st / _check_neg
# ---------------------------------------------------------------------------


class TestCheckSt:
    def test_valid_st(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group)
        _check_st(st)  # should not raise

    def test_missing_key_raises(self):
        st = {'Sigma_XX': None, 'Sigma_YY': None}
        with pytest.raises(ValueError, match="Missing keys"):
            _check_st(st)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            _check_st({})


class TestCheckNeg:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="negative pair statistics"):
            _check_neg(None)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing neg keys"):
            _check_neg({'wrong_key': None})

    def test_valid_neg(self):
        neg = {'Sigma_cross_neg': torch.zeros(4, 4)}
        _check_neg(neg)  # should not raise


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_output_keys(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group)
        expected = {'Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean', 'n_pairs'}
        assert set(st.keys()) == expected

    def test_matrix_shapes(self, simple_data):
        embs, ids, id_to_group = simple_data
        d = embs.shape[1]
        st = compute_stats(embs, ids, id_to_group)
        for key in ['Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross']:
            assert st[key].shape == (d, d), f"{key} has wrong shape"
        assert st['X_mean'].shape == (d,)

    def test_sigma_total_is_sum(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group)
        diff = (st['Sigma_total'] - (st['Sigma_XX'] + st['Sigma_YY'])).abs().max()
        assert diff < 1e-7

    def test_sigma_cross_is_symmetric(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group)
        diff = (st['Sigma_cross'] - st['Sigma_cross'].T).abs().max()
        assert diff < 1e-7

    def test_symmetrize_sxx_equals_syy(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group, symmetrize=True)
        diff = (st['Sigma_XX'] - st['Sigma_YY']).abs().max()
        assert diff < 1e-6

    def test_no_symmetrize_sxx_different_syy(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group, symmetrize=False)
        # Without symmetrization, SXX and SYY are generally different
        diff = (st['Sigma_XX'] - st['Sigma_YY']).abs().max()
        assert diff > 1e-5

    def test_n_pairs_with_symmetrize(self, simple_data):
        embs, ids, id_to_group = simple_data
        st_sym = compute_stats(embs, ids, id_to_group, symmetrize=True)
        st_no = compute_stats(embs, ids, id_to_group, symmetrize=False)
        assert st_sym['n_pairs'] == 2 * st_no['n_pairs']

    def test_device_and_dtype(self, simple_data):
        embs, ids, id_to_group = simple_data
        st = compute_stats(embs, ids, id_to_group)
        for key in ['Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean']:
            assert st[key].device == DEVICE
            assert st[key].dtype == DTYPE

    def test_single_group_no_pairs(self):
        """Each item in its own group produces no pairs."""
        torch.manual_seed(0)
        embs = torch.randn(5, 4, device=DEVICE, dtype=DTYPE)
        ids = list(range(5))
        id_to_group = {i: i for i in range(5)}  # each in own group
        st = compute_stats(embs, ids, id_to_group)
        # No pairs → n_pairs should be 0 and matrices should be zero
        assert st['n_pairs'] == 0


# ---------------------------------------------------------------------------
# compute_all_stats
# ---------------------------------------------------------------------------


class TestComputeAllStats:
    def test_returns_st_and_neg(self, simple_data):
        embs, ids, id_to_group = simple_data
        st, neg = compute_all_stats(embs, ids, id_to_group)
        _check_st(st)
        _check_neg(neg)

    def test_neg_has_sigma_cross_neg(self, simple_data):
        embs, ids, id_to_group = simple_data
        _, neg = compute_all_stats(embs, ids, id_to_group)
        assert 'Sigma_cross_neg' in neg
        d = embs.shape[1]
        assert neg['Sigma_cross_neg'].shape == (d, d)

    def test_neg_sigma_cross_is_symmetric(self, simple_data):
        embs, ids, id_to_group = simple_data
        _, neg = compute_all_stats(embs, ids, id_to_group)
        diff = (neg['Sigma_cross_neg'] - neg['Sigma_cross_neg'].T).abs().max()
        assert diff < 1e-6

    def test_st_matches_compute_stats(self, simple_data):
        embs, ids, id_to_group = simple_data
        st_direct = compute_stats(embs, ids, id_to_group)
        st_all, _ = compute_all_stats(embs, ids, id_to_group)
        for key in ['Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'X_mean']:
            diff = (st_direct[key] - st_all[key]).abs().max()
            assert diff < 1e-7, f"{key} differs"
