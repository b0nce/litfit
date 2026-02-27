"""Tests for compute_stats_streaming."""

import warnings
from collections import defaultdict
from unittest.mock import patch

import pytest
import torch

from litfit.device import DEVICE, DTYPE, to_torch
from litfit.stats import (
    _needs_fp64,
    compute_stats,
    compute_stats_streaming,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pairs(embs, ids, id_to_group):
    """Reproduce the pair-building logic from compute_stats."""
    embs_t = to_torch(embs)
    g2i = defaultdict(list)
    for i, qid in enumerate(ids):
        g2i[id_to_group[qid]].append(i)
    xs, ys = [], []
    for indices in g2i.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                xs.append(indices[i])
                ys.append(indices[j])
    return embs_t[xs], embs_t[ys]


def _batched_iter(X, Y, batch_size):
    """Yield (X_batch, Y_batch) chunks."""
    for start in range(0, X.shape[0], batch_size):
        yield X[start : start + batch_size], Y[start : start + batch_size]


# Fixed test data
_SEED = 42
_D = 8
_N = 20
_N_GROUPS = 5


@pytest.fixture()
def pair_data():
    torch.manual_seed(_SEED)
    embs = torch.randn(_N, _D)
    ids = list(range(_N))
    id_to_group = {i: i // (_N // _N_GROUPS) for i in range(_N)}
    X, Y = _build_pairs(embs, ids, id_to_group)
    return embs, ids, id_to_group, X, Y


# ---------------------------------------------------------------------------
# Tests: matches compute_stats
# ---------------------------------------------------------------------------

STAT_KEYS = ['Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean']
TOL = 1e-5


@pytest.mark.parametrize("symmetrize", [True, False])
def test_matches_compute_stats(pair_data, symmetrize):
    embs, ids, id_to_group, X, Y = pair_data
    ref = compute_stats(embs, ids, id_to_group, symmetrize=symmetrize)
    streaming = compute_stats_streaming(_batched_iter(X, Y, batch_size=5), symmetrize=symmetrize)
    for key in STAT_KEYS:
        diff = (ref[key] - streaming[key]).abs().max().item()
        assert diff < TOL, f"{key}: max_diff={diff:.2e} (symmetrize={symmetrize})"
    assert ref['n_pairs'] == streaming['n_pairs']


@pytest.mark.parametrize("symmetrize", [True, False])
def test_single_batch_matches(pair_data, symmetrize):
    """Streaming with one big batch should be identical to multi-batch."""
    embs, ids, id_to_group, X, Y = pair_data
    ref = compute_stats(embs, ids, id_to_group, symmetrize=symmetrize)
    streaming = compute_stats_streaming(_batched_iter(X, Y, batch_size=X.shape[0]), symmetrize=symmetrize)
    for key in STAT_KEYS:
        diff = (ref[key] - streaming[key]).abs().max().item()
        assert diff < TOL, f"{key}: max_diff={diff:.2e}"


@pytest.mark.parametrize("batch_size", [1, 3, 7, 30])
def test_various_batch_sizes(pair_data, batch_size):
    embs, ids, id_to_group, X, Y = pair_data
    ref = compute_stats(embs, ids, id_to_group, symmetrize=True)
    streaming = compute_stats_streaming(_batched_iter(X, Y, batch_size=batch_size), symmetrize=True)
    for key in STAT_KEYS:
        diff = (ref[key] - streaming[key]).abs().max().item()
        assert diff < TOL, f"{key}: max_diff={diff:.2e} (batch_size={batch_size})"


# ---------------------------------------------------------------------------
# Tests: output format
# ---------------------------------------------------------------------------


def test_output_keys(pair_data):
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
    expected_keys = {'Sigma_XX', 'Sigma_YY', 'Sigma_XY', 'Sigma_total', 'Sigma_cross', 'X_mean', 'n_pairs'}
    assert set(result.keys()) == expected_keys


def test_output_dtype(pair_data):
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
    for key in STAT_KEYS:
        assert result[key].dtype == DTYPE, f"{key} has dtype {result[key].dtype}"


def test_output_device(pair_data):
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
    for key in STAT_KEYS:
        assert result[key].device == DEVICE, f"{key} on {result[key].device}"


def test_n_pairs_symmetrize(pair_data):
    _, _, _, X, Y = pair_data
    n = X.shape[0]
    sym = compute_stats_streaming(_batched_iter(X, Y, batch_size=5), symmetrize=True)
    nosym = compute_stats_streaming(_batched_iter(X, Y, batch_size=5), symmetrize=False)
    assert sym['n_pairs'] == 2 * n
    assert nosym['n_pairs'] == n


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_empty_iterator_raises():
    with pytest.raises(ValueError, match="no data"):
        compute_stats_streaming(iter([]))


def test_single_pair():
    X = torch.randn(1, 4)
    Y = torch.randn(1, 4)
    result = compute_stats_streaming(iter([(X, Y)]), symmetrize=False)
    assert result['n_pairs'] == 1
    assert result['Sigma_XX'].shape == (4, 4)


def test_numpy_input():
    """Iterator can yield numpy arrays."""
    import numpy as np

    rng = np.random.RandomState(0)
    X = rng.randn(10, 4).astype(np.float32)
    Y = rng.randn(10, 4).astype(np.float32)

    def pair_iter():
        for i in range(0, 10, 5):
            yield X[i : i + 5], Y[i : i + 5]

    result = compute_stats_streaming(pair_iter())
    assert result['n_pairs'] == 20  # symmetrize=True doubles
    for key in STAT_KEYS:
        assert result[key].dtype == DTYPE


# ---------------------------------------------------------------------------
# Tests: symmetry properties
# ---------------------------------------------------------------------------


def test_sigma_total_is_sum(pair_data):
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
    diff = (result['Sigma_total'] - (result['Sigma_XX'] + result['Sigma_YY'])).abs().max()
    assert diff < 1e-7


def test_sigma_cross_is_symmetric(pair_data):
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
    diff = (result['Sigma_cross'] - result['Sigma_cross'].T).abs().max()
    assert diff < 1e-7


def test_symmetrize_sxx_equals_syy(pair_data):
    """With symmetrize=True, SXX and SYY should be identical."""
    _, _, _, X, Y = pair_data
    result = compute_stats_streaming(_batched_iter(X, Y, batch_size=5), symmetrize=True)
    diff = (result['Sigma_XX'] - result['Sigma_YY']).abs().max()
    assert diff == 0.0


# ---------------------------------------------------------------------------
# Tests: fp64 upcast and warning
# ---------------------------------------------------------------------------


def test_needs_fp64():
    assert not _needs_fp64(1000, torch.float32)
    assert not _needs_fp64(10**6, torch.float32)
    # float32 max exponent ~3.4e38; 1/n underflows for astronomically large n
    # but we can test the helper directly with float16 where range is small
    assert _needs_fp64(10**6, torch.float16)


def test_fp64_upcast_warns(pair_data):
    """When _needs_fp64 returns True, a warning should be emitted."""
    _, _, _, X, Y = pair_data
    with patch('litfit.stats._needs_fp64', return_value=True):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
            precision_warnings = [x for x in w if "underflows" in str(x.message)]
            assert len(precision_warnings) == 1


def test_no_warning_normal_n(pair_data):
    """Normal-sized data should not trigger the precision warning."""
    _, _, _, X, Y = pair_data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_stats_streaming(_batched_iter(X, Y, batch_size=5))
        precision_warnings = [x for x in w if "underflows" in str(x.message)]
        assert len(precision_warnings) == 0
