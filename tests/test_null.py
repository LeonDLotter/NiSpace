"""Tests for null map generation functions:
- nulls_random (moved from test_parcellation_null.py)
- nulls_moran
- nulls_variomoran
- compute_mem / moran_randomization (from _brainspace_moran.py)

All tests use synthetic data only — no data-repo access, no external BrainSpace
dependency (MoranRandomization is a bundled pure-numpy/scipy copy).
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from nispace.nulls import (
    nulls_random, nulls_moran, nulls_variomoran,
    nulls_burt2018, nulls_burt2020,
    _BRAINSMASH_AVAILABLE,
    generate_null_maps,
)
from nispace._brainspace_moran import compute_mem, moran_randomization

requires_brainsmash = pytest.mark.skipif(
    not _BRAINSMASH_AVAILABLE, reason="brainsmash not installed"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_dist_mat():
    """20-point 1D Euclidean distance matrix — small, well-conditioned."""
    pts = np.linspace(0, 1, 20)[:, None]
    return cdist(pts, pts).astype(np.float64)


@pytest.fixture
def smooth_map(linear_dist_mat):
    """Spatially smooth 1D map: sin wave over 20 points, low SA."""
    return np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)


@pytest.fixture
def white_noise_map(rng):
    """Spatially uncorrelated 20-point map (near-zero Moran's I)."""
    return rng.standard_normal(20).astype(np.float32)


# ---------------------------------------------------------------------------
# nulls_random
# ---------------------------------------------------------------------------

def test_nulls_random_output_shape(rng):
    x = rng.standard_normal(50).astype(np.float32)
    nulls = nulls_random(x, n_nulls=30, seed=0)
    assert nulls.shape == (30, 50)


def test_nulls_random_each_row_is_permutation(rng):
    x = rng.standard_normal(40).astype(np.float32)
    nulls = nulls_random(x, n_nulls=20, seed=1)
    x_sorted = np.sort(x)
    for row in nulls:
        np.testing.assert_array_equal(np.sort(row), x_sorted)


def test_nulls_random_nan_positions_preserved():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan], dtype=np.float32)
    nulls = nulls_random(x, n_nulls=10, seed=0)
    assert nulls.shape == (10, 6)
    assert np.all(np.isnan(nulls[:, 2]))
    assert np.all(np.isnan(nulls[:, 5]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 3, 4]]))


def test_nulls_random_reproducible_with_same_seed(rng):
    x = rng.standard_normal(30).astype(np.float32)
    a = nulls_random(x, n_nulls=10, seed=99)
    b = nulls_random(x, n_nulls=10, seed=99)
    np.testing.assert_array_equal(a, b)


def test_nulls_random_different_seeds_differ(rng):
    x = rng.standard_normal(30).astype(np.float32)
    a = nulls_random(x, n_nulls=10, seed=1)
    b = nulls_random(x, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_nulls_random_ignores_dist_mat(rng):
    x = rng.standard_normal(20).astype(np.float32)
    dm = rng.random((20, 20)).astype(np.float32)
    a = nulls_random(x, dist_mat=None, n_nulls=5, seed=7)
    b = nulls_random(x, dist_mat=dm,   n_nulls=5, seed=7)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# compute_mem
# ---------------------------------------------------------------------------

def test_compute_mem_output_shapes(linear_dist_mat):
    n = linear_dist_mat.shape[0]
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem, ev = compute_mem(W)
    assert mem.ndim == 2
    assert mem.shape[0] == n
    assert ev.ndim == 1
    assert mem.shape[1] == ev.shape[0]


def test_compute_mem_eigenvalues_descending(linear_dist_mat):
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    _, ev = compute_mem(W)
    assert np.all(np.diff(ev) <= 0), "Eigenvalues must be in descending order"


def test_compute_mem_truncated_matches_full_top_k(linear_dist_mat):
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem_full, ev_full = compute_mem(W)
    k = 5
    mem_trunc, ev_trunc = compute_mem(W, n_components=k)
    assert mem_trunc.shape[1] <= k
    # top-k eigenvalues should match (up to sign flip on eigenvectors)
    np.testing.assert_allclose(
        np.abs(ev_trunc), np.abs(ev_full[:k]), rtol=1e-3
    )


def test_compute_mem_sparse_requires_n_components(linear_dist_mat):
    import scipy.sparse as ssp
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    W_sparse = ssp.csr_matrix(W)
    with pytest.raises(ValueError, match="n_components"):
        compute_mem(W_sparse)


def test_compute_mem_sparse_with_n_components(linear_dist_mat):
    import scipy.sparse as ssp
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    W_sparse = ssp.csr_matrix(W)
    mem, ev = compute_mem(W_sparse, n_components=5)
    assert mem.shape[0] == linear_dist_mat.shape[0]
    assert mem.shape[1] == ev.shape[0]
    assert mem.shape[1] <= 5


# ---------------------------------------------------------------------------
# moran_randomization
# ---------------------------------------------------------------------------

def _make_W_mem(dist_mat):
    dm = np.where(dist_mat == 0, np.inf, dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem, mev = compute_mem(W, n_components=10)
    return mem, mev


def test_moran_randomization_output_shape(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=25, seed=0)
    assert out.shape == (25, len(smooth_map))


def test_moran_randomization_reproducible(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    a = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=5)
    b = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=5)
    np.testing.assert_array_equal(a, b)


def test_moran_randomization_different_seeds_differ(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    a = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=1)
    b = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_moran_randomization_pair_procedure_shape(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=10,
                               procedure="pair", seed=0)
    assert out.shape == (10, len(smooth_map))


def test_moran_randomization_invalid_procedure(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    with pytest.raises(ValueError, match="procedure"):
        moran_randomization(smooth_map, mem, mev, procedure="unknown")


def test_moran_randomization_singleton_mean_approximately_preserved(
        linear_dist_mat, smooth_map):
    # Singleton flips only signs of MEM coefficients — mean and variance of
    # the distribution of null means should be close to the original mean.
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=200, seed=0)
    null_means = out.mean(axis=1)
    np.testing.assert_allclose(null_means.mean(), smooth_map.mean(), atol=0.3)


def test_moran_randomization_joint_output_shape(linear_dist_mat):
    rng = np.random.default_rng(0)
    x2d = rng.standard_normal((20, 3)).astype(np.float32)
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(x2d, mem, mev, n_nulls=10, joint=True, seed=0)
    assert out.shape == (10, 20, 3)


# ---------------------------------------------------------------------------
# nulls_moran
# ---------------------------------------------------------------------------

def test_nulls_moran_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert nulls.shape == (20, len(smooth_map))


def test_nulls_moran_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_moran_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[3] = np.nan
    x[15] = np.nan
    nulls = nulls_moran(x, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, 20)
    assert np.all(np.isnan(nulls[:, 3]))
    assert np.all(np.isnan(nulls[:, 15]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 4]]))


def test_nulls_moran_reproducible(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=7)
    b = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=7)
    np.testing.assert_array_equal(a, b)


def test_nulls_moran_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=1)
    b = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_nulls_moran_nulls_differ_from_input(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert not np.all(nulls == smooth_map)


# ---------------------------------------------------------------------------
# nulls_variomoran
# ---------------------------------------------------------------------------

def test_nulls_variomoran_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert nulls.shape == (20, len(smooth_map))


def test_nulls_variomoran_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_variomoran_reproducible(linear_dist_mat, smooth_map):
    a = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=3)
    b = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=3)
    np.testing.assert_array_equal(a, b)


def test_nulls_variomoran_fallback_for_white_noise(linear_dist_mat, white_noise_map):
    # White noise has near-zero Moran's I → should fall back to standard 1/d W.
    # Smoke test: must complete without error and return correct shape.
    nulls = nulls_variomoran(
        white_noise_map, linear_dist_mat,
        n_nulls=10, seed=0,
        variogram_threshold=0.5,  # force fallback for near-zero SA
    )
    assert nulls.shape == (10, len(white_noise_map))
    assert np.all(np.isfinite(nulls))


def test_nulls_variomoran_and_moran_same_shape(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=15, seed=0)
    b = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=15, seed=0)
    assert a.shape == b.shape


# ---------------------------------------------------------------------------
# nulls_burt2018 (neuromaps.nulls.batch_surrogates — always available)
# ---------------------------------------------------------------------------

def test_nulls_burt2018_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, len(smooth_map))


def test_nulls_burt2018_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_burt2018_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[5] = np.nan
    nulls = nulls_burt2018(x, linear_dist_mat, n_nulls=5, seed=0)
    assert nulls.shape == (5, 20)
    assert np.all(np.isnan(nulls[:, 5]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 3, 4]]))


def test_nulls_burt2018_reproducible(linear_dist_mat, smooth_map):
    a = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=11)
    b = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=11)
    np.testing.assert_array_equal(a, b)


def test_nulls_burt2018_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=1)
    b = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# nulls_burt2020 (brainsmash — optional, skip if absent)
# ---------------------------------------------------------------------------

@requires_brainsmash
def test_nulls_burt2020_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, len(smooth_map))


@requires_brainsmash
def test_nulls_burt2020_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


@requires_brainsmash
def test_nulls_burt2020_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[7] = np.nan
    nulls = nulls_burt2020(x, linear_dist_mat, n_nulls=5, seed=0)
    assert nulls.shape == (5, 20)
    assert np.all(np.isnan(nulls[:, 7]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 3, 4]]))


@requires_brainsmash
def test_nulls_burt2020_reproducible(linear_dist_mat, smooth_map):
    a = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=13)
    b = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=13)
    np.testing.assert_array_equal(a, b)


@requires_brainsmash
def test_nulls_burt2020_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=1)
    b = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# generate_null_maps: requirement-check regression tests
# ---------------------------------------------------------------------------

def test_generate_null_maps_random_no_parc_no_distmat(rng):
    # Bug fix: method="random" needs no parcellation and no dist_mat
    data = rng.standard_normal(20).astype(np.float32)
    nulls, result_mat = generate_null_maps(
        method="random",
        data=data,
        parcellation=None,
        dist_mat=None,
        n_nulls=10,
        seed=0,
        verbose=False,
    )
    assert nulls.shape == (1, 10, 20)
    assert result_mat is None


def test_generate_null_maps_random_with_tuple_distmat_does_not_crash(rng):
    # Bug fix: method="random" + tuple dist_mat should not crash (dist_mat is ignored)
    pts = np.linspace(0, 1, 10)[:, None]
    dm = cdist(pts, pts).astype(np.float64)
    data = rng.standard_normal(20).astype(np.float32)
    nulls, _ = generate_null_maps(
        method="random",
        data=data,
        parcellation=None,
        dist_mat=(dm, dm),  # tuple — was crashing before the fix
        n_nulls=5,
        seed=0,
        verbose=False,
    )
    assert nulls.shape == (1, 5, 20)


def test_generate_null_maps_none_parc_none_distmat_raises_clearly(rng):
    # Bug fix: parcellation=None + dist_mat=None for distance method must raise ValueError
    # (not an opaque TypeError "data type not defined")
    data = rng.standard_normal(20).astype(np.float32)
    with pytest.raises(ValueError, match="parcellation.*dist_mat.*None"):
        generate_null_maps(
            method="moran",
            data=data,
            parcellation=None,
            dist_mat=None,
            n_nulls=5,
            verbose=False,
        )
